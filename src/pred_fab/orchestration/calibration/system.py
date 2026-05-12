from dataclasses import dataclass
from typing import Any, Callable
import functools

import numpy as np
import torch

from ...core import DataModule, Dataset, DatasetSchema
from ...core import DataInt, DataObject, DataBool, DataCategorical, DataDomainAxis
from ...core import ParameterProposal, ParameterTrajectory, ExperimentSpec
from ...utils import PfabLogger, Mode, NormMethod, SourceStep, SplitType, combined_score, profiler
from ..base_system import BaseOrchestrationSystem
from .engine import OptimizationEngine, _OptResult
from .bounds import BoundsManager
from .space import SolutionSpace

@dataclass
class EvidenceBackend:
    """Δ∫E callbacks the acquisition objective dispatches to.

    ``batched_tensor`` covers per-candidate single-point Δ∫E; ``joint_batched_tensor``
    covers per-candidate joint multi-point (trajectory / N-batch) Δ∫E. Both
    are gradient-traversable. Either may be ``None`` when unavailable —
    the κ-blend skips that arm.
    """
    batched_tensor: Callable[..., torch.Tensor] | None = None
    joint_batched_tensor: Callable[..., torch.Tensor] | None = None


# ======================================================================
# CalibrationSystem — orchestrator, composes the other two
# ======================================================================

class CalibrationSystem(BaseOrchestrationSystem):
    """Active-learning calibration engine: acquisition-driven exploration, inference, batch baseline, and joint schedule optimization."""

    def __init__(
        self,
        schema: DatasetSchema,
        logger: PfabLogger,
        uncertainty_fn: Callable[[np.ndarray], float],
        *,
        evidence: EvidenceBackend | None = None,
        n_exp_fn: Callable[[], int] | None = None,
        fit_empty_kde_fn: Callable[[DataModule, int], None] | None = None,
        push_virtual_fn: Callable[[list[dict[str, Any]], list[float] | None, DataModule | None], None] | None = None,
        pop_virtual_fn: Callable[[], None] | None = None,
        perf_fn_tensor: Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]] | None = None,
        random_seed: int | None = None,
    ):
        super().__init__(logger, random_seed=random_seed)
        self.perf_fn_tensor = perf_fn_tensor
        self.uncertainty_fn = uncertainty_fn
        self.evidence = evidence if evidence is not None else EvidenceBackend()
        self._n_exp_fn = n_exp_fn
        self._fit_empty_kde_fn = fit_empty_kde_fn
        self._push_virtual_fn = push_virtual_fn
        self._pop_virtual_fn = pop_virtual_fn

        # Composed subsystems
        self.engine = OptimizationEngine(logger, random_seed=random_seed)
        self.bounds = BoundsManager(schema, logger)

        # Active datamodule — set before each optimization run so that
        # _acquisition_objective can call array_to_params.
        self._active_datamodule: DataModule | None = None

        # Set after each optimization call for external inspection.
        self.last_opt_nfev: int = 0
        self.last_opt_n_starts: int = 0
        self.last_opt_score: float = 0.0
        self.last_opt_perf: float = 0.0
        self.last_opt_unc: float = 0.0
        self.last_trajectory: list[dict[str, Any]] | None = None
        self.convergence_history: dict[str, list[float]] = {}  # label → per-iteration convergence
        # Phase data for validation plots
        self.last_domain_values: list[dict[str, int]] | None = None
        self.last_process_points: list[dict[str, Any]] | None = None
        self.last_trajectory_points: np.ndarray | None = None
        self.last_trajectory_exp_ids: list[int] | None = None
        self.acquisition_scale: float = 1000.0
        self.post_global_callback: Callable[[list[ExperimentSpec]], None] | None = None
        self.derive_L_fn: Callable[[dict[str, Any]], int] | None = None

        # Set ordered weights
        self.schema = schema
        self.perf_names_order = list(schema.performance_attrs.keys())
        self.performance_weights: dict[str, float] = {perf: 1.0 for perf in self.perf_names_order}
        self.parameters = schema.parameters


        # Persistent κ default for acquisition_step / exploration_step. Overridable
        # per call. inference_step ignores it (κ=0 is the inference semantic).
        self.kappa_default: float = 0.5

        # Running min/max of predicted system performance across training data.
        self._perf_range_min: float | None = None
        self._perf_range_max: float | None = None

        self._schedule_joint_var_limit: int = 200  # threshold for auto-selecting joint vs sequential
        self._suppress_opt_print: bool = False

    # ==================================================================
    # § Properties — proxy to BoundsManager
    # ==================================================================

    @property
    def data_objects(self) -> dict[str, DataObject]:
        return self.bounds.data_objects

    @property
    def schema_bounds(self) -> dict[str, tuple[float, float]]:
        return self.bounds.schema_bounds

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return self.bounds.param_bounds

    @property
    def fixed_params(self) -> dict[str, Any]:
        return self.bounds.fixed_params

    @property
    def trust_regions(self) -> dict[str, float]:
        return self.bounds.trust_regions

    @property
    def trajectory_configs(self) -> dict[str, str]:
        return self.bounds.trajectory_configs

    # ==================================================================
    # § Configuration
    # ==================================================================

    @property
    def random_seed(self) -> int | None:
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value: int | None) -> None:
        self._random_seed = value
        self.rng = np.random.RandomState(value)
        self.engine._random_seed = value
        self.engine.rng = np.random.RandomState(value)

    # ==================================================================
    # § Delegated config methods
    # ==================================================================

    def configure_param_bounds(self, bounds: dict[str, tuple[float, float]], force: bool = False) -> None:
        """Configure parameter ranges for offline calibration."""
        self.bounds.configure_param_bounds(bounds, force=force)

    def configure_fixed_params(self, fixed_params: dict[str, Any], force: bool = False) -> None:
        """Configure fixed parameter values."""
        self.bounds.configure_fixed_params(fixed_params, force=force)

    def configure_adaptation_delta(self, deltas: dict[str, float], force: bool = False) -> None:
        """Configure trust region deltas for online calibration."""
        self.bounds.configure_adaptation_delta(deltas, force=force)

    def configure_trajectory_parameter(self, code: str, dimension_code: str, force: bool = False) -> None:
        """Declare that a runtime-adjustable parameter should be re-optimised at each step of the given dimension."""
        self.bounds.configure_trajectory_parameter(code, dimension_code, force=force)

    def get_tunable_params(self, datamodule: DataModule) -> list[str]:
        """Return codes of parameters the optimizer can actually vary."""
        return self.bounds.get_tunable_params(datamodule)

    def _get_hierarchical_bounds_for_code(self, code: str) -> tuple[float, float]:
        return self.bounds._get_hierarchical_bounds_for_code(code)

    def _get_global_bounds(self, datamodule: DataModule) -> np.ndarray:
        return self.bounds._get_global_bounds(datamodule)

    def _get_trust_region_bounds(self, datamodule: DataModule, current_params: dict[str, Any]) -> np.ndarray:
        return self.bounds._get_trust_region_bounds(datamodule, current_params)

    def _get_n_exp(self) -> int:
        """Current experiment count from the prediction system."""
        if self._n_exp_fn is not None:
            return self._n_exp_fn()
        return 1

    def state_report(self) -> None:
        """Log the current calibration configuration state."""
        _B = "\033[1m"
        _D = "\033[2m"
        _R = "\033[0m"

        lines = [f"\n  {_B}Calibration{_R}"]

        pw_parts = [f"{k}={v:g}" for k, v in self.performance_weights.items()]
        lines.append(f"    {_D}Weights: {', '.join(pw_parts)}{_R}")


        lines.append(f"\n    {_D}{'Parameter':<20s} {'Bounds':<20s} {'Delta':<8s}{_R}")
        for code in self.data_objects.keys():
            low, high = self.bounds._get_hierarchical_bounds_for_code(code)
            bounds_str = f"[{low}, {high}]"
            delta = self.trust_regions.get(code, "\u2500")
            lines.append(f"    {code:<20s} {bounds_str:<20s} {delta:<8}")

        self.logger.console_summary("\n".join(lines))
        self.logger.console_new_line()

    def set_performance_weights(self, weights: dict[str, float]) -> None:
        """Set weights for system performance calculation. Default is 1.0 for all."""
        for name, value in weights.items():
            if name in self.performance_weights:
                self.performance_weights[name] = value
                self.logger.debug(f"Set performance weight: {name} -> {value}")
            else:
                self.logger.console_warning(f"Performance attribute '{name}' not in schema; ignoring weight.")

    # ==================================================================
    # § Acquisition objectives — κ-blended evidence + performance
    # ==================================================================
    #
    # Unified acquisition (higher is better):
    #
    #     score(batch) = (1 − κ) · mean_k combined_score(perf(z_k), w)
    #                   + κ · Δ∫_[0,1]^D E(z | data_old ∪ batch) − E(z | data_old) dz
    #
    # κ=1 baseline drops the perf term; κ=0 inference drops the Δ∫E term.
    # Single-candidate phases pass batch with shape (1, D).

    def _compute_perf_dict_for_params(self, params: dict[str, Any]) -> dict[str, float | None]:
        """Compute a single-candidate ``{perf_code: float}`` via the tensor perf closure."""
        if self.perf_fn_tensor is None:
            return {}
        try:
            with torch.no_grad():
                t_out = self.perf_fn_tensor([params])
        except Exception:
            return {}
        out: dict[str, float | None] = {}
        for code, t in t_out.items():
            v = float(t[0].item())
            out[code] = None if v != v else v  # NaN → None
        return out

    def _compute_normalised_perf_for_params(
        self,
        params: dict[str, Any],
        perf_range: tuple[float, float] | None = None,
    ) -> float:
        """Predict + evaluate single-candidate weighted system perf; normalise into ``[0, 1]`` if ``perf_range`` is given."""
        perf_dict = self._compute_perf_dict_for_params(params)
        return self._normalize_perf_dict(perf_dict, perf_range)

    @property
    def _perf_range(self) -> tuple[float, float] | None:
        """Tuple form of ``(_perf_range_min, _perf_range_max)`` if both are set, else None."""
        if self._perf_range_min is None or self._perf_range_max is None:
            return None
        return (self._perf_range_min, self._perf_range_max)

    def _per_candidate_perf_batched(
        self,
        X_SD: np.ndarray,
        perf_range: tuple[float, float] | None,
    ) -> np.ndarray:
        """No-grad numpy shim around `_per_candidate_perf_tensor`."""
        S = X_SD.shape[0]
        if S == 0:
            return np.empty((0,), dtype=np.float64)
        with torch.no_grad():
            out_t = self._per_candidate_perf_tensor(
                torch.from_numpy(np.ascontiguousarray(X_SD)).double(), perf_range,
            )
        return out_t.detach().cpu().numpy().astype(np.float64)

    def _normalize_perf_dict(
        self,
        perf_dict: dict[str, float | None],
        perf_range: tuple[float, float] | None,
    ) -> float:
        """Combine per-feature perf into a system score, then normalise into [0, 1]."""
        perf_values = [
            float(perf_dict[name]) if perf_dict.get(name) is not None else 0.0  # type: ignore[arg-type]
            for name in self.perf_names_order
            if name in perf_dict
        ]
        sys_perf = self._compute_system_performance(perf_values) if perf_values else 0.0
        if perf_range is not None:
            pmin, pmax = perf_range
            span = pmax - pmin
            sys_perf = (sys_perf - pmin) / span if span > 1e-10 else 0.5
        return float(sys_perf)

    def _acquisition_objective(
        self,
        x_flat: np.ndarray,
        kappa: float,
        perf_range: tuple[float, float] | None = None,
    ) -> float:
        """DE-compatible (negated) acquisition for single-candidate phases.

        Thin no-grad numpy shim around `_acquisition_objective_tensor` —
        wraps the 1-D x_flat as a (1, D) batch and pulls out the scalar.
        """
        with torch.no_grad():
            X_SD = torch.from_numpy(np.atleast_2d(x_flat)).double()
            out = self._acquisition_objective_tensor(X_SD, kappa, perf_range)
        return float(out[0].item())

    def _kappa_blend(self, scores, perfs, evidences, kappa: float):
        """Negated κ-weighted blend scaled by ``acquisition_scale``."""
        if perfs is not None and kappa < 1.0:
            scores = scores + (1.0 - kappa) * perfs
        if evidences is not None and kappa > 0.0:
            scores = scores + kappa * evidences
        return -scores * self.acquisition_scale

    def _per_candidate_perf_tensor(
        self,
        X_SD: torch.Tensor,
        perf_range: tuple[float, float] | None,
    ) -> torch.Tensor:
        """Per-candidate weighted performance ``(S,)``, gradient-traversable.

        Routes through ``perf_fn_tensor`` for autograd. When the tensor closure
        is unavailable (test fixtures) or raises, falls back to a scalar loop
        — gradient is lost on that path.
        """
        S = int(X_SD.shape[0])
        if S == 0:
            return torch.zeros(0, dtype=X_SD.dtype)
        dm = self._active_datamodule
        if dm is None:
            return torch.zeros(S, dtype=X_SD.dtype)
        if self.perf_fn_tensor is None:
            # No perf closure registered → 0 perf for all candidates.
            return torch.zeros(S, dtype=X_SD.dtype)

        # Build per-candidate params dicts. Continuous values stay as 0-D
        # tensors for autograd; categorical / int / domain values resolve
        # to concrete Python types at decode time (no grad through them).
        params_list: list[dict[str, Any]] = []
        for s in range(S):
            try:
                # array_to_params is numpy-typed — but we want tensor values
                # for continuous params. Strategy: build dict from numpy
                # decode for concrete int / cat resolution, then overwrite
                # continuous numerics with the differentiable tensor entries.
                p_np = dm.array_to_params(X_SD[s].detach().cpu().numpy().reshape(-1))
                p_with_grad = self._reattach_tensor_continuous(p_np, X_SD[s], dm)
                params_list.append(p_with_grad)
            except (ValueError, KeyError):
                params_list.append(None)  # type: ignore[arg-type]

        valid_idx = [i for i, p in enumerate(params_list) if p is not None]
        if not valid_idx:
            return torch.zeros(S, dtype=X_SD.dtype)

        try:
            perf_dict_S = self.perf_fn_tensor(
                [params_list[i] for i in valid_idx]  # type: ignore[index]
            )
        except Exception:
            return torch.zeros(S, dtype=X_SD.dtype)

        # System-perf aggregation: weighted sum across perf codes, normalised by perf_range.
        out_valid = torch.zeros(len(valid_idx), dtype=X_SD.dtype)
        weight_sum = sum(self.performance_weights.get(name, 1.0) for name in self.perf_names_order)
        for name in self.perf_names_order:
            if name not in perf_dict_S:
                continue
            w = float(self.performance_weights.get(name, 1.0))
            vals = perf_dict_S[name].to(dtype=X_SD.dtype)
            # NaN entries → 0 contribution (matches numpy nanmean handling).
            vals = torch.where(torch.isnan(vals), torch.zeros_like(vals), vals)
            out_valid = out_valid + (w / weight_sum) * vals
        if perf_range is not None:
            pmin, pmax = perf_range
            span = pmax - pmin
            if span > 1e-10:
                out_valid = (out_valid - float(pmin)) / float(span)
            else:
                out_valid = torch.full_like(out_valid, 0.5)

        out_full = torch.zeros(S, dtype=X_SD.dtype)
        for k, i in enumerate(valid_idx):
            out_full[i] = out_valid[k]
        return out_full

    def _reattach_tensor_continuous(
        self,
        params_np: dict[str, Any],
        x_norm: torch.Tensor,
        dm: DataModule,
    ) -> dict[str, Any]:
        """Replace continuous numeric values in ``params_np`` with their
        gradient-bearing counterparts decoded from ``x_norm``.

        ``array_to_params`` returns Python floats / ints — the gradient on
        ``x_norm`` is lost during that decode. For autograd, we preserve the
        tensor entries for continuous params by reading them off ``x_norm``
        directly via the schema's denormalisation map. Categorical / integer /
        domain params stay as Python (no gradient through discrete choices).
        """
        # Walk DataModule input columns; for each continuous one, replace the
        # decoded float with the gradient-traversable tensor value.
        out = dict(params_np)
        for j, col_name in enumerate(dm.input_columns):
            if col_name not in out:
                continue
            stats = getattr(dm, "_parameter_stats", {}).get(col_name)
            if stats is None:
                continue
            # Only continuous numeric stats define a min/max for affine reverse.
            lo = stats.get("min") if isinstance(stats, dict) else getattr(stats, "min", None)
            hi = stats.get("max") if isinstance(stats, dict) else getattr(stats, "max", None)
            if lo is None or hi is None:
                continue
            span = float(hi) - float(lo)
            if span <= 0:
                continue
            out[col_name] = x_norm[j] * span + float(lo)
        return out

    def _acquisition_objective_tensor(
        self,
        X_SD: torch.Tensor,
        kappa: float,
        perf_range: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Vectorised (negated) acquisition for the gradient optimiser.

        Takes ``(S, D)`` torch tensor — one row per candidate — and returns
        ``(S,)`` of negated scores (lower = better). Mirrors
        ``_acquisition_objective_vectorized`` but routes through the tensor
        APIs added in so gradient flows from each
        candidate's score back through the per-candidate parameters.
        """
        with profiler.section("acq._acquisition_objective_tensor"):
            S = int(X_SD.shape[0])

            perfs: torch.Tensor | None = None
            if kappa < 1.0:
                perfs = self._per_candidate_perf_tensor(X_SD, perf_range)

            evidences: torch.Tensor | None = None
            if kappa > 0.0 and self.evidence.batched_tensor is not None:
                evidences = self.evidence.batched_tensor(X_SD).to(dtype=X_SD.dtype)

            return self._kappa_blend(torch.zeros(S, dtype=X_SD.dtype), perfs, evidences, kappa)

    def _compute_system_performance(self, performance: list[float]) -> float:
        """Compute weighted system performance from an ordered list of scores."""
        if not performance:
            return 0.0
        perf_dict = {
            name: performance[i]
            for i, name in enumerate(self.perf_names_order)
            if i < len(performance)
        }
        return combined_score(perf_dict, self.performance_weights)

    # ==================================================================
    # § Helpers — spec construction, datamodule, perf range
    # ==================================================================

    def update_perf_range(self, datamodule: DataModule) -> None:
        """Update performance normalization range from training data.

        Called after each train(). Computes predicted performance at each
        training experiment. Uses raw min/max directly — no buffer.
        """
        self._active_datamodule = datamodule
        train_codes = datamodule.get_split_codes(SplitType.TRAIN)
        if not train_codes:
            return

        raw_min: float | None = None
        raw_max: float | None = None

        for code in train_codes:
            exp = datamodule.dataset.get_experiment(code)
            params_dict = exp.parameters.get_values_dict()
            try:
                sys_perf = self._compute_normalised_perf_for_params(params_dict)
            except Exception:
                continue

            if raw_min is None or sys_perf < raw_min:
                raw_min = sys_perf
            if raw_max is None or sys_perf > raw_max:
                raw_max = sys_perf

        if raw_min is not None and raw_max is not None:
            self._perf_range_min = raw_min
            self._perf_range_max = raw_max
            self.logger.debug(
                f"Performance range: [{raw_min:.3f}, {raw_max:.3f}]"
            )

    def _get_acquisition_ranges(self) -> tuple[tuple[float, float] | None, None]:
        """Return (perf_range, unc_range) for acquisition normalization."""
        if self._perf_range_min is not None and self._perf_range_max is not None:
            perf_range = (self._perf_range_min, self._perf_range_max)
        else:
            perf_range = None
        return perf_range, None

    def _build_objective(self, mode: Mode, kappa: float) -> Callable:
        """Return the objective function for the given calibration mode."""
        if mode == Mode.BASELINE:
            raise ValueError(
                "Mode.BASELINE uses run_baseline() — call calibration_system.run_baseline(n) directly."
            )
        # Inference is acquisition with kappa=0 (pure performance, no uncertainty)
        effective_kappa = 0.0 if mode == Mode.INFERENCE else kappa
        perf_range, _ = self._get_acquisition_ranges()
        return functools.partial(
            self._acquisition_objective,
            kappa=effective_kappa,
            perf_range=perf_range,
        )

    def _build_step_grid(self, current_params: dict[str, Any]) -> list[dict[str, int]]:
        """Build flattened Cartesian grid of {dim_code: step_index} dicts, coarsest dimension first.

        Returns [{}] when no schedule configs are configured (experiment-level use case).
        """
        import itertools as _it

        dim_key_order = {code: i for i, code in enumerate(self.data_objects.keys())}
        dim_codes = sorted(
            {dc for dc in self.trajectory_configs.values()},
            key=lambda dc: dim_key_order.get(dc, 999),
        )
        if not dim_codes:
            return [{}]

        sizes = [int(current_params[dc]) for dc in dim_codes]
        return [
            dict(zip(dim_codes, idx))
            for idx in _it.product(*(range(s) for s in sizes))
        ]

    def _build_experiment_spec(
        self,
        proposals: list[dict[str, Any]],
        step_grid: list[dict[str, int]],
        source_step: str,
    ) -> ExperimentSpec:
        """Assemble per-step proposals into an ExperimentSpec with initial_params and dimension schedules."""
        initial = ParameterProposal.from_dict(proposals[0], source_step=source_step)

        trajectories: dict[str, ParameterTrajectory] = {}
        if len(proposals) > 1 and step_grid and step_grid[0]:
            dim_key_order_spec = {code: i for i, code in enumerate(self.data_objects.keys())}
            dim_codes = sorted(
                {dc for dc in self.trajectory_configs.values()},
                key=lambda dc: dim_key_order_spec.get(dc, 999),
            )
            for dim_code in dim_codes:
                traj_codes = [
                    c for c, dc in self.trajectory_configs.items() if dc == dim_code
                ]
                entries: list[tuple[int, ParameterProposal]] = []
                prev_idx: int | None = None
                for flat_i, indices in enumerate(step_grid):
                    cur_idx = indices.get(dim_code, 0)
                    if flat_i == 0:
                        prev_idx = cur_idx
                        continue
                    if cur_idx != prev_idx:
                        seg_vals = {c: proposals[flat_i][c] for c in traj_codes if c in proposals[flat_i]}
                        if seg_vals:
                            entries.append((
                                cur_idx,
                                ParameterProposal.from_dict(seg_vals, source_step=source_step),
                            ))
                    prev_idx = cur_idx

                if entries:
                    trajectories[dim_code] = ParameterTrajectory(
                        dimension=dim_code, entries=entries,
                    )

        return ExperimentSpec(initial_params=initial, trajectories=trajectories)

    def _build_schema_datamodule(self) -> DataModule:
        """Build a schema-only DataModule (no training data) for baseline generation.

        Uses NormMethod.NONE; excludes params with infinite bounds (logged as warnings).
        """
        active_codes: list[str] = []
        for code, data_obj in self.data_objects.items():
            if code in self.fixed_params:
                continue
            if isinstance(data_obj, (DataCategorical, DataBool)):
                active_codes.append(code)
                continue
            try:
                lo, hi = self.bounds._get_hierarchical_bounds_for_code(code)
            except ValueError:
                continue
            if lo == -np.inf or hi == np.inf:
                self.logger.warning(
                    f"Parameter '{code}' has infinite bounds; "
                    "skipping in baseline generation."
                )
                continue
            active_codes.append(code)

        dataset = Dataset(schema=self.schema, debug_flag=True)
        datamodule = DataModule(dataset, normalize=NormMethod.NONE)
        datamodule.initialize(
            input_parameters=active_codes,
            input_features=[],
            output_columns=list(self.schema.performance_attrs.keys()),
        )
        datamodule.fit_without_data()
        return datamodule

    # ==================================================================
    # § Baseline — batch proposal generation (run_baseline)
    # ==================================================================

    def run_baseline(self, n: int) -> list["ExperimentSpec"]:
        """Generate n baseline proposals via batch acquisition (κ=1, joint over N points).

        Two- or three-phase optimization:
          Domain   — DataDomainAxis params only (when ``split_domain_phase`` is True)
          Process  — continuous + integer params (with domain values held if Domain ran)
          Schedule — per-layer offsets for scheduled params (if applicable)
        """
        if n == 0:
            return []

        # --- Collect parameters (ALL types, including domain axes) ---
        continuous_params: list[tuple[str, float, float]] = []
        integer_params: list[tuple[str, int, int]] = []
        categorical_params: list[tuple[str, list[Any]]] = []

        for code, data_obj in self.data_objects.items():
            if code in self.fixed_params:
                continue
            if isinstance(data_obj, DataCategorical):
                categorical_params.append((code, list(data_obj.constraints["categories"])))
            elif isinstance(data_obj, DataBool):
                categorical_params.append((code, [False, True]))
            elif isinstance(data_obj, (DataInt, DataDomainAxis)):
                try:
                    lo, hi = self.bounds._get_hierarchical_bounds_for_code(code)
                except ValueError:
                    continue
                if lo == -np.inf or hi == np.inf:
                    self.logger.warning(f"Parameter '{code}' has infinite bounds; skipping in baseline.")
                    continue
                if lo == hi:
                    # Degenerate bounds: structurally fixed. Pin to fixed_params
                    # so downstream spec construction picks it up; nothing for
                    # the optimiser to choose, no extra DE/KDE dimensionality.
                    self.bounds.fixed_params[code] = int(lo)
                    self.logger.debug(f"Auto-fixed integer param '{code}' = {int(lo)} (degenerate bounds).")
                    continue
                integer_params.append((code, int(lo), int(hi)))
            else:
                try:
                    lo, hi = self.bounds._get_hierarchical_bounds_for_code(code)
                except ValueError:
                    continue
                if lo == -np.inf or hi == np.inf:
                    self.logger.warning(f"Parameter '{code}' has infinite bounds; skipping in baseline.")
                    continue
                if lo == hi:
                    self.bounds.fixed_params[code] = float(lo)
                    self.logger.debug(f"Auto-fixed continuous param '{code}' = {lo} (degenerate bounds).")
                    continue
                continuous_params.append((code, lo, hi))

        numeric_params: list[tuple[str, float, float]] = [
            *continuous_params,
            *[(c, float(lo), float(hi)) for c, lo, hi in integer_params],
        ]
        n_numeric = len(numeric_params)

        if not numeric_params and not categorical_params:
            self.logger.warning("No valid parameters for baseline generation.")
            return []

        # --- Stratify categoricals ---
        if categorical_params:
            import itertools as _it
            cat_combos = list(_it.product(*(cats for _, cats in categorical_params)))
            cat_codes = [code for code, _ in categorical_params]
            n_strata = len(cat_combos)
        else:
            cat_combos = [()]
            cat_codes = []
            n_strata = 1

        n_per_stratum = [n // n_strata] * n_strata
        for i in range(n % n_strata):
            n_per_stratum[i] += 1

        cat_assignments: list[tuple[Any, ...]] = []
        for combo, n_i in zip(cat_combos, n_per_stratum):
            cat_assignments.extend([combo] * n_i)

        # --- Integer bookkeeping ---
        int_indices: list[int] = []
        int_ranges_map: dict[int, int] = {}
        for d_int, (_, lo, hi) in enumerate(integer_params):
            col = len(continuous_params) + d_int
            int_indices.append(col)
            int_ranges_map[col] = int(hi - lo)
        int_set = set(int_indices)

        # --- Initialize numeric positions via recursive bisection ---
        init_norm = np.zeros((n, n_numeric)) if n_numeric > 0 else np.zeros((n, 0))
        if n_numeric > 0:
            idx = 0
            for _combo, n_i in zip(cat_combos, n_per_stratum):
                if n_i == 0:
                    continue
                bounds_list = [(lo, hi) for _, lo, hi in numeric_params]
                original_ranges = [hi - lo for lo, hi in bounds_list]
                points = self._recursive_split(n_i, bounds_list, original_ranges)
                for point in points:
                    for d, (_, lo, hi) in enumerate(numeric_params):
                        span = hi - lo
                        init_norm[idx, d] = (point[d] - lo) / span if span > 0 else 0.5
                    idx += 1
            if n_strata > 1:
                jitter = self.engine.rng.normal(0, 0.02, size=init_norm.shape)
                init_norm = np.clip(init_norm + jitter, 0.01, 0.99)

        # --- Detect schedule config ---
        traj_set = set(self.trajectory_configs.keys())
        domain_axis_sched_dims: set[str] = set()

        if self.trajectory_configs:
            sched_dims = set(self.trajectory_configs.values())
            unfixed = [d for d in sched_dims if d not in self.fixed_params]
            if unfixed:
                domain_axis_sched_dims = {
                    d for d in unfixed
                    if d in self.data_objects and isinstance(self.data_objects[d], DataDomainAxis)
                }
                non_domain_unfixed = [d for d in unfixed if d not in domain_axis_sched_dims]
                if non_domain_unfixed:
                    self.logger.warning(
                        f"Schedule dimensions {sorted(non_domain_unfixed)} must be fixed to "
                        f"determine the number of steps. Call configure_fixed_params() for each. "
                        f"Proceeding without schedule for those dimensions."
                    )
                if not domain_axis_sched_dims:
                    traj_set = {
                        code for code in traj_set
                        if self.trajectory_configs[code] in self.fixed_params
                    }

        # --- Global: joint DE over all dims (domain + process) ---
        domain_axis_codes = {
            code for code, _, _ in numeric_params
            if code in self.data_objects and isinstance(self.data_objects[code], DataDomainAxis)
        }

        # Detect trajectory config early — determines whether we use slope path
        has_slope_traj = bool(traj_set)

        if numeric_params and not has_slope_traj:
            flat_specs, flat_params, optimized = self._run_acquisition_phase(
                n, numeric_params, integer_params, int_set, int_ranges_map,
                init_norm, cat_codes, cat_assignments,
                structural_values=None,
                label=f"Global (D={len(numeric_params)}, V={n * len(numeric_params)})", init_evidence=True,
            )
        elif numeric_params and has_slope_traj:
            # Single-pass: slopes handle everything — build warm start specs from bisection
            structural_values = None
            flat_specs = []
            for i in range(n):
                bp: dict[str, Any] = dict(self.fixed_params)
                for d, (code, lo, hi) in enumerate(numeric_params):
                    bp[code] = float(init_norm[i, d] * (hi - lo) + lo)
                for d_cat, code in enumerate(cat_codes):
                    bp[code] = cat_assignments[i][d_cat]
                for _, (code_i, lo_i, hi_i) in enumerate(integer_params):
                    if code_i in bp:
                        bp[code_i] = int(np.clip(np.round(bp[code_i]), lo_i, hi_i))
                bp = self.schema.parameters.sanitize_values(bp, ignore_unknown=True)
                flat_specs.append(ExperimentSpec(
                    initial_params=ParameterProposal.from_dict(bp, source_step=SourceStep.BASELINE),
                    trajectories={},
                ))
            flat_params = init_norm.copy()
            optimized = init_norm.copy()

        else:
            # No numeric params anywhere — fall back to centre-of-bounds + cats.
            optimized = np.zeros((n, 0))
            flat_params = np.zeros((n, 0))
            flat_specs = []
            for i in range(n):
                params: dict[str, Any] = dict(self.fixed_params)
                for d, (code, lo, hi) in enumerate(numeric_params):
                    params[code] = (lo + hi) / 2.0
                for d_cat, code in enumerate(cat_codes):
                    params[code] = cat_assignments[i][d_cat]
                params = self.schema.parameters.sanitize_values(params, ignore_unknown=True)
                proposal = ParameterProposal.from_dict(params, source_step=SourceStep.BASELINE)
                flat_specs.append(ExperimentSpec(initial_params=proposal, trajectories={}))

        # Extract domain axis values from Global results (needed by Trajectory)
        structural_values: list[dict[str, int]] | None = None
        if domain_axis_codes and flat_specs:
            structural_values = []
            for spec in flat_specs:
                p = spec.initial_params.to_dict()
                structural_values.append({c: int(p[c]) for c in domain_axis_codes if c in p})

        # Store phase points for validation plot
        self.last_process_points = [spec.initial_params.to_dict() for spec in flat_specs] if flat_specs else None
        self.last_domain_values = list(structural_values) if structural_values is not None else None

        self.last_global_specs = list(flat_specs) if flat_specs else []

        if self.post_global_callback is not None:
            self.post_global_callback(self.last_global_specs)

        # --- Derive per_exp_L from domain axis values in Process output ---
        per_exp_L: list[int] | None = None
        if domain_axis_sched_dims and flat_specs:
            group_key_codes = sorted(domain_axis_sched_dims)
            per_exp_L = []
            for spec in flat_specs:
                p = spec.initial_params.to_dict()
                per_exp_L.append(max(int(p.get(c, 1)) for c in group_key_codes))
        elif traj_set:
            # Fixed dimension schedule case
            fixed_sched = [d for d in set(self.trajectory_configs.values()) if d in self.fixed_params]
            if fixed_sched:
                L = max(int(self.fixed_params[d]) for d in fixed_sched)
                per_exp_L = [L] * n

        # --- Schedule phase (if scheduled params exist and L > 1) ---
        if has_slope_traj:
            dim_codes_for_sched = sorted(set(self.trajectory_configs.values()) & domain_axis_sched_dims)
            if dim_codes_for_sched:
                primary_dim_code = dim_codes_for_sched[0]
            else:
                fixed_dim_codes = sorted(
                    d for d in set(self.trajectory_configs.values()) if d in self.fixed_params
                )
                primary_dim_code = fixed_dim_codes[0] if fixed_dim_codes else ""

            traj_params_list: list[tuple[str, float, float]] = []
            for code in sorted(traj_set):
                if code in domain_axis_sched_dims:
                    continue
                lo, hi = self.bounds._get_hierarchical_bounds_for_code(code)
                traj_params_list.append((code, lo, hi))

            if traj_params_list:
                # Initialize evidence model (normally done by _run_acquisition_phase)
                baseline_dm = self._build_schema_datamodule()
                self._active_datamodule = baseline_dm
                if self._fit_empty_kde_fn is not None:
                    self._fit_empty_kde_fn(baseline_dm, n)

                specs = self._run_slope_trajectory(
                    n, flat_specs, traj_params_list, per_exp_L,
                    primary_dim_code, integer_params, cat_codes, cat_assignments,
                    structural_values, continuous_params,
                )
            else:
                specs = flat_specs
        else:
            specs = flat_specs

        # --- Log summary ---
        if n > 1 and n_numeric > 0 and optimized.shape[1] > 0:
            min_dist = self._min_pairwise_distance(optimized, None)
            self.logger.info(
                f"Baseline: {n} experiments via batch acquisition (κ=1) "
                f"({len(continuous_params)} continuous, {len(integer_params)} integer, "
                f"{len(categorical_params)} categorical"
                f"{f', {n_strata} strata' if n_strata > 1 else ''}"
                f") \u2014 min dist = {min_dist:.4f}."
            )
        else:
            self.logger.info(
                f"Baseline: {n} experiment(s) "
                f"({len(continuous_params)} continuous, {len(integer_params)} integer, "
                f"{len(categorical_params)} categorical)."
            )

        return specs

    @staticmethod
    def _filter_phase_params(
        numeric_params: list[tuple[str, float, float]],
        init_norm: np.ndarray,
        int_set: set[int],
        int_ranges_map: dict[int, int],
        codes_keep: set[str],
    ) -> tuple[list[tuple[str, float, float]], np.ndarray, set[int], dict[int, int]]:
        """Slice ``numeric_params`` (and the matching int / init bookkeeping)
        down to the subset whose codes appear in ``codes_keep``. The returned
        int-set / range-map are re-keyed against the filtered list so the
        acquisition-phase helper sees consistent indices."""
        keep_indices = [d for d, (c, _, _) in enumerate(numeric_params) if c in codes_keep]
        sub_params = [numeric_params[d] for d in keep_indices]
        sub_init = init_norm[:, keep_indices] if init_norm.size > 0 else init_norm
        sub_int_set: set[int] = set()
        sub_int_ranges: dict[int, int] = {}
        for new_d, old_d in enumerate(keep_indices):
            if old_d in int_set:
                sub_int_set.add(new_d)
                sub_int_ranges[new_d] = int_ranges_map[old_d]
        return sub_params, sub_init, sub_int_set, sub_int_ranges

    def _run_acquisition_phase(
        self,
        n: int,
        numeric_params: list[tuple[str, float, float]],
        integer_params: list[tuple[str, int, int]],
        int_set: set[int],
        int_ranges_map: dict[int, int],
        init_norm: np.ndarray,
        cat_codes: list[str],
        cat_assignments: list[tuple[Any, ...]],
        structural_values: list[dict[str, int]] | None,
        *,
        label: str,
        init_evidence: bool,
    ) -> tuple[list[ExperimentSpec], np.ndarray, np.ndarray]:
        """Run one batch acquisition phase: maximize ΔI (κ=1) jointly over N points
        in the given parameter subspace, with prior-phase values fed via
        ``structural_values``. Used for both Domain and Process baseline phases.

        ``int_set`` and ``int_ranges_map`` are keyed against the passed-in
        ``numeric_params`` (not against any global merged list). ``init_evidence``
        should be True for the first phase of a baseline run; subsequent phases
        reuse the evidence model from the first.
        """
        all_phase_params: list[tuple[int, str, float, float]] = []
        for d, (code, lo, hi) in enumerate(numeric_params):
            all_phase_params.append((d, code, lo, hi))

        all_static_tuples = [(code, lo, hi) for _, code, lo, hi in all_phase_params]
        all_int_set: set[int] = set()
        all_int_ranges: dict[int, int] = {}
        for si, (d_i, _, _, _) in enumerate(all_phase_params):
            if d_i in int_set:
                all_int_set.add(si)
                all_int_ranges[si] = int_ranges_map[d_i]

        console = self.logger._console_output_enabled

        space = SolutionSpace(
            n_experiments=n,
            static_params=all_static_tuples,
            int_set=all_int_set,
            int_ranges_map=all_int_ranges,
        )

        # Build schema-only datamodule for batch uncertainty evaluation
        baseline_dm = self._build_schema_datamodule()
        self._active_datamodule = baseline_dm

        # Initialize empty evidence model (active_mask from schema bounds).
        # Skipped on later phases — the first phase already populated it.
        if init_evidence and self._fit_empty_kde_fn is not None:
            self._fit_empty_kde_fn(baseline_dm, n)

        # Map SolutionSpace indices to datamodule columns
        n_dm_cols = len(baseline_dm.input_columns)
        phase_col_map: list[tuple[int, int]] = []
        for si, (_, code, lo, hi) in enumerate(all_phase_params):
            if code in baseline_dm.input_columns:
                phase_col_map.append((si, baseline_dm.input_columns.index(code)))
        # Pre-fill batch with structural values from prior phases (e.g. Domain
        # values pinned when running Process after Domain). Filled at 0.5 for
        # any column the previous phase didn't touch.
        prior_fill = np.full((n, n_dm_cols), 0.5)
        if structural_values is not None:
            for i, sv in enumerate(structural_values):
                for code, val in sv.items():
                    if code in baseline_dm.input_columns:
                        col = baseline_dm.input_columns.index(code)
                        # Normalize against schema bounds
                        try:
                            lo, hi = self.bounds._get_hierarchical_bounds_for_code(code)
                            span = hi - lo
                            prior_fill[i, col] = (float(val) - lo) / span if span > 0 else 0.5
                        except ValueError:
                            prior_fill[i, col] = 0.5

        d_phase = len(all_phase_params)
        prior_fill_t = torch.from_numpy(prior_fill).to(dtype=torch.float64)
        phase_cols_t = torch.tensor(
            [c for _, c in phase_col_map], dtype=torch.long,
        ) if phase_col_map else None
        phase_si_t = torch.tensor(
            [s for s, _ in phase_col_map], dtype=torch.long,
        ) if phase_col_map else None

        self.logger.info(
            f"Phase ({label}): N={n}, D_static={len(all_phase_params)}, "
            f"D_traj=0, total_vars={space.total_vars}"
        )

        def _acquisition_batch_objective_tensor(x_flat_S: torch.Tensor) -> torch.Tensor:
            S = int(x_flat_S.shape[0])
            pts_S = x_flat_S.reshape(S, n, d_phase)
            X_batch_S = prior_fill_t.unsqueeze(0).expand(S, -1, -1).clone().to(dtype=x_flat_S.dtype)
            if phase_cols_t is not None and phase_si_t is not None:
                src = pts_S.index_select(-1, phase_si_t)
                if all_int_set:
                    for si_local in all_int_set:
                        if si_local < src.shape[-1]:
                            col_val = src[:, :, si_local]
                            src = src.clone()
                            src[:, :, si_local] = col_val + (col_val.round() - col_val).detach()
                idx = phase_cols_t.unsqueeze(0).unsqueeze(0).expand(S, n, -1)
                X_batch_S = X_batch_S.scatter(-1, idx, src)
            full_S_NL = X_batch_S.unsqueeze(2)  # (S, N, 1, D)
            return self._acquisition_joint_batched_tensor(full_S_NL, 1.0, None)

        opt = self.engine.run_acquisition_gradient(
            _acquisition_batch_objective_tensor,
            space.bounds,
            label=label,
            show_progress=console,
        )
        self.convergence_history[label] = opt.convergence_history
        if not hasattr(self, 'last_baseline_nfev'):
            self.last_baseline_nfev: int = opt.nfev
        else:
            self.last_baseline_nfev += opt.nfev

        best_x = opt.best_x if opt.best_x is not None else np.full(space.total_vars, 0.5)
        specs = space.decode_to_specs(
            best_x,
            fixed_params=dict(self.fixed_params),
            cat_codes=cat_codes,
            cat_assignments=cat_assignments,
            structural_values=structural_values,
            primary_dim_code="",
            schema_sanitize=lambda d: self.schema.parameters.sanitize_values(d, ignore_unknown=True),
            integer_params=integer_params,
            source_step=SourceStep.BASELINE,
        )

        optimized = space.decode_optimized_positions(best_x)
        return specs, best_x, optimized

    def _run_slope_trajectory(
        self,
        n: int,
        flat_specs: list[ExperimentSpec],
        traj_params: list[tuple[str, float, float]],
        per_exp_L_init: list[int],
        primary_dim_code: str,
        integer_params: list[tuple[str, int, int]],
        cat_codes: list[str],
        cat_assignments: list[tuple[Any, ...]],
        structural_values: list[dict[str, int]] | None,
        continuous_params: list[tuple[str, float, float]],
    ) -> list[ExperimentSpec]:
        """Joint Global+Trajectory optimization with sigmoid slopes.

        Adds one z_slope variable per trajectory dimension per experiment
        to the existing Global solution. Single LBFGS pass — no coordinate
        descent. Layer values decoded via sigmoid: z(k) = z_mid + offset * z_slope.
        """
        from .slope import decode_slope_trajectory, default_slope_max

        D_traj = len(traj_params)
        self.logger.info(f"_run_slope_trajectory: D_traj={D_traj}, per_exp_L_init={per_exp_L_init}")
        if D_traj == 0 or (per_exp_L_init is not None and max(per_exp_L_init) <= 1):
            self.logger.info("_run_slope_trajectory: early exit (no traj or L<=1)")
            return flat_specs
        if per_exp_L_init is None:
            self.logger.info("_run_slope_trajectory: early exit (per_exp_L_init is None)")
            return flat_specs

        baseline_dm = self._active_datamodule
        if baseline_dm is None:
            self.logger.info("_run_slope_trajectory: early exit (no active datamodule)")
            return flat_specs

        console = self.logger._console_output_enabled
        n_dm_cols = len(baseline_dm.input_columns)
        total_L = sum(per_exp_L_init)

        # Identify trajectory param columns in the datamodule
        traj_codes = [code for code, _, _ in traj_params]
        traj_dm_cols = [
            baseline_dm.input_columns.index(code)
            for code in traj_codes if code in baseline_dm.input_columns
        ]
        traj_dm_idxs = torch.tensor(traj_dm_cols, dtype=torch.long) if traj_dm_cols else None

        # All non-trajectory continuous params (static).
        static_params = [
            (code, lo, hi) for code, lo, hi in continuous_params
            if code not in set(traj_codes)
        ]
        D_static = len(static_params)

        # Column mapping for static params
        static_dm_cols = []
        static_si = []
        for si, (code, _, _) in enumerate(static_params):
            if code in baseline_dm.input_columns:
                static_dm_cols.append(baseline_dm.input_columns.index(code))
                static_si.append(si)
        static_dm_idxs = torch.tensor(static_dm_cols, dtype=torch.long) if static_dm_cols else None
        static_si_t = torch.tensor(static_si, dtype=torch.long) if static_si else None

        # Column mapping for trajectory midpoint params
        traj_si = []
        for ti, (code, _, _) in enumerate(traj_params):
            for si, (scode, _, _) in enumerate(static_params):
                if scode == code:
                    traj_si.append(si)
                    break
        # Actually traj params are NOT in static_params (they're excluded).
        # Midpoints are separate variables. Decision vector layout:
        # [static_0..D_static-1 | midpoint_0..D_traj-1 | slope_0..D_traj-1] per experiment
        D_per_exp = D_static + D_traj + D_traj  # static + midpoints + slopes

        # Bounds
        bounds_list: list[tuple[float, float]] = []
        for _exp in range(n):
            for _si, (_code, _lo, _hi) in enumerate(static_params):
                bounds_list.append((0.0, 1.0))
            for _ti, (_code, lo, hi) in enumerate(traj_params):
                bounds_list.append((0.0, 1.0))  # midpoint in [0, 1]
            for _ti, (code, lo, hi) in enumerate(traj_params):
                dim_code = self.trajectory_configs.get(code, "")
                sm = default_slope_max(dim_code, self.data_objects)
                bounds_list.append((-sm, sm))  # slope

        # Warm start from Global flat_specs: static params + midpoints from Global, slope = 0
        x0 = np.zeros(n * D_per_exp)
        for i, spec in enumerate(flat_specs):
            off = i * D_per_exp
            p = spec.initial_params.to_dict()
            for si, (code, lo, hi) in enumerate(static_params):
                raw = float(p.get(code, (lo + hi) / 2.0))
                span = hi - lo
                x0[off + si] = (raw - lo) / span if span > 0 else 0.5
            for ti, (code, lo, hi) in enumerate(traj_params):
                raw = float(p.get(code, (lo + hi) / 2.0))
                span = hi - lo
                x0[off + D_static + ti] = (raw - lo) / span if span > 0 else 0.5
            # slopes start at 0 (flat trajectory)

        # Pre-fill base rows with frozen params (integers, fixed, structural)
        prior_fill = np.full((n, n_dm_cols), 0.5)
        all_set = set(code for code, _, _ in static_params) | set(traj_codes)
        for i, spec in enumerate(flat_specs):
            p = spec.initial_params.to_dict()
            for c_idx, col in enumerate(baseline_dm.input_columns):
                if col in p and col not in all_set:
                    val = p[col]
                    try:
                        lo_s, hi_s = self.bounds._get_hierarchical_bounds_for_code(col)
                        span_s = hi_s - lo_s
                        prior_fill[i, c_idx] = (float(val) - lo_s) / span_s if span_s > 0 else 0.5
                    except (ValueError, KeyError):
                        prior_fill[i, c_idx] = 0.5
        prior_fill_t = torch.from_numpy(prior_fill).to(dtype=torch.float64)

        # Build mapping from static param → (code, lo, hi) for L derivation
        static_param_map = {code: (si, lo, hi) for si, (code, lo, hi) in enumerate(static_params)}
        derive_L = self.derive_L_fn

        self.logger.info(
            f"Global: N={n}, D_static={D_static}, D_traj={D_traj}, "
            f"D_slope={D_traj}, total_vars={n * D_per_exp}"
        )

        def _derive_L_per_exp(static_vals_s: torch.Tensor) -> list[int]:
            """Derive N_layers per experiment from current static param values."""
            if derive_L is None:
                return list(per_exp_L_init)
            Ls = []
            for i in range(n):
                # Denormalize static params to build param dict
                p: dict[str, Any] = {}
                for si, (code, lo, hi) in enumerate(static_params):
                    val_norm = float(static_vals_s[i, si].detach())
                    p[code] = val_norm * (hi - lo) + lo
                Ls.append(max(1, derive_L(p)))
            return Ls

        def _objective(x_flat_S: torch.Tensor) -> torch.Tensor:
            S = int(x_flat_S.shape[0])
            x = x_flat_S.reshape(S, n, D_per_exp)
            static_vals = x[:, :, :D_static]                    # (S, N, D_static)
            midpoints = x[:, :, D_static:D_static + D_traj]     # (S, N, D_traj)
            slopes = x[:, :, D_static + D_traj:]                # (S, N, D_traj)

            # Derive L_i from current H_layer (use first start for L computation)
            cur_L = _derive_L_per_exp(static_vals[0])

            # Build all layers for all experiments
            all_rows = prior_fill_t.unsqueeze(0).expand(S, n, n_dm_cols).clone().to(dtype=x_flat_S.dtype)

            # Set static params
            if static_dm_idxs is not None and static_si_t is not None:
                src_static = static_vals.index_select(-1, static_si_t)
                idx_s = static_dm_idxs.unsqueeze(0).unsqueeze(0).expand(S, n, -1)
                all_rows = all_rows.scatter(-1, idx_s, src_static)

            # Expand to per-layer tensor using sigmoid decode
            layer_rows: list[torch.Tensor] = []
            weights_dynamic: list[float] = []
            for i in range(n):
                L_i = cur_L[i]
                traj_layers = decode_slope_trajectory(
                    midpoints[:, i, :], slopes[:, i, :], L_i,
                )  # (S, L_i, D_traj)

                base_i = all_rows[:, i, :].unsqueeze(1).expand(S, L_i, n_dm_cols).clone()
                if traj_dm_idxs is not None:
                    idx_t = traj_dm_idxs.unsqueeze(0).unsqueeze(0).expand(S, L_i, -1)
                    base_i = base_i.scatter(-1, idx_t, traj_layers)
                layer_rows.append(base_i)
                weights_dynamic.extend([1.0 / L_i] * L_i)

            full = torch.cat(layer_rows, dim=1)  # (S, total_L, D)
            full_S_NL = full.unsqueeze(1)  # (S, 1, total_L, D)
            w = torch.tensor(weights_dynamic, dtype=x_flat_S.dtype).unsqueeze(0).expand(S, -1)

            return self._acquisition_joint_batched_tensor(full_S_NL, 1.0, None, w)

        opt = self.engine.run_acquisition_gradient(
            _objective, bounds_list, x0=x0,
            label=f"Global (D={D_per_exp}, V={n * D_per_exp})",
            show_progress=console,
        )
        self.convergence_history["Global"] = opt.convergence_history
        if not hasattr(self, 'last_baseline_nfev'):
            self.last_baseline_nfev: int = opt.nfev
        else:
            self.last_baseline_nfev += opt.nfev

        # Decode result into ExperimentSpecs with trajectories
        best = opt.best_x if opt.best_x is not None else x0

        # Derive final per_exp_L from the optimized H_layer values
        final_L: list[int] = []
        for i in range(n):
            off = i * D_per_exp
            if derive_L is not None:
                p_final: dict[str, Any] = {}
                for si, (code, lo, hi) in enumerate(static_params):
                    p_final[code] = float(best[off + si] * (hi - lo) + lo)
                final_L.append(max(1, derive_L(p_final)))
            else:
                final_L.append(per_exp_L_init[i])

        specs_out: list[ExperimentSpec] = []

        for i in range(n):
            off = i * D_per_exp
            L_i = final_L[i]

            # Decode static params
            bp: dict[str, Any] = dict(self.fixed_params)
            if structural_values is not None:
                for sv_code, sv_val in structural_values[i].items():
                    bp[sv_code] = sv_val
            for si, (code, lo, hi) in enumerate(static_params):
                val = best[off + si]
                bp[code] = float(val * (hi - lo) + lo)
            for _, (code_i, lo_i, hi_i) in enumerate(integer_params):
                if code_i in bp and isinstance(bp[code_i], (int, float)):
                    bp[code_i] = int(np.clip(np.round(bp[code_i]), lo_i, hi_i))
            for d_cat, code in enumerate(cat_codes):
                bp[code] = cat_assignments[i][d_cat]

            # Decode trajectory via sigmoid (single path: torch → numpy)
            midpoint_t = torch.tensor(best[off + D_static:off + D_static + D_traj], dtype=torch.float64).unsqueeze(0)
            slope_t = torch.tensor(best[off + D_static + D_traj:off + D_per_exp], dtype=torch.float64).unsqueeze(0)
            traj_vals = decode_slope_trajectory(midpoint_t, slope_t, L_i)[0].cpu().numpy()  # (L_i, D_traj)

            # Set initial params to layer 0 values
            for ti, (code, lo, hi) in enumerate(traj_params):
                bp[code] = float(traj_vals[0, ti] * (hi - lo) + lo)
            bp = self.schema.parameters.sanitize_values(bp, ignore_unknown=True)
            initial = ParameterProposal.from_dict(bp, source_step=SourceStep.BASELINE)

            # Build trajectory entries for layers 1..L-1
            entries: list[tuple[int, ParameterProposal]] = []
            for k in range(1, L_i):
                sp: dict[str, Any] = {}
                for ti, (code, lo, hi) in enumerate(traj_params):
                    sp[code] = float(traj_vals[k, ti] * (hi - lo) + lo)
                sp = self.schema.parameters.sanitize_values(sp, ignore_unknown=True)
                entries.append((k, ParameterProposal.from_dict(sp, source_step=SourceStep.BASELINE)))

            trajectories: dict[str, ParameterTrajectory] = {}
            if entries:
                trajectories[primary_dim_code] = ParameterTrajectory(
                    dimension=primary_dim_code, entries=entries,
                )

            specs_out.append(ExperimentSpec(initial_params=initial, trajectories=trajectories))

        # Store for plotting
        self.last_traj_norms = [
            decode_slope_trajectory(
                torch.tensor(best[i * D_per_exp + D_static:i * D_per_exp + D_static + D_traj], dtype=torch.float64).unsqueeze(0),
                torch.tensor(best[i * D_per_exp + D_static + D_traj:i * D_per_exp + D_per_exp], dtype=torch.float64).unsqueeze(0),
                final_L[i],
            )[0].cpu().numpy()
            for i in range(n)
        ]
        self.last_traj_params = list(traj_params)
        self.last_trajectory_per_exp_L = list(final_L)
        self.last_trajectory_points = np.concatenate(self.last_traj_norms, axis=0)
        exp_ids: list[int] = []
        for i in range(n):
            exp_ids.extend([i] * final_L[i])
        self.last_trajectory_exp_ids = exp_ids

        return specs_out

    @staticmethod
    def _recursive_split(
        n: int,
        bounds: list[tuple[float, float]],
        original_ranges: list[float],
    ) -> list[list[float]]:
        """Recursively bisect the parameter space into n cells, return cell centres."""
        if n == 1:
            return [[(lo + hi) / 2.0 for lo, hi in bounds]]

        d = len(bounds)
        dim = max(
            range(d),
            key=lambda i: (bounds[i][1] - bounds[i][0]) / original_ranges[i]
            if original_ranges[i] > 0 else 0,
        )

        n_left = n // 2
        n_right = n - n_left
        lo, hi = bounds[dim]
        split = lo + (hi - lo) * n_left / n

        bounds_left = list(bounds)
        bounds_left[dim] = (lo, split)
        bounds_right = list(bounds)
        bounds_right[dim] = (split, hi)

        return (
            CalibrationSystem._recursive_split(n_left, bounds_left, original_ranges)
            + CalibrationSystem._recursive_split(n_right, bounds_right, original_ranges)
        )

    @staticmethod
    def _min_pairwise_distance(
        positions: np.ndarray,
        cat_positions: np.ndarray | None,
    ) -> float:
        """Compute minimum pairwise Euclidean distance across all points."""
        X = np.hstack([positions, cat_positions]) if cat_positions is not None else positions
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        return float(np.sqrt(np.min(dist_sq)))

    def _classify_phase_codes(
        self,
        datamodule: DataModule,
        fixed_values: dict[str, Any] | None = None,
        bounds: np.ndarray | None = None,
        bound_eps: float = 1e-12,
    ) -> dict[str, list[str]]:
        """Group ``datamodule.input_columns`` into phase buckets, free codes only.

        Returns ``{"domain": [...], "process": [...]}`` — codes of *free*
        domain-axis params and *free* non-domain params respectively.

        A code is excluded from both buckets if any of:
          - it appears in ``self.fixed_params`` (always-pinned by design intent);
          - it appears in ``fixed_values`` (pinned by a prior phase or caller);
          - ``bounds`` is provided and the code's interval ``[lo, hi]`` is
            degenerate (``hi - lo <= bound_eps``) — covers trust-region pinning
            for online adaptation as well as schema-level constant params.

        One-hot expanded categorical columns (e.g. 'mat_clay') stay in 'process'
        — they don't have a ``DataObject`` entry under their column name, but
        they are free for the optimiser to choose unless their bounds are
        degenerate.
        """
        fixed_set = set(self.fixed_params.keys())
        if fixed_values:
            fixed_set.update(fixed_values.keys())

        domain_codes: list[str] = []
        process_codes: list[str] = []
        for col_idx, col in enumerate(datamodule.input_columns):
            if col in fixed_set:
                continue
            if bounds is not None and col_idx < bounds.shape[0]:
                lo = float(bounds[col_idx, 0])
                hi = float(bounds[col_idx, 1])
                if hi - lo <= bound_eps:
                    continue
            obj = self.data_objects.get(col)
            if obj is not None and isinstance(obj, DataDomainAxis):
                domain_codes.append(col)
            else:
                process_codes.append(col)
        return {"domain": domain_codes, "process": process_codes}

    def _acquisition_joint_batched_objective(
        self,
        full_S_NL: np.ndarray,
        kappa: float,
        perf_range: tuple[float, float] | None,
    ) -> np.ndarray:
        """No-grad numpy shim around `_acquisition_joint_batched_tensor`.

        ``full_S_NL`` has shape ``(S, N, L, D_global)``; returns ``(S,)``
        negated κ-weighted scores for DE minimisation. See the tensor
        variant for the per-arm semantics.
        """
        with profiler.section("acq._acquisition_joint_batched_objective [per gen]"):
            with torch.no_grad():
                out_t = self._acquisition_joint_batched_tensor(
                    torch.from_numpy(np.ascontiguousarray(full_S_NL)).double(),
                    kappa, perf_range,
                )
            return out_t.detach().cpu().numpy().astype(np.float64)

    def _acquisition_joint_batched_tensor(
        self,
        full_S_NL: torch.Tensor,
        kappa: float,
        perf_range: tuple[float, float] | None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Negated κ-weighted joint acquisition ``(S,)``, gradient-traversable.

        ``weights`` shape ``(S, NL)`` — per-point evidence weight. None = 1.0.
        """
        with profiler.section("acq._acquisition_joint_batched_tensor"):
            S, N, L, D = full_S_NL.shape
            NL = N * L
            dtype = full_S_NL.dtype

            perfs_S: torch.Tensor | None = None
            if kappa < 1.0:
                flat_rows = full_S_NL.reshape(S * NL, D)
                perfs_flat = self._per_candidate_perf_tensor(flat_rows, perf_range)
                perfs_S = perfs_flat.reshape(S, NL).mean(dim=-1).to(dtype=dtype)

            evidence_S: torch.Tensor | None = None
            if kappa > 0.0 and self.evidence.joint_batched_tensor is not None:
                flat_per_candidate = full_S_NL.reshape(S, NL, D)
                w = weights.reshape(S, NL) if weights is not None else None
                evidence_S = self.evidence.joint_batched_tensor(flat_per_candidate, w).to(dtype=dtype)

            return self._kappa_blend(torch.zeros(S, dtype=dtype), perfs_S, evidence_S, kappa)

    def _run_phase(
        self,
        phase_param_codes: list[str],
        fixed_values_per_n: list[dict[str, float]] | None = None,
        *,
        datamodule: DataModule,
        full_bounds: np.ndarray,
        kappa: float,
        perf_range: tuple[float, float] | None,
        label: str,
        show_progress: bool,
        n_proposals: int = 1,
        n_schedule_steps: int = 1,
    ) -> tuple[list[list[dict[str, float]]], _OptResult]:
        """Generalised phase-of-calibration DE: optimise N proposals × L steps over a code subset.

        Single mechanism handles all four cases through the same vectorised DE path:
          - ``(N=1, L=1)`` — single-point exploration / inference Process phase.
          - ``(N>1, L=1)`` — joint-batch baseline Domain / Process phase.
          - ``(N=1, L>1)`` — exploration / inference schedule phase.
          - ``(N>1, L>1)`` — joint-batch baseline schedule.

        ``phase_param_codes`` selects the dimensions the optimiser sees; the rest
        of each ``(proposal, step)`` row is filled from ``fixed_values_per_n[n]``
        (per-proposal codes → normalised values from prior phases) or 0.5 default.
        The DE optimises ``D_phase × N × L`` flat dims; the joint κ-acquisition
        objective evaluates Δ∫E + averaged perf over each candidate's (N × L)
        points in one broadcast tensor reduction.

        Returns ``(per_proposal_per_step_codes, opt)`` where the first element
        is shape ``(N, L)`` — ``list[list[dict[code, normalised_value]]]``,
        indexed as ``[proposal_i][step_k]``.

        ``fixed_values_per_n`` length should match ``n_proposals`` (caller's
        responsibility); shorter lists are padded with empty dicts.
        """
        n_input = len(datamodule.input_columns)
        code_to_idx = {c: i for i, c in enumerate(datamodule.input_columns)}

        # Filter phase codes to those actually in datamodule.input_columns
        used_codes: list[str] = []
        phase_idxs: list[int] = []
        phase_bounds_per_unit: list[tuple[float, float]] = []
        for code in phase_param_codes:
            if code not in code_to_idx:
                continue
            idx = code_to_idx[code]
            used_codes.append(code)
            phase_idxs.append(idx)
            phase_bounds_per_unit.append(
                (float(full_bounds[idx, 0]), float(full_bounds[idx, 1]))
            )

        D_phase = len(phase_idxs)
        if D_phase == 0:
            empty_results: list[list[dict[str, float]]] = [
                [{} for _ in range(n_schedule_steps)] for _ in range(n_proposals)
            ]
            return (empty_results, _OptResult(best_x=None, nfev=0, n_starts=0, score=0.0))

        phase_idxs_arr = np.asarray(phase_idxs, dtype=np.int64)

        # Per-proposal prefill: (N, n_input). Codes outside the phase get fixed
        # values from prior phases (per-proposal) or 0.5 default.
        prefill_per_n = np.full((n_proposals, n_input), 0.5, dtype=np.float64)
        if fixed_values_per_n is not None:
            for i in range(min(n_proposals, len(fixed_values_per_n))):
                for code, val in fixed_values_per_n[i].items():
                    if code in code_to_idx:
                        prefill_per_n[i, code_to_idx[code]] = float(val)

        phase_bounds = phase_bounds_per_unit * (n_proposals * n_schedule_steps)
        prefill_t = torch.from_numpy(prefill_per_n).to(dtype=torch.float64)
        phase_idxs_t = torch.tensor(phase_idxs, dtype=torch.long)

        def _tensor_obj(x_flat_S: torch.Tensor) -> torch.Tensor:
            S = int(x_flat_S.shape[0])
            candidates = x_flat_S.reshape(S, n_proposals, n_schedule_steps, D_phase)
            full_S_NL = prefill_t.unsqueeze(0).unsqueeze(2).expand(
                S, n_proposals, n_schedule_steps, n_input,
            ).clone().to(dtype=x_flat_S.dtype)
            idx = phase_idxs_t.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
                S, n_proposals, n_schedule_steps, D_phase,
            )
            full_S_NL = full_S_NL.scatter(-1, idx, candidates)
            return self._acquisition_joint_batched_tensor(full_S_NL, kappa, perf_range)

        opt = self.engine.run_acquisition_gradient(
            _tensor_obj,
            phase_bounds,
            label=label,
            show_progress=show_progress,
        )

        # Decode best_x → (N, L, D_phase) → list[list[dict]]
        if opt.best_x is None:
            result_x = np.array([0.5 * (lo + hi) for (lo, hi) in phase_bounds])
        else:
            result_x = opt.best_x
        decoded = result_x.reshape(n_proposals, n_schedule_steps, D_phase)

        result: list[list[dict[str, float]]] = []
        for n in range(n_proposals):
            steps_for_n: list[dict[str, float]] = []
            for k in range(n_schedule_steps):
                step_dict = {used_codes[d]: float(decoded[n, k, d]) for d in range(D_phase)}
                steps_for_n.append(step_dict)
            result.append(steps_for_n)

        return result, opt

    # ==================================================================
    # § Online calibration — run_calibration (exploration / inference)
    # ==================================================================

    def run_calibration(
        self,
        datamodule: DataModule,
        mode: Mode,
        current_params: dict[str, Any] | None = None,
        target_indices: dict[str, int] | None = None,
        kappa: float = 0.5,
        n_optimization_rounds: int = 10,
        mpc_lookahead: int | None = None,
        mpc_discount: float | None = None,
        source_step_override: SourceStep | None = None,
    ) -> ExperimentSpec:
        """Run calibration and return an ExperimentSpec.

        Offline (global bounds) when current_params/target_indices are omitted;
        online (trust-region) when both are provided.
        Multi-step schedules use joint optimization over all steps simultaneously.
        mpc_lookahead/mpc_discount: online-only MPC lookahead for adaptation.
        """
        self._active_datamodule = datamodule

        objective = self._build_objective(mode, kappa)

        if source_step_override is not None:
            source_step = source_step_override
        elif mode == Mode.EXPLORATION:
            source_step: SourceStep = SourceStep.EXPLORATION
        elif mode == Mode.INFERENCE:
            source_step = SourceStep.INFERENCE
        else:
            source_step = SourceStep.BASELINE

        is_online = current_params is not None and target_indices is not None

        # MPC lookahead wraps the objective for online adaptation
        if is_online and mpc_lookahead is not None and mpc_lookahead > 0:
            discount = mpc_discount if mpc_discount is not None else 0.9
            objective = self.engine._wrap_mpc_objective(
                objective, datamodule, self.bounds._get_trust_region_bounds, mpc_lookahead, discount,
            )
            self.logger.info(
                f"MPC lookahead enabled: depth={mpc_lookahead}, discount={discount:.3f}"
            )

        if target_indices is not None:
            step_grid: list[dict[str, int]] = [target_indices]  # type: ignore[list-item]
        elif current_params is not None:
            step_grid = self._build_step_grid(current_params)
        else:
            step_grid = [{}]

        self.logger.info(
            f"run_calibration: mode={mode.name}, {'online (trust-region)' if is_online else 'offline (global)'}, "
            f"kappa={kappa:.2f}, step_grid={len(step_grid)} step(s)"
        )

        # Validate trust regions for online/trust-region optimization
        if is_online:
            missing = [
                c for c, obj in self.data_objects.items()
                if obj.runtime_adjustable and c not in self.trust_regions
            ]
            if missing:
                raise RuntimeError(
                    f"Trust-region optimization cannot proceed: runtime-adjustable parameters "
                    f"{sorted(missing)} have no configured trust region. "
                    f"Call configure_adaptation_delta() for each."
                )

        if len(step_grid) > 1:
            sched_without_delta = [
                c for c in self.trajectory_configs if c not in self.trust_regions
            ]
            if sched_without_delta:
                raise RuntimeError(
                    f"Schedule parameters {sorted(sched_without_delta)} have no "
                    f"configured trust region. "
                    f"Call configure_adaptation_delta() for each before running."
                )

        is_schedule = len(step_grid) > 1
        console = self.logger._console_output_enabled and not self._suppress_opt_print

        working_params: dict[str, Any] | None = (
            dict(current_params) if current_params else None
        )
        fixed_for_step = dict(self.fixed_params)

        if is_online and working_params is not None:
            for code, val in working_params.items():
                if code not in self.trust_regions and code not in fixed_for_step:
                    fixed_for_step[code] = val

        # ------------------------------------------------------------------
        # Build per_step_fn, bounds, and engine parameters
        # ------------------------------------------------------------------
        n_input = len(datamodule.input_columns)
        all_global_bounds = self.bounds._get_global_bounds(datamodule)

        if is_schedule:
            L = len(step_grid)

            # categoricals appear once (parent) in input_columns
            # and behave as static (integer) params in the schedule split.
            cat_codes = set(datamodule.categorical_mappings.keys())
            context_codes = set(datamodule.context_feature_codes)
            traj_set = set(self.trajectory_configs.keys())
            static_codes: list[str] = []
            traj_codes: list[str] = []

            for code in datamodule.input_columns:
                if code in context_codes or code in cat_codes or code in self.fixed_params:
                    static_codes.append(code)
                elif code in traj_set:
                    traj_codes.append(code)
                else:
                    static_codes.append(code)

            D_static = len(static_codes)
            D_traj = len(traj_codes)
            code_to_idx = {c: i for i, c in enumerate(datamodule.input_columns)}

            # --- Refine (Process): single-point acquisition, no schedule ---
            # All params (static + sched) treated as flat static for Refine
            all_p2_codes = static_codes + traj_codes
            D_p2 = len(all_p2_codes)

            if is_online:
                p2_bounds = self.bounds._get_trust_region_bounds(datamodule, working_params)  # type: ignore[arg-type]
            else:
                p2_bounds = all_global_bounds

            # Refine (Process): vectorised single-point κ-acquisition over
            # all_p2_codes. Routes through _run_phase which uses
            # _acquisition_joint_batched_objective (perf + KDE batched across
            # the DE candidate dim). Eliminates the scalar per-candidate path
            # that was the main slowdown in schedule iteration.
            _eff_kappa_p2 = 0.0 if mode == Mode.INFERENCE else kappa
            _perf_range_p2, _ = self._get_acquisition_ranges()

            # Inject working_params into the per-proposal fixed values so the
            # full-D acquisition vector reflects the current state for codes
            # outside the trust region. Trust-region-pinned codes naturally
            # drop out of phase_param_codes via the bounds-aware logic below.
            fixed_for_p2: dict[str, float] = {}
            if working_params:
                full_arr = datamodule.params_to_array(working_params)
                for code in datamodule.input_columns:
                    if code in code_to_idx and code not in all_p2_codes:
                        fixed_for_p2[code] = float(full_arr[code_to_idx[code]])

            p2_results, opt_p2 = self._run_phase(
                all_p2_codes,
                [fixed_for_p2],
                datamodule=datamodule,
                full_bounds=p2_bounds,
                kappa=_eff_kappa_p2,
                perf_range=_perf_range_p2,
                label="Global",
                show_progress=console,
                n_proposals=1,
                n_schedule_steps=1,
            )

            self.last_opt_nfev = opt_p2.nfev
            self.last_opt_n_starts = opt_p2.n_starts
            self.last_opt_score = opt_p2.score

            # Build flat_x from Refine result + fixed-for-p2 prefill (codes
            # outside the p2 set, e.g. trust-region-pinned working_params).
            flat_x = np.zeros(n_input)
            if opt_p2.best_x is not None:
                p2_vals_dict = p2_results[0][0]
                for c, v in fixed_for_p2.items():
                    if c in code_to_idx:
                        flat_x[code_to_idx[c]] = v
                for c, v in p2_vals_dict.items():
                    if c in code_to_idx:
                        flat_x[code_to_idx[c]] = v

            # --- Trajectory (Schedule): fix static, optimize offsets ---
            if D_traj > 0:
                # Build sched DE bounds and delta norms in normalized space
                sched_de_bounds: list[tuple[float, float]] = []
                traj_delta_norms: list[float] = []
                sched_param_tuples: list[tuple[str, float, float]] = []
                for code in traj_codes:
                    idx = code_to_idx[code]
                    lo_norm, hi_norm = float(all_global_bounds[idx, 0]), float(all_global_bounds[idx, 1])
                    sched_de_bounds.append((lo_norm, hi_norm))
                    sched_param_tuples.append((code, lo_norm, hi_norm))
                    delta_raw = self.trust_regions.get(code, 0.0)
                    if delta_raw > 0:
                        _, delta_norm = datamodule.normalize_parameter_bounds(code, 0.0, delta_raw)
                        lo_zero, _ = datamodule.normalize_parameter_bounds(code, 0.0, 0.0)
                        traj_delta_norms.append(abs(delta_norm - lo_zero))
                    else:
                        traj_delta_norms.append(0.0)

                # gradient is the only schedule path.
                # Absolute-step encoding (each step k ∈ [0, 1] strict via sigmoid
                # reparam) — no offset cumulative drift, no soft bound penalty,
                # no smoothing factor. Delta constraint between adjacent steps
                # becomes a smooth quadratic penalty in the tensor objective.
                if self.evidence.joint_batched_tensor is None:
                    raise RuntimeError(
                        "Schedule optimisation requires the tensor Δ∫E closure. "
                        "PfabAgent wires it automatically; manual CalibrationSystem "
                        "users must pass `delta_integrated_evidence_joint_batched_tensor_fn`."
                    )

                def _pts_row_to_dm(pts_row: np.ndarray) -> np.ndarray:
                    """Map a sched-only row + fixed static to datamodule input."""
                    x = flat_x.copy()
                    for d_s, c in enumerate(traj_codes):
                        x[code_to_idx[c]] = pts_row[d_s]
                    return x

                sched_perf_range, _ = self._get_acquisition_ranges()

                # Absolute-step encoding: total_vars = L * D_traj.
                abs_bounds: list[tuple[float, float]] = []
                for _k in range(L):
                    for d_s in range(D_traj):
                        abs_bounds.append(sched_de_bounds[d_s])

                flat_x_t = torch.from_numpy(flat_x).float()
                sched_col_indices = torch.tensor(
                    [code_to_idx[c] for c in traj_codes], dtype=torch.long,
                )

                self.logger.debug(
                    f"Trajectory schedule optimization (gradient): L={L}, "
                    f"D_traj={D_traj}, total_vars={L * D_traj}"
                )

                def _schedule_objective_tensor(x_S: torch.Tensor) -> torch.Tensor:
                    """Tensor schedule objective — (S, L*D_traj) → (S,) negated scores."""
                    S = int(x_S.shape[0])
                    steps_SLD = x_S.reshape(S, L, D_traj)
                    full_S_NL = flat_x_t.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
                        S, 1, L, n_input,
                    ).clone()
                    full_S_NL[:, 0, :, sched_col_indices] = steps_SLD.to(dtype=full_S_NL.dtype)
                    scores_neg = self._acquisition_joint_batched_tensor(
                        full_S_NL, kappa, sched_perf_range,
                    )
                    # Soft delta-constraint penalty: λ · Σ_k max(0, |Δstep|−delta)²
                    if D_traj > 0 and L > 1:
                        sched_delta_t = torch.tensor(traj_delta_norms, dtype=x_S.dtype)
                        valid_delta = sched_delta_t > 0
                        if bool(valid_delta.any().item()):
                            step_diffs = (steps_SLD[:, 1:, :] - steps_SLD[:, :-1, :]).abs()
                            excess = (step_diffs - sched_delta_t).clamp(min=0.0)
                            excess = excess * valid_delta.to(dtype=excess.dtype)
                            delta_penalty_S = (excess ** 2).sum(dim=(1, 2))
                            scores_neg = scores_neg + 5.0 * delta_penalty_S
                    return scores_neg

                opt = self.engine.run_acquisition_gradient(
                    _schedule_objective_tensor, abs_bounds,
                    label="Trajectory", show_progress=console,
                )
                self.last_opt_nfev += opt.nfev
                self.convergence_history["Schedule"] = opt.convergence_history

                dm_input_set = set(datamodule.input_columns)
                non_dm_sched = {c for c in self.trajectory_configs if c not in dm_input_set}

                proposals: list[dict[str, Any]] = []
                if opt.best_x is not None:
                    # Commit 12b: gradient-only — absolute-step decode.
                    pts = opt.best_x.reshape(L, D_traj)
                    for k in range(L):
                        step_params = datamodule.array_to_params(_pts_row_to_dm(pts[k]))
                        step_params.update(self.fixed_params)
                        if current_params and non_dm_sched:
                            for code in non_dm_sched:
                                if code in current_params and code not in step_params:
                                    step_params[code] = current_params[code]
                        if current_params:
                            for k_p, v_p in current_params.items():
                                if k_p not in step_params:
                                    step_params[k_p] = v_p
                        proposals.append(self.schema.parameters.sanitize_values(step_params, ignore_unknown=True))
                else:
                    self.logger.warning("Schedule optimization failed, returning fallback.")
                    fallback = dict(self.fixed_params)
                    if current_params:
                        fallback.update(current_params)
                    proposals = [fallback] * L

                self.last_trajectory = list(proposals)

                if opt.best_x is not None:
                    try:
                        pts0 = opt.best_x.reshape(L, D_traj)[0]
                        x0_step = _pts_row_to_dm(pts0)
                        _params = datamodule.array_to_params(x0_step)
                        self.last_opt_perf = self._compute_normalised_perf_for_params(
                            _params, self._perf_range,
                        )
                        self.last_opt_unc = float(self.uncertainty_fn(x0_step))
                    except Exception:
                        self.last_opt_perf = 0.0
                        self.last_opt_unc = 0.0
                else:
                    self.last_opt_perf = 0.0
                    self.last_opt_unc = 0.0

                self.logger.info(
                    f"de: process={opt_p2.nfev} evals, schedule={opt.nfev} evals, "
                    f"score={opt.score:.6f}"
                )

            else:
                # D_traj == 0: sched params not in datamodule input_columns.
                # Replicate flat result across all steps via non-DM sched param handling.
                dm_input_set = set(datamodule.input_columns)
                non_dm_sched = {c for c in self.trajectory_configs if c not in dm_input_set}

                proposals = []
                for k in range(L):
                    step_params = datamodule.array_to_params(flat_x)
                    step_params.update(self.fixed_params)
                    if current_params and non_dm_sched:
                        for code in non_dm_sched:
                            if code in current_params and code not in step_params:
                                step_params[code] = current_params[code]
                    if current_params:
                        for k_p, v_p in current_params.items():
                            if k_p not in step_params:
                                step_params[k_p] = v_p
                    proposals.append(self.schema.parameters.sanitize_values(step_params, ignore_unknown=True))

                self.last_trajectory = list(proposals)

                try:
                    _params = datamodule.array_to_params(flat_x)
                    self.last_opt_perf = self._compute_normalised_perf_for_params(
                        _params, self._perf_range,
                    )
                    self.last_opt_unc = float(self.uncertainty_fn(flat_x))
                except Exception:
                    self.last_opt_perf = 0.0
                    self.last_opt_unc = 0.0

                self.logger.info(
                    f"de: process={opt_p2.nfev} evals, score={opt_p2.score:.6f} "
                    f"(no sched params in datamodule)"
                )

        # ------------------------------------------------------------------
        # SINGLE-STEP optimization (L=1) via optimization engine
        # ------------------------------------------------------------------
        else:
            if is_online:
                bounds = self.bounds._get_trust_region_bounds(datamodule, working_params)  # type: ignore[arg-type]
                n_rounds = 0
            else:
                bounds = all_global_bounds
                n_rounds = n_optimization_rounds

            _eff_kappa = 0.0 if mode == Mode.INFERENCE else kappa
            _perf_range, _ = self._get_acquisition_ranges()

            # Phase-decomposed acquisition (Domain → Process). Splitting keeps
            # each phase's search space smaller. Codes whose bounds
            # collapse (trust-region pin, fixed_params, schema constants) drop
            # out of both buckets; either bucket may be empty without breaking
            # the flow.
            phase_codes = self._classify_phase_codes(
                datamodule, fixed_for_step, bounds=bounds,
            )

            fixed_for_next: dict[str, float] = {}
            domain_opt: _OptResult | None = None
            process_opt: _OptResult | None = None

            if phase_codes["domain"]:
                domain_results, domain_opt = self._run_phase(
                    phase_codes["domain"],
                    [fixed_for_next],
                    datamodule=datamodule,
                    full_bounds=bounds,
                    kappa=_eff_kappa,
                    perf_range=_perf_range,
                    label=f"Global · domain (D={len(phase_codes['domain'])})",
                    show_progress=console,
                    n_proposals=1,
                    n_schedule_steps=1,
                )
                fixed_for_next.update(domain_results[0][0])

            if phase_codes["process"]:
                process_results, process_opt = self._run_phase(
                    phase_codes["process"],
                    [fixed_for_next],
                    datamodule=datamodule,
                    full_bounds=bounds,
                    kappa=_eff_kappa,
                    perf_range=_perf_range,
                    label=f"Global (D={len(phase_codes['process'])})",
                    show_progress=console,
                    n_proposals=1,
                    n_schedule_steps=1,
                )
                fixed_for_next.update(process_results[0][0])

            # Reassemble the per-experiment output vector. Codes absent from
            # any phase (all-fixed case) keep 0.5; the downstream overlay
            # ``proposed_params.update(fixed_for_step)`` pins them to their
            # actual design-intent values.
            static_out = np.full((1, n_input), 0.5)
            code_to_idx_full = {c: i for i, c in enumerate(datamodule.input_columns)}
            for code, val in fixed_for_next.items():
                if code in code_to_idx_full:
                    static_out[0, code_to_idx_full[code]] = val

            # Single canonical _OptResult.
            last_opt = process_opt if process_opt is not None else domain_opt
            total_nfev = (
                (domain_opt.nfev if domain_opt is not None else 0)
                + (process_opt.nfev if process_opt is not None else 0)
            )
            history: list[float] = []
            if domain_opt is not None:
                history.extend(domain_opt.convergence_history)
            if process_opt is not None:
                history.extend(process_opt.convergence_history)
            opt = _OptResult(
                best_x=static_out[0].copy(),
                nfev=total_nfev,
                n_starts=last_opt.n_starts if last_opt is not None else 1,
                score=last_opt.score if last_opt is not None else 0.0,
                convergence_history=history,
            )

            self.last_opt_nfev = opt.nfev
            self.last_opt_n_starts = opt.n_starts
            self.last_opt_score = opt.score

            best_x = static_out[0] if opt.best_x is not None else None
            if best_x is not None:
                try:
                    _params = datamodule.array_to_params(best_x)
                    self.last_opt_perf = self._compute_normalised_perf_for_params(
                        _params, self._perf_range,
                    )
                    self.last_opt_unc = float(self.uncertainty_fn(best_x))
                except Exception:
                    self.last_opt_perf = 0.0
                    self.last_opt_unc = 0.0
            else:
                self.last_opt_perf = 0.0
                self.last_opt_unc = 0.0

            self.logger.info(
                f"acq: {opt.n_starts} start(s), {opt.nfev} evals, score={opt.score:.6f}"
            )

            if opt.best_x is None:
                self.logger.warning("Optimization failed, returning fallback parameters.")
                if working_params:
                    proposals = [working_params]
                else:
                    raise RuntimeError("No valid parameters could be proposed.")
            else:
                proposed_params = datamodule.array_to_params(static_out[0])
                if fixed_for_step:
                    proposed_params.update(fixed_for_step)
                if working_params:
                    for k_p, v_p in working_params.items():
                        if k_p not in proposed_params:
                            proposed_params[k_p] = v_p
                proposed_params = self.schema.parameters.sanitize_values(
                    proposed_params, ignore_unknown=True,
                )
                proposals = [proposed_params]

            self.last_trajectory = None

        proposal_summary = {k: round(v, 4) if isinstance(v, float) else v for k, v in proposals[0].items()}
        self.logger.info(f"Calibration proposal: {proposal_summary}")

        return self._build_experiment_spec(proposals, step_grid, source_step)

    # ==================================================================
    # § Model wrappers
    # ==================================================================

    def get_models(self) -> list[Any]:
        """Return empty list (no internal ML models owned by CalibrationSystem)."""
        return []

    def get_model_specs(self) -> dict[str, list[str]]:
        return {}
