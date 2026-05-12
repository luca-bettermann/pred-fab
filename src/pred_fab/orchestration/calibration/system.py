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
from .space import SolutionSpace, StaticVariable, TrajectoryVariable, Variable

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
        """Generate n baseline proposals via Sobol -> LBFGS acquisition (kappa=1)."""
        if n == 0:
            return []

        categorical_params: list[tuple[str, list[Any]]] = []
        optimizable: list[tuple[str, DataObject]] = []
        traj_set = set(self.trajectory_configs.keys())

        for code, data_obj in self.data_objects.items():
            if code in self.fixed_params:
                continue
            if isinstance(data_obj, DataCategorical):
                categorical_params.append((code, list(data_obj.constraints["categories"])))
                continue
            if isinstance(data_obj, DataBool):
                categorical_params.append((code, [False, True]))
                continue
            try:
                lo, hi = self.bounds._get_hierarchical_bounds_for_code(code)
            except ValueError:
                continue
            if lo == -np.inf or hi == np.inf:
                self.logger.warning(f"Parameter '{code}' has infinite bounds; skipping.")
                continue
            if lo == hi:
                fixed_val = int(lo) if isinstance(data_obj, (DataInt, DataDomainAxis)) else float(lo)
                self.bounds.fixed_params[code] = fixed_val
                self.logger.debug(f"Auto-fixed '{code}' = {fixed_val} (degenerate bounds).")
                continue
            optimizable.append((code, data_obj))

        variables: list[Variable] = [
            TrajectoryVariable(data_object=obj, dimension_code=self.trajectory_configs[code])
            if code in traj_set
            else StaticVariable(data_object=obj, is_integer=isinstance(obj, (DataInt, DataDomainAxis)))
            for code, obj in optimizable
        ]

        if not variables and not categorical_params:
            self.logger.warning("No valid parameters for baseline generation.")
            return []

        # Stratify categoricals
        if categorical_params:
            import itertools as _it
            cat_combos = list(_it.product(*(cats for _, cats in categorical_params)))
            cat_codes = [code for code, _ in categorical_params]
        else:
            cat_combos = [()]
            cat_codes = []
        n_strata = len(cat_combos)

        n_per_stratum = [n // n_strata] * n_strata
        for i in range(n % n_strata):
            n_per_stratum[i] += 1

        cat_assignments: list[tuple[Any, ...]] = []
        for combo, n_i in zip(cat_combos, n_per_stratum):
            cat_assignments.extend([combo] * n_i)

        specs = self._optimize(
            n, variables, cat_codes, cat_assignments,
            kappa=1.0, source_step=SourceStep.BASELINE,
            label="Baseline",
        )

        n_static = sum(1 for v in variables if isinstance(v, StaticVariable))
        n_traj = sum(1 for v in variables if isinstance(v, TrajectoryVariable))
        self.logger.info(
            f"Baseline: {n} experiments "
            f"({n_static} static, {n_traj} trajectory, "
            f"{len(categorical_params)} categorical"
            f"{f', {n_strata} strata' if n_strata > 1 else ''})."
        )
        return specs

    def _optimize(
        self,
        n: int,
        variables: list[Variable],
        cat_codes: list[str],
        cat_assignments: list[tuple[Any, ...]],
        kappa: float,
        source_step: str | None = None,
        label: str = "Optimizing",
    ) -> list[ExperimentSpec]:
        """Single optimization path: SolutionSpace + Engine.

        Constructs the decision vector layout from Variable objects, defines
        the objective closure, runs Sobol -> LBFGS, and decodes the result.
        """
        baseline_dm = self._build_schema_datamodule()
        self._active_datamodule = baseline_dm

        if self._fit_empty_kde_fn is not None:
            self._fit_empty_kde_fn(baseline_dm, n)

        space = SolutionSpace(
            variables=variables,
            n_experiments=n,
            bounds_manager=self.bounds,
            datamodule=baseline_dm,
            fixed_params=dict(self.fixed_params),
            cat_codes=cat_codes,
            cat_assignments=cat_assignments,
            schema_sanitize=lambda d: self.schema.parameters.sanitize_values(d, ignore_unknown=True),
            derive_L_fn=self.derive_L_fn,
            source_step=source_step,
        )

        if space.total_vars == 0:
            return space.decode_to_specs(np.zeros(0))

        console = self.logger._console_output_enabled
        perf_range = self._perf_range if kappa < 1.0 else None

        def objective(x_flat: torch.Tensor) -> torch.Tensor:
            points, weights = space.decode(x_flat)
            full_S_NL = points.unsqueeze(1)  # (S, 1, total_points, D)
            return self._acquisition_joint_batched_tensor(full_S_NL, kappa, perf_range, weights)

        self.logger.info(
            f"{label}: N={n}, D={space._D_per_exp}, V={space.total_vars}"
        )

        opt = self.engine.optimize(
            objective, space.bounds,
            d_param=space._D_per_exp,
            label=label,
            show_progress=console,
        )
        self.convergence_history[label] = opt.convergence_history
        self.last_baseline_nfev = getattr(self, 'last_baseline_nfev', 0) + opt.nfev

        best_x = opt.best_x if opt.best_x is not None else np.zeros(space.total_vars)
        specs = space.decode_to_specs(best_x)

        # Store trajectory plot data
        if space._D_traj > 0:
            traj_norms, traj_params, L_per_exp = space.get_trajectory_plot_data(best_x)
            self.last_traj_norms = traj_norms
            self.last_traj_params = traj_params
            self.last_trajectory_per_exp_L = L_per_exp
            if any(L > 1 for L in L_per_exp):
                self.last_trajectory_points = np.concatenate(traj_norms, axis=0)
                exp_ids: list[int] = []
                for i, L_i in enumerate(L_per_exp):
                    exp_ids.extend([i] * L_i)
                self.last_trajectory_exp_ids = exp_ids

        self.last_global_specs = list(specs)
        if self.post_global_callback is not None:
            self.post_global_callback(self.last_global_specs)

        return specs


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

        opt = self.engine.optimize(
            _tensor_obj,
            phase_bounds,
            d_param=D_phase,
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

                opt = self.engine.optimize(
                    _schedule_objective_tensor, abs_bounds,
                    d_param=D_traj,
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
