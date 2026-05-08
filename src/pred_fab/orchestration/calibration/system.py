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


@dataclass
class _ScheduleState:
    """Per-call state for the Trajectory phase.

    Holds warm-starts, mutable optimisation state, and pre-computed lookups
    that the inner-DE machinery reads on every per-experiment call.
    """
    n: int
    flat_specs: list[ExperimentSpec]
    sched_params: list[tuple[str, float, float]]
    static_params: list[tuple[str, float, float]]
    per_exp_L: list[int]
    primary_dim_code: str
    sched_delta_norms: list[float]
    static_delta_norms: list[float]

    # Mutable state — updated each pass.
    static_norms: np.ndarray              # (n, D_static)
    schedule_norms: list[np.ndarray]      # n × (L_i, D_sched)

    # Pre-computed lookups for X_batch construction.
    n_dm_cols: int
    exp_base_rows: list[np.ndarray]       # n × (n_dm_cols,)
    sched_col_map: list[tuple[int, int]]  # (sched_idx, dm_col_idx)
    static_col_map: list[tuple[int, int]] # (static_idx, dm_col_idx)

    @property
    def D_sched(self) -> int:
        return len(self.sched_params)

    @property
    def D_static(self) -> int:
        return len(self.static_params)


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
        self.trajectory_locked_static: set[str] = set()
        self.smoothness_weight: float = 0.01
        self.trajectory_step_callback: Callable[[int, int, '_ScheduleState'], None] | None = None

        # Set ordered weights
        self.schema = schema
        self.perf_names_order = list(schema.performance_attrs.keys())
        self.performance_weights: dict[str, float] = {perf: 1.0 for perf in self.perf_names_order}
        self.parameters = schema.parameters

        # Backend is phase-deterministic — no user-facing optimizer choice:
        # Global (joint DE over all dims) = DE (handles
        # integers/cats natively). Refine (continuous LBFGS) and Trajectory
        # (trajectory) = LBFGS multi-start. The Domain → Process split inside
        # Global keeps each DE call small (joint over all dims at full DE
        # budget timed out at >5 min for N=5 + ~3 numeric params).

        # Persistent κ default for acquisition_step / exploration_step. Overridable
        # per call. inference_step ignores it (κ=0 is the inference semantic).
        self.kappa_default: float = 0.5

        # Running min/max of predicted system performance across training data.
        self._perf_range_min: float | None = None
        self._perf_range_max: float | None = None

        self._schedule_joint_var_limit: int = 200  # threshold for auto-selecting joint vs sequential
        self._suppress_opt_print: bool = False

    # ------------------------------------------------------------------
    # Proxy properties for backward compatibility
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Proxy properties for engine config (used by agent.py)
    # ------------------------------------------------------------------

    @property
    def de_maxiter(self) -> int:
        return self.engine.de_maxiter

    @de_maxiter.setter
    def de_maxiter(self, value: int) -> None:
        self.engine.de_maxiter = value

    @property
    def de_popsize(self) -> int:
        return self.engine.de_popsize

    @de_popsize.setter
    def de_popsize(self, value: int) -> None:
        self.engine.de_popsize = value



    # ------------------------------------------------------------------
    # random_seed property
    # ------------------------------------------------------------------

    @property
    def random_seed(self) -> int | None:
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value: int | None) -> None:
        self._random_seed = value
        self.rng = np.random.RandomState(value)
        self.engine._random_seed = value
        self.engine.rng = np.random.RandomState(value)

    # ------------------------------------------------------------------
    # Delegated config methods
    # ------------------------------------------------------------------

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

    # === OBJECTIVE FUNCTIONS ===
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

    @staticmethod
    def _kappa_blend(scores, perfs, evidences, kappa: float):
        """Negated κ-weighted blend: −[(1−κ)·perfs + κ·evidences].

        Backend-agnostic — both arms work for numpy ndarrays and torch tensors
        via operator overloading. ``perfs`` / ``evidences`` may be ``None`` when
        their respective branch is inactive (κ at boundary, or evidence_fn
        unavailable). Returns ``-scores`` so callers can directly hand it to
        a minimiser.
        """
        if perfs is not None and kappa < 1.0:
            scores = scores + (1.0 - kappa) * perfs
        if evidences is not None and kappa > 0.0:
            scores = scores + kappa * evidences
        return -scores

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

    # === PRIVATE HELPERS ===

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
                sched_codes = [
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
                        seg_vals = {c: proposals[flat_i][c] for c in sched_codes if c in proposals[flat_i]}
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

    # === OPTIMIZATION WORKFLOW ===

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
        sched_set = set(self.trajectory_configs.keys())
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
                    sched_set = {
                        code for code in sched_set
                        if self.trajectory_configs[code] in self.fixed_params
                    }

        # --- Global: joint DE over all dims (domain + process) ---
        domain_axis_codes = {
            code for code, _, _ in numeric_params
            if code in self.data_objects and isinstance(self.data_objects[code], DataDomainAxis)
        }

        if numeric_params:
            flat_specs, flat_params, optimized = self._run_acquisition_phase(
                n, numeric_params, integer_params, int_set, int_ranges_map,
                init_norm, cat_codes, cat_assignments,
                structural_values=None,
                label=f"Global (D={len(numeric_params)}, V={n * len(numeric_params)})", init_evidence=True,
            )
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

        # --- Derive per_exp_L from domain axis values in Process output ---
        per_exp_L: list[int] | None = None
        if domain_axis_sched_dims and flat_specs:
            group_key_codes = sorted(domain_axis_sched_dims)
            per_exp_L = []
            for spec in flat_specs:
                p = spec.initial_params.to_dict()
                per_exp_L.append(max(int(p.get(c, 1)) for c in group_key_codes))
        elif sched_set:
            # Fixed dimension schedule case
            fixed_sched = [d for d in set(self.trajectory_configs.values()) if d in self.fixed_params]
            if fixed_sched:
                L = max(int(self.fixed_params[d]) for d in fixed_sched)
                per_exp_L = [L] * n

        # --- Schedule phase (if scheduled params exist and L > 1) ---
        if sched_set and per_exp_L is not None and max(per_exp_L) > 1:
            # Determine primary dimension code: prefer an unfixed domain-axis
            # sched dim; fall back to a fixed-dim sched when --design-intent
            # pins the dimension (e.g. n_layers=4).
            dim_codes_for_sched = sorted(set(self.trajectory_configs.values()) & domain_axis_sched_dims)
            if dim_codes_for_sched:
                primary_dim_code = dim_codes_for_sched[0]
            else:
                fixed_dim_codes = sorted(
                    d for d in set(self.trajectory_configs.values()) if d in self.fixed_params
                )
                primary_dim_code = fixed_dim_codes[0] if fixed_dim_codes else ""

            sched_params_list: list[tuple[str, float, float]] = []
            for code in sorted(sched_set):
                if code in domain_axis_sched_dims:
                    continue
                lo, hi = self.bounds._get_hierarchical_bounds_for_code(code)
                sched_params_list.append((code, lo, hi))

            if sched_params_list:
                # TEST: disable static drift to isolate trajectory behavior
                static_params_list: list[tuple[str, float, float]] = []
                specs = self._phase3_trajectory(
                    n, flat_specs, sched_params_list, per_exp_L,
                    primary_dim_code, integer_params, cat_codes, cat_assignments,
                    None,
                    static_params=static_params_list,
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

        init_de = init_norm.copy()
        for d in int_set:
            if d < init_de.shape[1]:
                init_de[:, d] = np.round(init_de[:, d] * int_ranges_map[d])

        space = SolutionSpace(
            n_experiments=n,
            static_params=all_static_tuples,
            trust_regions={},
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

        def _acquisition_batch_objective_vec(x_flat_DS: np.ndarray) -> np.ndarray:
            """Vectorised: ``(D, S) → (S,)`` — one negated ΔE per population member."""
            x_flat_SD = torch.from_numpy(x_flat_DS.T).double()  # (S, total_vars)
            S = int(x_flat_SD.shape[0])
            pts_S = x_flat_SD.reshape(S, n, d_phase)
            X_batch_S = prior_fill_t.unsqueeze(0).expand(S, -1, -1).clone()
            if phase_cols_t is not None and phase_si_t is not None:
                src = pts_S.index_select(-1, phase_si_t)
                X_batch_S[:, :, phase_cols_t] = src
            full_S_NL = X_batch_S.unsqueeze(2)  # (S, N, 1, D)
            with torch.no_grad():
                scores_neg = self._acquisition_joint_batched_tensor(full_S_NL, 1.0, None)
            return scores_neg.cpu().numpy()

        # Build init_norm for the phase-local space (remap original indices)
        merged_init = np.zeros((n, len(all_phase_params)))
        for si, (d_i, _, _, _) in enumerate(all_phase_params):
            if d_i < init_de.shape[1]:
                merged_init[:, si] = init_de[:, d_i]

        init_pop = space.build_init_population(self.engine.rng, merged_init)

        self.logger.info(
            f"Phase ({label}): N={n}, D_static={len(all_phase_params)}, "
            f"D_sched=0, total_vars={space.total_vars}, maxiter={self.engine.de_maxiter}"
        )

        # Gradient when tensor evidence is wired + no integer dims.
        # DE fallback handles integer/categorical phases where gradients
        # don't apply. With ANOVA kernel, baseline evidence has strong
        # per-dimension gradients — no need for DE on continuous params.
        use_gradient = (
            self.evidence.joint_batched_tensor is not None
            and not (space.integrality is not None and any(space.integrality))
        )

        def _acquisition_batch_objective_tensor(x_flat_S: torch.Tensor) -> torch.Tensor:
            S = int(x_flat_S.shape[0])
            pts_S = x_flat_S.reshape(S, n, d_phase)
            # Out-of-place column replacement to preserve autograd.
            # Start from prior_fill (constant), then replace phase columns
            # with optimised values via index arithmetic (no in-place ops).
            X_batch_S = prior_fill_t.unsqueeze(0).expand(S, -1, -1).clone().to(dtype=x_flat_S.dtype)
            if phase_cols_t is not None and phase_si_t is not None:
                src = pts_S.index_select(-1, phase_si_t)
                idx = phase_cols_t.unsqueeze(0).unsqueeze(0).expand(S, n, -1)
                X_batch_S = X_batch_S.scatter(-1, idx, src)
            de = self.evidence.joint_batched_tensor(X_batch_S)  # type: ignore[misc]
            return -de.to(dtype=x_flat_S.dtype)

        if use_gradient:
            opt = self.engine.run_acquisition_gradient(
                _acquisition_batch_objective_tensor,
                space.bounds,
                label=label,
                show_progress=console,
            )
            self.convergence_history[label] = opt.convergence_history
        else:
            opt = self.engine._run_de(
                _acquisition_batch_objective_vec, space.bounds, init_pop=init_pop,
                integrality=space.integrality, label=label, show_progress=console,
                maxiter=self.engine.de_maxiter, vectorized=True,
            )
            self.convergence_history[label] = opt.convergence_history
            # Snapshot DE params before refinement (deep copy as plain dicts)
            de_best_x = opt.best_x if opt.best_x is not None else merged_init.ravel()
            de_specs = space.decode_to_specs(
                de_best_x,
                fixed_params=dict(self.fixed_params),
                cat_codes=cat_codes,
                cat_assignments=cat_assignments,
                structural_values=structural_values,
                primary_dim_code="",
                schema_sanitize=lambda d: self.schema.parameters.sanitize_values(d, ignore_unknown=True),
                integer_params=integer_params,
                source_step=SourceStep.BASELINE,
            )
            self.last_de_params = [dict(s.initial_params.to_dict()) for s in de_specs]
            # Refinement: gradient polish from DE warm start (continuous dims only)
            has_int = space.integrality is not None and any(space.integrality)
            if not has_int and self.evidence.joint_batched_tensor is not None and opt.best_x is not None:
                refine_label = f"Refine (D={d_phase}, V={space.total_vars})"
                refine_opt = self.engine.run_acquisition_gradient(
                    _acquisition_batch_objective_tensor,
                    space.bounds,
                    x0=opt.best_x,
                    label=refine_label,
                    show_progress=console,
                )
                self.convergence_history[refine_label] = refine_opt.convergence_history
                if refine_opt.best_x is not None and refine_opt.score >= opt.score:
                    opt = refine_opt
        if not hasattr(self, 'last_baseline_nfev'):
            self.last_baseline_nfev: int = opt.nfev
        else:
            self.last_baseline_nfev += opt.nfev

        best_x = opt.best_x if opt.best_x is not None else merged_init.ravel()
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

    def _phase3_trajectory(
        self,
        n: int,
        flat_specs: list[ExperimentSpec],
        sched_params: list[tuple[str, float, float]],
        per_exp_L: list[int],
        primary_dim_code: str,
        integer_params: list[tuple[str, int, int]],
        cat_codes: list[str],
        cat_assignments: list[tuple[Any, ...]],
        structural_values: list[dict[str, int]] | None,
        static_params: list[tuple[str, float, float]] | None = None,
    ) -> list[ExperimentSpec]:
        """Trajectory: joint multi-start LBFGS over all N experiments' trajectories.

        Decision vector: ``(D_static + L_i × D_sched)`` per experiment, all
        concatenated → ``Σᵢ (D_static + L_i × D_sched)`` total dims.
        Per-dim bounds: static params drift within the same trust-region
        band used for schedule params; schedule values are
        ``[0, 1]`` (sigmoid-reparam'd inside ``run_acquisition_gradient``).
        Per-experiment delta constraints are smooth quadratic penalties on
        adjacent step differences.

        Multi-start covers the trajectory-shape axis: Global's warm start
        anchors the static dims (and step-0 schedule values) but the
        trajectory shape over ``k=1..L-1`` is otherwise unspecified, so
        diverse starting shapes converge to different basins and we pick
        the best.
        """
        if len(sched_params) == 0 or max(per_exp_L) <= 1:
            return flat_specs

        state = self._init_trajectory_state(
            n, flat_specs, sched_params, static_params or [],
            per_exp_L, primary_dim_code,
        )
        if state is None:
            return flat_specs

        self._run_joint_trajectory_optimisation(state)
        return self._decode_trajectory_specs(state)

    def _init_trajectory_state(
        self,
        n: int,
        flat_specs: list[ExperimentSpec],
        sched_params: list[tuple[str, float, float]],
        static_params: list[tuple[str, float, float]],
        per_exp_L: list[int],
        primary_dim_code: str,
    ) -> _ScheduleState | None:
        """Build warm-starts, base rows, and column maps for the Schedule phase."""
        baseline_dm = self._active_datamodule
        if baseline_dm is None:
            return None

        # Trust-region delta in normalised space per sched param.
        sched_delta_norms = [
            (self.trust_regions.get(code, (hi - lo) / 10.0) / (hi - lo) if hi - lo > 0 else 0.0)
            for code, lo, hi in sched_params
        ]
        static_delta_norms = [
            (self.trust_regions.get(code, (hi - lo) / 10.0) / (hi - lo) if hi - lo > 0 else 0.0)
            for code, lo, hi in static_params
        ]

        # Phase-2 warm starts — step 0 is now optimisable; static is the trust-region centre.
        step0_warmstart = self._warmstart_from_specs(flat_specs, sched_params, n)
        static_warmstart = self._warmstart_from_specs(flat_specs, static_params, n)

        # Initial state — schedule starts flat at the step-0 warm start.
        static_norms = static_warmstart.copy()
        schedule_norms = [
            np.tile(step0_warmstart[i], (per_exp_L[i], 1)) for i in range(n)
        ]

        # Datamodule column index maps for X_batch construction.
        n_dm_cols = len(baseline_dm.input_columns)
        sched_col_map = [
            (si, baseline_dm.input_columns.index(code))
            for si, (code, _, _) in enumerate(sched_params)
            if code in baseline_dm.input_columns
        ]
        static_col_map = [
            (si, baseline_dm.input_columns.index(code))
            for si, (code, _, _) in enumerate(static_params)
            if code in baseline_dm.input_columns
        ]

        # Base rows hold frozen columns (integer static, fixed params).
        # Continuous static + sched columns are overlaid per-call.
        sched_code_set = {code for code, _, _ in sched_params}
        static_code_set = {code for code, _, _ in static_params}
        exp_base_rows: list[np.ndarray] = []
        for i_exp in range(n):
            row = np.full(n_dm_cols, 0.5)
            static_dict = flat_specs[i_exp].initial_params.to_dict()
            for c_idx, col in enumerate(baseline_dm.input_columns):
                if col in static_dict and col not in sched_code_set and col not in static_code_set:
                    val = static_dict[col]
                    try:
                        lo_s, hi_s = self.bounds._get_hierarchical_bounds_for_code(col)
                        span_s = hi_s - lo_s
                        row[c_idx] = (float(val) - lo_s) / span_s if span_s > 0 else 0.5
                    except (ValueError, KeyError):
                        row[c_idx] = 0.5
            exp_base_rows.append(row)

        return _ScheduleState(
            n=n, flat_specs=flat_specs,
            sched_params=sched_params, static_params=static_params,
            per_exp_L=per_exp_L, primary_dim_code=primary_dim_code,
            sched_delta_norms=sched_delta_norms, static_delta_norms=static_delta_norms,
            static_norms=static_norms, schedule_norms=schedule_norms,
            n_dm_cols=n_dm_cols, exp_base_rows=exp_base_rows,
            sched_col_map=sched_col_map, static_col_map=static_col_map,
        )

    @staticmethod
    def _warmstart_from_specs(
        flat_specs: list[ExperimentSpec],
        params: list[tuple[str, float, float]],
        n: int,
    ) -> np.ndarray:
        """Extract normalised values from Phase-2 specs as a warm start."""
        out = np.zeros((n, len(params)))
        for i, spec in enumerate(flat_specs):
            p_dict = spec.initial_params.to_dict()
            for si, (code, lo, hi) in enumerate(params):
                raw_val = float(p_dict.get(code, (lo + hi) / 2.0))
                span = hi - lo
                out[i, si] = (raw_val - lo) / span if span > 0 else 0.5
        return out

    def _run_joint_trajectory_optimisation(self, state: _ScheduleState) -> None:
        """Coordinate descent over N experiments' trajectories.

        Each round optimises one experiment at a time (~D_static + L×D_sched
        dims) while holding all others fixed. Iterates rounds until convergence
        or maxiter budget is exhausted.

        Mutates ``state.static_norms`` and ``state.schedule_norms`` in place.
        """
        n = state.n
        D_static = state.D_static
        D_sched = state.D_sched
        per_exp_L = list(state.per_exp_L)
        console = self.logger._console_output_enabled

        base_rows_t = torch.from_numpy(np.stack(state.exp_base_rows)).to(dtype=torch.float64)
        n_dm_cols = base_rows_t.shape[1]
        static_col_idxs = torch.tensor(
            [col_idx for _, col_idx in state.static_col_map], dtype=torch.long,
        )
        sched_col_idxs = torch.tensor(
            [col_idx for _, col_idx in state.sched_col_map], dtype=torch.long,
        )
        static_delta_norms_list = state.static_delta_norms
        sched_delta_norms_list = state.sched_delta_norms
        kappa = 1.0

        def _build_fixed_rows(exclude_idx: int) -> torch.Tensor:
            """Build (NL_fixed, n_dm_cols) for all experiments except exclude_idx."""
            blocks: list[torch.Tensor] = []
            for j in range(n):
                if j == exclude_idx:
                    continue
                L_j = per_exp_L[j]
                row_j = base_rows_t[j].unsqueeze(0).expand(L_j, n_dm_cols).clone()
                if D_static > 0:
                    stat_j = torch.from_numpy(state.static_norms[j]).to(dtype=torch.float64)
                    row_j = row_j.scatter(-1, static_col_idxs.unsqueeze(0).expand(L_j, -1),
                                          stat_j.unsqueeze(0).expand(L_j, -1))
                if D_sched > 0:
                    sched_j = torch.from_numpy(state.schedule_norms[j]).to(dtype=torch.float64)
                    row_j = row_j.scatter(-1, sched_col_idxs.unsqueeze(0).expand(L_j, -1), sched_j)
                blocks.append(row_j)
            return torch.cat(blocks, dim=0) if blocks else torch.empty(0, n_dm_cols, dtype=torch.float64)

        def _make_per_exp_objective(exp_idx: int):
            L_i = per_exp_L[exp_idx]
            # Layout: [D_static | D_sched (step0) | (L_i-1)*D_sched (deltas)]
            D_exp = D_static + L_i * D_sched

            def _objective(x_S: torch.Tensor) -> torch.Tensor:
                S = int(x_S.shape[0])
                stat = x_S[:, :D_static]

                # Delta reparameterization: step0 + cumsum(deltas) → absolute
                step0 = x_S[:, D_static:D_static + D_sched]
                traj = torch.zeros(S, L_i, D_sched, dtype=x_S.dtype)
                traj[:, 0, :] = step0
                if L_i > 1:
                    deltas = x_S[:, D_static + D_sched:].reshape(S, L_i - 1, D_sched)
                    traj[:, 1:, :] = step0.unsqueeze(1) + deltas.cumsum(dim=1)
                # No clamp — sigmoid reparam in run_acquisition_gradient
                # enforces bounds; clamping would kill gradients.

                cand_i = base_rows_t[exp_idx].unsqueeze(0).unsqueeze(0).expand(S, L_i, n_dm_cols).clone().to(dtype=x_S.dtype)
                if D_static > 0:
                    cand_i = cand_i.scatter(-1, static_col_idxs.unsqueeze(0).unsqueeze(0).expand(S, L_i, -1),
                                            stat.unsqueeze(1).expand(S, L_i, D_static))
                if D_sched > 0:
                    cand_i = cand_i.scatter(-1, sched_col_idxs.unsqueeze(0).unsqueeze(0).expand(S, L_i, -1), traj)

                full_S_NL = cand_i.unsqueeze(1)  # (S, 1, L_i, n_dm_cols)
                scores_neg = self._acquisition_joint_batched_tensor(full_S_NL, kappa, None)

                # R² smoothness (delta bounds replace the old soft delta constraint)
                if D_sched > 0 and L_i > 2:
                    t = torch.arange(L_i, dtype=x_S.dtype)
                    t_mean = t.mean()
                    t_var = ((t - t_mean) ** 2).sum()
                    y_mean = traj.mean(dim=1, keepdim=True)
                    ss_tot = ((traj - y_mean) ** 2).sum(dim=1)  # (S, D_sched)
                    # Only penalize when trajectory has meaningful variation;
                    # near-flat ss_tot ≈ 0 makes the ratio pathological.
                    has_variation = ss_tot > 1e-6
                    if bool(has_variation.any().item()):
                        cov = ((t[None, :, None] - t_mean) * (traj - y_mean)).sum(dim=1)
                        slope = cov / t_var.clamp(min=1e-12)
                        y_hat = y_mean + slope[:, None, :] * (t[None, :, None] - t_mean)
                        ss_res = ((traj - y_hat) ** 2).sum(dim=1)
                        r2 = torch.where(has_variation, 1.0 - ss_res / ss_tot.clamp(min=1e-6), torch.ones_like(ss_tot))
                        penalty = (1.0 - r2).clamp(min=0.0).sum(dim=-1)
                        scores_neg = scores_neg + scores_neg.detach().abs() * self.smoothness_weight * penalty

                return scores_neg

            return _objective, D_exp

        def _build_virtual_params(exclude_idx: int = -1) -> list[dict[str, Any]]:
            """Build param dicts for all experiments except exclude_idx (-1 = include all)."""
            params_list: list[dict[str, Any]] = []
            for j in range(n):
                if j == exclude_idx and exclude_idx >= 0:
                    continue
                for k in range(per_exp_L[j]):
                    p: dict[str, Any] = {}
                    for si, (code, lo, hi) in enumerate(state.static_params):
                        p[code] = float(state.static_norms[j, si] * (hi - lo) + lo)
                    for si, (code, lo, hi) in enumerate(state.sched_params):
                        p[code] = float(state.schedule_norms[j][k, si] * (hi - lo) + lo)
                    # Add fixed params
                    base_dict = state.flat_specs[j].initial_params.to_dict()
                    for code, val in base_dict.items():
                        if code not in p:
                            p[code] = val
                    params_list.append(p)
            return params_list

        max_rounds = min(5, max(1, self.engine.gradient_n_iters // max(n, 1)))
        total_iters = 0
        can_push = self._push_virtual_fn is not None and self._pop_virtual_fn is not None
        baseline_dm = self._active_datamodule

        # Push ALL experiments as virtual points initially
        if can_push:
            all_virtual = _build_virtual_params(-1)  # -1 = exclude none
            if all_virtual:
                self._push_virtual_fn(all_virtual, None, baseline_dm)  # type: ignore[misc]

        for round_idx in range(max_rounds):
            improved_this_round = False
            for exp_idx in range(n):
                L_i = per_exp_L[exp_idx]

                # Pop current experiment's layers, keep all others
                if can_push:
                    self._pop_virtual_fn()  # type: ignore[misc]
                    others = _build_virtual_params(exp_idx)
                    if others:
                        self._push_virtual_fn(others, None, baseline_dm)  # type: ignore[misc]

                objective, D_exp = _make_per_exp_objective(exp_idx)

                # Bounds: [static drift | step0 drift | deltas]
                bounds_i: list[tuple[float, float]] = []
                for si in range(D_static):
                    c = float(state.static_norms[exp_idx, si])
                    d = static_delta_norms_list[si]
                    bounds_i.append((max(0.0, c - d), min(1.0, c + d)))
                for si in range(D_sched):
                    c = float(state.schedule_norms[exp_idx][0, si])
                    d = sched_delta_norms_list[si]
                    bounds_i.append((max(0.0, c - d), min(1.0, c + d)))
                for _k in range(1, L_i):
                    for si in range(D_sched):
                        d = sched_delta_norms_list[si]
                        bounds_i.append((-d, d))

                # Warm start: step0 from Global, small random deltas to break
                # the zero-gradient saddle point at the flat trajectory.
                rng = self.engine.rng
                x0_i = np.zeros(D_exp)
                x0_i[:D_static] = state.static_norms[exp_idx]
                x0_i[D_static:D_static + D_sched] = state.schedule_norms[exp_idx][0]
                if L_i > 1 and D_sched > 0:
                    n_deltas = (L_i - 1) * D_sched
                    delta_start = D_static + D_sched
                    for di in range(n_deltas):
                        d_bound = bounds_i[delta_start + di][1]  # symmetric ±d
                        x0_i[delta_start + di] = rng.uniform(-0.1 * d_bound, 0.1 * d_bound)

                opt = self.engine.run_acquisition_gradient(
                    objective, bounds_i, x0=x0_i,
                    label=f"Traj {exp_idx+1}/{n} (r{round_idx+1})",
                    show_progress=console,
                    n_starts=1,
                )
                total_iters += len(opt.convergence_history)

                if opt.best_x is not None:
                    new_static = opt.best_x[:D_static]
                    # Decode step0 + cumsum(deltas) → absolute schedule
                    step0 = opt.best_x[D_static:D_static + D_sched]
                    new_sched = np.zeros((L_i, D_sched))
                    new_sched[0] = step0
                    if L_i > 1:
                        deltas = opt.best_x[D_static + D_sched:].reshape(L_i - 1, D_sched)
                        new_sched[1:] = step0 + np.cumsum(deltas, axis=0)
                    if not np.allclose(new_static, state.static_norms[exp_idx], atol=1e-6) or \
                       not np.allclose(new_sched, state.schedule_norms[exp_idx], atol=1e-6):
                        improved_this_round = True
                    state.static_norms[exp_idx] = new_static.copy()
                    state.schedule_norms[exp_idx] = new_sched.copy()

                # Push updated experiment back into virtual points
                if can_push:
                    self._pop_virtual_fn()  # type: ignore[misc]
                    all_current = _build_virtual_params(-1)
                    if all_current:
                        self._push_virtual_fn(all_current, None, baseline_dm)  # type: ignore[misc]

                if self.trajectory_step_callback is not None:
                    self.trajectory_step_callback(round_idx, exp_idx, state)

            if not improved_this_round:
                break

        # Clean up virtual points
        if can_push:
            self._pop_virtual_fn()  # type: ignore[misc]

        self.last_baseline_nfev += total_iters
        self.convergence_history["Trajectory"] = []



    def _decode_trajectory_specs(
        self,
        state: _ScheduleState,
    ) -> list[ExperimentSpec]:
        """Decode final state into a list of ExperimentSpec + emit per-layer console summary."""
        # Cache for plot validation.
        self.last_trajectory_points = np.concatenate(state.schedule_norms, axis=0)
        exp_ids: list[int] = []
        for i in range(state.n):
            exp_ids.extend([i] * state.per_exp_L[i])
        self.last_trajectory_exp_ids = exp_ids

        specs_out: list[ExperimentSpec] = []
        for i in range(state.n):
            L_i = state.per_exp_L[i]
            base_params = dict(state.flat_specs[i].initial_params.to_dict())
            traj = state.schedule_norms[i]

            # Apply schedule-phase refinement of continuous static params.
            for si, (code, lo, hi) in enumerate(state.static_params):
                base_params[code] = float(state.static_norms[i, si] * (hi - lo) + lo)
            # Update initial params with step-0 sched values.
            for si, (code, lo, hi) in enumerate(state.sched_params):
                base_params[code] = float(traj[0, si] * (hi - lo) + lo)

            base_params = self.schema.parameters.sanitize_values(base_params, ignore_unknown=True)
            initial = ParameterProposal.from_dict(base_params, source_step=SourceStep.BASELINE)

            entries: list[tuple[int, ParameterProposal]] = []
            for k in range(1, L_i):
                sp: dict[str, Any] = {}
                for si, (code, lo, hi) in enumerate(state.sched_params):
                    sp[code] = float(traj[k, si] * (hi - lo) + lo)
                sp = self.schema.parameters.sanitize_values(sp, ignore_unknown=True)
                entries.append((k, ParameterProposal.from_dict(sp, source_step=SourceStep.BASELINE)))

            trajectories: dict[str, ParameterTrajectory] = {}
            if entries:
                trajectories[state.primary_dim_code] = ParameterTrajectory(
                    dimension=state.primary_dim_code, entries=entries
                )

            specs_out.append(ExperimentSpec(initial_params=initial, trajectories=trajectories))

        # Store trajectory data for callers to display
        self.last_trajectory_schedule_norms = [s.copy() for s in state.schedule_norms]
        self.last_trajectory_sched_params = list(state.sched_params)
        self.last_trajectory_per_exp_L = list(state.per_exp_L)

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
    ) -> torch.Tensor:
        """Negated κ-weighted joint acquisition ``(S,)``, gradient-traversable through ``full_S_NL``."""
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
                evidence_S = self.evidence.joint_batched_tensor(flat_per_candidate).to(dtype=dtype)

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

        # DE bounds: phase_bounds_per_unit replicated N×L times
        # (flat layout: [unit_0, unit_1, ..., unit_{N*L-1}], each unit = D_phase dims)
        de_bounds = phase_bounds_per_unit * (n_proposals * n_schedule_steps)

        def _vec_obj(X_DS: np.ndarray) -> np.ndarray:
            """X_DS: (D_phase × N × L, S) → (S,) via decode, inject, joint κ-acquisition."""
            S = X_DS.shape[1]
            # Reshape: (D_phase * N * L, S) → (S, N, L, D_phase)
            candidates = X_DS.T.reshape(S, n_proposals, n_schedule_steps, D_phase)
            # Build full (S, N, L, n_input) by injecting candidates into prefill_per_n
            full_S_NL = np.broadcast_to(
                prefill_per_n[None, :, None, :],
                (S, n_proposals, n_schedule_steps, n_input),
            ).copy()
            full_S_NL[..., phase_idxs_arr] = candidates
            return self._acquisition_joint_batched_objective(full_S_NL, kappa, perf_range)

        opt = self.engine.run_acquisition_vectorized(
            _vec_obj,
            de_bounds,
            label=label,
            show_progress=show_progress,
        )

        # Decode best_x → (N, L, D_phase) → list[list[dict]]
        if opt.best_x is None:
            result_x = np.array([0.5 * (lo + hi) for (lo, hi) in de_bounds])
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
            sched_set = set(self.trajectory_configs.keys())
            static_codes: list[str] = []
            sched_codes: list[str] = []

            for code in datamodule.input_columns:
                if code in context_codes or code in cat_codes or code in self.fixed_params:
                    static_codes.append(code)
                elif code in sched_set:
                    sched_codes.append(code)
                else:
                    static_codes.append(code)

            D_static = len(static_codes)
            D_sched = len(sched_codes)
            code_to_idx = {c: i for i, c in enumerate(datamodule.input_columns)}

            # --- Refine (Process): single-point acquisition, no schedule ---
            # All params (static + sched) treated as flat static for Refine
            all_p2_codes = static_codes + sched_codes
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
            if D_sched > 0:
                # Build sched DE bounds and delta norms in normalized space
                sched_de_bounds: list[tuple[float, float]] = []
                sched_delta_norms: list[float] = []
                sched_param_tuples: list[tuple[str, float, float]] = []
                for code in sched_codes:
                    idx = code_to_idx[code]
                    lo_norm, hi_norm = float(all_global_bounds[idx, 0]), float(all_global_bounds[idx, 1])
                    sched_de_bounds.append((lo_norm, hi_norm))
                    sched_param_tuples.append((code, lo_norm, hi_norm))
                    delta_raw = self.trust_regions.get(code, 0.0)
                    if delta_raw > 0:
                        _, delta_norm = datamodule.normalize_parameter_bounds(code, 0.0, delta_raw)
                        lo_zero, _ = datamodule.normalize_parameter_bounds(code, 0.0, 0.0)
                        sched_delta_norms.append(abs(delta_norm - lo_zero))
                    else:
                        sched_delta_norms.append(0.0)

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
                    for d_s, c in enumerate(sched_codes):
                        x[code_to_idx[c]] = pts_row[d_s]
                    return x

                sched_perf_range, _ = self._get_acquisition_ranges()

                # Absolute-step encoding: total_vars = L * D_sched.
                abs_bounds: list[tuple[float, float]] = []
                for _k in range(L):
                    for d_s in range(D_sched):
                        abs_bounds.append(sched_de_bounds[d_s])

                flat_x_t = torch.from_numpy(flat_x).float()
                sched_col_indices = torch.tensor(
                    [code_to_idx[c] for c in sched_codes], dtype=torch.long,
                )

                self.logger.debug(
                    f"Trajectory schedule optimization (gradient): L={L}, "
                    f"D_sched={D_sched}, total_vars={L * D_sched}"
                )

                def _schedule_objective_tensor(x_S: torch.Tensor) -> torch.Tensor:
                    """Tensor schedule objective — (S, L*D_sched) → (S,) negated scores."""
                    S = int(x_S.shape[0])
                    steps_SLD = x_S.reshape(S, L, D_sched)
                    full_S_NL = flat_x_t.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
                        S, 1, L, n_input,
                    ).clone()
                    full_S_NL[:, 0, :, sched_col_indices] = steps_SLD.to(dtype=full_S_NL.dtype)
                    scores_neg = self._acquisition_joint_batched_tensor(
                        full_S_NL, kappa, sched_perf_range,
                    )
                    # Soft delta-constraint penalty: λ · Σ_k max(0, |Δstep|−delta)²
                    if D_sched > 0 and L > 1:
                        sched_delta_t = torch.tensor(sched_delta_norms, dtype=x_S.dtype)
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
                    pts = opt.best_x.reshape(L, D_sched)
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
                        pts0 = opt.best_x.reshape(L, D_sched)[0]
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
                # D_sched == 0: sched params not in datamodule input_columns.
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

            # Phase-decomposed acquisition (Domain → Process). Splitting bounds
            # the per-phase DE budget at O(D_domain²) + O(D_process²) instead
            # of O(D_total²) under de_maxiter, which empirically times out
            # at >5 minutes for joint regimes at typical N. Codes whose bounds
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

    # === WRAPPERS ===

    def get_models(self) -> list[Any]:
        """Return empty list (no internal ML models owned by CalibrationSystem)."""
        return []

    def get_model_specs(self) -> dict[str, list[str]]:
        return {}
