from dataclasses import dataclass
from typing import Any, Callable
import functools

import numpy as np

from ...core import DataModule, Dataset, DatasetSchema
from ...core import DataInt, DataObject, DataBool, DataCategorical, DataDomainAxis
from ...core import ParameterProposal, ParameterSchedule, ExperimentSpec
from ...utils import PfabLogger, Mode, NormMethod, SourceStep, SplitType, combined_score
from ..base_system import BaseOrchestrationSystem
from .engine import OptimizationEngine, Optimizer, _OptResult
from .bounds import BoundsManager
from .space import SolutionSpace


# How far static params can drift from their Phase-2 warm start during the
# Schedule phase, as a fraction of normalised range. Internal heuristic that
# anchors per-experiment refinement near Process's globally-coherent placement;
# intentionally not user-exposed (distinct concept from the per-step
# ``trust_regions`` configured for runtime params).
_STATIC_DRIFT_FRAC = 0.2


@dataclass
class _ScheduleState:
    """Per-call state for the iterative Schedule phase (Phase 3).

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
        perf_fn: Callable[[dict[str, Any]], dict[str, float | None]],
        uncertainty_fn: Callable[[np.ndarray], float],
        delta_integrated_evidence_fn: Callable[[np.ndarray], float] | None = None,
        push_virtual_points_fn: Callable[[list[dict[str, Any]], list[float], DataModule | None], None] | None = None,
        pop_virtual_points_fn: Callable[[], None] | None = None,
        n_exp_fn: Callable[[], int] | None = None,
        fit_empty_kde_fn: Callable[[DataModule, int], None] | None = None,
        random_seed: int | None = None,
    ):
        super().__init__(logger, random_seed=random_seed)
        self.perf_fn = perf_fn
        self.uncertainty_fn = uncertainty_fn
        self.delta_integrated_evidence_fn = delta_integrated_evidence_fn
        self.push_virtual_points_fn = push_virtual_points_fn
        self.pop_virtual_points_fn = pop_virtual_points_fn
        self._n_exp_fn = n_exp_fn
        self._fit_empty_kde_fn = fit_empty_kde_fn

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
        self.last_schedule: list[dict[str, Any]] | None = None
        self.convergence_history: dict[str, list[float]] = {}  # label → per-iteration convergence
        # Phase data for validation plots
        self.last_domain_values: list[dict[str, int]] | None = None
        self.last_process_points: list[dict[str, Any]] | None = None
        self.last_schedule_points: np.ndarray | None = None
        self.last_schedule_exp_ids: list[int] | None = None

        # Set ordered weights
        self.schema = schema
        self.perf_names_order = list(schema.performance_attrs.keys())
        self.performance_weights: dict[str, float] = {perf: 1.0 for perf in self.perf_names_order}
        self.parameters = schema.parameters

        self.optimizer: Optimizer = Optimizer.DE
        self.online_optimizer: Optimizer = Optimizer.LBFGSB

        # Running min/max of predicted system performance across training data.
        self._perf_range_min: float | None = None
        self._perf_range_max: float | None = None

        self._schedule_joint_var_limit: int = 200  # threshold for auto-selecting joint vs sequential
        self._suppress_opt_print: bool = False

        # Baseline phase strategy. True (default) = Domain phase first
        # (DataDomainAxis only), then Process phase with domain values held
        # fixed. False = single Process phase over all numeric params jointly.
        # Empirically the joint regime is intractable at typical baseline
        # sizes: N=5 with 3 numeric params (incl. domain axes) timed out at
        # 5+ minutes per call. Split keeps each per-phase DE small enough
        # to converge in seconds.
        self.split_domain_phase: bool = True

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
    def schedule_configs(self) -> dict[str, str]:
        return self.bounds.schedule_configs

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

    @property
    def de_tol(self) -> float:
        return self.engine.de_tol

    @de_tol.setter
    def de_tol(self, value: float) -> None:
        self.engine.de_tol = value

    @property
    def lbfgsb_maxfun(self) -> int | None:
        return self.engine.lbfgsb_maxfun

    @lbfgsb_maxfun.setter
    def lbfgsb_maxfun(self, value: int | None) -> None:
        self.engine.lbfgsb_maxfun = value

    @property
    def lbfgsb_eps(self) -> float:
        return self.engine.lbfgsb_eps

    @lbfgsb_eps.setter
    def lbfgsb_eps(self, value: float) -> None:
        self.engine.lbfgsb_eps = value

    @property
    def schedule_smoothing(self) -> float:
        return self.engine.schedule_smoothing

    @schedule_smoothing.setter
    def schedule_smoothing(self, value: float) -> None:
        self.engine.schedule_smoothing = value

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

    def configure_schedule_parameter(self, code: str, dimension_code: str, force: bool = False) -> None:
        """Declare that a runtime-adjustable parameter should be re-optimised at each step of the given dimension."""
        self.bounds.configure_schedule_parameter(code, dimension_code, force=force)

    def get_tunable_params(self, datamodule: DataModule) -> list[str]:
        """Return codes of parameters the optimizer can actually vary."""
        return self.bounds.get_tunable_params(datamodule)

    def _get_hierarchical_bounds_for_code(self, code: str) -> tuple[float, float]:
        return self.bounds._get_hierarchical_bounds_for_code(code)

    def _get_global_bounds(self, datamodule: DataModule) -> np.ndarray:
        return self.bounds._get_global_bounds(datamodule)

    def _get_trust_region_bounds(self, datamodule: DataModule, current_params: dict[str, Any]) -> np.ndarray:
        return self.bounds._get_trust_region_bounds(datamodule, current_params)

    # ------------------------------------------------------------------
    # Console / reporting
    # ------------------------------------------------------------------

    def _print_optimized_line(self, nfev: int, suffix: str = "") -> None:
        """Deprecated — progress bar finish now handles this."""
        pass  # kept for backward compat; DE/LBFGSB bars show iter+nfev

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

    def _per_candidate_perf(
        self,
        x_norm: np.ndarray,
        perf_range: tuple[float, float] | None,
    ) -> float:
        """Weighted performance score for a single normalized parameter vector."""
        dm = self._active_datamodule
        if dm is None:
            return 0.0
        try:
            params_dict = dm.array_to_params(x_norm.reshape(-1))
        except (ValueError, KeyError):
            return 0.0
        try:
            perf_dict = self.perf_fn(params_dict)
        except Exception:
            return 0.0
        perf_values = [
            float(perf_dict[name]) if perf_dict.get(name) is not None else 0.0  # type: ignore
            for name in self.perf_names_order
            if name in perf_dict
        ]
        sys_perf = self._compute_system_performance(perf_values) if perf_values else 0.0
        if perf_range is not None:
            pmin, pmax = perf_range
            span = pmax - pmin
            sys_perf = (sys_perf - pmin) / span if span > 1e-10 else 0.5
        return float(sys_perf)

    def _acquisition(
        self,
        batch_norm: np.ndarray,
        kappa: float,
        perf_range: tuple[float, float] | None = None,
    ) -> float:
        """Unified acquisition score (higher is better). batch_norm: (L, D_dm)."""
        if batch_norm.ndim != 2:
            batch_norm = batch_norm.reshape(1, -1)
        L = batch_norm.shape[0]
        if L == 0:
            return 0.0

        score = 0.0
        if kappa < 1.0:
            perfs = [self._per_candidate_perf(batch_norm[k], perf_range) for k in range(L)]
            score += (1.0 - kappa) * float(np.mean(perfs))

        if kappa > 0.0 and self.delta_integrated_evidence_fn is not None:
            de = float(self.delta_integrated_evidence_fn(batch_norm))
            score += kappa * de

        return score

    def _acquisition_objective(
        self,
        x_flat: np.ndarray,
        kappa: float,
        perf_range: tuple[float, float] | None = None,
    ) -> float:
        """DE-compatible (negated) acquisition for single-candidate phases.

        x_flat: 1-D normalized parameter vector. Inference and exploration use this path.
        """
        return -self._acquisition(x_flat.reshape(1, -1), kappa, perf_range)

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
                perf_dict = self.perf_fn(params_dict)
                pv = [
                    float(perf_dict[name]) if perf_dict.get(name) is not None else 0.0  # type: ignore
                    for name in self.perf_names_order if name in perf_dict
                ]
                sys_perf = self._compute_system_performance(pv) if pv else 0.0
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
            {dc for dc in self.schedule_configs.values()},
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

        schedules: dict[str, ParameterSchedule] = {}
        if len(proposals) > 1 and step_grid and step_grid[0]:
            dim_key_order_spec = {code: i for i, code in enumerate(self.data_objects.keys())}
            dim_codes = sorted(
                {dc for dc in self.schedule_configs.values()},
                key=lambda dc: dim_key_order_spec.get(dc, 999),
            )
            for dim_code in dim_codes:
                sched_codes = [
                    c for c, dc in self.schedule_configs.items() if dc == dim_code
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
                    schedules[dim_code] = ParameterSchedule(
                        dimension=dim_code, entries=entries,
                    )

        return ExperimentSpec(initial_params=initial, schedules=schedules)

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
                integer_params.append((code, int(lo), int(hi)))
            else:
                try:
                    lo, hi = self.bounds._get_hierarchical_bounds_for_code(code)
                except ValueError:
                    continue
                if lo == -np.inf or hi == np.inf:
                    self.logger.warning(f"Parameter '{code}' has infinite bounds; skipping in baseline.")
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
        sched_set = set(self.schedule_configs.keys())
        domain_axis_sched_dims: set[str] = set()

        if self.schedule_configs:
            sched_dims = set(self.schedule_configs.values())
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
                        if self.schedule_configs[code] in self.fixed_params
                    }

        # --- Detect domain axes among unfixed numeric params ---
        domain_axis_codes = {
            code for code, _, _ in numeric_params
            if code in self.data_objects and isinstance(self.data_objects[code], DataDomainAxis)
        }
        do_split = (
            self.split_domain_phase and bool(domain_axis_codes) and n_numeric > 0
        )

        # --- Phase: Domain (only when split is on and domain axes exist) ---
        structural_values: list[dict[str, int]] | None = None
        domain_specs: list[ExperimentSpec] = []
        if do_split:
            d_params, d_init, d_int_set, d_int_ranges = self._filter_phase_params(
                numeric_params, init_norm, int_set, int_ranges_map, domain_axis_codes
            )
            domain_specs, _, _ = self._run_acquisition_phase(
                n, d_params, integer_params, d_int_set, d_int_ranges,
                d_init, cat_codes, cat_assignments,
                structural_values=None,
                label=f"Domain (D={len(d_params)})", init_evidence=True,
            )
            structural_values = []
            for spec in domain_specs:
                p = spec.initial_params.to_dict()
                structural_values.append({c: int(p[c]) for c in domain_axis_codes if c in p})

        # --- Phase: Process (single call; inputs branched on do_split) ---
        if do_split:
            process_codes = {code for code, _, _ in numeric_params} - domain_axis_codes
            p_params, p_init, p_int_set, p_int_ranges = self._filter_phase_params(
                numeric_params, init_norm, int_set, int_ranges_map, process_codes
            )
            p_init_evidence = False
        else:
            p_params = numeric_params
            p_init = init_norm
            p_int_set = int_set
            p_int_ranges = int_ranges_map
            p_init_evidence = True

        if p_params:
            flat_specs, flat_params, optimized = self._run_acquisition_phase(
                n, p_params, integer_params, p_int_set, p_int_ranges,
                p_init, cat_codes, cat_assignments,
                structural_values=structural_values,
                label=f"Process (D={len(p_params)})", init_evidence=p_init_evidence,
            )
        elif do_split:
            # Pure-domain case: all non-domain params fixed. Use Domain results as-is.
            flat_specs = domain_specs
            flat_params = np.zeros((n, 0))
            optimized = np.zeros((n, 0))
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
                flat_specs.append(ExperimentSpec(initial_params=proposal, schedules={}))

        # Store phase points for validation plot
        self.last_process_points = [spec.initial_params.to_dict() for spec in flat_specs] if flat_specs else None
        self.last_domain_values = list(structural_values) if structural_values is not None else None

        # Console: show process params per experiment
        console = self.logger._console_output_enabled
        if console and flat_specs:
            _D = "\033[2m"
            _R = "\033[0m"
            _S = "\033[38;2;39;39;42m"  # Zinc-800
            for i, spec in enumerate(flat_specs):
                p = spec.initial_params.to_dict()
                parts = [
                    f"{k[:3]}={v:.3f}" if isinstance(v, float) else f"{k[:3]}={v}"
                    for k, v in p.items() if k not in self.fixed_params
                ]
                print(f"    {_S}baseline_{i+1:02d}{_R}  {_D}{'  '.join(parts)}{_R}")

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
            fixed_sched = [d for d in set(self.schedule_configs.values()) if d in self.fixed_params]
            if fixed_sched:
                L = max(int(self.fixed_params[d]) for d in fixed_sched)
                per_exp_L = [L] * n

        # --- Schedule phase (if scheduled params exist and L > 1) ---
        if sched_set and per_exp_L is not None and max(per_exp_L) > 1:
            # Determine primary dimension code: prefer an unfixed domain-axis
            # sched dim; fall back to a fixed-dim sched when --design-intent
            # pins the dimension (e.g. n_layers=4).
            dim_codes_for_sched = sorted(set(self.schedule_configs.values()) & domain_axis_sched_dims)
            if dim_codes_for_sched:
                primary_dim_code = dim_codes_for_sched[0]
            else:
                fixed_dim_codes = sorted(
                    d for d in set(self.schedule_configs.values()) if d in self.fixed_params
                )
                primary_dim_code = fixed_dim_codes[0] if fixed_dim_codes else ""

            sched_params_list: list[tuple[str, float, float]] = []
            for code in sorted(sched_set):
                if code in domain_axis_sched_dims:
                    continue
                lo, hi = self.bounds._get_hierarchical_bounds_for_code(code)
                sched_params_list.append((code, lo, hi))

            if sched_params_list:
                # Continuous static params eligible for schedule-phase
                # drift refinement (excludes integer / domain-axis / sched).
                static_params_list = [
                    (code, lo, hi) for code, lo, hi in continuous_params
                    if code not in sched_set
                ]
                specs = self._phase3_schedule(
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
            sched_params=[],
            per_exp_L=[1] * n,
            trust_regions={},
            int_set=all_int_set,
            int_ranges_map=all_int_ranges,
            schedule_smoothing=0.0,
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

        def _acquisition_batch_objective(x_flat: np.ndarray) -> float:
            """Maximize Δ∫E over joint N-batch placement (κ=1)."""
            pts = space.decode(x_flat)  # (N, D_point)
            X_batch = prior_fill.copy()
            for si, col_idx in phase_col_map:
                for i in range(n):
                    X_batch[i, col_idx] = pts[i, si]
            return -self._acquisition(X_batch, kappa=1.0)

        # Build init_norm for the phase-local space (remap original indices)
        merged_init = np.zeros((n, len(all_phase_params)))
        for si, (d_i, _, _, _) in enumerate(all_phase_params):
            if d_i < init_de.shape[1]:
                merged_init[:, si] = init_de[:, d_i]

        init_pop = space.build_init_population(self.engine.rng, merged_init)

        self.logger.info(
            f"Phase ({label}): N={n}, D_static={len(all_phase_params)}, "
            f"D_sched=0, total_vars={space.total_vars}"
        )

        opt = self.engine._run_de(
            _acquisition_batch_objective, space.bounds, init_pop=init_pop,
            integrality=space.integrality, label=label, show_progress=console,
        )
        if not hasattr(self, 'last_baseline_nfev'):
            self.last_baseline_nfev: int = opt.nfev
        else:
            self.last_baseline_nfev += opt.nfev
        self.convergence_history[label] = opt.convergence_history

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

    def _phase3_schedule(
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
        """Phase 3 (Schedule): per-experiment joint over (static + schedule).

        For each experiment, jointly optimise its continuous static params (in
        a trust region around Phase-2 warm start), step 0 of the schedule
        (free in [0, 1]), and the L_i-1 offsets — all in one DE call. Other
        experiments' current full designs are held in the X_batch background.
        Outer passes iterate until max change across (static, step 0, offsets)
        falls below tol (typical: 2-3 passes converge). Integer static and
        domain-axis params stay frozen at their Process / Domain values.

        Smoothness retains its full lookahead within the per-experiment DE.
        Cross-experiment coupling becomes symmetric at convergence.
        """
        if len(sched_params) == 0 or max(per_exp_L) <= 1:
            return flat_specs

        state = self._init_schedule_state(
            n, flat_specs, sched_params, static_params or [],
            per_exp_L, primary_dim_code,
        )
        if state is None:
            return flat_specs

        self._run_schedule_passes(state)
        return self._decode_schedule_specs(state)

    def _init_schedule_state(
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
            sched_delta_norms=sched_delta_norms,
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

    def _run_schedule_passes(
        self,
        state: _ScheduleState,
        max_passes: int = 4,
        tol: float = 0.001,
    ) -> None:
        """Outer pass loop — mutates state.static_norms / schedule_norms in place."""
        console = self.logger._console_output_enabled
        if console:
            print(f"\n  Schedule (D={state.D_static}+{state.D_sched})")

        converged_pass: int | None = None

        for pass_idx in range(max_passes):
            if console:
                print(f"\n  Pass {pass_idx + 1}/{max_passes}")

            max_static_delta = 0.0
            max_sched_delta = 0.0
            for i_exp in range(state.n):
                old_static = state.static_norms[i_exp].copy()
                old_sched = state.schedule_norms[i_exp].copy()
                new_static, new_sched = self._optimise_schedule_for_experiment(state, i_exp)
                if state.D_static > 0:
                    max_static_delta = max(
                        max_static_delta, float(np.max(np.abs(new_static - old_static)))
                    )
                max_sched_delta = max(
                    max_sched_delta, float(np.max(np.abs(new_sched - old_sched)))
                )
                state.static_norms[i_exp] = new_static
                state.schedule_norms[i_exp] = new_sched

            self.convergence_history[f"Schedule pass {pass_idx + 1}"] = []
            self.logger.info(
                f"Schedule pass {pass_idx + 1}: max Δstatic={max_static_delta:.4f}, "
                f"max Δsched={max_sched_delta:.4f}"
            )
            max_delta = max(max_static_delta, max_sched_delta)

            if max_delta < tol:
                converged_pass = pass_idx + 1
                if console:
                    print(f"\n  Converged at pass {converged_pass}")
                break
        if converged_pass is None and console:
            print(f"\n  Hit max_passes={max_passes}")

    def _optimise_schedule_for_experiment(
        self,
        state: _ScheduleState,
        i_exp: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the inner DE for one experiment; return (new_static, new_trajectory).

        Variable layout in the DE vector:
          [0 .. D_static)                   continuous static (drift trust region)
          [D_static .. D_static+D_sched)    step 0 (free in [0, 1])
          [D_static+D_sched ..)             offsets for steps 1..L_i-1 (in [-δ, +δ])
        """
        bounds = self._build_schedule_per_exp_bounds(state, i_exp)
        init = self._build_schedule_per_exp_init(state, i_exp)
        background = self._build_schedule_background(state, skip=i_exp)
        objective = self._make_schedule_objective(state, i_exp, background)
        n_vars = len(init)

        # Inner DE sizing — respect user de_maxiter/de_popsize as upper bounds
        # so test runs (e.g. de-maxiter=10) actually throttle the inner loop.
        # scipy DE requires population > 4, so floor at 5.
        popsize = min(max(10, 4 * n_vars), max(self.de_popsize, 5))
        init_pop = np.tile(init, (popsize, 1))
        jitter = self.engine.rng.normal(0, 0.02, size=init_pop.shape)
        init_pop = init_pop + jitter
        for v_idx, (lo_b, hi_b) in enumerate(bounds):
            init_pop[:, v_idx] = np.clip(init_pop[:, v_idx], lo_b, hi_b)
        init_pop[0] = init  # keep one centre exact

        inner_maxiter = min(max(40, 15 * n_vars), max(self.de_maxiter, 5))
        opt = self.engine._run_de(
            objective, bounds, init_pop=init_pop,
            label=f"Schedule {i_exp + 1}/{state.n}",
            show_progress=self.logger._console_output_enabled,
            maxiter=inner_maxiter,
            popsize=popsize,
        )
        best_x = opt.best_x if opt.best_x is not None else init
        self.last_baseline_nfev += opt.nfev
        return self._decode_schedule_per_exp_x(state, best_x, i_exp)

    def _build_schedule_background(
        self,
        state: _ScheduleState,
        skip: int,
    ) -> np.ndarray:
        """All experiments' current rows (static + schedule), except experiment ``skip``."""
        rows: list[np.ndarray] = []
        for i_exp in range(state.n):
            if i_exp == skip:
                continue
            for k in range(state.per_exp_L[i_exp]):
                row = state.exp_base_rows[i_exp].copy()
                for si, col_idx in state.static_col_map:
                    row[col_idx] = state.static_norms[i_exp, si]
                for si, col_idx in state.sched_col_map:
                    row[col_idx] = state.schedule_norms[i_exp][k, si]
                rows.append(row)
        return np.stack(rows) if rows else np.zeros((0, state.n_dm_cols))

    def _build_schedule_per_exp_bounds(
        self,
        state: _ScheduleState,
        i_exp: int,
    ) -> list[tuple[float, float]]:
        """Bounds for the per-experiment DE vector — order matches variable layout."""
        bounds: list[tuple[float, float]] = []
        # Static stays within ±_STATIC_DRIFT_FRAC of its Phase-2 warm start.
        for si in range(state.D_static):
            centre = state.static_norms[i_exp, si]
            lo_b = max(0.0, centre - _STATIC_DRIFT_FRAC)
            hi_b = min(1.0, centre + _STATIC_DRIFT_FRAC)
            if hi_b <= lo_b:
                hi_b = min(1.0, lo_b + 1e-6)
            bounds.append((lo_b, hi_b))
        # Step 0 free in [0, 1].
        bounds.extend([(0.0, 1.0)] * state.D_sched)
        # Offsets for steps 1..L_i-1 in [-δ, +δ].
        n_free_steps = state.per_exp_L[i_exp] - 1
        for _k in range(n_free_steps):
            for dnorm in state.sched_delta_norms:
                d_eff = dnorm if dnorm > 0 else 1.0
                bounds.append((-d_eff, d_eff))
        return bounds

    def _build_schedule_per_exp_init(
        self,
        state: _ScheduleState,
        i_exp: int,
    ) -> np.ndarray:
        """Init vector — current static + step 0, with offsets from current trajectory diffs."""
        L_i = state.per_exp_L[i_exp]
        n_vars = state.D_static + state.D_sched + (L_i - 1) * state.D_sched
        init = np.zeros(n_vars)
        init[:state.D_static] = state.static_norms[i_exp]
        init[state.D_static:state.D_static + state.D_sched] = state.schedule_norms[i_exp][0]
        traj_curr = state.schedule_norms[i_exp]
        for k in range(1, L_i):
            base = state.D_static + state.D_sched + (k - 1) * state.D_sched
            init[base:base + state.D_sched] = traj_curr[k] - traj_curr[k - 1]
        return init

    def _make_schedule_objective(
        self,
        state: _ScheduleState,
        i_exp: int,
        background: np.ndarray,
    ) -> Callable[[np.ndarray], float]:
        """Build the per-experiment DE objective (closure over state + background)."""
        L_i = state.per_exp_L[i_exp]
        D_static = state.D_static
        D_sched = state.D_sched
        lam_smooth = float(self.schedule_smoothing)

        def objective(x_flat: np.ndarray) -> float:
            stat = x_flat[:D_static]
            step0 = x_flat[D_static:D_static + D_sched]
            # Reconstruct full trajectory by integrating offsets from step 0.
            traj = np.empty((L_i, D_sched))
            traj[0] = step0
            for k in range(1, L_i):
                base = D_static + D_sched + (k - 1) * D_sched
                traj[k] = traj[k - 1] + x_flat[base:base + D_sched]

            # Build candidate rows: base row + new static + per-step sched.
            cand = np.empty((L_i, state.n_dm_cols))
            for k in range(L_i):
                cand[k] = state.exp_base_rows[i_exp]
                for si, col_idx in state.static_col_map:
                    cand[k, col_idx] = stat[si]
                for si, col_idx in state.sched_col_map:
                    cand[k, col_idx] = traj[k, si]

            X_batch = np.concatenate([background, cand], axis=0)
            neg_score = -self._acquisition(X_batch, kappa=1.0)

            # Trust-region penalty: step-to-step jumps within delta_norm.
            # Offset bounds enforce this hard, but a soft penalty near the
            # boundary keeps the search stable.
            tr_penalty = 0.0
            for si, dnorm in enumerate(state.sched_delta_norms):
                if dnorm <= 0:
                    continue
                diffs = np.abs(np.diff(traj[:, si]))
                excess = np.maximum(diffs - dnorm, 0.0)
                tr_penalty += float((excess * excess).sum())

            # Second-difference smoothness: penalises zigzag, allows steady drift.
            smooth = 0.0
            if lam_smooth > 0 and L_i >= 3:
                sd = traj[2:] - 2.0 * traj[1:-1] + traj[:-2]
                smooth = float(lam_smooth * (sd * sd).sum())

            return neg_score + tr_penalty * 10.0 + smooth

        return objective

    def _decode_schedule_per_exp_x(
        self,
        state: _ScheduleState,
        x_flat: np.ndarray,
        i_exp: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode an inner-DE result vector into (new_static, new_trajectory)."""
        L_i = state.per_exp_L[i_exp]
        new_static = x_flat[:state.D_static].copy()
        new_traj = np.empty((L_i, state.D_sched))
        new_traj[0] = x_flat[state.D_static:state.D_static + state.D_sched]
        for k in range(1, L_i):
            base = state.D_static + state.D_sched + (k - 1) * state.D_sched
            new_traj[k] = new_traj[k - 1] + x_flat[base:base + state.D_sched]
        return new_static, new_traj

    def _decode_schedule_specs(
        self,
        state: _ScheduleState,
    ) -> list[ExperimentSpec]:
        """Decode final state into a list of ExperimentSpec + emit per-layer console summary."""
        # Cache for plot validation.
        self.last_schedule_points = np.concatenate(state.schedule_norms, axis=0)
        exp_ids: list[int] = []
        for i in range(state.n):
            exp_ids.extend([i] * state.per_exp_L[i])
        self.last_schedule_exp_ids = exp_ids

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

            schedules: dict[str, ParameterSchedule] = {}
            if entries:
                schedules[state.primary_dim_code] = ParameterSchedule(
                    dimension=state.primary_dim_code, entries=entries
                )

            specs_out.append(ExperimentSpec(initial_params=initial, schedules=schedules))

        # Per-layer schedule values grouped by param.
        if self.logger._console_output_enabled:
            _D = "\033[2m"
            _R = "\033[0m"
            _S = "\033[38;2;39;39;42m"  # Zinc-800
            for si, (code, lo, hi) in enumerate(state.sched_params):
                print(f"    {code}")
                for i in range(state.n):
                    vals = [float(state.schedule_norms[i][k, si] * (hi - lo) + lo)
                            for k in range(state.per_exp_L[i])]
                    vals_str = " → ".join(f"{v:.1f}" for v in vals)
                    print(f"    {_S}baseline_{i + 1:02d}{_R}  {_D}{vals_str}{_R}")

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
                c for c in self.schedule_configs if c not in self.trust_regions
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

            col_map = datamodule.get_onehot_column_map()
            context_codes = set(datamodule.context_feature_codes)
            sched_set = set(self.schedule_configs.keys())
            static_codes: list[str] = []
            sched_codes: list[str] = []

            for code in datamodule.input_columns:
                if code in context_codes or code in col_map or code in self.fixed_params:
                    static_codes.append(code)
                elif code in sched_set:
                    sched_codes.append(code)
                else:
                    static_codes.append(code)

            D_static = len(static_codes)
            D_sched = len(sched_codes)
            code_to_idx = {c: i for i, c in enumerate(datamodule.input_columns)}

            # --- Phase 2 (Process): single-point acquisition, no schedule ---
            # All params (static + sched) treated as flat static for Phase 2
            all_p2_codes = static_codes + sched_codes
            D_p2 = len(all_p2_codes)

            if is_online:
                p2_bounds = self.bounds._get_trust_region_bounds(datamodule, working_params)  # type: ignore[arg-type]
            else:
                p2_bounds = all_global_bounds

            # Build p2 bounds for the merged flat vector
            p2_static_bounds: list[tuple[float, float]] = []
            for code in all_p2_codes:
                idx = code_to_idx[code]
                p2_static_bounds.append((float(p2_bounds[idx, 0]), float(p2_bounds[idx, 1])))

            def _process_acquisition_objective(pts: np.ndarray) -> float:
                """Exploration/inference Process DE objective: single-point UCB."""
                x = np.zeros(n_input)
                for i_s, c in enumerate(all_p2_codes):
                    x[code_to_idx[c]] = pts[0, i_s]
                return objective(x)

            opt_for_p2 = self.online_optimizer if is_online else None
            n_rounds_p2 = 0 if is_online else n_optimization_rounds

            x0_p2: np.ndarray | None = None
            if working_params:
                full_arr = datamodule.params_to_array(working_params)
                x0_p2 = np.array([full_arr[code_to_idx[c]] for c in all_p2_codes])

            opt_p2, static_out_p2, _ = self.engine.run(
                _process_acquisition_objective, N=1, D_static=D_p2, D_sched=0, L=1,
                static_bounds=p2_static_bounds, sched_bounds=[], sched_deltas=np.array([]),
                optimizer=opt_for_p2,
                default_optimizer=self.optimizer,
                x0=x0_p2,
                n_restarts=n_rounds_p2,
                label="Process", show_progress=console,
            )

            self.last_opt_nfev = opt_p2.nfev
            self.last_opt_n_starts = opt_p2.n_starts
            self.last_opt_score = opt_p2.score

            # Build flat params from Phase 2
            flat_x = np.zeros(n_input)
            if opt_p2.best_x is not None:
                p2_vals = static_out_p2[0]
                for i_s, c in enumerate(all_p2_codes):
                    flat_x[code_to_idx[c]] = p2_vals[i_s]

            # --- Phase 3 (Schedule): fix static, optimize offsets ---
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

                sched_space = SolutionSpace(
                    n_experiments=1,
                    static_params=[],
                    sched_params=sched_param_tuples,
                    per_exp_L=[L],
                    trust_regions={},
                    int_set=set(),
                    int_ranges_map={},
                    schedule_smoothing=self.schedule_smoothing,
                    sched_de_bounds=sched_de_bounds,
                    sched_delta_norms=sched_delta_norms,
                )

                def _pts_row_to_dm(pts_row: np.ndarray) -> np.ndarray:
                    """Map decoded sched-only row + fixed static to datamodule input."""
                    x = flat_x.copy()
                    for d_s, c in enumerate(sched_codes):
                        x[code_to_idx[c]] = pts_row[d_s]
                    return x

                sched_perf_range, _ = self._get_acquisition_ranges()

                def _schedule_acquisition_objective(x_flat: np.ndarray) -> float:
                    """Exploration/inference schedule objective: joint over L layers.

                    Adds a small step-to-step smoothness penalty as a tie-breaker
                    under the integrated-evidence objective.
                    """
                    pts = sched_space.decode(x_flat)
                    X_batch = np.stack([_pts_row_to_dm(pts[k]) for k in range(L)])
                    neg_score = -self._acquisition(X_batch, kappa, perf_range=sched_perf_range)
                    neg_score += sched_space.smoothing_penalty(x_flat, neg_score)
                    return neg_score

                self.logger.debug(
                    f"Phase 3 schedule optimization: L={L}, D_static=0, "
                    f"D_sched={D_sched}, total_vars={sched_space.total_vars}"
                )

                opt = self.engine._run_de(
                    _schedule_acquisition_objective, sched_space.bounds, label="Schedule", show_progress=console,
                )
                self.last_opt_nfev += opt.nfev
                self.convergence_history["Schedule"] = opt.convergence_history

                dm_input_set = set(datamodule.input_columns)
                non_dm_sched = {c for c in self.schedule_configs if c not in dm_input_set}

                proposals: list[dict[str, Any]] = []
                if opt.best_x is not None:
                    pts = sched_space.decode(opt.best_x)
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

                self.last_schedule = list(proposals)

                if opt.best_x is not None:
                    try:
                        x0_step = _pts_row_to_dm(sched_space.decode(opt.best_x)[0])
                        _params = datamodule.array_to_params(x0_step)
                        _perf_dict = self.perf_fn(_params)
                        _pv = [
                            float(v) if (v := _perf_dict.get(n)) is not None else 0.0
                            for n in self.perf_names_order if n in _perf_dict
                        ]
                        raw_perf = self._compute_system_performance(_pv) if _pv else 0.0
                        if self._perf_range_min is not None and self._perf_range_max is not None:
                            span = self._perf_range_max - self._perf_range_min
                            self.last_opt_perf = (raw_perf - self._perf_range_min) / span if span > 1e-10 else 0.5
                        else:
                            self.last_opt_perf = raw_perf
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
                non_dm_sched = {c for c in self.schedule_configs if c not in dm_input_set}

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

                self.last_schedule = list(proposals)

                try:
                    _params = datamodule.array_to_params(flat_x)
                    _perf_dict = self.perf_fn(_params)
                    _pv = [
                        float(v) if (v := _perf_dict.get(n)) is not None else 0.0
                        for n in self.perf_names_order if n in _perf_dict
                    ]
                    raw_perf = self._compute_system_performance(_pv) if _pv else 0.0
                    if self._perf_range_min is not None and self._perf_range_max is not None:
                        span = self._perf_range_max - self._perf_range_min
                        self.last_opt_perf = (raw_perf - self._perf_range_min) / span if span > 1e-10 else 0.5
                    else:
                        self.last_opt_perf = raw_perf
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

            opt_for_step = self.online_optimizer if is_online else None

            def acq_single(pts: np.ndarray) -> float:
                """Evaluate acquisition/inference at a single point."""
                return objective(pts[0])

            opt, static_out, _sched_out = self.engine.run(
                acq_single, N=1, D_static=n_input, D_sched=0, L=1,
                static_bounds=bounds.tolist(), sched_bounds=[], sched_deltas=np.array([]),
                optimizer=opt_for_step,
                default_optimizer=self.optimizer,
                x0=datamodule.params_to_array(working_params) if working_params else None,
                n_restarts=n_rounds,
                label="Optimizing", show_progress=console,
            )

            self.last_opt_nfev = opt.nfev
            if console:
                self._print_optimized_line(opt.nfev)
            self.last_opt_n_starts = opt.n_starts
            self.last_opt_score = opt.score

            best_x = static_out[0] if opt.best_x is not None else None
            if best_x is not None:
                try:
                    _params = datamodule.array_to_params(best_x)
                    _perf_dict = self.perf_fn(_params)
                    _perf_values = [
                        float(v) if (v := _perf_dict.get(n)) is not None else 0.0
                        for n in self.perf_names_order if n in _perf_dict
                    ]
                    raw_perf = self._compute_system_performance(_perf_values) if _perf_values else 0.0
                    if self._perf_range_min is not None and self._perf_range_max is not None:
                        span = self._perf_range_max - self._perf_range_min
                        self.last_opt_perf = (raw_perf - self._perf_range_min) / span if span > 1e-10 else 0.5
                    else:
                        self.last_opt_perf = raw_perf
                    self.last_opt_unc = float(self.uncertainty_fn(best_x))
                except Exception:
                    self.last_opt_perf = 0.0
                    self.last_opt_unc = 0.0
            else:
                self.last_opt_perf = 0.0
                self.last_opt_unc = 0.0

            active_optimizer = opt_for_step or self.optimizer
            self.logger.info(
                f"{active_optimizer.value}: {opt.n_starts} start(s), {opt.nfev} evals, score={opt.score:.6f}"
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

            self.last_schedule = None

        proposal_summary = {k: round(v, 4) if isinstance(v, float) else v for k, v in proposals[0].items()}
        self.logger.info(f"Calibration proposal: {proposal_summary}")

        return self._build_experiment_spec(proposals, step_grid, source_step)

    # === WRAPPERS ===

    def get_models(self) -> list[Any]:
        """Return empty list (no internal ML models owned by CalibrationSystem)."""
        return []

    def get_model_specs(self) -> dict[str, list[str]]:
        return {}
