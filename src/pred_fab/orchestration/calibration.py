from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable
import warnings
import functools

import numpy as np
from scipy.optimize import minimize, differential_evolution

from ..core import DataModule, Dataset, DatasetSchema
from ..core import DataInt, DataReal, DataObject, DataBool, DataCategorical, DataDomainAxis
from ..core import ParameterProposal, ParameterSchedule, ExperimentSpec
from ..utils import PfabLogger, ProgressBar, Mode, NormMethod, SourceStep, SplitType, combined_score
from .base_system import BaseOrchestrationSystem


class Optimizer(Enum):
    """Optimization backend for the calibration acquisition function."""
    LBFGSB = "lbfgsb"  # gradient-based multi-start (fast, local)
    DE     = "de"       # differential evolution (global, slower)


@dataclass
class _OptResult:
    """Raw output from an optimizer backend."""
    best_x: np.ndarray | None
    nfev: int
    n_starts: int
    score: float  # negated objective (higher = better)


# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ======================================================================
# OptimizationEngine — pure optimization, no schema/calibration knowledge
# ======================================================================

class OptimizationEngine:
    """Numerical optimization backend: DE and L-BFGS-B with joint schedule support."""

    def __init__(self, logger: PfabLogger, random_seed: int | None = None):
        self.logger = logger
        self._random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        # DE optimizer parameters (global, population-based + L-BFGS-B polish)
        self.de_maxiter: int = 1000
        self.de_popsize: int = 15

        # L-BFGS-B optimizer parameters (gradient-based, multi-start)
        self.lbfgsb_maxfun: int | None = None
        self.lbfgsb_eps: float = 1e-3

        # Schedule smoothing: penalizes speed changes between adjacent layers
        self.schedule_smoothing: float = 0.25

    def run(
        self,
        per_step_fn: Callable[[np.ndarray], float],
        N: int,
        D_static: int,
        D_sched: int,
        L: int,
        static_bounds: list[tuple[float, float]],
        sched_bounds: list[tuple[float, float]],
        sched_deltas: np.ndarray,
        *,
        optimizer: Optimizer | None = None,
        default_optimizer: Optimizer = Optimizer.DE,
        init_pop: np.ndarray | None = None,
        integrality_static: list[bool] | None = None,
        x0: np.ndarray | None = None,
        n_restarts: int = 0,
        label: str = "Optimizing",
        show_progress: bool = False,
    ) -> tuple[_OptResult, np.ndarray, np.ndarray]:
        """Single optimization engine for all calibration use cases.

        Vector layout per unit: [static, sched_step0, offset_1, ..., offset_{L-1}]
        Total vars: N x (D_static + D_sched + (L-1) x D_sched)

        Returns (opt_result, static_out[N, D_static], sched_out[N, L, D_sched]).
        """
        active_optimizer = optimizer or default_optimizer
        D_unit = D_static + D_sched + max(L - 1, 0) * D_sched
        n_vars = N * D_unit

        # --- 1. Build bounds ---
        all_bounds: list[tuple[float, float]] = []
        integrality: list[bool] | None = None
        if integrality_static is not None and any(integrality_static):
            integrality = []

        for _u in range(N):
            for d in range(D_static):
                all_bounds.append(static_bounds[d])
                if integrality is not None:
                    integrality.append(integrality_static[d])  # type: ignore[index]
            for d in range(D_sched):
                all_bounds.append(sched_bounds[d])
                if integrality is not None:
                    integrality.append(False)
            for _k in range(1, L):
                for d in range(D_sched):
                    dn = float(sched_deltas[d])
                    all_bounds.append((-dn, dn))
                    if integrality is not None:
                        integrality.append(False)

        # --- 2. Build objective wrapper ---
        schedule_smoothing = self.schedule_smoothing

        def _objective(x_flat: np.ndarray) -> float:
            units = x_flat.reshape(N, D_unit)
            step_sum = 0.0

            for k in range(L):
                pts = np.zeros((N, D_static + D_sched))
                for u in range(N):
                    # Static dims: same for all steps
                    pts[u, :D_static] = units[u, :D_static]
                    if D_sched > 0:
                        # Scheduled dims: step0 + cumulative offsets, clipped to sched_bounds
                        step0 = units[u, D_static:D_static + D_sched]
                        abs_val = step0.copy()
                        for kk in range(1, k + 1):
                            off_start = D_static + D_sched + (kk - 1) * D_sched
                            abs_val = abs_val + units[u, off_start:off_start + D_sched]
                        # Clip to sched_bounds
                        for d in range(D_sched):
                            lo, hi = sched_bounds[d]
                            if abs_val[d] < lo or abs_val[d] > hi:
                                abs_val[d] = np.clip(abs_val[d], lo, hi)
                        pts[u, D_static:] = abs_val

                step_sum += per_step_fn(pts)

            step_avg = step_sum / L

            # Smoothing penalty
            if L > 1 and D_sched > 0 and schedule_smoothing > 0:
                total_penalty = 0.0
                for u in range(N):
                    sched_vals = np.zeros((L, D_sched))
                    step0 = units[u, D_static:D_static + D_sched]
                    sched_vals[0] = step0
                    for kk in range(1, L):
                        off_start = D_static + D_sched + (kk - 1) * D_sched
                        sched_vals[kk] = sched_vals[kk - 1] + units[u, off_start:off_start + D_sched]
                    sf = OptimizationEngine._schedule_smoothing_factor(sched_vals, sched_deltas, schedule_smoothing)
                    total_penalty += abs(step_avg) * (1.0 - sf)
                return step_avg + total_penalty / N
            return step_avg

        # --- 3. Run optimizer ---
        if active_optimizer == Optimizer.DE:
            opt = self._run_de(
                _objective,
                all_bounds,
                init_pop=init_pop,
                integrality=integrality,
                label=label,
                show_progress=show_progress,
            )
        else:
            # L-BFGS-B multi-start
            x0_list: list[np.ndarray] = []
            if x0 is not None:
                # Pad x0 to full vector (assumes N=1, L=1 for L-BFGS-B)
                if x0.size == D_static:
                    x0_list.append(x0)
                else:
                    x0_list.append(x0[:n_vars] if x0.size >= n_vars else x0)
            bounds_arr = np.array(all_bounds)
            for _ in range(n_restarts):
                x0_list.append(self.rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1]))
            if not x0_list:
                x0_list.append(self.rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1]))

            n_dims = n_vars
            max_fun = self.lbfgsb_maxfun if self.lbfgsb_maxfun is not None else max(100, 10 * (n_dims + 1))
            eps = self.lbfgsb_eps
            total_starts = len(x0_list)

            best_x, best_val = None, np.inf
            total_nfev = 0
            bar = ProgressBar(label, max_iter=total_starts) if show_progress else None
            for i, x0_i in enumerate(x0_list):
                if bar:
                    bar.step()
                try:
                    res = minimize(
                        fun=_objective, x0=x0_i, bounds=bounds_arr, method='L-BFGS-B',
                        options={'eps': eps, 'maxfun': max_fun},
                    )
                    total_nfev += res.nfev
                    if res.fun < best_val:
                        best_val = res.fun
                        best_x = res.x
                    self.logger.debug(
                        f"  start {i + 1}/{total_starts}: val={res.fun:.6f}, nfev={res.nfev}, converged={res.success}"
                    )
                except Exception as e:
                    self.logger.warning(f"L-BFGS-B round {i + 1} failed: {e}")

            if bar:
                bar.finish(nfev=total_nfev)

            opt = _OptResult(
                best_x=best_x,
                nfev=total_nfev,
                n_starts=total_starts,
                score=float(-best_val) if best_val != np.inf else 0.0,
            )

        # --- 4. Decode result ---
        if opt.best_x is not None:
            units = opt.best_x.reshape(N, D_unit)
        else:
            units = np.full((N, D_unit), 0.5)

        static_out = units[:, :D_static]
        sched_out = np.zeros((N, L, D_sched))
        if D_sched > 0:
            for u in range(N):
                step0 = units[u, D_static:D_static + D_sched]
                sched_out[u, 0] = step0
                for k in range(1, L):
                    off_start = D_static + D_sched + (k - 1) * D_sched
                    sched_out[u, k] = sched_out[u, k - 1] + units[u, off_start:off_start + D_sched]
                    # Clip to sched_bounds
                    for d in range(D_sched):
                        lo, hi = sched_bounds[d]
                        sched_out[u, k, d] = np.clip(sched_out[u, k, d], lo, hi)

        return opt, static_out, sched_out

    def _run_de(
        self,
        objective: Callable,
        bounds: list[tuple[float, float]],
        *,
        init_pop: np.ndarray | None = None,
        integrality: list[bool] | None = None,
        label: str = "Optimizing",
        show_progress: bool = False,
    ) -> _OptResult:
        """Unified differential evolution wrapper."""
        maxiter = self.de_maxiter
        popsize = self.de_popsize
        has_int = integrality is not None and any(integrality)
        bar = ProgressBar(label, max_iter=maxiter) if show_progress else None

        def _progress(xk: Any, convergence: Any) -> None:
            if bar:
                bar.step()

        de_kwargs: dict[str, Any] = dict(
            func=objective,
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            mutation=(0.5, 1.0),
            recombination=0.7,
            tol=0.001,
            polish=not has_int,
            callback=_progress,
        )
        if init_pop is not None:
            de_kwargs["init"] = init_pop
        else:
            de_kwargs["init"] = "latinhypercube"
        if integrality is not None:
            de_kwargs["integrality"] = integrality
        if self._random_seed is not None:
            de_kwargs["seed"] = int(self.rng.randint(0, 2**31 - 1))

        result = differential_evolution(**de_kwargs)  # type: ignore[call-overload]

        if bar:
            bar.finish(nfev=result.nfev)

        return _OptResult(
            best_x=result.x,
            nfev=result.nfev,
            n_starts=1,
            score=float(-result.fun),
        )

    @staticmethod
    def _schedule_smoothing_factor(
        scheduled_values: np.ndarray,
        deltas: np.ndarray,
        lam: float,
    ) -> float:
        """Multiplicative penalty in (0, 1] for schedule jumps."""
        if lam <= 0 or scheduled_values.shape[0] <= 1:
            return 1.0
        factor = 1.0
        for k in range(1, scheduled_values.shape[0]):
            for d in range(scheduled_values.shape[1]):
                if deltas[d] <= 0:
                    continue
                change = abs(scheduled_values[k, d] - scheduled_values[k - 1, d])
                frac = min(change / deltas[d], 1.0)
                factor *= (1.0 - lam * frac)
        return factor

    def _wrap_mpc_objective(
        self,
        base_objective: Callable,
        datamodule: DataModule,
        bounds_fn: Callable[[DataModule, dict[str, Any]], np.ndarray],
        depth: int,
        discount: float,
    ) -> Callable:
        """Wrap base_objective with MPC lookahead for online adaptation.

        At each candidate X, simulates `depth` future steps via inner L-BFGS-B
        solves within trust-region bounds. Returns discounted sum of scores.
        """
        if depth <= 0:
            return base_objective

        class _MpcObjective:
            """Callable MPC wrapper that tracks total inner evaluations."""

            def __init__(self, weight_sum: float):
                self._eval_counter = [0]
                self._weight_sum = weight_sum

            def __call__(self, X: np.ndarray) -> float:
                base_score = base_objective(X)
                self._eval_counter[0] += 1
                total = base_score
                X_cur = X.copy()
                for j in range(depth):
                    try:
                        params_cur = datamodule.array_to_params(X_cur)
                        bounds_ahead = bounds_fn(datamodule, params_cur)
                        res = minimize(
                            fun=base_objective,
                            x0=X_cur,
                            bounds=bounds_ahead,
                            method='L-BFGS-B',
                            options={'maxfun': 20},
                        )
                        self._eval_counter[0] += res.nfev
                        X_cur = res.x
                        step_score = base_objective(X_cur)
                        self._eval_counter[0] += 1
                        total += discount ** (j + 1) * step_score
                    except Exception:
                        break
                return total / self._weight_sum

        weight_sum = 1.0 + sum(discount ** (j + 1) for j in range(depth))
        return _MpcObjective(weight_sum)


# ======================================================================
# BoundsManager — schema-aware bounds computation
# ======================================================================

class BoundsManager:
    """Schema-aware parameter bounds, fixed params, trust regions, and schedule configs."""

    def __init__(self, schema: DatasetSchema, logger: PfabLogger):
        self.schema = schema
        self.logger = logger

        self.data_objects: dict[str, DataObject] = {}
        self.schema_bounds: dict[str, tuple[float, float]] = {}
        self.param_bounds: dict[str, tuple[float, float]] = {}
        self.fixed_params: dict[str, Any] = {}
        self.trust_regions: dict[str, float] = {}
        self.schedule_configs: dict[str, str] = {}  # param_code -> dimension_code

        self._set_param_constraints_from_schema(schema)

    def _set_param_constraints_from_schema(self, schema: DatasetSchema) -> None:
        """Extract parameter constraints from dataset schema."""
        for code, data_obj in schema.parameters.data_objects.items():
            if isinstance(data_obj, (DataBool, DataCategorical)):
                min_val, max_val = 0.0, 1.0
            elif issubclass(type(data_obj), DataObject):
                min_val = data_obj.constraints.get("min", -np.inf)
                max_val = data_obj.constraints.get("max", np.inf)
            else:
                raise TypeError(f"Expected DataObject type for parameter '{code}', got {type(data_obj).__name__}")

            self.data_objects[code] = data_obj
            self.schema_bounds[code] = (min_val, max_val)

    def configure_param_bounds(self, bounds: dict[str, tuple[float, float]], force: bool = False) -> None:
        """Configure parameter ranges for offline calibration."""
        for code, (low, high) in bounds.items():
            if not self._validate_and_clean_config(
                code, (DataReal, DataInt), ['fixed_params'], force
            ):
                continue

            schema_min, schema_max = self.schema_bounds[code]
            if low < schema_min or high > schema_max:
                raise ValueError(
                    f"Bounds for object '{code}' exceed schema constraints: "
                    f"[{low}, {high}] vs schema [{schema_min}, {schema_max}]"
                )

            self.param_bounds[code] = (low, high)
            self.logger.debug(f"Set parameter bounds: {code} -> [{low}, {high}]")

    def configure_fixed_params(self, fixed_params: dict[str, Any], force: bool = False) -> None:
        """Configure fixed parameter values."""
        for code, value in (fixed_params or {}).items():
            if not self._validate_and_clean_config(
                code, None, ['param_bounds', 'trust_regions'], force
            ):
                continue

            self.fixed_params[code] = value
            self.logger.debug(f"Set fixed parameter: {code} -> {value}")

    def configure_adaptation_delta(self, deltas: dict[str, float], force: bool = False) -> None:
        """Configure trust region deltas for online calibration."""
        for code, delta in deltas.items():
            if not self._validate_and_clean_config(
                code, (DataReal, DataInt), ['fixed_params'], force
            ):
                continue

            obj = self.data_objects[code]
            if not obj.runtime_adjustable:
                raise ValueError(
                    f"Parameter '{code}' is not runtime-adjustable. Trust regions can only be "
                    f"configured for parameters declared with runtime=True in the schema. "
                    f"Either mark '{code}' as runtime=True in the schema definition, or remove "
                    f"this configure_adaptation_delta() call."
                )

            self.trust_regions[code] = delta

    def configure_schedule_parameter(self, code: str, dimension_code: str, force: bool = False) -> None:
        """Declare that a runtime-adjustable parameter should be re-optimised at each step of the given dimension."""
        if code not in self.data_objects:
            self.logger.console_warning(
                f"Object '{code}' not found in schema; ignoring configure_schedule_parameter."
            )
            return

        obj = self.data_objects[code]

        if not obj.runtime_adjustable:
            raise ValueError(
                f"Parameter '{code}' is not runtime-adjustable. configure_schedule_parameter() "
                f"requires a parameter declared with runtime=True in the schema."
            )

        if not isinstance(obj, (DataReal, DataInt)):
            raise ValueError(
                f"Parameter '{code}' type {type(obj).__name__} is not supported for "
                f"dimension stepping. Only DataReal and DataInt parameters can be step parameters."
            )

        if dimension_code not in self.data_objects:
            raise ValueError(
                f"Dimension '{dimension_code}' not found in schema."
            )
        dim_obj = self.data_objects[dimension_code]
        if not isinstance(dim_obj, DataDomainAxis):
            raise ValueError(
                f"'{dimension_code}' is not a DataDomainAxis parameter "
                f"(got {type(dim_obj).__name__})."
            )

        if code in self.schedule_configs and not force:
            self.logger.console_warning(
                f"Parameter '{code}' already has a schedule configuration for "
                f"'{self.schedule_configs[code]}'; ignoring. Use force=True to overwrite."
            )
            return

        self.schedule_configs[code] = dimension_code
        if code not in self.trust_regions:
            lo, hi = self.schema_bounds.get(code, (0.0, 1.0))
            if lo != -np.inf and hi != np.inf:
                self.trust_regions[code] = (hi - lo) / 10.0
        self.logger.debug(
            f"Configured schedule for '{code}' stepping through '{dimension_code}'."
        )

    def _validate_and_clean_config(
        self,
        code: str,
        allowed_types: tuple[type, ...] | None,
        conflicting_collections: list[str],
        force: bool
    ) -> bool:
        """Validate parameter against schema and check for conflicting configurations."""
        if code not in self.data_objects:
            self.logger.console_warning(f"Object '{code}' not found in schema; ignoring.")
            return False

        if allowed_types:
            obj = self.data_objects[code]
            if not isinstance(obj, allowed_types):
                 self.logger.console_warning(
                     f"Object '{code}' type {type(obj).__name__} not supported for this configuration; ignoring."
                )
                 return False

        for collection_name in conflicting_collections:
            collection = getattr(self, collection_name)
            if code in collection:
                if force:
                    del collection[code]
                    self.logger.debug(f"Removed '{code}' from {collection_name} due to force=True.")
                else:
                    self.logger.console_warning(
                        f"Object '{code}' is already configured in {collection_name}; ignoring. Use force=True to overwrite."
                    )
                    return False
        return True

    def get_tunable_params(self, datamodule: DataModule) -> list[str]:
        """Return codes of parameters the optimizer can actually vary.

        Excludes: context features, fixed params, features (lag values),
        and one-hot columns. Only returns the original parameter codes
        that have non-zero-width bounds.
        """
        context_codes = set(datamodule.context_feature_codes)
        col_map = datamodule.get_onehot_column_map()
        schema_params = set(datamodule.dataset.schema.parameters.data_objects.keys())
        tunable = []
        for code in datamodule.input_columns:
            if code in context_codes:
                continue
            if code in col_map:
                continue
            if code not in schema_params:
                continue
            if code in self.fixed_params:
                continue
            lo, hi = self._get_hierarchical_bounds_for_code(code)
            if hi - lo < 1e-12:
                continue
            tunable.append(code)
        return tunable

    def _get_global_bounds(self, datamodule: DataModule) -> np.ndarray:
        """Return normalized optimization bounds over the full parameter space."""
        bounds_list = []
        col_map = datamodule.get_onehot_column_map()
        context_codes = set(datamodule.context_feature_codes)

        for code in datamodule.input_columns:
            if code in context_codes:
                bounds_list.append(self._normalize_bounds(code, 0.0, 0.0, datamodule))
                continue

            if code in col_map:
                parent_param, cat_val = col_map[code]
                if parent_param in self.fixed_params:
                    val = 1.0 if self.fixed_params[parent_param] == cat_val else 0.0
                    low, high = val, val
                else:
                    low, high = 0.0, 1.0
            else:
                low, high = self._get_hierarchical_bounds_for_code(code)

            n_low, n_high = self._normalize_bounds(code, low, high, datamodule)
            bounds_list.append((n_low, n_high))

        return np.array(bounds_list)

    def _get_trust_region_bounds(self, datamodule: DataModule, current_params: dict[str, Any]) -> np.ndarray:
        """Return normalized trust-region bounds centred on current_params."""
        bounds_list = []
        col_map = datamodule.get_onehot_column_map()

        for code in datamodule.input_columns:
            curr = 0.0
            is_one_hot = code in col_map

            if is_one_hot:
                parent_param, cat_val = col_map[code]
                if parent_param and parent_param in current_params:
                     curr = 1.0 if current_params[parent_param] == cat_val else 0.0
                elif code in current_params:
                    curr = current_params[code]
            else:
                if code in current_params:
                    curr = current_params[code]

            if code in self.trust_regions:
                delta = self.trust_regions[code]
                low, high = curr - delta, curr + delta
            else:
                low, high = curr, curr

            bounds_list.append(self._normalize_bounds(code, low, high, datamodule))

        return np.array(bounds_list)

    def _get_hierarchical_bounds_for_code(self, code: str) -> tuple[float, float]:
        if code in self.fixed_params:
            val = self.fixed_params[code]
            low, high = val, val
        elif code in self.param_bounds:
            low, high = self.param_bounds[code]
        elif code in self.schema_bounds:
            low, high = self.schema_bounds[code]
        else:
            raise ValueError(f"No bounds found for '{code}'. Cannot determine optimization bounds.")
        return low, high

    def _normalize_bounds(self, col: str, low: float, high: float, datamodule: DataModule) -> tuple[float, float]:
        """Normalize bounds to [0, 1] based on schema constraints."""
        n_low, n_high = datamodule.normalize_parameter_bounds(col, low, high)
        if (n_low, n_high) != (low, high):
            self.logger.debug(f"Processed bounds for '{col}': raw [{low}, {high}] -> normalized [{n_low}, {n_high}]")
        else:
            self.logger.debug(f"No normalization stats for '{col}'. Using raw bounds [{low}, {high}].")
        return n_low, n_high


# ======================================================================
# CalibrationSystem — orchestrator, composes the other two
# ======================================================================

class CalibrationSystem(BaseOrchestrationSystem):
    """Active-learning calibration engine: UCB exploration, inference, LHS baseline, and joint schedule optimization."""

    def __init__(
        self,
        schema: DatasetSchema,
        logger: PfabLogger,
        perf_fn: Callable[[dict[str, Any]], dict[str, float | None]],
        uncertainty_fn: Callable[[np.ndarray], float],
        similarity_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
        n_exp_fn: Callable[[], int] | None = None,
        n_decay_fn: Callable[[int], float] | None = None,
        base_buffer_fn: Callable[[], float] | None = None,
        random_seed: int | None = None,
    ):
        super().__init__(logger)
        self.perf_fn = perf_fn
        self.uncertainty_fn = uncertainty_fn
        self.similarity_fn = similarity_fn
        self._n_exp_fn = n_exp_fn
        self._n_decay_fn = n_decay_fn
        self._base_buffer_fn = base_buffer_fn

        self._random_seed = random_seed

        # Composed subsystems
        self.engine = OptimizationEngine(logger, random_seed=random_seed)
        self.bounds = BoundsManager(schema, logger)

        # Active datamodule — set before each optimization run so that
        # _inference_func / _acquisition_func can call array_to_params.
        self._active_datamodule: DataModule | None = None

        # Set after each optimization call for external inspection.
        self.last_opt_nfev: int = 0
        self.last_opt_n_starts: int = 0
        self.last_opt_score: float = 0.0
        self.last_opt_perf: float = 0.0
        self.last_opt_unc: float = 0.0
        self.last_schedule: list[dict[str, Any]] | None = None

        # Set ordered weights
        self.schema = schema
        self.perf_names_order = list(schema.performance_attrs.keys())
        self.performance_weights: dict[str, float] = {perf: 1.0 for perf in self.perf_names_order}
        self.parameters = schema.parameters

        self.optimizer: Optimizer = Optimizer.DE
        self.online_optimizer: Optimizer = Optimizer.LBFGSB

        # Baseline particle repulsion: Riesz energy exponent.
        self.baseline_riesz_p: float = 2.0

        # Running min/max of predicted system performance across training data.
        self._perf_range_min: float | None = None
        self._perf_range_max: float | None = None

        # Boundary buffer: penalise acquisition scores near parameter bounds.
        self._exploration_radius: float = 0.20
        self._buffer: float = 0.5
        self._decay_exp: float = 0.5
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
        """Print the optimized line with eval count."""
        bar = "\u2588" * 12
        print(f"\033[32m\u2713\033[0m {'Optimized':<10s} [{bar}] \033[2m{nfev} evals\033[0m{suffix}", flush=True)

    def _get_n_exp(self) -> int:
        """Current experiment count from the prediction system."""
        if self._n_exp_fn is not None:
            return self._n_exp_fn()
        return 1

    def _n_decay(self, n: int) -> float:
        """N-dependent decay factor, delegated to prediction system for single source of truth."""
        if self._n_decay_fn is not None:
            return self._n_decay_fn(n)
        return 1.0 / max(n, 1) ** (1/3)

    def state_report(self) -> None:
        """Log the current calibration configuration state."""
        _B = "\033[1m"
        _D = "\033[2m"
        _R = "\033[0m"

        lines = [f"\n  {_B}Calibration{_R}"]

        pw_parts = [f"{k}={v:g}" for k, v in self.performance_weights.items()]
        lines.append(f"    {_D}Weights: {', '.join(pw_parts)}{_R}")

        explore_parts = [f"radius={self._exploration_radius:g}",
                         f"buffer={self._buffer:g}",
                         f"decay_exp={self._decay_exp:g}"]
        lines.append(f"    {_D}Exploration: {', '.join(explore_parts)}{_R}")

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

    def _inference_func(self, X: np.ndarray) -> float:
        """INFERENCE objective: negative predicted performance (for minimisation)."""
        dm = self._active_datamodule
        if dm is None:
            return 0.0
        try:
            params_dict = dm.array_to_params(X.reshape(-1))
        except (ValueError, KeyError):
            return 0.0
        try:
            perf_dict = self.perf_fn(params_dict)
        except Exception:
            return 0.0
        perf_values = [
            float(perf_dict[name]) if perf_dict.get(name) is not None else 0.0 # type: ignore
            for name in self.perf_names_order
            if name in perf_dict
        ]
        sys_perf = self._compute_system_performance(perf_values)
        return -sys_perf

    def _acquisition_func(
        self,
        X: np.ndarray,
        kappa: float,
        bounds: np.ndarray | None = None,
        perf_range: tuple[float, float] | None = None,
        unc_range: tuple[float, float] | None = None,
    ) -> float:
        """EXPLORATION objective: negative UCB score with min-max normalized components."""
        dm = self._active_datamodule
        if dm is None:
            return 0.0
        try:
            params_dict = dm.array_to_params(X.reshape(-1))
        except (ValueError, KeyError):
            return 0.0
        try:
            perf_dict = self.perf_fn(params_dict)
        except Exception:
            perf_dict = {}
        perf_values = [
            float(perf_dict[name]) if perf_dict.get(name) is not None else 0.0 # type: ignore
            for name in self.perf_names_order
            if name in perf_dict
        ]
        sys_perf = self._compute_system_performance(perf_values) if perf_values else 0.0
        u = float(self.uncertainty_fn(X.reshape(-1)))

        if perf_range is not None:
            pmin, pmax = perf_range
            span = pmax - pmin
            sys_perf = (sys_perf - pmin) / span if span > 1e-10 else 0.5

        score = (1.0 - kappa) * sys_perf + kappa * u

        if bounds is not None:
            score *= self._boundary_factor(X.reshape(-1), bounds)

        return -score

    def _boundary_factor(
        self,
        X: np.ndarray,
        bounds: np.ndarray,
        strength: float = 0.5,
        exponent: float = 2.0,
    ) -> float:
        """Multiplicative penalty in (0, 1] that decays near parameter boundaries."""
        radius = self._exploration_radius
        if radius <= 0:
            return 1.0

        n_exp = self._get_n_exp()
        extent = radius * self._n_decay(n_exp)

        factor = 1.0
        for i in range(len(X)):
            lo, hi = bounds[i]
            span = hi - lo
            if span < 1e-12:
                continue
            d_lo = (X[i] - lo) / span
            d_hi = (hi - X[i]) / span
            d = min(d_lo, d_hi)
            if d >= extent:
                continue
            t = max(d / extent, 0.0)
            factor *= 1.0 - strength * (1.0 - t ** exponent)

        return factor

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
        training experiment, then applies an absolute buffer that decays
        with the number of experiments:

            half_buffer = (base_buffer / N^decay) / 2
            perf_min = raw_min - half_buffer
            perf_max = raw_max + half_buffer

        Early on (few experiments), the buffer is large and performance is
        compressed — uncertainty naturally dominates. As data grows, the
        buffer shrinks and performance fills [0, 1].
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
            n = len(train_codes)
            base_buffer = self._base_buffer_fn() if self._base_buffer_fn else 0.2
            half_buffer = (base_buffer * self._n_decay(n)) / 2
            self._perf_range_min = raw_min - half_buffer
            self._perf_range_max = raw_max + half_buffer
            self.logger.debug(
                f"Performance range: [{raw_min:.3f}, {raw_max:.3f}] "
                f"+ half_buffer {half_buffer:.3f} (N={n}) "
                f"\u2192 [{self._perf_range_min:.3f}, {self._perf_range_max:.3f}]"
            )

    def _get_acquisition_ranges(self) -> tuple[tuple[float, float] | None, None]:
        """Return (perf_range, unc_range) for acquisition normalization."""
        if self._perf_range_min is not None and self._perf_range_max is not None:
            perf_range = (self._perf_range_min, self._perf_range_max)
        else:
            perf_range = None
        return perf_range, None

    def _build_objective(self, mode: Mode, kappa: float, bounds: np.ndarray | None = None) -> Callable:
        """Return the objective function for the given calibration mode."""
        if mode == Mode.BASELINE:
            raise ValueError(
                "Mode.BASELINE uses run_baseline() — call calibration_system.run_baseline(n) directly."
            )
        if mode == Mode.EXPLORATION:
            perf_range, unc_range = self._get_acquisition_ranges()
            return functools.partial(
                self._acquisition_func,
                kappa=kappa,
                bounds=bounds,
                perf_range=perf_range,
                unc_range=unc_range,
            )
        elif mode == Mode.INFERENCE:
            return self._inference_func
        raise ValueError(f"Unknown calibration mode: {mode}")

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

    def run_baseline(
        self,
        n: int,
    ) -> list["ExperimentSpec"]:
        """Generate n baseline proposals via joint particle repulsion.

        Places all N points simultaneously in the normalized parameter space and
        minimizes Riesz energy (pairwise + boundary repulsion) to maximize spread.
        """
        if n == 0:
            return []

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
            elif isinstance(data_obj, DataInt):
                try:
                    lo, hi = self.bounds._get_hierarchical_bounds_for_code(code)
                except ValueError:
                    continue
                if lo == -np.inf or hi == np.inf:
                    self.logger.warning(
                        f"Parameter '{code}' has infinite bounds; skipping in baseline."
                    )
                    continue
                integer_params.append((code, int(lo), int(hi)))
            else:
                try:
                    lo, hi = self.bounds._get_hierarchical_bounds_for_code(code)
                except ValueError:
                    continue
                if lo == -np.inf or hi == np.inf:
                    self.logger.warning(
                        f"Parameter '{code}' has infinite bounds; skipping in baseline."
                    )
                    continue
                continuous_params.append((code, lo, hi))

        numeric_params: list[tuple[str, float, float]] = [
            *continuous_params,
            *[(c, float(lo), float(hi)) for c, lo, hi in integer_params],
        ]
        n_numeric = len(numeric_params)
        n_integer = len(integer_params)

        if not numeric_params and not categorical_params:
            self.logger.warning("No valid parameters for baseline generation.")
            return []

        # --- Assign categorical values (stratified buckets, no ordinal distance) ---
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

        # --- Initialize numeric positions via recursive bisection ---
        if n_numeric > 0:
            init_norm = np.zeros((n, n_numeric))
            idx = 0
            for _combo, n_i in zip(cat_combos, n_per_stratum):
                if n_i == 0:
                    continue
                bounds = [(lo, hi) for _, lo, hi in numeric_params]
                original_ranges = [hi - lo for lo, hi in bounds]
                points = self._recursive_split(n_i, bounds, original_ranges)
                for point in points:
                    for d, (_, lo, hi) in enumerate(numeric_params):
                        span = hi - lo
                        init_norm[idx, d] = (point[d] - lo) / span if span > 0 else 0.5
                    idx += 1

            if n_strata > 1:
                jitter = self.engine.rng.normal(0, 0.02, size=init_norm.shape)
                init_norm = np.clip(init_norm + jitter, 0.01, 0.99)

            int_indices: list[int] = []
            int_ranges_map: dict[int, int] = {}
            for d_int, (_, lo, hi) in enumerate(integer_params):
                col = len(continuous_params) + d_int
                int_indices.append(col)
                int_ranges_map[col] = int(hi - lo)

            # --- Determine schedule dimensions ---
            sched_set = set(self.schedule_configs.keys())
            L = 1
            if self.schedule_configs:
                sched_dims = set(self.schedule_configs.values())
                unfixed = [d for d in sched_dims if d not in self.fixed_params]
                if unfixed:
                    raise RuntimeError(
                        f"Schedule dimensions {sorted(unfixed)} must be fixed to determine "
                        f"the number of steps. Call configure_fixed_params() for each."
                    )
                L = max(int(self.fixed_params[d]) for d in sched_dims)

            static_indices: list[int] = []
            sched_indices: list[int] = []
            sched_deltas_norm: list[float] = []
            for d, (code, lo, hi) in enumerate(numeric_params):
                if code in sched_set:
                    sched_indices.append(d)
                    delta = self.trust_regions.get(code, 0.0)
                    span = hi - lo
                    sched_deltas_norm.append(delta / span if span > 0 else 0.0)
                else:
                    static_indices.append(d)

            has_schedule = L > 1 and len(sched_indices) > 0
            D_static = len(static_indices)
            D_sched = len(sched_indices) if has_schedule else 0

            int_set = set(int_indices)

            static_bounds_list: list[tuple[float, float]] = []
            integrality_mask: list[bool] = []
            for si, d in enumerate(static_indices):
                if d in int_set:
                    static_bounds_list.append((0.0, float(int_ranges_map[d])))
                    integrality_mask.append(True)
                else:
                    static_bounds_list.append((0.01, 0.99))
                    integrality_mask.append(False)

            sched_bounds_list: list[tuple[float, float]] = []
            if has_schedule:
                for d in sched_indices:
                    if d in int_set:
                        sched_bounds_list.append((0.0, float(int_ranges_map[d])))
                    else:
                        sched_bounds_list.append((0.01, 0.99))

            sched_delta_arr = np.array(sched_deltas_norm) if has_schedule else np.array([])

            # Build riesz per-step function
            p = self.baseline_riesz_p
            iu = np.triu_indices(n, k=1)

            def riesz_per_step(pts: np.ndarray) -> float:
                """Pairwise + boundary energy for N points in D dims (normalized [0,1])."""
                pts_norm = pts.copy()
                for si, d in enumerate(static_indices):
                    if d in int_set:
                        r = int_ranges_map[d]
                        pts_norm[:, si] = pts_norm[:, si] / r if r > 0 else 0.5
                diff = pts_norm[:, np.newaxis, :] - pts_norm[np.newaxis, :, :]
                dsq = np.maximum(np.sum(diff ** 2, axis=2)[iu], 1e-20)
                pairwise = float(np.sum(dsq ** (-p / 2)))
                clip = np.clip(pts_norm, 1e-10, 1.0 - 1e-10)
                boundary = float(np.sum((2.0 * clip) ** (-p)))
                boundary += float(np.sum((2.0 * (1.0 - clip)) ** (-p)))
                return pairwise + boundary

            # Convert init positions: integer dims from [0,1] to [0, range].
            init_de = init_norm.copy()
            for d in int_set:
                if d < init_de.shape[1]:
                    init_de[:, d] = np.round(init_de[:, d] * int_ranges_map[d])

            # Reorder init to engine layout: [static, sched] per unit
            n_vars_per_unit = D_static + D_sched + max(L - 1, 0) * D_sched
            n_total_vars = n * n_vars_per_unit
            init_flat = np.zeros(n_total_vars)
            for i in range(n):
                offset = i * n_vars_per_unit
                for si, d in enumerate(static_indices):
                    init_flat[offset + si] = init_de[i, d]
                if has_schedule:
                    for si, d in enumerate(sched_indices):
                        init_flat[offset + D_static + si] = init_de[i, d]

            popsize = max(self.de_popsize, 2)
            pop_total = popsize * n_total_vars
            de_bounds_all: list[tuple[float, float]] = []
            for _u in range(n):
                de_bounds_all.extend(static_bounds_list)
                de_bounds_all.extend(sched_bounds_list)
                for _k in range(1, L):
                    for d in range(D_sched):
                        dn = sched_deltas_norm[d]
                        de_bounds_all.append((-dn, dn))

            init_pop = np.empty((pop_total, n_total_vars))
            for i in range(pop_total):
                for v in range(n_total_vars):
                    lo_b, hi_b = de_bounds_all[v]
                    init_pop[i, v] = self.engine.rng.uniform(lo_b + 0.001, hi_b - 0.001)
            init_pop[0] = init_flat
            n_jittered = min(pop_total - 1, 5)
            for j in range(1, 1 + n_jittered):
                jitter = self.engine.rng.normal(0, 0.05, size=n_total_vars)
                candidate = init_flat + jitter
                for v in range(n_total_vars):
                    lo_b, hi_b = de_bounds_all[v]
                    candidate[v] = np.clip(candidate[v], lo_b + 0.001, hi_b - 0.001)
                init_pop[j] = candidate

            console = self.logger._console_output_enabled
            self.logger.info(
                f"Baseline: N={n}, L={L}, D_static={D_static}, D_sched={D_sched}, "
                f"total_vars={n_total_vars}"
            )

            opt, static_out, sched_out = self.engine.run(
                riesz_per_step, N=n, D_static=D_static, D_sched=D_sched, L=L,
                static_bounds=static_bounds_list, sched_bounds=sched_bounds_list,
                sched_deltas=sched_delta_arr,
                default_optimizer=self.optimizer,
                init_pop=init_pop,
                integrality_static=integrality_mask if any(integrality_mask) else None,
                label="Baseline", show_progress=console,
            )
            self.last_baseline_nfev: int = opt.nfev

            # --- Decode results into ExperimentSpecs ---
            specs: list[ExperimentSpec] = []
            dim_code = next(iter(set(self.schedule_configs.values())), "") if has_schedule else ""

            for i in range(n):
                base_params: dict[str, Any] = dict(self.fixed_params)

                for si, d in enumerate(static_indices):
                    code, lo, hi = numeric_params[d]
                    if d in int_set:
                        base_params[code] = int(np.round(static_out[i, si]) + lo)
                    else:
                        base_params[code] = float(static_out[i, si] * (hi - lo) + lo)

                for d_cat, code in enumerate(cat_codes):
                    base_params[code] = cat_assignments[i][d_cat]

                if has_schedule:
                    for si, d in enumerate(sched_indices):
                        code, lo, hi = numeric_params[d]
                        base_params[code] = float(sched_out[i, 0, si] * (hi - lo) + lo)

                    base_params = self.schema.parameters.sanitize_values(base_params, ignore_unknown=True)
                    initial = ParameterProposal.from_dict(base_params, source_step=SourceStep.BASELINE)

                    entries: list[tuple[int, ParameterProposal]] = []
                    for k in range(1, L):
                        step_params: dict[str, Any] = {}
                        for si, d in enumerate(sched_indices):
                            code, lo, hi = numeric_params[d]
                            step_params[code] = float(sched_out[i, k, si] * (hi - lo) + lo)
                        step_params = self.schema.parameters.sanitize_values(
                            step_params, ignore_unknown=True,
                        )
                        entries.append((k, ParameterProposal.from_dict(
                            step_params, source_step=SourceStep.BASELINE,
                        )))

                    schedules: dict[str, ParameterSchedule] = {}
                    if entries:
                        schedules[dim_code] = ParameterSchedule(dimension=dim_code, entries=entries)
                    specs.append(ExperimentSpec(initial_params=initial, schedules=schedules))
                else:
                    for d_int_i, (code_i, lo_i, hi_i) in enumerate(integer_params):
                        if code_i in base_params:
                            base_params[code_i] = int(np.clip(np.round(base_params[code_i]), lo_i, hi_i))

                    base_params = self.schema.parameters.sanitize_values(base_params, ignore_unknown=True)
                    proposal = ParameterProposal.from_dict(base_params, source_step=SourceStep.BASELINE)
                    specs.append(ExperimentSpec(initial_params=proposal, schedules={}))

            optimized = static_out.copy()
            for si, d in enumerate(static_indices):
                if d in int_set:
                    r = int_ranges_map[d]
                    optimized[:, si] = optimized[:, si] / r if r > 0 else 0.5

        else:
            optimized = np.zeros((n, 0))
            specs = []
            for i in range(n):
                params: dict[str, Any] = dict(self.fixed_params)
                for d, (code, lo, hi) in enumerate(numeric_params):
                    params[code] = (lo + hi) / 2.0
                for d_cat, code in enumerate(cat_codes):
                    params[code] = cat_assignments[i][d_cat]
                params = self.schema.parameters.sanitize_values(params, ignore_unknown=True)
                proposal = ParameterProposal.from_dict(params, source_step=SourceStep.BASELINE)
                specs.append(ExperimentSpec(initial_params=proposal, schedules={}))

        if n > 1 and n_numeric > 0 and optimized.shape[1] > 0:
            min_dist = self._min_pairwise_distance(optimized, None)
            self.logger.info(
                f"Baseline: {n} experiments via particle repulsion "
                f"({len(continuous_params)} continuous, {n_integer} integer, "
                f"{len(categorical_params)} categorical"
                f"{f', {n_strata} strata' if n_strata > 1 else ''}"
                f", riesz_p={self.baseline_riesz_p:.1f}"
                f") \u2014 min dist = {min_dist:.4f}."
            )
        else:
            self.logger.info(
                f"Baseline: {n} experiment(s) "
                f"({len(continuous_params)} continuous, {n_integer} integer, "
                f"{len(categorical_params)} categorical)."
            )

        return specs

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

        global_bounds = self.bounds._get_global_bounds(datamodule) if mode == Mode.EXPLORATION else None
        objective = self._build_objective(mode, kappa, bounds=global_bounds)

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

            static_bounds_list: list[tuple[float, float]] = []
            for code in static_codes:
                idx = code_to_idx[code]
                static_bounds_list.append((float(all_global_bounds[idx, 0]), float(all_global_bounds[idx, 1])))

            sched_bounds_list: list[tuple[float, float]] = []
            sched_deltas_norm = np.zeros(D_sched)
            for d, code in enumerate(sched_codes):
                idx = code_to_idx[code]
                lo_norm, hi_norm = float(all_global_bounds[idx, 0]), float(all_global_bounds[idx, 1])
                sched_bounds_list.append((lo_norm, hi_norm))
                delta_raw = self.trust_regions.get(code, 0.0)
                if delta_raw > 0:
                    _, delta_norm = datamodule.normalize_parameter_bounds(code, 0.0, delta_raw)
                    lo_zero, _ = datamodule.normalize_parameter_bounds(code, 0.0, 0.0)
                    delta_norm = abs(delta_norm - lo_zero)
                else:
                    delta_norm = 0.0
                sched_deltas_norm[d] = delta_norm

            def acq_per_step(pts: np.ndarray) -> float:
                """Evaluate acquisition/inference at (1, D_full) points for one step."""
                x_full = np.zeros(n_input)
                for i, code in enumerate(static_codes):
                    x_full[code_to_idx[code]] = pts[0, i]
                for d, code in enumerate(sched_codes):
                    x_full[code_to_idx[code]] = pts[0, D_static + d]
                return objective(x_full)

            self.logger.debug(
                f"Joint schedule optimization: L={L}, "
                f"D_static={D_static}, D_sched={D_sched}, "
                f"total_vars={D_static + D_sched + (L - 1) * D_sched}"
            )

            opt, static_out, sched_out = self.engine.run(
                acq_per_step, N=1, D_static=D_static, D_sched=D_sched, L=L,
                static_bounds=static_bounds_list, sched_bounds=sched_bounds_list,
                sched_deltas=sched_deltas_norm,
                default_optimizer=self.optimizer,
                label="Schedule", show_progress=console,
            )

            self.last_opt_nfev = opt.nfev
            self.last_opt_n_starts = opt.n_starts
            self.last_opt_score = opt.score

            if console:
                self._print_optimized_line(opt.nfev)

            dm_input_set = set(datamodule.input_columns)
            non_dm_sched = {
                c for c in self.schedule_configs if c not in dm_input_set
            }

            proposals: list[dict[str, Any]] = []
            if opt.best_x is not None:
                for k in range(L):
                    x_step = np.zeros(n_input)
                    for i, code in enumerate(static_codes):
                        x_step[code_to_idx[code]] = static_out[0, i]
                    for d, code in enumerate(sched_codes):
                        x_step[code_to_idx[code]] = sched_out[0, k, d]

                    step_params = datamodule.array_to_params(x_step)
                    step_params.update(self.fixed_params)
                    if current_params and non_dm_sched:
                        for code in non_dm_sched:
                            if code in current_params and code not in step_params:
                                step_params[code] = current_params[code]
                    if current_params:
                        for k_param, v_param in current_params.items():
                            if k_param not in step_params:
                                step_params[k_param] = v_param
                    step_params = self.schema.parameters.sanitize_values(
                        step_params, ignore_unknown=True,
                    )
                    proposals.append(step_params)
            else:
                self.logger.warning("Joint schedule optimization failed, returning fallback.")
                fallback = dict(self.fixed_params)
                if current_params:
                    fallback.update(current_params)
                proposals = [fallback] * L

            self.last_schedule = list(proposals)

            if opt.best_x is not None:
                try:
                    x0_step = np.zeros(n_input)
                    for i, code in enumerate(static_codes):
                        x0_step[code_to_idx[code]] = static_out[0, i]
                    for d, code in enumerate(sched_codes):
                        x0_step[code_to_idx[code]] = sched_out[0, 0, d]
                    _params = datamodule.array_to_params(x0_step)
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
                    self.last_opt_unc = float(self.uncertainty_fn(x0_step))
                except Exception:
                    self.last_opt_perf = 0.0
                    self.last_opt_unc = 0.0
            else:
                self.last_opt_perf = 0.0
                self.last_opt_unc = 0.0

            self.logger.info(
                f"de: 1 start(s), {opt.nfev} evals, score={opt.score:.6f}"
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
                    raw_unc = float(self.uncertainty_fn(best_x))
                    _gbounds = self.bounds._get_global_bounds(datamodule)
                    self.last_opt_unc = raw_unc * self._boundary_factor(best_x, _gbounds)
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
