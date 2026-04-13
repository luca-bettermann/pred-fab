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
from ..utils import PfabLogger, Mode, NormMethod, SourceStep, SplitType, combined_score
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

class CalibrationSystem(BaseOrchestrationSystem):
    """Active-learning calibration engine: UCB exploration, inference, LHS baseline, and MPC lookahead."""

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

        # Virtual KDE point callbacks for within-trajectory spacing.
        # Set by PfabAgent to inject proposed points into the KDE between
        # trajectory steps, causing subsequent layers to see lower uncertainty
        # at already-proposed speeds and naturally spacing out.
        self._add_virtual_point_fn: Callable[[dict[str, Any]], None] | None = None
        self._clear_virtual_points_fn: Callable[[], None] | None = None
        self._random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.default_mpc_lookahead: int = 0
        self.default_mpc_discount: float = 0.9

        # Active datamodule — set before each optimization run so that
        # _inference_func / _acquisition_func can call array_to_params.
        self._active_datamodule: DataModule | None = None

        # Set after each _run_optimization call for external inspection.
        self.last_opt_nfev: int = 0
        self.last_opt_n_starts: int = 0
        self.last_opt_score: float = 0.0
        self.last_opt_perf: float = 0.0      # normalized performance component
        self.last_opt_unc: float = 0.0       # uncertainty component (with buffer)
        self.last_trajectory: list[dict[str, Any]] | None = None  # per-layer params for trajectory

        # Set ordered weights
        self.schema = schema
        self.perf_names_order = list(schema.performance_attrs.keys())
        self.performance_weights: dict[str, float] = {perf: 1.0 for perf in self.perf_names_order}
        self.parameters = schema.parameters

        # Configure data_objects, bounds and fixed params
        self.data_objects: dict[str, DataObject] = {}
        self.schema_bounds: dict[str, tuple[float, float]] = {}
        self.param_bounds: dict[str, tuple[float, float]] = {}
        self.fixed_params: dict[str, Any] = {}
        self.trust_regions: dict[str, float] = {}
        self.trajectory_configs: dict[str, str] = {}   # param_code → dimension_code

        # OFAT (One-Factor-At-a-Time) state for online calibration.
        # Empty list = all_at_once (default); populated via configure_ofat_strategy().
        self._ofat_codes: list[str] = []
        self._ofat_index: int = 0

        self.optimizer: Optimizer = Optimizer.DE           # offline (exploration + inference)
        self.online_optimizer: Optimizer = Optimizer.LBFGSB  # online (adaptation / trust-region)

        # DE optimizer parameters (global, population-based + L-BFGS-B polish)
        self.de_maxiter: int = 100     # maximum generations
        self.de_popsize: int = 10      # population size per dimension

        # L-BFGS-B optimizer parameters (gradient-based, multi-start)
        self.lbfgsb_maxfun: int | None = None  # max evals per start (None = auto)
        self.lbfgsb_eps: float = 1e-3           # finite-difference step size

        # Running min/max of predicted system performance across training data.
        # Updated after each train() call, used to normalize acquisition scores.
        self._perf_range_min: float | None = None  # lowest predicted performance seen so far
        self._perf_range_max: float | None = None  # highest predicted performance seen so far

        # Trajectory smoothing: penalizes speed changes between adjacent layers
        # to encourage monotonic trajectories. penalty = 1 - lambda * |delta| / trust_width.
        self.trajectory_smoothing: float = 0.0  # 0 = disabled, 0.1 = mild, 0.3 = strong

        # Boundary buffer: penalise acquisition scores near parameter bounds to
        # counteract KDE edge effects (no evidence outside the search space).
        # Derived from exploration_radius: extent = radius / √N (tracks KDE bandwidth).
        # Enabled automatically when exploration_radius > 0.
        self._exploration_radius: float = 0.20  # mirrored from PredictionSystem for buffer calc
        self._buffer: float = 0.5              # mirrored from PredictionSystem
        self._decay_exp: float = 0.5           # mirrored from PredictionSystem
        self._suppress_opt_print: bool = False  # suppress per-step print during trajectory

        # Extract parameter constraints from schema
        self._set_param_constraints_from_schema(schema)

    def _print_optimized_line(self, nfev: int, suffix: str = "") -> None:
        """Print the ✓ Optimized [...] line. Suffix allows merging with proposal components."""
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

    def _set_param_constraints_from_schema(self, schema: DatasetSchema) -> None:
        """Extract parameter constraints from dataset schema."""
        for code, data_obj in schema.parameters.data_objects.items():
            # Set the appropriate constraints for bool and one-hot encodings
            if isinstance(data_obj, (DataBool, DataCategorical)):
                min_val, max_val = 0.0, 1.0
            # Get constraints for continuous parameters
            elif issubclass(type(data_obj), DataObject):
                min_val = data_obj.constraints.get("min", -np.inf)
                max_val = data_obj.constraints.get("max", np.inf)
            else:
                raise TypeError(f"Expected DataObject type for parameter '{code}', got {type(data_obj).__name__}")

            # Store constraints
            self.data_objects[code] = data_obj
            self.schema_bounds[code] = (min_val, max_val)

    def state_report(self) -> None:
        """Log the current calibration configuration state."""
        _B = "\033[1m"
        _D = "\033[2m"
        _R = "\033[0m"

        lines = [f"\n  {_B}Calibration{_R}"]

        # Performance weights
        pw_parts = [f"{k}={v:g}" for k, v in self.performance_weights.items()]
        lines.append(f"    {_D}Weights: {', '.join(pw_parts)}{_R}")

        # Exploration settings
        explore_parts = [f"radius={self._exploration_radius:g}",
                         f"buffer={self._buffer:g}",
                         f"decay_exp={self._decay_exp:g}"]
        lines.append(f"    {_D}Exploration: {', '.join(explore_parts)}{_R}")

        # Parameter bounds and deltas
        lines.append(f"\n    {_D}{'Parameter':<20s} {'Bounds':<20s} {'Delta':<8s}{_R}")
        for code in self.data_objects.keys():
            low, high = self._get_hierarchical_bounds_for_code(code)
            bounds_str = f"[{low}, {high}]"
            delta = self.trust_regions.get(code, "─")
            lines.append(f"    {code:<20s} {bounds_str:<20s} {delta:<8}")

        self.logger.console_summary("\n".join(lines))
        self.logger.console_new_line()

    # === CONFIGURATION METHODS ===

    def set_performance_weights(self, weights: dict[str, float]) -> None:
        """Set weights for system performance calculation. Default is 1.0 for all."""
        # set according to order in perf_names_order
        for name, value in weights.items():
            if name in self.performance_weights:
                self.performance_weights[name] = value
                self.logger.debug(f"Set performance weight: {name} -> {value}")
            else:
                self.logger.console_warning(f"Performance attribute '{name}' not in schema; ignoring weight.")

    def configure_param_bounds(self, bounds: dict[str, tuple[float, float]], force: bool = False) -> None:
        """Configure parameter ranges for offline calibration."""
        for code, (low, high) in bounds.items():

            # Helper Validation
            if not self._validate_and_clean_config(
                code,
                (DataReal, DataInt),
                ['fixed_params'],
                force
            ):
                continue

            # Method-Specific Validation: Check vs Schema
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

            # Helper Validation
            if not self._validate_and_clean_config(
                code,
                None,  # All types allow fixing
                ['param_bounds', 'trust_regions'],
                force
            ):
                continue

            self.fixed_params[code] = value
            self.logger.debug(f"Set fixed parameter: {code} -> {value}")

    def configure_adaptation_delta(self, deltas: dict[str, float], force: bool = False) -> None:
        """Configure trust region deltas for online calibration."""
        for code, delta in deltas.items():

            # Helper Validation
            if not self._validate_and_clean_config(
                code,
                (DataReal, DataInt),
                ['fixed_params'],
                force
            ):
                continue

            # Runtime-adjustability check: trust regions are exclusively for runtime params.
            obj = self.data_objects[code]
            if not obj.runtime_adjustable:
                raise ValueError(
                    f"Parameter '{code}' is not runtime-adjustable. Trust regions can only be "
                    f"configured for parameters declared with runtime=True in the schema. "
                    f"Either mark '{code}' as runtime=True in the schema definition, or remove "
                    f"this configure_adaptation_delta() call."
                )

            self.trust_regions[code] = delta


    def configure_step_parameter(self, code: str, dimension_code: str, force: bool = False) -> None:
        """Declare that a runtime-adjustable parameter should be re-optimised at each step of the given dimension.

        When run_calibration iterates over the step-grid, only parameters whose mapped dimension
        transitions at that step are free to move; the rest are fixed to the previous result.
        """
        if code not in self.data_objects:
            self.logger.console_warning(
                f"Object '{code}' not found in schema; ignoring configure_step_parameter."
            )
            return

        obj = self.data_objects[code]

        if not obj.runtime_adjustable:
            raise ValueError(
                f"Parameter '{code}' is not runtime-adjustable. configure_step_parameter() "
                f"requires a parameter declared with runtime=True in the schema."
            )

        if not isinstance(obj, (DataReal, DataInt)):
            raise ValueError(
                f"Parameter '{code}' type {type(obj).__name__} is not supported for "
                f"dimension stepping. Only DataReal and DataInt parameters can be step parameters."
            )

        # Validate dimension_code
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

        if code in self.trajectory_configs and not force:
            self.logger.console_warning(
                f"Parameter '{code}' already has a trajectory configuration for "
                f"'{self.trajectory_configs[code]}'; ignoring. Use force=True to overwrite."
            )
            return

        self.trajectory_configs[code] = dimension_code
        self.logger.debug(
            f"Configured trajectory for '{code}' stepping through '{dimension_code}'."
        )

    def configure_ofat_strategy(self, codes: list[str]) -> None:
        """Configure OFAT cycling: only one parameter in ``codes`` is freed per online step.

        Parameters must already have trust regions configured. An empty list resets to
        all_at_once (default). The cycle index resets to 0 on each call.
        """
        for code in codes:
            if code not in self.trust_regions:
                raise ValueError(
                    f"OFAT parameter '{code}' must have a trust region configured first. "
                    f"Call configure_adaptation_delta() before configure_ofat_strategy()."
                )
        self._ofat_codes = list(codes)
        self._ofat_index = 0
        if codes:
            self.logger.info(f"OFAT strategy configured: {codes} (starting at index 0)")
        else:
            self.logger.info("OFAT strategy cleared (all_at_once mode).")

    def _get_active_ofat_code(self) -> str | None:
        """Return the currently active OFAT parameter code, or None if OFAT is not active."""
        if not self._ofat_codes:
            return None
        return self._ofat_codes[self._ofat_index % len(self._ofat_codes)]

    def _advance_ofat(self) -> None:
        """Advance the OFAT cycle to the next parameter."""
        if self._ofat_codes:
            prev = self._ofat_codes[self._ofat_index % len(self._ofat_codes)]
            self._ofat_index = (self._ofat_index + 1) % len(self._ofat_codes)
            next_code = self._ofat_codes[self._ofat_index]
            self.logger.debug(f"OFAT advance: {prev} -> {next_code}")

    def _validate_and_clean_config(
        self,
        code: str,
        allowed_types: tuple[type, ...] | None,
        conflicting_collections: list[str],
        force: bool
    ) -> bool:
        """Validate parameter against schema and check for conflicting configurations."""
        # 1. Schema Existence
        if code not in self.data_objects:
            self.logger.console_warning(f"Object '{code}' not found in schema; ignoring.")
            return False

        # 2. Type Check
        if allowed_types:
            obj = self.data_objects[code]
            if not isinstance(obj, allowed_types):
                 self.logger.console_warning(
                     f"Object '{code}' type {type(obj).__name__} not supported for this configuration; ignoring."
                )
                 return False

        # 3. Conflict Resolution
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

    # === OBJECTIVE FUNCTIONS ===

    def _inference_func(self, X: np.ndarray) -> float:
        """INFERENCE objective: negative predicted performance (for minimisation)."""
        dm = self._active_datamodule
        if dm is None:
            return 0.0
        try:
            params_dict = dm.array_to_params(X.reshape(-1))
        except (ValueError, KeyError):
            return 0.0  # out-of-bounds point from optimizer overshoot
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
            return 0.0  # out-of-bounds point from optimizer overshoot
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

        # Normalize performance to its observed training range so kappa
        # balances meaningfully even when raw scores occupy a narrow band.
        # Values can exceed [0, 1] when the model extrapolates beyond training data.
        if perf_range is not None:
            pmin, pmax = perf_range
            span = pmax - pmin
            sys_perf = (sys_perf - pmin) / span if span > 1e-10 else 0.5

        score = (1.0 - kappa) * sys_perf + kappa * u

        # Apply boundary buffer to counteract KDE edge effects
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
        """Multiplicative penalty in (0, 1] that decays near parameter boundaries.

        Derived from the exploration radius — no separate configuration needed.
        extent   = exploration_radius / N^decay  (tracks KDE bandwidth decay)
        strength = penalty at boundary: score *= (1 - strength)
        exponent = curve shape (2.0 = quadratic, matching KDE kernel)

        Per-dimension factors are multiplied so corners (near multiple boundaries)
        receive a stronger penalty.
        """
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
        """Compute weighted system performance from an ordered list of scores.

        Delegates to the shared combined_score utility, converting the
        ordered list to a dict keyed by perf_names_order.
        """
        if not performance:
            return 0.0
        perf_dict = {
            name: performance[i]
            for i, name in enumerate(self.perf_names_order)
            if i < len(performance)
        }
        return combined_score(perf_dict, self.performance_weights)


    # === PRIVATE HELPERS ===

    def _wrap_mpc_objective(
        self,
        base_objective: Callable,
        datamodule: DataModule,
        depth: int,
        discount: float,
    ) -> Callable:
        """Wrap base_objective with MPC lookahead: MPC(X) = Σ γʲ·score(Xⱼ) over depth steps.

        Returns base_objective unchanged when depth <= 0.
        """
        if depth <= 0:
            return base_objective

        _mpc_evals = [0]  # mutable counter for total inner evals
        # Normalization: sum of discount weights so MPC score stays in same range as single score
        _weight_sum = 1.0 + sum(discount ** (j + 1) for j in range(depth))

        def mpc_obj(X: np.ndarray) -> float:
            base_score = base_objective(X)
            _mpc_evals[0] += 1
            total = base_score
            X_cur = X.copy()
            for j in range(depth):
                try:
                    params_cur = datamodule.array_to_params(X_cur)
                    bounds_ahead = self._get_trust_region_bounds(datamodule, params_cur)
                    res = minimize(
                        fun=base_objective,
                        x0=X_cur,
                        bounds=bounds_ahead,
                        method='L-BFGS-B',
                        options={'maxfun': 20},  # cheap inner solve
                    )
                    _mpc_evals[0] += res.nfev
                    X_cur = res.x
                    step_score = base_objective(X_cur)
                    _mpc_evals[0] += 1
                    discount_factor = discount ** (j + 1)
                    total += discount_factor * step_score
                except Exception:
                    break
            return total / _weight_sum

        mpc_obj._eval_counter = _mpc_evals  # expose counter

        return mpc_obj

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
                f"→ [{self._perf_range_min:.3f}, {self._perf_range_max:.3f}]"
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

        Returns [{}] when no trajectory configs are configured (experiment-level use case).
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

    def _get_eligible_params(
        self,
        prev_indices: dict[str, int] | None,
        curr_indices: dict[str, int],
    ) -> set:
        """Return trajectory params whose mapped dimension transitions at this step."""
        transitioning_dims: set = set()
        for dim_code in curr_indices:
            if prev_indices is None or curr_indices[dim_code] != prev_indices.get(dim_code):
                transitioning_dims.add(dim_code)
        return {
            code for code, dim_code in self.trajectory_configs.items()
            if dim_code in transitioning_dims
        }

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
            # Collect unique dims
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
                        # Dimension transitioned — record schedule entry
                        seg_vals = {c: proposals[flat_i][c] for c in traj_codes if c in proposals[flat_i]}
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
        Caller must apply any param_bounds overrides to self.param_bounds before calling.
        """
        active_codes: list[str] = []
        for code, data_obj in self.data_objects.items():
            if code in self.fixed_params:
                continue
            # Categoricals and bools always have finite schema bounds — include.
            if isinstance(data_obj, (DataCategorical, DataBool)):
                active_codes.append(code)
                continue
            # Continuous/integer: skip parameters whose effective bounds are unbounded.
            try:
                lo, hi = self._get_hierarchical_bounds_for_code(code)
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
        """Generate n baseline proposals with structured spacing via recursive bisection.

        Recursively splits the parameter space into equal-volume cells and places
        one point at each cell centre. Splits along the dimension with the largest
        relative range, producing a clean grid-like arrangement for any N and d.

        Categorical parameters are stratified: N is distributed evenly across
        all categorical combinations, with continuous spacing optimized per stratum.

        Deterministic, instant (O(N log N)), works for any N, d, and variable types.
        """
        if n == 0:
            return []

        # Collect active parameters (skip fixed and infinite-bounds).
        continuous_params: list[tuple[str, float, float]] = []  # (code, lo, hi)
        categorical_params: list[tuple[str, list[Any]]] = []    # (code, categories)

        for code, data_obj in self.data_objects.items():
            if code in self.fixed_params:
                continue
            if isinstance(data_obj, DataCategorical):
                categorical_params.append((code, list(data_obj.constraints["categories"])))
            elif isinstance(data_obj, DataBool):
                categorical_params.append((code, [False, True]))
            else:
                try:
                    lo, hi = self._get_hierarchical_bounds_for_code(code)
                except ValueError:
                    continue
                if lo == -np.inf or hi == np.inf:
                    self.logger.warning(
                        f"Parameter '{code}' has infinite bounds; skipping in baseline."
                    )
                    continue
                continuous_params.append((code, lo, hi))

        if not continuous_params and not categorical_params:
            self.logger.warning("No valid parameters for baseline generation.")
            return []

        # Build categorical combinations for stratification
        if categorical_params:
            import itertools as _it
            cat_combos = list(_it.product(*(cats for _, cats in categorical_params)))
            cat_codes = [code for code, _ in categorical_params]
        else:
            cat_combos = [()]
            cat_codes = []

        # Distribute n across categorical strata
        n_per_stratum = [n // len(cat_combos)] * len(cat_combos)
        for i in range(n % len(cat_combos)):
            n_per_stratum[i] += 1

        # Generate points per stratum via recursive bisection
        specs: list[ExperimentSpec] = []
        for combo, n_i in zip(cat_combos, n_per_stratum):
            if n_i == 0:
                continue
            bounds = [(lo, hi) for _, lo, hi in continuous_params]
            original_ranges = [hi - lo for lo, hi in bounds]
            points = self._recursive_split(n_i, bounds, original_ranges)

            for point in points:
                params: dict[str, Any] = dict(self.fixed_params)
                for k, (code, _, _) in enumerate(continuous_params):
                    params[code] = float(point[k])
                for k, code in enumerate(cat_codes):
                    params[code] = combo[k]
                params = self.schema.parameters.sanitize_values(params, ignore_unknown=True)
                proposal = ParameterProposal.from_dict(params, source_step=SourceStep.BASELINE)
                specs.append(ExperimentSpec(initial_params=proposal, schedules={}))

        self.logger.info(
            f"Baseline: {n} experiments via recursive bisection "
            f"({len(continuous_params)} continuous, {len(categorical_params)} categorical, "
            f"{len(cat_combos)} strata)."
        )
        return specs

    @staticmethod
    def _recursive_split(
        n: int,
        bounds: list[tuple[float, float]],
        original_ranges: list[float],
    ) -> list[list[float]]:
        """Recursively bisect the parameter space into n cells, return cell centres.

        Splits along the dimension with the largest relative range
        (current range / original range) so narrow dimensions still get split
        proportionally. Produces a deterministic grid-like arrangement.
        """
        if n == 1:
            return [[(lo + hi) / 2.0 for lo, hi in bounds]]

        d = len(bounds)
        # Split along dimension with largest relative range
        dim = max(
            range(d),
            key=lambda i: (bounds[i][1] - bounds[i][0]) / original_ranges[i]
            if original_ranges[i] > 0 else 0,
        )

        n_left = n // 2
        n_right = n - n_left
        lo, hi = bounds[dim]
        split = lo + (hi - lo) * n_left / n  # proportional split

        bounds_left = list(bounds)
        bounds_left[dim] = (lo, split)
        bounds_right = list(bounds)
        bounds_right[dim] = (split, hi)

        return (
            CalibrationSystem._recursive_split(n_left, bounds_left, original_ranges)
            + CalibrationSystem._recursive_split(n_right, bounds_right, original_ranges)
        )

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
        """Run a single calibration pass and return an ExperimentSpec.

        Offline (global bounds + random restarts) when current_params/target_indices are omitted;
        online (trust-region) when both are provided.
        mpc_lookahead=0 → greedy single-step; mpc_lookahead=N → N-step discounted lookahead.
        mpc_discount (γ): weight for future steps in the MPC sum Σ γʲ·score(Xⱼ).
          γ=0.9 means step j=1 counts at 90%, j=2 at 81%, etc. — nearer steps matter more.
        """
        lookahead = self.default_mpc_lookahead if mpc_lookahead is None else mpc_lookahead
        discount = self.default_mpc_discount if mpc_discount is None else mpc_discount

        self._active_datamodule = datamodule

        # For exploration, compute global bounds first so we can estimate ranges
        global_bounds = self._get_global_bounds(datamodule) if mode == Mode.EXPLORATION else None
        objective = self._build_objective(mode, kappa, bounds=global_bounds)

        if lookahead > 0:
            objective = self._wrap_mpc_objective(objective, datamodule, lookahead, discount)
            self.logger.info(
                f"MPC lookahead enabled: depth={lookahead}, discount={discount:.3f}"
            )

        if source_step_override is not None:
            source_step = source_step_override
        elif mode == Mode.EXPLORATION:
            source_step: SourceStep = SourceStep.EXPLORATION
        elif mode == Mode.INFERENCE:
            source_step = SourceStep.INFERENCE
        else:
            source_step = SourceStep.BASELINE

        # Online = both current_params AND target_indices provided (target_indices may be empty dict)
        is_online = current_params is not None and target_indices is not None
        if target_indices is not None:
            step_grid: list[dict[str, int]] = [target_indices]  # type: ignore[list-item]
        elif current_params is not None:
            step_grid = self._build_step_grid(current_params)
        else:
            step_grid = [{}]

        self.logger.info(
            f"run_calibration: mode={mode.name}, {'online (trust-region)' if is_online else 'offline (global)'}, "
            f"kappa={kappa:.2f}, mpc_lookahead={lookahead}, step_grid={len(step_grid)} step(s)"
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

        # Validate trajectory params have trust regions if multi-step
        if len(step_grid) > 1:
            traj_without_delta = [
                c for c in self.trajectory_configs if c not in self.trust_regions
            ]
            if traj_without_delta:
                raise RuntimeError(
                    f"Trajectory parameters {sorted(traj_without_delta)} have no "
                    f"configured trust region. "
                    f"Call configure_adaptation_delta() for each before running."
                )

        working_params: dict[str, Any] | None = (
            dict(current_params) if current_params else None
        )
        prev_indices: dict[str, int] | None = None
        proposals: list[dict[str, Any]] = []

        # Track previous step's params for trajectory smoothing penalty.
        _prev_step_params: dict[str, Any] | None = None

        is_trajectory = len(step_grid) > 1
        console = self.logger._console_output_enabled
        if is_trajectory and console:
            self._suppress_opt_print = True
        total_traj_nfev = 0

        for step_i, curr_indices in enumerate(step_grid):
            # Build fixed params for this step
            fixed_for_step = dict(self.fixed_params)
            eligible: set = set()
            if working_params and self.trajectory_configs and curr_indices:
                eligible = self._get_eligible_params(prev_indices, curr_indices)
                for code in self.trajectory_configs:
                    if code not in eligible and code in working_params:
                        fixed_for_step[code] = working_params[code]

            # Online: explicitly pin all params outside the trust region to current values.
            # This avoids denormalization drift for params that have zero-width bounds.
            if is_online and working_params is not None:
                for code, val in working_params.items():
                    if code not in self.trust_regions and code not in fixed_for_step:
                        fixed_for_step[code] = val

            # Use trust-region when online (current_params + target_indices), or when we have a
            # reference point on a non-experiment-level step (subsequent trajectory steps).
            # Otherwise use global bounds with random restarts.
            if is_online or (working_params is not None and bool(curr_indices)):
                bounds = self._get_trust_region_bounds(datamodule, working_params)  # type: ignore[arg-type]
                n_rounds = 0
                bounds_mode = "trust-region"
            else:
                bounds = self._get_global_bounds(datamodule)
                n_rounds = n_optimization_rounds
                bounds_mode = "global"

            self.logger.debug(
                f"Step {step_i + 1}/{len(step_grid)}: indices={curr_indices}, "
                f"bounds={bounds_mode}, eligible_traj={sorted(eligible) if eligible else '—'}, "
                f"fixed={sorted(fixed_for_step.keys())}"
            )

            # Wrap objective with trajectory smoothing penalty for non-first steps.
            # Multiplicative penalty on the acquisition score (before negation):
            #   score *= (1 - lambda * |delta| / trust_width)
            # This directly reduces the score for large jumps, keeping it in [0, 1].
            step_objective = objective
            if (self.trajectory_smoothing > 0 and _prev_step_params is not None
                    and len(step_grid) > 1):
                _lam = self.trajectory_smoothing
                _prev = _prev_step_params
                _tr = self.trust_regions
                _dm = datamodule

                def _smoothed_objective(X, _base=step_objective, _l=_lam, _p=_prev, _t=_tr, _d=_dm):
                    base_score = _base(X)  # negative (minimization)
                    try:
                        params_dict = _d.array_to_params(X.reshape(-1))
                    except (ValueError, KeyError):
                        return base_score
                    # Penalty: reduce score proportionally to jump size
                    for code, delta in _t.items():
                        if code in _p and code in params_dict and delta > 0:
                            change = abs(float(params_dict[code]) - float(_p[code]))
                            frac = min(change / delta, 1.0)
                            # base_score is negative; multiply penalty makes it less negative (worse)
                            base_score *= (1.0 - _l * frac)
                    return base_score

                step_objective = _smoothed_objective

            # Online (trust-region) uses the online_optimizer; offline uses the main optimizer.
            opt_for_step = self.online_optimizer if is_online else None

            result = self._run_optimization(
                datamodule,
                x0_params=working_params,
                bounds=bounds,
                objective_func=step_objective,
                n_rounds=n_rounds,
                fixed_param_values=fixed_for_step,
                optimizer_override=opt_for_step,
            )
            proposals.append(result)
            _prev_step_params = dict(result)
            working_params = result
            prev_indices = curr_indices
            total_traj_nfev += self.last_opt_nfev

            # Trajectory: layer progress bar
            if is_trajectory and console:
                n_steps = len(step_grid)
                bar_len = 12
                filled = int(bar_len * (step_i + 1) / n_steps)
                bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
                if step_i + 1 == n_steps:
                    print(f"\033[32m\u2713\033[0m {'Optimized':<10s} [{bar}] "
                          f"\033[2m{total_traj_nfev} evals ({n_steps} layers)\033[0m", flush=True)
                else:
                    print(f"  {'Optimizing':<10s} [{bar}] "
                          f"\033[2m{step_i+1}/{n_steps} layers, {total_traj_nfev} evals\033[0m",
                          end="\r", flush=True)

            # Inject virtual KDE point so subsequent trajectory steps see lower
            # uncertainty at this speed, naturally spacing out layer proposals.
            if is_trajectory and self._add_virtual_point_fn is not None:
                self._add_virtual_point_fn(result)

        # Clear virtual KDE points after trajectory completes.
        if is_trajectory and self._clear_virtual_points_fn is not None:
            self._clear_virtual_points_fn()

        if is_trajectory:
            self._suppress_opt_print = False
            self.last_trajectory = list(proposals)
        else:
            self.last_trajectory = None

        proposal_summary = {k: round(v, 4) if isinstance(v, float) else v for k, v in proposals[0].items()}
        self.logger.info(f"Calibration proposal: {proposal_summary}")

        # Advance OFAT cycle after each online step so the next call targets the next parameter.
        if is_online:
            self._advance_ofat()

        return self._build_experiment_spec(proposals, step_grid, source_step)

    def _run_optimization(
        self,
        datamodule: DataModule,
        x0_params: dict[str, Any] | None,
        bounds: np.ndarray,
        objective_func: Callable,
        n_rounds: int,
        fixed_param_values: dict[str, Any],
        optimizer_override: Optimizer | None = None,
    ) -> dict[str, Any]:
        """Run the acquisition function optimization and return proposed parameters."""
        active_optimizer = optimizer_override or self.optimizer

        console = self.logger._console_output_enabled and not self._suppress_opt_print

        if active_optimizer == Optimizer.DE:
            opt = self._optimize_de(bounds, objective_func, show_progress=console)
        else:
            opt = self._optimize_lbfgsb(
                datamodule, x0_params, bounds, objective_func, n_rounds,
                show_progress=console,
            )

        # Publish result bookkeeping — include MPC inner evals if applicable
        total_nfev = opt.nfev
        if hasattr(objective_func, '_eval_counter'):
            total_nfev = objective_func._eval_counter[0]
        self.last_opt_nfev = total_nfev

        if console:
            self._print_optimized_line(total_nfev)
        self.last_opt_n_starts = opt.n_starts
        self.last_opt_score = opt.score

        # Compute individual components at the optimal point for reporting
        if opt.best_x is not None:
            try:
                _params = datamodule.array_to_params(opt.best_x)
                _perf_dict = self.perf_fn(_params)
                _perf_values = [
                    float(_perf_dict[n]) if _perf_dict.get(n) is not None else 0.0
                    for n in self.perf_names_order if n in _perf_dict
                ]
                raw_perf = self._compute_system_performance(_perf_values) if _perf_values else 0.0
                # Normalize performance same as acquisition function
                if self._perf_range_min is not None and self._perf_range_max is not None:
                    span = self._perf_range_max - self._perf_range_min
                    self.last_opt_perf = (raw_perf - self._perf_range_min) / span if span > 1e-10 else 0.5
                else:
                    self.last_opt_perf = raw_perf
                raw_unc = float(self.uncertainty_fn(opt.best_x))
                _bounds = self._get_global_bounds(datamodule)
                self.last_opt_unc = raw_unc * self._boundary_factor(opt.best_x, _bounds)
            except Exception:
                self.last_opt_perf = 0.0
                self.last_opt_unc = 0.0
        else:
            self.last_opt_perf = 0.0
            self.last_opt_unc = 0.0

        self.logger.info(
            f"{active_optimizer.value}: {opt.n_starts} start(s), {opt.nfev} evals, score={opt.score:.6f}"
        )

        # Handle failure
        if opt.best_x is None:
            self.logger.warning("Optimization failed, returning fallback parameters.")
            if x0_params:
                return x0_params
            raise RuntimeError("No valid parameters could be proposed.")

        # Decode, merge fixed/carry-over params, and sanitize
        proposed_params = datamodule.array_to_params(opt.best_x)
        if fixed_param_values:
            proposed_params.update(fixed_param_values)
        # Carry over params from x0 that aren't in the optimization space
        # (e.g. runtime params not used by any prediction model).
        if x0_params:
            for k, v in x0_params.items():
                if k not in proposed_params:
                    proposed_params[k] = v
        return datamodule.dataset.schema.parameters.sanitize_values(
            proposed_params, ignore_unknown=True,
        )

    def _optimize_lbfgsb(
        self,
        datamodule: DataModule,
        x0_params: dict[str, Any] | None,
        bounds: np.ndarray,
        objective_func: Callable,
        n_rounds: int,
        show_progress: bool = False,
    ) -> _OptResult:
        """Multi-start L-BFGS-B (gradient-based, local)."""
        x0_list = []
        if x0_params:
            x0_list.append(datamodule.params_to_array(x0_params))
        for _ in range(n_rounds):
            x0_list.append(self.rng.uniform(bounds[:, 0], bounds[:, 1]))
        if not x0_list:
            x0_list.append(self.rng.uniform(bounds[:, 0], bounds[:, 1]))

        # Cap evals per start. Default: ~10 gradient steps (each needs d+1 finite-diff evals).
        # Without a cap, L-BFGS-B defaults to 15000*d — prohibitive for expensive objectives.
        n_dims = len(x0_list[0])
        max_fun = self.lbfgsb_maxfun if self.lbfgsb_maxfun is not None else max(100, 10 * (n_dims + 1))
        eps = self.lbfgsb_eps
        total_starts = len(x0_list)

        best_x, best_val = None, np.inf
        total_nfev = 0
        for i, x0 in enumerate(x0_list):
            if show_progress:
                bar_len = 12
                filled = int(bar_len * (i + 1) / total_starts)
                bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
                print(f"  {'Optimizing':<10s} [{bar}] \033[2m{i+1}/{total_starts}\033[0m", end="\r", flush=True)
            try:
                res = minimize(
                    fun=objective_func, x0=x0, bounds=bounds, method='L-BFGS-B',
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

        return _OptResult(
            best_x=best_x,
            nfev=total_nfev,
            n_starts=len(x0_list),
            score=float(-best_val) if best_val != np.inf else 0.0,
        )

    def _optimize_de(
        self,
        bounds: np.ndarray,
        objective_func: Callable,
        show_progress: bool = False,
    ) -> _OptResult:
        """Differential evolution (global, population-based, with L-BFGS-B polish)."""
        seed = int(self.rng.randint(0, 2**31 - 1))
        maxiter = self.de_maxiter
        popsize = self.de_popsize

        _iter = [0]
        def _progress(xk, convergence):
            _iter[0] += 1
            if show_progress:
                bar_len = 12
                filled = int(bar_len * _iter[0] / maxiter)
                bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
                print(f"  {'Optimizing':<10s} [{bar}] \033[2m{_iter[0]}/{maxiter}\033[0m", end="\r", flush=True)

        result = differential_evolution(
            func=objective_func,
            bounds=bounds.tolist(),
            maxiter=maxiter, popsize=popsize,
            mutation=(0.5, 1.0), recombination=0.7, tol=1e-4,
            polish=True, init='latinhypercube',
            callback=_progress,
        )

        return _OptResult(
            best_x=result.x,
            nfev=result.nfev,
            n_starts=1,
            score=float(-result.fun),
        )

    # === BOUNDS FOR OPTIMIZATION ===

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
                continue  # one-hot column, not a raw param
            if code not in schema_params:
                continue  # feature (lag value), not a parameter
            if code in self.fixed_params:
                continue
            lo, hi = self._get_hierarchical_bounds_for_code(code)
            if hi - lo < 1e-12:
                continue  # zero-width bounds = constant
            tunable.append(code)
        return tunable

    def _get_global_bounds(self, datamodule: DataModule) -> np.ndarray:
        """Return normalized optimization bounds over the full parameter space."""
        bounds_list = []
        col_map = datamodule.get_onehot_column_map()
        context_codes = set(datamodule.context_feature_codes)

        for code in datamodule.input_columns:

            # Context features are fixed at 0 during optimization (uncontrollable).
            if code in context_codes:
                bounds_list.append(self._normalize_bounds(code, 0.0, 0.0, datamodule))
                continue

            # Fast One-Hot Check
            if code in col_map:
                parent_param, cat_val = col_map[code]
                if parent_param in self.fixed_params:
                    # Fixed One-hot: [0, 0] or [1, 1] depending on fixed category
                    val = 1.0 if self.fixed_params[parent_param] == cat_val else 0.0
                    low, high = val, val
                else:
                    # Unfixed Categorical (one-hot): [0, 1]
                    low, high = 0.0, 1.0
            else:
                low, high = self._get_hierarchical_bounds_for_code(code)

            # Process & Append
            n_low, n_high = self._normalize_bounds(code, low, high, datamodule)
            bounds_list.append((n_low, n_high))

        return np.array(bounds_list)

    def _get_trust_region_bounds(self, datamodule: DataModule, current_params: dict[str, Any]) -> np.ndarray:
        """Return normalized trust-region bounds centred on current_params."""
        bounds_list = []
        col_map = datamodule.get_onehot_column_map()

        for code in datamodule.input_columns:

            # Determine Center (Current Value) & Check One-hot
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

            # Determine Bounds from Trust Region
            # Note: Trust Regions (deltas) are typically only for continuous parameters.
            # If a parameter is not in trust_regions, it is fixed to current.
            # OFAT: if active, only the currently active parameter is freed; others fixed.
            active_ofat = self._get_active_ofat_code()
            if code in self.trust_regions:
                if active_ofat is not None and code != active_ofat:
                    # OFAT mode: this param has a trust region but is not the active one → fix.
                    low, high = curr, curr
                else:
                    delta = self.trust_regions[code]
                    low, high = curr - delta, curr + delta
            else:
                # No trust region -> Fixed to current
                low, high = curr, curr

            # Process & Append
            bounds_list.append(self._normalize_bounds(code, low, high, datamodule))

        return np.array(bounds_list)

    def _get_hierarchical_bounds_for_code(self, code: str) -> tuple[float, float]:
        # 1. Check Fixed Context
        if code in self.fixed_params:
            val = self.fixed_params[code]
            low, high = val, val
        # 2. Check Explicit Param Bounds
        elif code in self.param_bounds:
            low, high = self.param_bounds[code]
        # 3. Default to Schema Constraints (could be infinite)
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

    # === WRAPPERS ===

    def get_models(self) -> list[Any]:
        """Return empty list (no internal ML models owned by CalibrationSystem)."""
        return []

    def get_model_specs(self) -> dict[str, list[str]]:
        return {}
