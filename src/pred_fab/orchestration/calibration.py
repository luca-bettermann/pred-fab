from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize

import warnings
import functools

from ..core import DataModule, DatasetSchema
from ..core import DataInt, DataReal, DataObject, DataBool, DataCategorical, DataDimension
from ..core import ParameterProposal, ParameterSchedule, ExperimentSpec
from ..utils import PfabLogger, Mode, SamplingStrategy
from .base_system import BaseOrchestrationSystem

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

class CalibrationSystem(BaseOrchestrationSystem):
    """
    Orchestrates calibration and active learning.

    - Owns System Performance definition and active-learning acquisition logic
    - Generates baseline experiments (LHS)
    - Proposes new experiments via Bayesian Optimization (UCB with KDE uncertainty)
    - Supports Online (Trust Region) and Offline (Global) modes
    - Level 2 trajectory exploration with diversity discounting via similarity_fn
    """

    def __init__(
        self,
        schema: DatasetSchema,
        logger: PfabLogger,
        perf_fn: Callable[[Dict[str, Any]], Dict[str, Optional[float]]],
        uncertainty_fn: Callable[[np.ndarray], float],
        similarity_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            schema: Dataset schema defining parameters and performance attributes.
            logger: Logger instance.
            perf_fn: Callable mapping a raw params dict to a performance dict
                ``{perf_code: value_or_None}``.  Encapsulates predict + evaluate.
            uncertainty_fn: Callable mapping a normalized parameter array (1-D) to
                a scalar epistemic uncertainty in [0, 1].
            similarity_fn: Optional callable mapping two normalized parameter arrays
                to a scalar similarity in [0, 1].  Used for Level 2 trajectory
                diversity discounting.  When None, diversity discounting is skipped.
            random_seed: Seed for reproducible random sampling.
        """
        super().__init__(logger)
        self.perf_fn = perf_fn
        self.uncertainty_fn = uncertainty_fn
        self.similarity_fn = similarity_fn
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        # Active datamodule — set before each optimization run so that
        # _inference_func / _acquisition_func can call array_to_params.
        self._active_datamodule: Optional[DataModule] = None

        # Set ordered weights
        self.perf_names_order = list(schema.performance_attrs.keys())
        self.performance_weights: Dict[str, float] = {perf: 1.0 for perf in self.perf_names_order}
        self.parameters = schema.parameters

        # Configure data_objects, bounds and fixed params
        self.data_objects: Dict[str, DataObject] = {}
        self.schema_bounds: Dict[str, Tuple[float, float]] = {}
        self.param_bounds: Dict[str, Tuple[float, float]] = {}
        self.fixed_params: Dict[str, Any] = {}
        self.trust_regions: Dict[str, float] = {}
        self.trajectory_configs: Dict[str, int] = {}   # code → dimension_level

        # Extract parameter constraints from schema
        self._set_param_constraints_from_schema(schema)

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
        summary = ["===== Calibration System State =====\n"]
        width = 20
        # Columns: Input, Bounds, Delta
        header = f"{'Input':<{width}} | {'Bounds':<{width}} | {'Delta':<{8}}"
        summary.append(header)
        summary.append("-" * len(header))

        for code in self.data_objects.keys():

            # Determine Bounds
            # Priority: Fixed -> Configured Bounds -> Schema Constraints
            low, high = self._get_hierarchical_bounds_for_code(code)
            bounds_str = f"[{low}, {high}]"

            # Determine Delta
            delta = self.trust_regions.get(code, "-")

            summary.append(f"{code:<{width}} | {bounds_str:<{width}} | {delta:<{8}}")

        self.logger.console_new_line()
        self.logger.console_info("\n".join(summary))
        self.logger.console_new_line()

    # === CONFIGURATION METHODS ===

    def set_performance_weights(self, weights: Dict[str, float]) -> None:
        """Set weights for system performance calculation. Default is 1.0 for all."""
        # set according to order in perf_names_order
        for name, value in weights.items():
            if name in self.performance_weights:
                self.performance_weights[name] = value
                self.logger.debug(f"Set performance weight: {name} -> {value}")
            else:
                self.logger.console_warning(f"Performance attribute '{name}' not in schema; ignoring weight.")

    def configure_param_bounds(self, bounds: Dict[str, Tuple[float, float]], force: bool = False) -> None:
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

    def configure_fixed_params(self, fixed_params: Dict[str, Any], force: bool = False) -> None:
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

    def configure_adaptation_delta(self, deltas: Dict[str, float], force: bool = False) -> None:
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


    def configure_trajectory(self, code: str, dimension_level: int, force: bool = False) -> None:
        """Configure a runtime parameter for trajectory-based exploration at a given dimension level."""
        if code not in self.data_objects:
            self.logger.console_warning(
                f"Object '{code}' not found in schema; ignoring configure_trajectory."
            )
            return

        obj = self.data_objects[code]

        if not obj.runtime_adjustable:
            raise ValueError(
                f"Parameter '{code}' is not runtime-adjustable. configure_trajectory() "
                f"requires a parameter declared with runtime=True in the schema."
            )

        if not isinstance(obj, (DataReal, DataInt)):
            raise ValueError(
                f"Parameter '{code}' type {type(obj).__name__} is not supported for "
                f"trajectory exploration. Only DataReal and DataInt parameters can have "
                f"trajectories."
            )

        if code in self.trajectory_configs and not force:
            self.logger.console_warning(
                f"Parameter '{code}' already has a trajectory configuration at level "
                f"{self.trajectory_configs[code]}; ignoring. Use force=True to overwrite."
            )
            return

        self.trajectory_configs[code] = dimension_level
        self.logger.debug(
            f"Configured trajectory for '{code}' at dimension level {dimension_level}."
        )

    def _validate_and_clean_config(
        self,
        code: str,
        allowed_types: Optional[Tuple[type, ...]],
        conflicting_collections: List[str],
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
        """
        Objective for INFERENCE: Maximize predicted performance.
        Returns negative performance for minimization.
        """
        dm = self._active_datamodule
        if dm is None:
            return 0.0
        params_dict = dm.array_to_params(X.reshape(-1))
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

    def _acquisition_func(self, X: np.ndarray, w_explore: float) -> float:
        """
        Objective for exploration: Maximize UCB score.
        Score = (1 - w) * predicted_performance + w * epistemic_uncertainty
        Returns negative Score for minimization.
        """
        dm = self._active_datamodule
        if dm is None:
            return 0.0
        params_dict = dm.array_to_params(X.reshape(-1))
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
        u = self.uncertainty_fn(X.reshape(-1))
        score = (1.0 - w_explore) * sys_perf + w_explore * float(u)
        return -score

    def _compute_system_performance(self, performance: List[float]) -> float:
        """Compute weighted system performance [0, 1]."""
        if not performance:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        # make sure to order performance weights by the performance names in dataset
        ordered_weights = [self.performance_weights.get(name, 0.0) for name in self.perf_names_order]

        for i, weight in enumerate(ordered_weights):
            # Assume performance metrics are [0, 1]
            total_score += performance[i] * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0


    # === PRIVATE HELPERS ===

    def _lhs_unit_samples(self, d: int, n: int) -> np.ndarray:
        """Return (n, d) array of LHS samples in [0, 1]^d."""
        return qmc.LatinHypercube(d=d, seed=self.random_seed).random(n=n)

    # === BASELINE EXPERIMENT GENERATION ===

    def generate_baseline_experiments(
        self,
        n_samples: int,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        n_trajectory_segments: int = 3,
    ) -> List[ExperimentSpec]:
        """Generate initial design using Latin Hypercube Sampling."""
        self.logger.info(
            f"Generating {n_samples} baseline experiments using Latin Hypercube Sampling..."
        )

        # 1. Define Sampling Space (Physics)
        sampling_specs = self._get_sampling_specs(param_bounds)
        if not sampling_specs:
            self.logger.warning("No valid parameters for baseline generation.")
            return []

        param_names = sorted(sampling_specs.keys())
        d = len(param_names)

        # 2. Generate Stratified Samples (Geometry)
        # Returns (n_samples, d) matrix with values in [0, 1]
        lhs_samples = self._lhs_unit_samples(d, n_samples)

        # 3. Build ExperimentSpec per row
        experiments: List[ExperimentSpec] = []
        for row in lhs_samples:
            initial_dict = self._transform_lhs_sample(row, param_names, sampling_specs)
            initial_proposal = ParameterProposal.from_dict(
                initial_dict, source_step="baseline_sampling"
            )

            # 4. Generate trajectory schedules for configured runtime params
            schedules: Dict[str, ParameterSchedule] = {}
            if self.trajectory_configs and n_trajectory_segments > 1:
                schedules = self._generate_baseline_schedules(
                    initial_dict, n_trajectory_segments
                )

            spec = ExperimentSpec(initial_params=initial_proposal, schedules=schedules)
            experiments.append(spec)
            self.logger.debug(
                f"Generated baseline experiment: {initial_dict}, "
                f"schedules={list(schedules.keys())}"
            )

        return experiments

    def _generate_baseline_schedules(
        self,
        initial_dict: Dict[str, Any],
        n_segments: int,
    ) -> Dict[str, ParameterSchedule]:
        """Sample trajectory schedules for trajectory-configured runtime parameters."""
        level_to_dim: Dict[int, Tuple[str, int]] = {}
        for code, data_obj in self.data_objects.items():
            if isinstance(data_obj, DataDimension):
                dim_size = int(initial_dict.get(code, data_obj.constraints.get("min", 1)))
                level_to_dim[data_obj.level] = (code, dim_size)

        params_per_level: Dict[int, List[str]] = defaultdict(list)
        for code, level in self.trajectory_configs.items():
            params_per_level[level].append(code)

        schedules: Dict[str, ParameterSchedule] = {}
        for level, param_codes in params_per_level.items():
            if level not in level_to_dim:
                self.logger.warning(
                    f"No dimension found at level {level} for trajectory params "
                    f"{param_codes}. Skipping schedule generation."
                )
                continue

            dim_code, dim_size = level_to_dim[level]
            n_triggers = n_segments - 1
            # Evenly spaced trigger steps (exclude step 0, covered by initial_params)
            trigger_steps = sorted({
                max(1, int((k + 1) * dim_size / n_segments))
                for k in range(n_triggers)
            })

            entries: List[Tuple[int, ParameterProposal]] = []
            for step_idx in trigger_steps:
                step_values: Dict[str, Any] = {}
                for code in param_codes:
                    low, high = self.schema_bounds.get(code, (-np.inf, np.inf))
                    if low == -np.inf or high == np.inf:
                        try:
                            low, high = self._get_hierarchical_bounds_for_code(code)
                        except ValueError:
                            continue
                    step_values[code] = float(self.rng.uniform(low, high))
                if step_values:
                    entries.append((
                        step_idx,
                        ParameterProposal.from_dict(
                            step_values, source_step="baseline_trajectory"
                        ),
                    ))

            if entries:
                schedules[dim_code] = ParameterSchedule(
                    dimension=dim_code, entries=entries
                )

        return schedules

    def _get_sampling_specs(self, param_bounds: Optional[Dict[str, Tuple[float, float]]]) -> Dict[str, Dict[str, Any]]:
        """Build sampling specifications for each parameter."""
        sampling_specs = {}

        for code, data_obj in self.data_objects.items():

            # 1. Determine Effective Bounds (Continuous Only)
            if param_bounds and code in param_bounds:
                low, high = param_bounds[code]
            else:
                low, high = self._get_hierarchical_bounds_for_code(code)

            # 2. Check for Infinite Bounds (Safeguard)
            if isinstance(data_obj, (DataReal, DataInt)) and (low == -np.inf or high == np.inf):
                self.logger.warning(f"Parameter '{code}' has infinite bounds; skipping in baseline generation.")
                continue

            # 3. Create Specification
            if isinstance(data_obj, DataCategorical):
                 # Use retrieved bounds to detect fixed functionality

                 sampling_specs[code] = {
                     'type': SamplingStrategy.CATEGORICAL,
                     'categories': data_obj.constraints['categories'] if low != high else [low]
                }
            elif isinstance(data_obj, DataBool):
                 # Use retrieved bounds to detect fixed functionality
                 if low == high:
                     sampling_specs[code] = {
                         'type': SamplingStrategy.CATEGORICAL,
                         'categories': [low]
                     }
                 else:
                     sampling_specs[code] = {
                         'type': SamplingStrategy.BOOL
                    }
            else:
                 # Continuous / Integer
                 dtype = int if isinstance(data_obj, DataInt) else float
                 sampling_specs[code] = {
                     'type': SamplingStrategy.NUMERICAL,
                     'low': low,
                     'high': high,
                     'dtype': dtype
                }

            self.logger.debug(f"Included '{code}' in baseline generation specs.")

        return sampling_specs

    def _transform_lhs_sample(self, row: np.ndarray, param_names: List[str], sampling_specs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single [0, 1] LHS vector into a valid parameter dictionary."""
        params = {}
        for val, name in zip(row, param_names):
            spec = sampling_specs[name]

            if spec['type'] == SamplingStrategy.NUMERICAL:
                # Scale: [0, 1] -> [low, high]

                scaled_val = spec['low'] + val * (spec['high'] - spec['low'])
                params[name] = spec['dtype'](scaled_val)

            elif spec['type'] == SamplingStrategy.BOOL:
                # Scale: [0, 1] -> {True, False}
                params[name] = bool(val > 0.5)

            elif spec['type'] == SamplingStrategy.CATEGORICAL:
                # Scale: [0, 1] -> Category Index
                cats = spec['categories']
                idx = int(val * len(cats))
                idx = min(idx, len(cats) - 1) # clip to be safe
                params[name] = cats[idx]

        # Reuse canonical parameter coercion/rounding rules.
        return self.parameters.sanitize_values(params, ignore_unknown=True)


    # === TRAJECTORY EXPLORATION ===

    def run_trajectory_exploration(
        self,
        datamodule: DataModule,
        current_params: Dict[str, Any],
        w_explore: float = 0.5,
        n_segments: int = 3,
        n_lhs_candidates: int = 20,
    ) -> ExperimentSpec:
        """Propose an optimised trajectory schedule for runtime parameters via LHS warm start + SLSQP.

        Uses Level 2 diversity discounting when ``similarity_fn`` is provided: each
        trajectory segment's acquisition score is multiplied by
        ``max(0, 1 - max_{j<k} sim(X_j, X_k))`` so that the optimizer is incentivised
        to explore different regions of the latent space across segments.
        """
        if not self.trajectory_configs:
            raise RuntimeError(
                "run_trajectory_exploration() requires at least one trajectory parameter "
                "configured via configure_trajectory()."
            )

        missing_deltas = [
            code for code in self.trajectory_configs
            if code not in self.trust_regions
        ]
        if missing_deltas:
            raise RuntimeError(
                f"run_trajectory_exploration() cannot proceed: trajectory parameters "
                f"{sorted(missing_deltas)} have no configured trust region. "
                f"Call configure_adaptation_delta() for each before running trajectory "
                f"exploration."
            )

        self._active_datamodule = datamodule

        traj_codes = sorted(self.trajectory_configs.keys())
        static_params = {k: v for k, v in current_params.items() if k not in self.trajectory_configs}

        # Per-parameter bounds for the trajectory decision variable
        traj_bounds_list: List[Tuple[float, float]] = []
        for code in traj_codes:
            low, high = self._get_hierarchical_bounds_for_code(code)
            for _ in range(n_segments):
                traj_bounds_list.append((low, high))
        traj_bounds_arr = np.array(traj_bounds_list)

        def _traj_objective(v: np.ndarray) -> float:
            """Average (Level-2) acquisition score across segments (returns negative for minimisation)."""
            per_seg_acq: List[float] = []
            per_seg_X: List[Optional[np.ndarray]] = []
            for seg_idx in range(n_segments):
                seg_params = dict(static_params)
                for p_idx, code in enumerate(traj_codes):
                    seg_params[code] = float(v[p_idx * n_segments + seg_idx])
                try:
                    X = datamodule.params_to_array(seg_params)
                    per_seg_acq.append(self._acquisition_func(X, w_explore))  # already negative
                    per_seg_X.append(X)
                except Exception:
                    per_seg_acq.append(0.0)
                    per_seg_X.append(None)

            # Level 2: discount each segment by (1 - max_sim to all prior segments)
            # acq values are negative; multiplying by discount in [0,1] moves them
            # closer to zero (less contribution), incentivising diversity.
            if self.similarity_fn is not None:
                discounted: List[float] = []
                for k, acq in enumerate(per_seg_acq):
                    if k == 0 or per_seg_X[k] is None:
                        discounted.append(acq)
                        continue
                    valid_prior = [X_j for X_j in per_seg_X[:k] if X_j is not None]
                    max_sim = max(
                        (self.similarity_fn(X_j, per_seg_X[k]) for X_j in valid_prior),  # type: ignore
                        default=0.0,
                    )
                    discounted.append(acq * max(0.0, 1.0 - max_sim))
                total_neg = sum(discounted)
            else:
                total_neg = sum(per_seg_acq)

            count = sum(1 for X in per_seg_X if X is not None)
            return total_neg / count if count > 0 else 0.0

        # SLSQP trust region constraints: |v[k+1] - v[k]| <= delta
        constraints: List[Dict] = []
        for p_idx, code in enumerate(traj_codes):
            delta = self.trust_regions[code]
            for seg_idx in range(n_segments - 1):
                i_a = p_idx * n_segments + seg_idx
                i_b = p_idx * n_segments + seg_idx + 1
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda v, a=i_a, b=i_b, d=delta: d - (v[b] - v[a])
                })
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda v, a=i_a, b=i_b, d=delta: d - (v[a] - v[b])
                })

        # Phase 1: LHS warm start
        n_dims = len(traj_codes) * n_segments
        lhs_rows = self._lhs_unit_samples(n_dims, n_lhs_candidates)

        best_x0: Optional[np.ndarray] = None
        best_score = np.inf
        for row in lhs_rows:
            v = traj_bounds_arr[:, 0] + row * (traj_bounds_arr[:, 1] - traj_bounds_arr[:, 0])
            score = _traj_objective(v)
            if score < best_score:
                best_score = score
                best_x0 = v.copy()

        if best_x0 is None:
            raise RuntimeError(
                "LHS warm start produced no valid candidates for trajectory exploration."
            )

        # Phase 2: SLSQP refinement with step-change constraints
        best_v = best_x0
        try:
            result = minimize(
                fun=_traj_objective,
                x0=best_x0,
                bounds=traj_bounds_arr,
                method='SLSQP',
                constraints=constraints,
            )
            if result.success or result.fun < best_score:
                best_v = result.x
        except Exception as e:
            self.logger.warning(
                f"SLSQP trajectory optimisation failed: {e}. Using LHS best candidate."
            )

        # Decode best_v: segment 0 → initial_params, segments 1..K-1 → schedule entries.
        initial_dict = dict(current_params)
        for p_idx, code in enumerate(traj_codes):
            initial_dict[code] = float(best_v[p_idx * n_segments + 0])

        # Collect dimension info (level → (param_code, size))
        level_to_dim: Dict[int, Tuple[str, int]] = {}
        for code, data_obj in self.data_objects.items():
            if isinstance(data_obj, DataDimension):
                dim_size = int(current_params.get(code, data_obj.constraints.get("min", 1)))
                level_to_dim[data_obj.level] = (code, dim_size)

        params_per_level: Dict[int, List[str]] = defaultdict(list)
        for code, level in self.trajectory_configs.items():
            params_per_level[level].append(code)

        schedules: Dict[str, ParameterSchedule] = {}
        for level, param_codes in params_per_level.items():
            if level not in level_to_dim or n_segments <= 1:
                continue
            dim_code, dim_size = level_to_dim[level]
            n_triggers = n_segments - 1
            trigger_steps = sorted({
                max(1, int((k + 1) * dim_size / n_segments))
                for k in range(n_triggers)
            })
            entries: List[Tuple[int, ParameterProposal]] = []
            for t_idx, step in enumerate(trigger_steps[:n_triggers]):
                seg_values: Dict[str, Any] = {}
                for code in param_codes:
                    p_idx = traj_codes.index(code)
                    seg_values[code] = float(best_v[p_idx * n_segments + t_idx + 1])
                if seg_values:
                    entries.append((
                        step,
                        ParameterProposal.from_dict(
                            seg_values, source_step="trajectory_exploration"
                        ),
                    ))
            if entries:
                schedules[dim_code] = ParameterSchedule(
                    dimension=dim_code, entries=entries
                )

        initial_proposal = ParameterProposal.from_dict(
            initial_dict, source_step="trajectory_exploration"
        )
        return ExperimentSpec(initial_params=initial_proposal, schedules=schedules)

    # === OPTIMIZATION WORKFLOW ===

    def run_calibration(
        self,
        datamodule: DataModule,
        mode: Mode,
        w_explore: float = 0.5,
        n_optimization_rounds: int = 10,
    ) -> Dict[str, Any]:
        """
        Run calibration (Offline) to propose new parameters.
        Uses global parameter bounds and fixed context.
        """
        self._active_datamodule = datamodule

        # 1. Get Offline Bounds
        bounds_array = self._get_offline_bounds(datamodule)

        # 2. Select Objective Function
        if mode == Mode.EXPLORATION:
            objective_func = functools.partial(self._acquisition_func, w_explore=w_explore)
        elif mode == Mode.INFERENCE:
            objective_func = self._inference_func
        else:
            raise ValueError(f"Unknown phase: {mode}")

        # 3. Run Unified Optimization
        return self._run_optimization(
            datamodule,
            x0_params=None,
            bounds=bounds_array,
            objective_func=objective_func,
            n_rounds=n_optimization_rounds,
            fixed_param_values=self.fixed_params
        )

    def run_adaptation(
        self,
        datamodule: DataModule,
        mode: Mode,
        current_params: Dict[str, Any],
        w_explore: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Run adaptation (Online) to propose new parameters.
        Uses trust regions around current_params.
        """
        # Fail fast if any runtime parameter has no trust region configured.
        runtime_without_delta = [
            code
            for code, obj in self.data_objects.items()
            if obj.runtime_adjustable and code not in self.trust_regions
        ]
        if runtime_without_delta:
            raise RuntimeError(
                f"run_adaptation() cannot proceed: the following runtime-adjustable "
                f"parameters have no configured trust region: "
                f"{sorted(runtime_without_delta)}. "
                f"Call configure_adaptation_delta() for each before running adaptation."
            )

        self._active_datamodule = datamodule

        # 1. Get Online Bounds
        bounds_array = self._get_online_bounds(datamodule, current_params)

        # 2. Select Objective Function
        if mode == Mode.EXPLORATION:
            objective_func = functools.partial(self._acquisition_func, w_explore=w_explore)
        elif mode == Mode.INFERENCE:
            objective_func = self._inference_func
        else:
            raise ValueError(f"Unknown phase: {mode}")

        # 3. Prepare Fixed Parameters (parameters without trust regions are fixed to current)
        fixed_subset = {
            k: v for k, v in current_params.items()
            if k not in self.trust_regions
        }

        # 4. Run Unified Optimization
        return self._run_optimization(
            datamodule,
            x0_params=current_params,
            bounds=bounds_array,
            objective_func=objective_func,
            n_rounds=0, # No random restarts for adaptation
            fixed_param_values=fixed_subset
        )

    def _run_optimization(
        self,
        datamodule: DataModule,
        x0_params: Optional[Dict[str, Any]],
        bounds: np.ndarray,
        objective_func: Callable,
        n_rounds: int,
        fixed_param_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the acquisition function optimization."""
        # Start from current params if available
        x0_list = []
        if x0_params:
            x0_list.append(datamodule.params_to_array(x0_params))

        # Random restarts
        for _ in range(n_rounds):
            x0_list.append(self.rng.uniform(bounds[:, 0], bounds[:, 1]))

        if not x0_list:
             # Fallback if no x0_params and n_rounds=0 (unlikely)
             x0_list.append(self.rng.uniform(bounds[:, 0], bounds[:, 1]))

        # Run optimization from each starting point
        best_x, best_val = None, np.inf
        for x0 in x0_list:
            try:
                res = minimize(
                    fun=objective_func,
                    x0=x0,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
                self.logger.debug(f"Optimization round result: val={res.fun}, x={res.x}")
            except Exception as e:
                self.logger.warning(f"Optimization round failed with error: {e}")
                continue

        # Handle failure
        if best_x is None:
            self.logger.warning("Optimization failed, returning fallback parameters.")
            if x0_params:
                return x0_params
            else:
                raise RuntimeError("No valid parameters could be proposed.")
        else:
            self.logger.info(f"Optimization succeeded: best_val={best_val}, best_x={best_x}")

        # Convert result back
        proposed_params = datamodule.array_to_params(best_x)
        if fixed_param_values:
            proposed_params.update(fixed_param_values)
        # Carry over any params from the starting point that aren't in the
        # optimization space (e.g. runtime params not used by any prediction model).
        if x0_params:
            for k, v in x0_params.items():
                if k not in proposed_params:
                    proposed_params[k] = v
        return datamodule.dataset.schema.parameters.sanitize_values(
            proposed_params,
            ignore_unknown=True
        )

    # === BOUNDS FOR OPTIMIZATION ===

    def _get_offline_bounds(self, datamodule: DataModule) -> np.ndarray:
        """Calculate optimization bounds for OFFLINE domain (Global + Fixed Context)."""
        bounds_list = []
        col_map = datamodule.get_onehot_column_map()

        for code in datamodule.input_columns:

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

    def _get_online_bounds(self, datamodule: DataModule, current_params: Dict[str, Any]) -> np.ndarray:
        """Calculate optimization bounds for ONLINE domain (Trust Regions around Current)."""
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
            if code in self.trust_regions:
                delta = self.trust_regions[code]
                low, high = curr - delta, curr + delta
            else:
                # No trust region -> Fixed to current
                low, high = curr, curr

            # Process & Append
            bounds_list.append(self._normalize_bounds(code, low, high, datamodule))

        return np.array(bounds_list)

    def _get_hierarchical_bounds_for_code(self, code: str) -> Tuple[float, float]:
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

    def _normalize_bounds(self, col: str, low: float, high: float, datamodule: DataModule) -> Tuple[float, float]:
        """Normalize bounds to [0, 1] based on schema constraints."""
        n_low, n_high = datamodule.normalize_parameter_bounds(col, low, high)
        if (n_low, n_high) != (low, high):
            self.logger.debug(f"Processed bounds for '{col}': raw [{low}, {high}] -> normalized [{n_low}, {n_high}]")
        else:
            self.logger.debug(f"No normalization stats for '{col}'. Using raw bounds [{low}, {high}].")
        return n_low, n_high

    # === WRAPPERS ===

    def get_models(self) -> List[Any]:
        """Return empty list (no internal ML models owned by CalibrationSystem)."""
        return []

    def get_model_specs(self) -> Dict[str, List[str]]:
        return {}
