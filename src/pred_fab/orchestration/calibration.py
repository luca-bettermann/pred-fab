from typing import Dict, List, Optional, Any, Set, Tuple, Callable, cast
import numpy as np
from scipy.optimize import minimize

import warnings
import functools

from ..core import DataModule, DatasetSchema
from ..core import DataInt, DataReal, DataObject, DataBool, DataCategorical, DataDimension
from ..core import ParameterProposal, ParameterSchedule, ExperimentSpec
from ..utils import PfabLogger, Mode, SourceStep
from .base_system import BaseOrchestrationSystem
from ..utils._calib_baseline import BaselineSampler

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
        self._random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.default_mpc_lookahead_depth: int = 0
        self.default_mpc_discount: float = 0.9

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
        self.trajectory_configs: Dict[str, str] = {}   # param_code → dimension_code

        # Extract parameter constraints from schema
        self._set_param_constraints_from_schema(schema)

        # Baseline sampler — holds references to the shared config dicts so that
        # configure_* mutations are automatically visible without re-wiring.
        self.baseline_sampler = BaselineSampler(
            parameters=self.parameters,
            data_objects=self.data_objects,
            schema_bounds=self.schema_bounds,
            fixed_params=self.fixed_params,
            param_bounds=self.param_bounds,
            trajectory_configs=self.trajectory_configs,
            rng=self.rng,
            logger=self.logger,
            random_seed=self._random_seed,
        )

    # ------------------------------------------------------------------
    # random_seed property — propagates updates to the baseline sampler.
    # ------------------------------------------------------------------

    @property
    def random_seed(self) -> Optional[int]:
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value: Optional[int]) -> None:
        self._random_seed = value
        if hasattr(self, 'baseline_sampler') and self.baseline_sampler is not None:
            self.baseline_sampler.random_seed = value

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


    def configure_trajectory(self, code: str, dimension_code: str, force: bool = False) -> None:
        """Configure a runtime parameter for trajectory-based stepping over a dimension.

        Args:
            code: Runtime-adjustable parameter code (e.g., ``"speed"``).
            dimension_code: Dimension parameter code to step through (e.g., ``"dim_1"``).
                Must refer to a ``DataDimension`` in the schema.
            force: Overwrite an existing trajectory configuration for *code*.
        """
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

        # Validate dimension_code
        if dimension_code not in self.data_objects:
            raise ValueError(
                f"Dimension '{dimension_code}' not found in schema."
            )
        dim_obj = self.data_objects[dimension_code]
        if not isinstance(dim_obj, DataDimension):
            raise ValueError(
                f"'{dimension_code}' is not a DataDimension parameter "
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

    def _wrap_mpc_objective(
        self,
        base_objective: Callable,
        datamodule: DataModule,
        depth: int,
        discount: float,
    ) -> Callable:
        """Wrap base_objective with a depth-step MPC lookahead.

        Accumulates discounted scores: MPC(X) = score(X) + γ¹·score(X₁) + … + γᵈ·score(Xᵈ),
        where each Xⱼ₊₁ is obtained by a trust-region L-BFGS-B step from Xⱼ.
        Returns base_objective unchanged when depth <= 0.
        """
        if depth <= 0:
            return base_objective

        def mpc_obj(X: np.ndarray) -> float:
            total = base_objective(X)
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
                    )
                    X_cur = res.x
                    total += (discount ** (j + 1)) * base_objective(X_cur)
                except Exception:
                    break
            return total

        return mpc_obj

    def _build_objective(self, mode: Mode, w_explore: float) -> Callable:
        """Return the objective function for the given calibration mode."""
        if mode == Mode.EXPLORATION:
            return functools.partial(self._acquisition_func, w_explore=w_explore)
        elif mode == Mode.INFERENCE:
            return self._inference_func
        raise ValueError(f"Unknown calibration mode: {mode}")

    def _build_step_grid(self, current_params: Dict[str, Any]) -> List[Dict[str, int]]:
        """Build flattened grid of ``{dim_code: step_index}`` dicts.

        Dimensions are ordered coarsest-first (by ``DataDimension.level``).
        Returns ``[{}]`` when no trajectory configs exist (experiment-level).
        """
        import itertools as _it

        dim_codes = sorted(
            {dc for dc in self.trajectory_configs.values()},
            key=lambda dc: cast(DataDimension, self.data_objects[dc]).level,
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
        prev_indices: Optional[Dict[str, int]],
        curr_indices: Dict[str, int],
    ) -> set:
        """Return trajectory param codes eligible for optimization at this step.

        A param is eligible when the dimension it is mapped to *transitions*
        (i.e. its index differs from the previous step, or it is the first step).
        """
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
        proposals: List[Dict[str, Any]],
        step_grid: List[Dict[str, int]],
        source_step: str,
    ) -> ExperimentSpec:
        """Assemble per-step result dicts into an ``ExperimentSpec``.

        ``proposals[0]`` becomes ``initial_params``.  For each dimension in the
        grid, collect proposals where that dimension transitions and build a
        ``ParameterSchedule`` with entries keyed by step index.
        """
        initial = ParameterProposal.from_dict(proposals[0], source_step=source_step)

        schedules: Dict[str, ParameterSchedule] = {}
        if len(proposals) > 1 and step_grid and step_grid[0]:
            # Collect unique dims
            dim_codes = sorted(
                {dc for dc in self.trajectory_configs.values()},
                key=lambda dc: cast(DataDimension, self.data_objects[dc]).level,
            )
            for dim_code in dim_codes:
                traj_codes = [
                    c for c, dc in self.trajectory_configs.items() if dc == dim_code
                ]
                entries: List[Tuple[int, ParameterProposal]] = []
                prev_idx: Optional[int] = None
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

    # === OPTIMIZATION WORKFLOW ===

    def run_calibration(
        self,
        datamodule: DataModule,
        mode: Mode,
        current_params: Optional[Dict[str, Any]] = None,
        target_indices: Optional[Dict[str, int]] = None,
        w_explore: float = 0.5,
        n_optimization_rounds: int = 10,
        mpc_lookahead_depth: Optional[int] = None,
        mpc_discount: Optional[float] = None,
    ) -> ExperimentSpec:
        """Run calibration and return an ExperimentSpec.

        Pass current_params and target_indices for online (trust-region) optimization.
        Omit both for offline (global bounds with random restarts) optimization.
        When trajectory parameters are configured, iterates over a step grid derived
        from current_params dimensions.
        """
        # Resolve instance-level MPC defaults
        lookahead = self.default_mpc_lookahead_depth if mpc_lookahead_depth is None else mpc_lookahead_depth
        discount = self.default_mpc_discount if mpc_discount is None else mpc_discount

        self._active_datamodule = datamodule
        objective = self._build_objective(mode, w_explore)

        if lookahead > 0:
            objective = self._wrap_mpc_objective(objective, datamodule, lookahead, discount)

        source_step = (
            SourceStep.EXPLORATION if mode == Mode.EXPLORATION else SourceStep.INFERENCE
        )

        # Online = both current_params AND target_indices provided (target_indices may be empty dict)
        is_online = current_params is not None and target_indices is not None
        if target_indices is not None:
            step_grid: List[Dict[str, int]] = [target_indices]  # type: ignore[list-item]
        elif current_params is not None:
            step_grid = self._build_step_grid(current_params)
        else:
            step_grid = [{}]

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

        working_params: Optional[Dict[str, Any]] = (
            dict(current_params) if current_params else None
        )
        prev_indices: Optional[Dict[str, int]] = None
        proposals: List[Dict[str, Any]] = []

        for curr_indices in step_grid:
            # Build fixed params for this step
            fixed_for_step = dict(self.fixed_params)
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
            else:
                bounds = self._get_global_bounds(datamodule)
                n_rounds = n_optimization_rounds

            result = self._run_optimization(
                datamodule,
                x0_params=working_params,
                bounds=bounds,
                objective_func=objective,
                n_rounds=n_rounds,
                fixed_param_values=fixed_for_step,
            )
            proposals.append(result)
            working_params = result
            prev_indices = curr_indices

        return self._build_experiment_spec(proposals, step_grid, source_step)

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

    def _get_global_bounds(self, datamodule: DataModule) -> np.ndarray:
        """Calculate global optimization bounds (full parameter space + fixed context)."""
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

    def _get_trust_region_bounds(self, datamodule: DataModule, current_params: Dict[str, Any]) -> np.ndarray:
        """Calculate trust-region optimization bounds centred on current_params."""
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
