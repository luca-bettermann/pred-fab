from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize

import warnings
import functools

from ..core import DataModule, DatasetSchema
from ..core import DataInt, DataReal, DataObject, DataBool, DataCategorical
from ..utils import PfabLogger, Mode, SamplingStrategy
from ..interfaces import ISurrogateModel, GaussianProcessSurrogate
from .base_system import BaseOrchestrationSystem

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

class CalibrationSystem(BaseOrchestrationSystem):
    """
    Orchestrates calibration and active learning.
    
    - Owns Exploration Model (GP) and System Performance definition
    - Generates baseline experiments (LHS)
    - Proposes new experiments via Bayesian Optimization
    - Supports Online (Trust Region) and Offline (Global) modes
    """
    
    def __init__(
        self, 
        schema: DatasetSchema,
        logger: PfabLogger, 
        predict_fn: Callable, 
        residual_predict_fn: Callable,
        evaluate_fn: Callable, 
        random_seed: Optional[int] = None,
        surrogate_model: Optional[ISurrogateModel] = None,
    ):
        super().__init__(logger)
        self.predict_fn = predict_fn
        self.evaluate_fn = evaluate_fn
        self.residual_predict_fn = residual_predict_fn
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        # Initialize Surrogate Model
        if surrogate_model:
            self.model = surrogate_model
        else:
            self.model = GaussianProcessSurrogate(logger, random_seed or 42)

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
            
            self.trust_regions[code] = delta


    def _validate_and_clean_config(
        self, 
        code: str, 
        allowed_types: Optional[Tuple[type, ...]], 
        conflicting_collections: List[str], 
        force: bool
    ) -> bool:
        """
        Validate parameter against schema and conflicting configurations.
        
        Args:
            code: Parameter code
            allowed_types: Tuple of allowed DataObject types
            conflicting_collections: Names of dict attributes to check for conflicts
            force: If True, remove conflicts. If False, return False on conflict.
            
        Returns:
            bool: True if validation passed (and conflicts resolved), False if execution should stop.
        """
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
        # X is (n_features,)
        # predict_fn expects (n_samples, n_features)
        X_reshaped = X.reshape(1, -1)
        pred_features = self.predict_fn(X_reshaped)
        
        # Apply residual correction if available (Online Adaptation)
        # TODO: make this cleaner
        if self.residual_predict_fn is not None:
            # Prepare inputs for residual model: [X, BasePredictions]
            X_residual_input = np.hstack([X_reshaped, pred_features])
            residuals = self.residual_predict_fn(X_residual_input)
            pred_features = pred_features + residuals
            
        pred_performance = self.evaluate_fn(pred_features)
        
        # Extract values from dict and compute score
        # Note: evaluate_fn returns dict of arrays/scalars. 
        # We assume single sample here.
        perf_values = [float(val) if np.isscalar(val) else float(val[0]) for val in pred_performance.values()] # type: ignore
        
        sys_perf = self._compute_system_performance(perf_values)
        return -sys_perf
    
    def _acquisition_func(self, X: np.ndarray, w_explore: float) -> float:
        """
        Objective for exploration: Maximize Weighted Score.
        Score = (1 - w) * Mean + w * Std
        Returns negative Score for minimization.
        """
        # Predict mean and std from surrogate
        mean, std = self.model.predict(X.reshape(1, -1))
        
        weighted_mu = self._compute_system_performance(mean[0].tolist())
        weighted_sigma = self._compute_system_performance(std[0].tolist())
        
        # Weighted Blend
        # w=0 -> Pure Mean (Exploitation)
        # w=1 -> Pure Std (Exploration)
        score = (1.0 - w_explore) * weighted_mu + w_explore * weighted_sigma
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
    
    # === SURROGATE TRAINING ===

    def train_surrogate_model(
        self,
        datamodule: DataModule
    ) -> None:
        """Train surrogate model on existing experiment data. We assume the datamodule is fitted."""
        X_train, y_train = datamodule.build_calibration_training_arrays(
            performance_order=self.perf_names_order,
            strict=False
        )
        if len(X_train) > 0:
            self.model.fit(X_train, y_train)
        else:
            self.logger.warning("No valid data to train surrogate model.")


    # === BASELINE EXPERIMENT GENERATION ===

    def generate_baseline_experiments(
        self, 
        n_samples: int,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> List[Dict[str, Any]]:
        """Generate initial design using Latin Hypercube Sampling."""

        self.logger.info(f"Generating {n_samples} baseline experiments using Latin Hypercube Sampling...")
        
        # 1. Define Sampling Space (Physics)
        sampling_specs = self._get_sampling_specs(param_bounds)
        if not sampling_specs:
            self.logger.warning("No valid parameters for baseline generation.")
            return []
        
        param_names = sorted(sampling_specs.keys())
        d = len(param_names)
        
        # 2. Generate Stratified Samples (Geometry)
        # Returns (n_samples, d) matrix with values in [0, 1]
        sampler = qmc.LatinHypercube(d=d, seed=self.random_seed)
        lhs_samples = sampler.random(n=n_samples) 
        
        # 3. Transform Samples to Parameter Space (Mapping)
        experiments = []
        for row in lhs_samples:
            params = self._transform_lhs_sample(row, param_names, sampling_specs)
            experiments.append(params)
            self.logger.debug(f"Generated baseline experiment: {params}")
            
        return experiments

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
        # 1. Get Offline Bounds
        bounds_array = self._get_offline_bounds(datamodule)
        
        # 2. Select Objective Function
        if mode == Mode.EXPLORATION:
            self.train_surrogate_model(datamodule)
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
        # 1. Get Online Bounds
        bounds_array = self._get_online_bounds(datamodule, current_params)
        
        # 2. Select Objective Function
        if mode == Mode.EXPLORATION:
            self.train_surrogate_model(datamodule)
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
        """Return Surrogate Model (required by BaseOrchestrationSystem)."""
        return [self.model]
    
    def get_model_specs(self) -> Dict[str, List[str]]:
        return {}
