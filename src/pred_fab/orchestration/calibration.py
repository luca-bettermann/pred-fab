from os import name
from typing import Dict, List, Optional, Any, Literal, Tuple, Callable, Type
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize

import warnings
import functools

from ..core import DataModule, DatasetSchema
from ..core import DataInt, DataReal, DataObject, DataBool, DataCategorical
from ..utils import PfabLogger, SplitType, Domain, Mode
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
        
        # Set static parameter constraints from schema
        self.data_objects: Dict[str, DataObject] = {}
        self.param_constraints: Dict[str, Tuple[float, float]] = {}

        # Configure bounds and fixed params
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
            self.param_constraints[code] = (min_val, max_val)

    def state_report(self) -> None:
        """Log the current calibration configuration state."""
        summary = ["===== Calibration System State =====\n"]
        width = 20
        # Columns: Input, Bounds, Delta
        header = f"{'Input':<{width}} | {'Bounds':<{width}} | {'Delta':<{8}}"
        summary.append(header)
        summary.append("-" * len(header))

        for code, (s_min, s_max) in self.param_constraints.items():
            
            # Determine Bounds
            # Priority: Fixed -> Configured Bounds -> Schema Constraints
            if code in self.fixed_params:
                val = self.fixed_params[code]
                bounds_str = f"fixed = {val}"
            elif code in self.param_bounds:
                low, high = self.param_bounds[code]
                bounds_str = f"[{low}, {high}]"
            else:
                bounds_str = f"[{s_min}, {s_max}]"
            
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
            schema_min, schema_max = self.param_constraints[code]
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

    def _get_train_arrays(self, datamodule: DataModule) -> Tuple[np.ndarray, np.ndarray]:
        """Get train arrays of X for the parameters and y for the performance attributes."""
        X_train = []
        y_train = []

        for code in datamodule.get_split_codes(split=SplitType.TRAIN):
            exp = datamodule.dataset.get_experiment(code)
            if exp.performance:
                params = exp.parameters.get_values_dict()
                x_arr = datamodule.params_to_array(params)
                
                X_train.append(x_arr)
                y_train.append(exp.performance.values)
            else:
                self.logger.warning(f"Performance data missing for experiment {code}. Skipping.")
                continue

        return np.array(X_train), np.array(y_train)
    
    def train_surrogate_model(
        self,
        datamodule: DataModule
    ) -> None:
        """Train surrogate model on existing experiment data. We assume the datamodule is fitted."""
        X_train, y_train = self._get_train_arrays(datamodule)
        if len(X_train) > 0:
            self.model.fit(X_train, y_train)
        else:
            self.logger.warning("No valid data to train surrogate model.")


    # === BASELINE EXPERIMENT GENERATION ===

    def generate_baseline_experiments(
        self, 
        n_samples: int, 
        param_ranges: Dict[str, Tuple[float, float]]
    ) -> List[Dict[str, Any]]:
        """Generate initial design using Latin Hypercube Sampling."""
        param_names = sorted(param_ranges.keys())
        bounds = np.array([param_ranges[name] for name in param_names])
        
        sampler = qmc.LatinHypercube(d=len(param_names), seed=self.random_seed)
        sample = sampler.random(n=n_samples)
        
        # Scale samples to bounds
        scaled = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
        
        experiments = []
        for row in scaled:
            params = {name: float(val) for name, val in zip(param_names, row)}
            experiments.append(params)
            
        return experiments

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
            fixed_param_values=context
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
        return proposed_params

    # === BOUNDS FOR OPTIMIZATION ===

    def _get_onehot_info(self, col: str, datamodule: DataModule) -> Tuple[Optional[str], Optional[Any]]:
        """Identify if a column is part of a one-hot encoding component."""
        mappings = getattr(datamodule, '_categorical_mappings', {})
        for p, cats in mappings.items():
            for cat in cats:
                if col == f"{p}_{cat}":
                    return p, cat
        return None, None

    def _get_offline_bounds(self, datamodule: DataModule) -> np.ndarray:
        """Calculate optimization bounds for OFFLINE domain (Global + Fixed Context)."""
        bounds_list = []
        for col in datamodule.input_columns:
            parent_param, cat_val = self._get_onehot_info(col, datamodule)
            
            low, high = -np.inf, np.inf
            
            # 1. Check Fixed Context
            if col in self.fixed_params:
                val = self.fixed_params[col]
                low, high = val, val
            elif parent_param and parent_param in self.fixed_params:
                # Fixed Categorical
                val = 1.0 if self.fixed_params[parent_param] == cat_val else 0.0
                low, high = val, val
            # 2. Check Explicit Param Bounds
            elif col in self.param_bounds:
                low, high = self.param_bounds[col]
            # 3. Check One-Hot (implicit [0, 1])
            elif parent_param:
                low, high = 0.0, 1.0
            
            # Process & Append
            bounds_list.append(self._process_single_bound(col, low, high, datamodule, parent_param))
            
        return np.array(bounds_list)

    def _get_online_bounds(self, datamodule: DataModule, current_params: Dict[str, Any]) -> np.ndarray:
        """Calculate optimization bounds for ONLINE domain (Trust Regions around Current)."""
        bounds_list = []
        for col in datamodule.input_columns:
            parent_param, cat_val = self._get_onehot_info(col, datamodule)
            
            # Determine Center (Current Value)
            curr = 0.0
            if current_params:
                if col in current_params:
                    curr = current_params[col]
                elif parent_param and parent_param in current_params:
                    curr = 1.0 if current_params[parent_param] == cat_val else 0.0
            
            # Determine Bounds from Trust Region
            if col in self.trust_regions:
                delta = self.trust_regions[col]
                low, high = curr - delta, curr + delta
            else:
                # No trust region -> Fixed to current
                low, high = curr, curr
                
            # Process & Append
            bounds_list.append(self._process_single_bound(col, low, high, datamodule, parent_param))
            
        return np.array(bounds_list)

    def _process_single_bound(
        self, 
        col: str, 
        low: float, 
        high: float, 
        datamodule: DataModule,
        parent_param: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Helper to:
        1. Fallback to Schema max/min if bounds are -inf/inf
        2. Clamp bounds to Schema min/max
        3. Normalize bounds
        """
        # === SCHEMA FALLBACK & CLAMPING ===
        # 1. Retrieve Schema Constraints
        # For one-hot columns, we use the parent parameter's constraints (which are 0.0, 1.0 for categorical)
        lookup_key = parent_param if parent_param else col
        
        if lookup_key in self.param_constraints:
             schema_min, schema_max = self.param_constraints[lookup_key]
        else:
            raise ValueError(f"Could not find schema constraints for parameter '{lookup_key}'.")


        # TODO: fallback is already configured in the parameter constraints? -> adjust this function!
        # Fallback if still infinite (e.g. offline mode with no bounds set)
        if low == -np.inf: low = schema_min
        if high == np.inf: high = schema_max
            
        # Clamp
        orig_low, orig_high = low, high
        low = max(low, schema_min)
        high = min(high, schema_max)
        
        if low != orig_low or high != orig_high:
             self.logger.warning(
                 f"Bounds for parameter '{col}' constrained to schema limits: "
                 f"[{orig_low}, {orig_high}] -> [{low}, {high}]"
            )
        
        if low == -np.inf or high == np.inf:
             raise ValueError(f"Could not determine finite bounds for parameter '{col}'. Please configure bounds or fixed parameters.")

        # === NORMALIZATION ===
        if col in datamodule._parameter_stats:
            # Continuous variable with stats
            stats = datamodule._parameter_stats[col]
            n_low = datamodule._apply_normalization(np.array([low]), stats)[0]
            n_high = datamodule._apply_normalization(np.array([high]), stats)[0]
            # Handle flipping if normalization (e.g. -1 factor?) - usually linear monotonic
            return (min(n_low, n_high), max(n_low, n_high))
        else:
            # Maybe one-hot or not normalized
            return (low, high)

    # === WRAPPERS ===

    def get_models(self) -> List[Any]:
        """Return Surrogate Model (required by BaseOrchestrationSystem)."""
        return [self.model]
    
    def get_model_specs(self) -> Dict[str, List[str]]:
        return {}