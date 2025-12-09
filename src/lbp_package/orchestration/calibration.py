from os import name
from typing import Dict, List, Optional, Any, Literal, Tuple, Callable
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize
import warnings
import functools

from ..core import Dataset, ExperimentData, PerformanceAttributes, DataModule, DatasetSchema
from ..utils import LBPLogger, SplitType, Mode
from ..interfaces.calibration import IExplorationModel, GaussianProcessExploration
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
        logger: LBPLogger, 
        schema: DatasetSchema, 
        predict_fn: Callable, 
        evaluate_fn: Callable, 
        random_seed: Optional[int] = None,
        model: Optional[IExplorationModel] = None
    ):
        super().__init__(logger)
        # self.schema = schema
        self.predict_fn = predict_fn
        self.evaluate_fn = evaluate_fn
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
        # Configuration
        self.param_bounds: Dict[str, Tuple[float, float]] = {}
        self.trust_regions: Dict[str, float] = {}
        self.fixed_params: Dict[str, Any] = {}

        # Set ordered weights
        self.perf_names_order = list(schema.performance.keys())
        self.performance_weights: Dict[str, float] = {perf: 1.0 for perf in self.perf_names_order}
        
        # Initialize Surrogate Model
        if model:
            self.model = model
        else:
            self.model = GaussianProcessExploration(logger, random_seed or 42)
        
    def set_performance_weights(self, weights: Dict[str, float]) -> None:
        """Set weights for system performance calculation. Default is 1.0 for all."""
        # set according to order in perf_names_order
        for name, value in weights.items():
            if name in self.performance_weights:
                self.performance_weights[name] = value
            else:
                self.logger.console_warning(f"Performance attribute '{name}' not in schema; ignoring weight.")
        
    def configure_param_bounds(
        self, 
        bounds: Dict[str, Tuple[float, float]], 
        fixed_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Configure parameter ranges for offline calibration."""
        self.param_bounds = bounds
        self.fixed_params = fixed_params or {}
        
    def configure_trust_regions(
        self, 
        deltas: Dict[str, float]
    ) -> None:
        """
        Configure trust region deltas for online calibration.
        Non-configured parameters have no trust region applied and are fixed.
        """
        self.trust_regions = deltas

    def compute_system_performance(self, performance: List[float]) -> float:
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
    
    def _train_exploration_model(
        self,
        datamodule: DataModule
    ) -> None:
        """Train surrogate model on existing experiment data. We assume the datamodule is fitted."""
        X_train, y_train = self._get_train_arrays(datamodule)
        if len(X_train) > 0:
            self.model.fit(X_train, y_train)
        else:
            self.logger.warning("No valid data to train surrogate model.")
    
    def _exploitation_func(self, X: np.ndarray) -> float:
        """Exploitation function using prediction and evaluation functions."""
        pred_features = self.predict_fn(X)
        pred_performance = self.evaluate_fn(pred_features)
        pred_sys_performance = self.compute_system_performance(list(pred_performance.values()))
        return -pred_sys_performance
    
    def propose_params(
        self,
        datamodule: DataModule,
        mode: Mode,
        current_params: Dict[str, Any],
        w_explore: float = 0.5,
        n_optimization_rounds: int = 10,
    ) -> Dict[str, Any]:
        # 1. Fit Surrogate on latest data (only in offline mode)
        if mode == Mode.OFFLINE:
            self._train_exploration_model(datamodule)

        # 2. Get Bounds
        bounds = self._get_bounds_for_step(datamodule, current_params, mode)

        # 3 Define objective function for optimization
        if mode == Mode.OFFLINE:
            # retrieve exploration function and fix arguments
            objective_func = functools.partial(
                self.model.exploration_func, 
                sys_perf=self.compute_system_performance,
                w_explore=w_explore
            )
        else:
            objective_func = self._exploitation_func
        
        # 3. Run Optimization (use exploration function from IExplorationModel)
        return self._run_optimization(
            datamodule, 
            current_params,
            bounds, 
            objective_func,
            n_optimization_rounds, 
            )

    def _get_bounds_for_step(
        self, 
        datamodule: DataModule, 
        current_params: Dict[str, Any],
        mode: Mode = Mode.OFFLINE, 
    ) -> np.ndarray:
        """Calculate optimization bounds based on mode, fixed params, and config."""
        bounds_list = []
        
        for col in datamodule.input_columns:
            # 1. Determine Physical Bounds
            low, high = self._get_physical_bounds_for_col(col, current_params, mode, datamodule)
            
            # 2. Normalize
            if col in datamodule._parameter_stats:
                # Continuous variable with stats
                n_low = datamodule._apply_normalization(np.array([low]), datamodule._parameter_stats[col])[0]
                n_high = datamodule._apply_normalization(np.array([high]), datamodule._parameter_stats[col])[0]
                bounds_list.append((min(n_low, n_high), max(n_low, n_high)))
            else:
                # Likely one-hot or no normalization (already [0, 1])
                bounds_list.append((low, high))
                
        return np.array(bounds_list)

    def _get_physical_bounds_for_col(
        self, 
        col: str, 
        current_params: Dict[str, Any], 
        mode: Mode, 
        datamodule: DataModule
    ) -> Tuple[float, float]:
        """Determine physical (unnormalized) bounds for a single column."""
        # 1. Identify if One-Hot
        parent_param = None
        cat_val = None
        mappings = getattr(datamodule, '_categorical_mappings', {})
        
        for p, cats in mappings.items():
            for cat in cats:
                if col == f"{p}_{cat}":
                    parent_param = p
                    cat_val = cat
                    break
            if parent_param: break
        
        # === FIXED PARAMS CHECK (Priority 1) ===
        # Handle Continuous Fixed
        if col in self.fixed_params:
            val = self.fixed_params[col]
            return val, val
            
        # Handle Categorical Fixed (One-Hot)
        if parent_param and parent_param in self.fixed_params:
            fixed_val = self.fixed_params[parent_param]
            # If fixed to this category -> 1.0, else -> 0.0
            val = 1.0 if fixed_val == cat_val else 0.0
            return val, val

        # === SCHEMA CONSTRAINTS (Priority 3/Fallback) ===
        schema_min, schema_max = -np.inf, np.inf
        
        # If it's a direct parameter in schema
        if datamodule.dataset.schema.parameters.has(col):
            data_obj = datamodule.dataset.schema.parameters.get(col)
            schema_min = data_obj.constraints.get("min", -np.inf)
            schema_max = data_obj.constraints.get("max", np.inf)
        elif parent_param:
            # One-hot columns are bounded [0, 1]
            schema_min, schema_max = 0.0, 1.0

        # === MODE SPECIFIC LOGIC ===
        low, high = -np.inf, np.inf
        
        if mode == Mode.ONLINE:
            # Online Hierarchy
            # 1. Current Param as Center
            curr = 0.0
            if current_params:
                if col in current_params:
                    curr = current_params[col]
                elif parent_param and parent_param in current_params:
                    # One-hot from current params
                    curr = 1.0 if current_params[parent_param] == cat_val else 0.0
            
            # 2. Trust Regions
            if col in self.trust_regions:
                delta = self.trust_regions[col]
                low = curr - delta
                high = curr + delta
            else:
                # "take curr_param as fixed params"
                low = high = curr
                
        elif mode == Mode.OFFLINE:
            # Offline Hierarchy
            # 2. Param Bounds
            if col in self.param_bounds:
                low, high = self.param_bounds[col]
            else:
                # Fallback to schema bounds if no explicit bounds
                low, high = schema_min, schema_max

        # === CLAMPING (Priority 3) ===
        low = max(low, schema_min)
        high = min(high, schema_max)
        
        # === VALIDATION (Priority 4) ===
        if low == -np.inf or high == np.inf:
             raise ValueError(f"Could not determine finite bounds for parameter '{col}'. Please configure bounds or fixed parameters.")
             
        return low, high

    def _run_optimization(
        self, 
        datamodule: DataModule, 
        current_params: Dict[str, Any],
        bounds: np.ndarray, 
        objective_func: Callable,
        n_rounds: int, 
    ) -> Dict[str, Any]:
        """Run the acquisition function optimization."""
        # Start from current params if available
        x0_list = [datamodule.params_to_array(current_params)]
        
        # Random restarts
        for _ in range(n_rounds):
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
            if current_params:
                return current_params
            else:
                raise RuntimeError("No valid parameters could be proposed.")
        else:
            self.logger.info(f"Optimization succeeded: best_val={best_val}, best_x={best_x}")
                
        # Convert result back
        proposed_params = datamodule.array_to_params(best_x)
        proposed_params.update(self.fixed_params)
        return proposed_params

    # === WRAPPERS ===

    def get_models(self) -> List[Any]:
        """Return Surrogate Model (required by BaseOrchestrationSystem)."""
        return [self.model]
    
    def get_model_specs(self) -> Dict[str, List[str]]:
        return {}