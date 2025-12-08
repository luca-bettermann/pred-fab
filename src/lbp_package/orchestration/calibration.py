from typing import Dict, List, Optional, Any, Literal, Tuple, Callable
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize
import warnings

from ..core import Dataset, ExperimentData, PerformanceAttributes, DataModule
from ..utils.logger import LBPLogger
from ..interfaces.calibration import ISurrogateModel, GaussianProcessSurrogate
from .base_system import BaseOrchestrationSystem

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

class CalibrationSystem(BaseOrchestrationSystem):
    """
    Orchestrates calibration and active learning.
    
    - Owns Surrogate Model (GP) and System Performance definition
    - Generates baseline experiments (LHS)
    - Proposes new experiments via Bayesian Optimization
    - Supports Online (Trust Region) and Offline (Global) modes
    """
    
    def __init__(
        self, 
        logger: LBPLogger, 
        dataset: Dataset, 
        predict_fn: Callable, 
        evaluate_fn: Callable, 
        random_seed: Optional[int] = None,
        surrogate_model: Optional[ISurrogateModel] = None
    ):
        super().__init__(logger)
        self.dataset = dataset
        self.predict_fn = predict_fn
        self.evaluate_fn = evaluate_fn
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
        # Configuration
        self.performance_weights: Dict[str, float] = {}
        self.offline_bounds: Dict[str, Tuple[float, float]] = {}
        self.online_deltas: Dict[str, float] = {}
        self.fixed_params: Dict[str, Any] = {}
        
        # Initialize Surrogate Model
        if surrogate_model:
            self.surrogate_model = surrogate_model
        else:
            self.surrogate_model = GaussianProcessSurrogate(logger, random_seed or 42)
        
    def set_performance_weights(self, weights: Dict[str, float]) -> None:
        """Set weights for system performance calculation."""
        self.performance_weights = weights
        
    def configure_offline_ranges(
        self, 
        bounds: Dict[str, Tuple[float, float]], 
        fixed_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Configure parameter ranges for offline calibration."""
        self.offline_bounds = bounds
        self.fixed_params = fixed_params or {}
        
    def configure_online_deltas(
        self, 
        deltas: Dict[str, float],
        fixed_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Configure trust region deltas for online calibration."""
        self.online_deltas = deltas
        if fixed_params:
            self.fixed_params.update(fixed_params)

    def compute_system_performance(self, performance: PerformanceAttributes) -> Optional[float]:
        """Compute weighted system performance [0, 1]."""
        if not performance:
            return 0.0
            
        total_score = 0.0
        total_weight = 0.0
        
        for name, weight in self.performance_weights.items():
            if performance.has_value(name):
                val = performance.get_value(name)
                # Assume performance metrics are already [0, 1] or normalized
                total_score += val * weight
                total_weight += weight
            else:
                # self.logger.console_warning(f"Performance attribute '{name}' missing.")
                return None

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
    
    def train_surrogate_model(
        self,
        datamodule: DataModule
    ) -> None:
        """Train surrogate model on existing experiment data. We assume the datamodule is fitted."""
        X_hist = []
        y_hist = []

        train_batches = datamodule.get_batches('train')

        
        for code in self.dataset.get_experiment_codes():
            exp = self.dataset.get_experiment(code)
            if exp.performance:
                # Check fixed params
                match = True
                for k, v in self.fixed_params.items():
                    if exp.parameters.get_value(k) != v:
                        match = False
                        break
                if not match:
                    continue
                
                # Convert to array
                try:
                    params = exp.parameters.get_values_dict()
                    x_arr = datamodule.params_to_array(params)
                    perf = self.compute_system_performance(exp.performance)
                    
                    if perf is not None:
                        X_hist.append(x_arr)
                        y_hist.append(perf)
                except Exception:
                    continue
                    
        X_train = np.array(X_hist)
        y_train = np.array(y_hist)
        
        if len(X_train) > 0:
            self.surrogate_model.fit(X_train, y_train)
        else:
            self.logger.warning("No valid data to train surrogate model.")
        
    def propose_new_parameters(
        self,
        datamodule: DataModule,
        current_params: Optional[Dict[str, Any]] = None,
        online: bool = False,
        exploration_weight: float = 0.5,
        n_optimization_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Propose next parameter set using Bayesian Optimization.
        
        Args:
            datamodule: Fitted DataModule for normalization
            current_params: Current parameters (required for online mode)
            online: If True, use Trust Region (deltas) and force exploitation
            exploration_weight: 0.0 (Exploitation) to 1.0 (Exploration)
            n_optimization_steps: Number of random restarts for acquisition optimization
        """
        if online:
            exploration_weight = 0.0  # Force exploitation in online mode
            if current_params is None:
                raise ValueError("current_params required for online calibration")
        
        # 1. Prepare History Data (X, y)
        X_hist = []
        y_hist = []
        
        # We need to know which parameters are being optimized to build X correctly?
        # Actually, DataModule handles the full parameter vector.
        # But we only optimize a subset (free parameters).
        # The GP should probably model the full normalized parameter space?
        # Or just the free parameters?
        # Using DataModule.params_to_array gives us the full normalized vector.
        # This is good because it handles correlations between all parameters.
        
        for code in self.dataset.get_experiment_codes():
            exp = self.dataset.get_experiment(code)
            if exp.performance:
                # Check fixed params
                match = True
                for k, v in self.fixed_params.items():
                    if exp.parameters.get_value(k) != v:
                        match = False
                        break
                if not match:
                    continue
                
                # Convert to array
                try:
                    params = exp.parameters.get_values_dict()
                    x_arr = datamodule.params_to_array(params)
                    perf = self.compute_system_performance(exp.performance)
                    
                    if perf is not None:
                        X_hist.append(x_arr)
                        y_hist.append(perf)
                except Exception:
                    continue
                    
        X_train = np.array(X_hist)
        y_train = np.array(y_hist)
        
        # Fit Surrogate
        if len(X_train) > 0:
            self.surrogate_model.fit(X_train, y_train)
            
        # 2. Define Bounds for Optimization
        # We optimize in the normalized space of DataModule.
        # But we need to respect the physical bounds defined in config.
        
        # Get all input columns from DataModule
        input_cols = datamodule.input_columns
        bounds_list = []
        
        # Helper to normalize a single value for a column
        def norm_val(col, val):
            if col in datamodule._parameter_stats:
                return datamodule._apply_normalization(np.array([val]), datamodule._parameter_stats[col])[0]
            return val

        for col in input_cols:
            # Determine physical bounds for this column
            lower, upper = -np.inf, np.inf
            
            # Check if it's a fixed parameter
            if col in self.fixed_params:
                val = self.fixed_params[col]
                lower = upper = val
            
            # Check online trust region
            elif online and col in self.online_deltas and current_params:
                curr = current_params.get(col)
                delta = self.online_deltas[col]
                if curr is not None:
                    lower = curr - delta
                    upper = curr + delta
            
            # Check offline bounds
            elif not online and col in self.offline_bounds:
                lower, upper = self.offline_bounds[col]
            
            # Fallback: DataObject bounds from Schema?
            # (Skipped for brevity, assuming config is primary)
            
            # If still infinite, use safe defaults for normalized space
            if lower == -np.inf: lower = -10.0 # Normalized space assumption
            if upper == np.inf: upper = 10.0
            
            # Normalize bounds
            # Note: This is tricky for One-Hot columns. 
            # For One-Hot, we should probably constrain them to [0, 1].
            # But params_to_array produces normalized values.
            # If a column is one-hot derived, it's binary 0/1.
            # Optimization in continuous space [0, 1] is okay, array_to_params handles argmax.
            
            # If it's a one-hot column, we don't have explicit bounds usually.
            # We just bound it 0-1.
            # How to detect one-hot col?
            # It's not in offline_bounds keys usually (those are original names).
            
            # Simplified: Just use -5, 5 for normalized space if no explicit bound.
            # If explicit bound exists (continuous), normalize it.
            
            if col in self.offline_bounds or (online and col in self.online_deltas):
                n_lower = norm_val(col, lower)
                n_upper = norm_val(col, upper)
                bounds_list.append((min(n_lower, n_upper), max(n_lower, n_upper)))
            else:
                # Default safe bounds for normalized space
                bounds_list.append((-3.0, 3.0))

        bounds = np.array(bounds_list)
        
        # 3. Define Objective Function
        kappa = 1.96 if exploration_weight > 0 else 0.0 # 95% CI
        
        # Refined Objective using Surrogate only:
        def objective_surrogate(x):
            # Predict with Surrogate
            mu, sigma = self.surrogate_model.predict(x.reshape(1, -1))
            
            if exploration_weight == 0.0:
                return -mu[0]
            elif exploration_weight == 1.0:
                return -sigma[0]
            else:
                return -(mu[0] + kappa * sigma[0])

        # 4. Optimize
        best_x = None
        best_val = np.inf
        
        # Start from current params if available
        x0_list = []
        if current_params:
            x0_list.append(datamodule.params_to_array(current_params))
        
        # Random restarts
        for _ in range(n_optimization_steps):
            x0_list.append(self.rng.uniform(bounds[:, 0], bounds[:, 1]))
            
        for x0 in x0_list:
            res = minimize(
                objective_surrogate,
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        
        if best_x is None:
            # Fallback if optimization failed completely (unlikely with restarts)
            # Just return current params or random point
            if current_params:
                return current_params
            else:
                # Return random point from bounds
                random_x = self.rng.uniform(bounds[:, 0], bounds[:, 1])
                best_x = random_x
                
        # 5. Convert result back
        proposed_params = datamodule.array_to_params(best_x)
        proposed_params.update(self.fixed_params)
        
        return proposed_params


    # === WRAPPERS ===

    def get_models(self) -> List[Any]:
        """Return Surrogate Model (required by BaseOrchestrationSystem)."""
        return [self.surrogate_model]
    
    def get_model_specs(self) -> Dict[str, List[str]]:
        return {}