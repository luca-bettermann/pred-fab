from typing import Dict, Tuple, Callable
from scipy.optimize import differential_evolution
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler

from lbp_package.interfaces import ICalibrationModel


class RandomSearchCalibration(ICalibrationModel):
    """Random search - crystal clear and simple."""
    
    def __init__(self, n_evaluations: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.n_evaluations = n_evaluations
        
    def optimize(self, param_ranges: Dict[str, Tuple[float, float]], 
                 objective_fn: Callable[[Dict[str, float]], float]) -> Dict[str, float]:
        """Try n_evaluations random points, return the best."""
        from scipy.stats import uniform
        
        # Set up parameter distributions  
        param_dists = {k: uniform(loc=min_val, scale=max_val-min_val) 
                      for k, (min_val, max_val) in param_ranges.items()}
        
        best_params = None
        best_objective = float('-inf')
        
        # Try random samples
        for params in ParameterSampler(param_dists, n_iter=self.n_evaluations):
            objective_val = objective_fn(params)
            if objective_val > best_objective:
                best_objective = objective_val  
                best_params = params.copy()
                
        if best_params is None:
            raise RuntimeError("No valid parameters found")
        return best_params


class DifferentialEvolutionCalibration(ICalibrationModel):
    """Differential evolution - direct scipy integration."""
    
    def __init__(self, maxiter: int = 50, popsize: int = 15, **kwargs):
        super().__init__(**kwargs)
        self.maxiter = maxiter
        self.popsize = popsize
        
    def optimize(self, param_ranges: Dict[str, Tuple[float, float]], 
                 objective_fn: Callable[[Dict[str, float]], float]) -> Dict[str, float]:
        """Use scipy's differential evolution directly."""
        
        # Convert param_ranges to bounds format for scipy
        param_names = list(param_ranges.keys())
        bounds = [param_ranges[name] for name in param_names]
        
        # Wrapper function: convert array to dict and back
        def scipy_objective(x):
            params = {name: val for name, val in zip(param_names, x)}
            return -objective_fn(params)  # Minimize negative (scipy minimizes, we want to maximize)
        
        # Use scipy differential evolution
        result = differential_evolution(
            scipy_objective, 
            bounds, 
            maxiter=self.maxiter, 
            popsize=self.popsize,
            seed=42
        )
        
        # Convert result back to parameter dict
        best_params = {name: val for name, val in zip(param_names, result.x)}
        return best_params
