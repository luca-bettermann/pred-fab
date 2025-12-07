from typing import Dict, List, Optional, Any, Literal, Tuple, Callable
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import warnings

from ..core.dataset import Dataset, ExperimentData
from ..utils.logger import LBPLogger
from ..interfaces.calibration import ICalibrationStrategy, CalibrationModes
from .base import BaseOrchestrationSystem

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

class BayesianCalibrationStrategy(ICalibrationStrategy):
    """
    Bayesian Optimization strategy using Gaussian Processes.
    
    - Uses GP to model system performance and uncertainty
    - Proposes points using UCB acquisition function
    """
    
    def __init__(self, logger: LBPLogger, predict_fn: Callable, evaluate_fn: Callable, random_seed: Optional[int] = None):
        super().__init__(logger, predict_fn, evaluate_fn)
        self.rng = np.random.RandomState(random_seed)
        
        # Initialize GP
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=random_seed,
            normalize_y=True
        )
        
    def propose_next_points(
        self,
        X_history: np.ndarray,
        y_history: np.ndarray,
        bounds: np.ndarray,
        n_points: int = 1,
        mode: CalibrationModes = 'exploration',
        exploration_weight: float = 0.5
        ) -> np.ndarray:
        """Propose next points using Bayesian Optimization."""

        kappa = self._set_kappa(exploration_weight)

        # Fit GP
        if len(X_history) > 0:
            self.gp.fit(X_history, y_history)
            
        # Define acquisition function
        def acquisition(x):
            x = x.reshape(1, -1)
            _, sigma = self.gp.predict(x, return_std=True) # type: ignore
            
            if mode == 'optimization':
                return -mu  # Minimize negative mean (maximize mean)
            else:
                # UCB: mu + kappa * sigma
                # kappa=2.0 (~95% CI)
                return -(mu + kappa * sigma)

        # Optimize acquisition
        proposed = []
        for _ in range(n_points):
            # Random restarts
            best_x = None
            best_val = np.inf
            
            for _ in range(10):
                x0 = self.rng.uniform(bounds[:, 0], bounds[:, 1])
                res = minimize(
                    acquisition,
                    x0,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            
            proposed.append(best_x)
            
        return np.array(proposed)
    
    def _set_kappa(self, exploration_weight: float) -> float:
        """Map exploration weight [0, 1] to kappa [0, ∞)."""
        if exploration_weight < 0.0 or exploration_weight > 1.0:
            raise ValueError("exploration_weight must be in [0, 1]")
        
        # exploration weight [0, 1], maps to kappa in UCB [0, ∞)
        if exploration_weight == 1.0:
            return 0.0
        elif exploration_weight == 0.0:
            return 1.0
        else:
            return 1 / (1.0 - exploration_weight + 1e-6) - 1.0


class CalibrationSystem(BaseOrchestrationSystem):
    """
    Orchestrates calibration and active learning.
    
    - Owns Surrogate Model (GP) and System Performance definition
    - Generates baseline experiments (LHS)
    - Proposes new experiments via Strategy
    """
    
    def __init__(self, dataset: Dataset, logger: LBPLogger, predict_fn: Callable, evaluate_fn: Callable, random_seed: Optional[int] = None):
        super().__init__(dataset, logger)
        self.strategy = BayesianCalibrationStrategy(logger, predict_fn, evaluate_fn, random_seed)
        self.performance_weights: Dict[str, float] = {}
        self.random_seed = random_seed
        
    def get_models(self) -> Any:
        """Return strategy (required by BaseOrchestrationSystem)."""
        return self.strategy
        
    def set_performance_weights(self, weights: Dict[str, float]) -> None:
        """Set weights for system performance calculation."""
        self.performance_weights = weights
        
    def compute_system_performance(self, exp_data: ExperimentData) -> Optional[float]:
        """Compute weighted system performance [0, 1]."""
        if not exp_data.performance:
            return 0.0
            
        total_score = 0.0
        total_weight = 0.0
        
        for name, weight in self.performance_weights.items():
            if exp_data.performance.has_value(name):
                val = exp_data.performance.get_value(name)
                # Assume performance metrics are already [0, 1] or normalized
                total_score += val * weight
                total_weight += weight
            else:
                self.logger.warning(f"Performance attribute '{name}' missing in experiment '{exp_data.exp_code}'. Aborting performance computation.")
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
        
    def propose_new_experiments(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        n_points: int = 1,
        mode: Literal['exploration', 'optimization'] = 'exploration',
        fixed_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Propose next experiments using strategy."""
        fixed_params = fixed_params or {}
        
        # Prepare history data
        X_hist = []
        y_hist = []
        
        param_names = sorted(param_ranges.keys())
        
        for code in self.dataset.get_experiment_codes():
            exp = self.dataset.get_experiment(code)
            if exp.performance: # Only use evaluated experiments
                # Check if fixed params match
                match = True
                for k, v in fixed_params.items():
                    if exp.parameters.get_value(k) != v:
                        match = False
                        break
                if not match:
                    continue
                    
                # Extract free params
                x_row = [exp.parameters.get_value(name) for name in param_names]
                perf = self.compute_system_performance(exp)
                if perf is not None:
                    X_hist.append(x_row)
                    y_hist.append(perf)
                else:
                    self.logger.warning(f"Experiment '{code}' missing performance, skipping in history.")
                
        X_arr = np.array(X_hist)
        y_arr = np.array(y_hist)
        bounds = np.array([param_ranges[name] for name in param_names])
        
        # Delegate to strategy
        proposed_X = self.strategy.propose_next_points(
            X_arr, y_arr, bounds, n_points, mode, **kwargs
        )
        
        # Convert back to dicts
        experiments = []
        for row in proposed_X:
            params = {name: float(val) for name, val in zip(param_names, row)}
            params.update(fixed_params)
            experiments.append(params)
            
        return experiments
