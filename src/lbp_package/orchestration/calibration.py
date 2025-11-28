from typing import Dict, Tuple, Callable, Optional, Any, List, Union
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import minimize
import warnings

from ..utils.logger import LBPLogger

# Suppress sklearn warnings about convergence if they occur
warnings.filterwarnings("ignore", category=UserWarning)

from ..interfaces.calibration import ICalibrationModel

class BayesianCalibration(ICalibrationModel):
    """
    Implements Bayesian Optimization for calibration using Gaussian Processes.
    
    This class handles the 'Active Learning' strategy where a surrogate model (GP)
    is used to model the objective function and uncertainty. It balances exploration
    (searching high-uncertainty regions) and exploitation (searching high-performance regions)
    using the Upper Confidence Bound (UCB) acquisition function.
    """
    
    def __init__(self, 
                 logger: LBPLogger,
                 n_iterations: int = 20, 
                 n_initial_points: int = 5,
                 exploration_weight: float = 0.5,
                 random_seed: Optional[int] = None):
        """
        Args:
            logger: Logger instance.
            n_iterations: Number of optimization steps after initialization.
            n_initial_points: Number of random points to evaluate before starting GP optimization.
            exploration_weight: Exploration preference in range [0.0, 1.0].
                                0.0 = Pure exploitation (standard optimization)
                                0.5 = Balanced exploration (approx 95% CI)
                                1.0 = Aggressive exploration
            random_seed: Seed for reproducibility.
        """
        super().__init__(logger)
        self.n_iterations = n_iterations
        self.n_initial_points = n_initial_points
        
        # Scale [0, 1] to Kappa [0, 5.0]
        # 0.0 -> 0.0
        # 0.5 -> 2.5 (slightly more than 1.96, but safe)
        # 1.0 -> 5.0
        self.exploration_weight = exploration_weight * 5.0
        
        self.rng = np.random.RandomState(random_seed)
        
        # Initialize GP with Matern kernel (good for smooth functions) + WhiteKernel (for noise/jitter)
        # nu=2.5 allows for moderate flexibility in the function shape
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
        self.gp = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=5, 
            random_state=random_seed,
            normalize_y=True # Important since objective values can have arbitrary scales
        )
        
        self.X_history: List[np.ndarray] = []
        self.y_history: List[float] = []

    def _acquisition_function(self, X: np.ndarray) -> Union[float, np.ndarray]:
        """
        Upper Confidence Bound (UCB) acquisition function.
        We want to MAXIMIZE this function.
        UCB(x) = mu(x) + kappa * sigma(x)
        """
        # GP predict returns mean and std
        # We cast to tuple to satisfy type checker if needed, or just unpack
        # return_std=True ensures we get (mean, std)
        mu, sigma = self.gp.predict(X, return_std=True) # type: ignore
        
        # We want to maximize the objective, so we look for high mean + high uncertainty
        return mu + self.exploration_weight * sigma

    def _propose_next_point(self, bounds: np.ndarray) -> np.ndarray:
        """
        Optimizes the acquisition function to find the next point to query.
        """
        n_params = bounds.shape[0]
        
        # Random restarts to avoid local optima in the acquisition function
        n_restarts = 10
        best_x = np.zeros(n_params) # Default fallback
        best_acq_value = -np.inf
        
        for _ in range(n_restarts):
            x0 = self.rng.uniform(bounds[:, 0], bounds[:, 1], size=(1, n_params))
            
            # We use minimize with a negative sign because scipy minimizes
            res = minimize(
                lambda x: -self._acquisition_function(x.reshape(1, -1)), # type: ignore
                x0=x0.flatten(),
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if -res.fun > best_acq_value:
                best_acq_value = -res.fun
                best_x = res.x
                
        return best_x

    def optimize(self, 
                 param_ranges: Dict[str, Tuple[float, float]], 
                 objective_fn: Callable[[Dict[str, float]], float],
                 fixed_params: Optional[Dict[str, Any]] = None,
                 uncertainty_fn: Optional[Callable[[Dict[str, float]], float]] = None) -> Dict[str, float]:
        
        fixed_params = fixed_params or {}
        
        # Sort keys to ensure consistent ordering for array conversion
        param_names = sorted(list(param_ranges.keys()))
        bounds = np.array([param_ranges[name] for name in param_names])
        
        self.logger.info(f"Starting Bayesian Optimization with {len(param_names)} free parameters.")
        self.logger.info(f"Fixed parameters: {list(fixed_params.keys())}")

        # === DIRECT OPTIMIZATION PATH (User provides uncertainty) ===
        if uncertainty_fn is not None:
            self.logger.info("Using direct optimization with user-provided uncertainty (skipping GP surrogate).")
            
            def direct_acquisition(x: np.ndarray) -> float:
                # Convert array back to dict
                params = {name: val for name, val in zip(param_names, x)}
                params.update(fixed_params)
                
                mu = objective_fn(params)
                sigma = uncertainty_fn(params)
                return mu + self.exploration_weight * sigma

            # Optimize direct acquisition
            n_restarts = 10
            best_x = np.zeros(len(param_names))
            best_acq_value = -np.inf
            
            for _ in range(n_restarts):
                x0 = self.rng.uniform(bounds[:, 0], bounds[:, 1])
                res = minimize(
                    lambda x: -direct_acquisition(x),
                    x0=x0,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                if -res.fun > best_acq_value:
                    best_acq_value = -res.fun
                    best_x = res.x
            
            best_params = {name: val for name, val in zip(param_names, best_x)}
            best_params.update(fixed_params)
            self.logger.info(f"Direct optimization finished. Best acquisition value: {best_acq_value:.4f}")
            return best_params
        
        # 1. Initial Random Exploration
        for i in range(self.n_initial_points):
            # Sample random point within bounds
            x_next = self.rng.uniform(bounds[:, 0], bounds[:, 1])
            
            # Construct full parameter dict
            params = {name: val for name, val in zip(param_names, x_next)}
            params.update(fixed_params)
            
            # Evaluate
            score = objective_fn(params)
            
            self.X_history.append(x_next)
            self.y_history.append(score)
            self.logger.debug(f"Init iter {i+1}/{self.n_initial_points}: score={score:.4f}")

        # 2. Active Learning Loop
        for i in range(self.n_iterations):
            # Update GP
            X_train = np.array(self.X_history)
            y_train = np.array(self.y_history)
            
            # Fit GP to observed data
            self.gp.fit(X_train, y_train)
            
            # Propose next point by maximizing acquisition function
            x_next = self._propose_next_point(bounds)
            
            # Construct full parameter dict
            params = {name: val for name, val in zip(param_names, x_next)}
            params.update(fixed_params)
            
            # Evaluate
            score = objective_fn(params)
            
            self.X_history.append(x_next)
            self.y_history.append(score)
            
            self.logger.info(f"Optimization iter {i+1}/{self.n_iterations}: score={score:.4f}, params={params}")

        # 3. Return best result found
        if not self.y_history:
             # Should not happen given n_initial_points > 0, but safe fallback
             return {**{k: v[0] for k, v in param_ranges.items()}, **fixed_params}

        best_idx = np.argmax(self.y_history)
        best_x = self.X_history[best_idx]
        best_score = self.y_history[best_idx]
        
        best_params = {name: val for name, val in zip(param_names, best_x)}
        best_params.update(fixed_params)
        
        self.logger.info(f"Optimization finished. Best score: {best_score:.4f}")
        return best_params
