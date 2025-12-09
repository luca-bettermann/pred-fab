from abc import ABC, abstractmethod
from typing import Tuple, Type, Any, final, Callable
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from ..utils import LBPLogger

# Should we call this IActiveLearningModel?
class IExplorationModel(ABC):
    """
    Interface for exploration mode used in calibration. 
    Explores the solution space for active learning.
    
    Provides performance predictions and uncertainty estimates.
    """
    
    def __init__(self, logger: LBPLogger, random_seed: int = 42):
        self.logger = logger
        self.random_seed = random_seed

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the surrogate model to the data.
        
        Args:
            X: Input parameter array (n_samples, n_parameters)
            y: Output performance values (n_samples,)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict performance attributes and the uncertainty of the predictions.
        Make sure the uncertainty is normalized to be comparable across different features.

        Performance attributes -> exploitation
        Uncertainty -> exploration
        
        Args:
            X: Input array (n_samples, n_params)
            
        Returns:
            Tuple of (prediction, uncertainty) arrays with shape (n_samples, n_performance)
        """
        pass

    @final
    def exploration_func(self, X: np.ndarray, sys_perf: Callable, w_explore: float) -> float:

        # Predict with Surrogate
        pred_perf, uncertainty = self.predict(X.reshape(1, -1))

        # Compute system performance for the exploitation term
        perf_term = sys_perf(list(pred_perf.flatten()))

        # Exploration term from the weighted uncertainty
        uncertainty_term = sys_perf(list(uncertainty.flatten()))
        
        # Weighted combination
        acquisition = (1 - w_explore) * perf_term + w_explore * uncertainty_term
        
        # Return negative because minimize seeks minima
        return -acquisition


class GaussianProcessExploration(IExplorationModel):
    """
    Default Gaussian Process implementation of IExplorationModel.
    Uses Matern kernel + WhiteKernel for noise handling.
    """
    
    def __init__(self, logger: LBPLogger, random_seed: int = 42, **kwargs):
        super().__init__(logger, random_seed)
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
        self.surrogate_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=random_seed,
            normalize_y=True
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit GP to data."""
        if len(X) == 0:
            return
        self.surrogate_model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and std."""
        mean, std = self.surrogate_model.predict(X, return_std=True) # type: ignore
        return mean, std



