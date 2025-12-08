from abc import ABC, abstractmethod
from typing import Literal, Tuple
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from ..utils import LBPLogger

CalibrationModes = Literal['exploration', 'exploitation']

class ISurrogateModel(ABC):
    """
    Interface for surrogate models used in calibration.
    
    Provides performance predictions and uncertainty estimates.
    """
    
    def __init__(self, logger: LBPLogger, random_seed: int = 42):
        self.logger = logger
        self.random_seed = random_seed

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the surrogate model to history data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict feature and uncertainty of the predictions.
        Make sure the uncertainty is normalized to be comparable across different features.
        
        Args:
            X: Input array (n_samples, n_features)
            
        Returns:
            Tuple of (prediction, uncertainty) arrays
        """
        pass


class GaussianProcessSurrogate(ISurrogateModel):
    """
    Default Gaussian Process implementation of ISurrogateModel.
    Uses Matern kernel + WhiteKernel for noise handling.
    """
    
    def __init__(self, logger: LBPLogger, random_seed: int = 42, **kwargs):
        super().__init__(logger, random_seed)
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
        self.gp = GaussianProcessRegressor(
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
        self.gp.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and std."""
        mean, std = self.gp.predict(X, return_std=True) # type: ignore
        return mean, std



