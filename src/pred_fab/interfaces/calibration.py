"""Surrogate model interfaces for GP-based uncertainty estimation in calibration."""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from ..utils.logger import PfabLogger


class ISurrogateModel(ABC):
    """Abstract interface for surrogate models providing uncertainty estimates for calibration."""

    def __init__(self, logger: PfabLogger, random_seed: int = 42):
        self.logger = logger
        self.random_seed = random_seed
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the surrogate to experiment-level (params → performance) data."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) arrays of shape (n_samples, n_performance)."""
        ...


class GaussianProcessSurrogate(ISurrogateModel):
    """GP surrogate using a Matérn-5/2 + WhiteKernel for noise. Multi-output via independent GPs."""

    def __init__(self, logger: PfabLogger, random_seed: int = 42):
        super().__init__(logger, random_seed)
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=random_seed,
            normalize_y=True,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit GP to (n_experiments, n_params) input and (n_experiments, n_perf) targets."""
        if len(X) == 0:
            return
        self._gp.fit(X, y)
        self.is_fitted = True
        self.logger.info(f"GP surrogate fitted on {len(X)} experiment(s).")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) each of shape (n_samples, n_perf)."""
        result = self._gp.predict(X, return_std=True)  # type: ignore[call-overload]
        mean_raw: np.ndarray = np.asarray(result[0])  # type: ignore[index]
        std_raw: np.ndarray = np.asarray(result[1])   # type: ignore[index]
        # Ensure (n_samples, n_perf) shape regardless of sklearn output format
        mean = mean_raw.reshape(len(X), -1) if mean_raw.ndim == 1 else mean_raw
        std = std_raw.reshape(len(X), -1) if std_raw.ndim == 1 else std_raw
        return mean, std
