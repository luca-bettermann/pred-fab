"""Residual model interface for online adaptation — corrects base model predictions during fabrication."""

from abc import ABC, abstractmethod
from typing import Any, Literal
import numpy as np
from sklearn.neural_network import MLPRegressor

from .base_interface import BaseInterface
from ..utils.logger import PfabLogger


class IResidualModel(BaseInterface):
    """Abstract base for residual models that learn the error between base model predictions and ground truth."""

    @abstractmethod
    def fit(self, X: np.ndarray, residuals: np.ndarray, **kwargs) -> None:
        """Train on input features X and target residuals (y_true - y_pred_base)."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict residual corrections for input features X."""
        pass


class MLPResidualModel(IResidualModel):
    """MLP-based residual model using sklearn MLPRegressor with warm_start for online updates."""

    def __init__(
        self,
        logger: PfabLogger,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        activation: Literal['relu', 'identity', 'logistic', 'tanh'] = 'relu',
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        random_state: int | None = None
    ):
        super().__init__(logger)
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
            solver='adam', # Efficient for larger datasets, 'lbfgs' better for small
            warm_start=True # Allow online updates
        )
        self._is_fitted = False
        
    @property
    def input_parameters(self) -> list[str]:
        """Residual model is dynamic, no fixed schema inputs."""
        return []

    @property
    def input_features(self) -> list[str]:
        """Residual model is dynamic, no fixed schema inputs."""
        return []

    @property
    def outputs(self) -> list[str]:
        """Residual model is dynamic, no fixed schema outputs."""
        return []

    def fit(self, X: np.ndarray, residuals: np.ndarray, **kwargs) -> None:
        """Fit MLP on (X, residuals) batch; warm_start allows incremental updates."""
        self.model.fit(X, residuals)
        self._is_fitted = True
        self.logger.info(f"Residual model fitted on {len(X)} samples.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict residuals; returns zeros of shape (n, 1) if not yet fitted."""
        if not self._is_fitted:
            return np.zeros((X.shape[0], 1))
        return self.model.predict(X)  # type: ignore[return-value]
