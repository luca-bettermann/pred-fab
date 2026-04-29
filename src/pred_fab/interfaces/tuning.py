"""Residual model interface for online adaptation — corrects base model predictions during fabrication."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import torch
import torch.nn as nn

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
    """Torch-native MLP residual model with warm-start for online updates.

    Strategy D: replaced sklearn ``MLPRegressor`` with a tiny ``nn.Sequential``
    + Adam loop. Fit-and-call semantics match the sklearn variant; ``fit``
    keeps the network state across calls (warm-start equivalent), so online
    updates retain prior weights.
    """

    def __init__(
        self,
        logger: PfabLogger,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        activation: str = 'relu',
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        random_state: int | None = None,
    ):
        super().__init__(logger)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_name = activation
        self.lr = float(learning_rate_init)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self._model: nn.Module | None = None
        self._is_fitted = False

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        return {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "logistic": nn.Sigmoid(),
            "identity": nn.Identity(),
        }.get(name, nn.ReLU())

    def _build_network(self, n_inputs: int, n_outputs: int) -> nn.Module:
        layers: list[nn.Module] = []
        prev = n_inputs
        for h in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(self._build_activation(self.activation_name))
            prev = h
        layers.append(nn.Linear(prev, n_outputs))
        return nn.Sequential(*layers)

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
        """Fit MLP on (X, residuals); keeps prior weights for warm-start."""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        y_arr = np.asarray(residuals, dtype=np.float32)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        y_t = torch.from_numpy(y_arr)

        n_inputs = int(X_t.shape[1])
        n_outputs = int(y_t.shape[1])

        if self._model is None:
            self._model = self._build_network(n_inputs, n_outputs)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self._model.train()
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            loss = loss_fn(self._model(X_t), y_t)
            loss.backward()
            optimizer.step()
        self._model.eval()
        self._is_fitted = True
        self.logger.info(f"Residual model fitted on {len(X)} samples.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict residuals; returns zeros of shape (n, 1) if not yet fitted."""
        if not self._is_fitted or self._model is None:
            return np.zeros((X.shape[0], 1))
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            y_t = self._model(X_t)
        return y_t.detach().cpu().numpy()
