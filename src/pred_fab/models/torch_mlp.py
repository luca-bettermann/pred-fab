"""Convenience base for feed-forward MLP prediction models on PyTorch.

Subclasses provide:
- ``HIDDEN``: tuple of hidden-layer widths (class attribute, override per model).
- ``input_parameters``, ``input_features``, ``outputs``: the standard
  IPredictionModel properties.

The base provides a Linear/ReLU stack with an Adam + MSE training loop, plus
``forward_pass`` and ``encode`` (penultimate-layer activations for KDE).
Inputs are assumed to arrive normalised from ``DataModule`` — no internal scaler.

Torch is imported lazily inside the methods that need it; ``import pred_fab``
stays torch-free, and importing this module when torch is missing yields a
clear actionable error only when training/inference is attempted.

Other model shapes (Random Forest, GP, transformer, …) should subclass
``IPredictionModel`` directly. ``TorchMLPModel`` is a convenience for the
common feed-forward case, not a replacement for the contract.
"""

from typing import Any

import numpy as np

from ..interfaces import IPredictionModel
from ..utils import PfabLogger


class TorchMLPModel(IPredictionModel):
    """Feed-forward MLP base. Subclasses set ``HIDDEN`` and the IPredictionModel properties."""

    HIDDEN: tuple[int, ...] = (32, 16)

    EPOCHS: int = 1500
    LR: float = 5e-3
    WEIGHT_DECAY: float = 1e-3
    SEED: int = 0

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)
        self._model: Any = None
        self._is_trained: bool = False

    def train(
        self,
        train_batches: list[tuple[np.ndarray, np.ndarray]],
        val_batches: list[tuple[np.ndarray, np.ndarray]],
        **kwargs: Any,
    ) -> None:
        if not train_batches:
            return
        torch, nn = _require_torch()
        n_outputs = len(self.outputs)

        X = np.vstack([b[0] for b in train_batches])
        y = np.vstack([b[1] for b in train_batches])
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        torch.manual_seed(self.SEED)
        net = self._build_network(X.shape[1], n_outputs, nn)
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32))

        optimizer = torch.optim.Adam(net.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)
        loss_fn = nn.MSELoss()
        net.train()
        for _ in range(self.EPOCHS):
            optimizer.zero_grad()
            loss = loss_fn(net(X_t), y_t)
            loss.backward()
            optimizer.step()
        net.eval()

        self._model = net
        self._is_trained = True

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        n_outputs = len(self.outputs)
        if self._model is None or not self._is_trained:
            return np.zeros((X.shape[0], n_outputs))
        torch, _ = _require_torch()
        with torch.no_grad():
            X_t = torch.from_numpy(X.astype(np.float32))
            return self._model(X_t).numpy().reshape(-1, n_outputs)

    def encode(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or not self._is_trained:
            return X
        torch, _ = _require_torch()
        with torch.no_grad():
            X_t = torch.from_numpy(X.astype(np.float32))
            layers = list(self._model.children())
            for layer in layers[:-1]:
                X_t = layer(X_t)
            return X_t.numpy()

    def _build_network(self, n_inputs: int, n_outputs: int, nn: Any) -> Any:
        """Construct nn.Sequential([Linear, ReLU]*len(HIDDEN) + [Linear(_, n_outputs)])."""
        layers: list[Any] = []
        prev = n_inputs
        for h in self.HIDDEN:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_outputs))
        return nn.Sequential(*layers)


def _require_torch() -> tuple[Any, Any]:
    """Lazy import torch; raise a clear actionable error if not installed."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise ImportError(
            "TorchMLPModel requires torch. Install via: pip install 'pred-fab[torch]'"
        ) from e
    return torch, nn
