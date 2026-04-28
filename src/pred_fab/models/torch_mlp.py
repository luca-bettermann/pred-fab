"""Convenience base for feed-forward MLP prediction models on PyTorch.

Subclasses provide:
- ``HIDDEN``: tuple of hidden-layer widths (class attribute, override per model).
- ``input_parameters``, ``input_features``, ``outputs``: the standard
  IPredictionModel properties.

The base provides a Linear/ReLU stack with an Adam + MSE training loop, plus
``forward_pass`` and ``encode`` (penultimate-layer activations for KDE).
Inputs are assumed to arrive normalised from ``DataModule`` — no internal scaler.

The framework's contract is tensor-native: ``forward_pass`` and ``encode`` take
and return ``torch.Tensor`` directly; no numpy↔tensor conversion happens here.

Other model shapes (transformer, GNN, LSTM) should subclass ``TorchMLPModel``
and override ``_build_network`` to return a different ``nn.Module``.
"""

from typing import Any

import torch
import torch.nn as nn

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
        self._model: nn.Module | None = None
        self._is_trained: bool = False

    def train(
        self,
        train_batches: list[tuple[torch.Tensor, torch.Tensor]],
        val_batches: list[tuple[torch.Tensor, torch.Tensor]],
        **kwargs: Any,
    ) -> None:
        if not train_batches:
            return
        n_outputs = len(self.outputs)

        X = torch.cat([b[0] for b in train_batches], dim=0)
        y = torch.cat([b[1] for b in train_batches], dim=0)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        torch.manual_seed(self.SEED)
        net = self._build_network(X.shape[1], n_outputs)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)
        loss_fn = nn.MSELoss()
        net.train()
        for _ in range(self.EPOCHS):
            optimizer.zero_grad()
            loss = loss_fn(net(X), y)
            loss.backward()
            optimizer.step()
        net.eval()

        self._model = net
        self._is_trained = True

    def forward_pass(self, X: torch.Tensor) -> torch.Tensor:
        n_outputs = len(self.outputs)
        if self._model is None or not self._is_trained:
            return torch.zeros((X.shape[0], n_outputs), dtype=X.dtype)
        with torch.no_grad():
            return self._model(X).reshape(-1, n_outputs)

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        if self._model is None or not self._is_trained:
            return X
        with torch.no_grad():
            h = X
            layers = list(self._model.children())
            for layer in layers[:-1]:
                h = layer(h)
            return h

    def _build_network(self, n_inputs: int, n_outputs: int) -> nn.Module:
        """Construct nn.Sequential([Linear, ReLU]*len(HIDDEN) + [Linear(_, n_outputs)])."""
        layers: list[nn.Module] = []
        prev = n_inputs
        for h in self.HIDDEN:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_outputs))
        return nn.Sequential(*layers)
