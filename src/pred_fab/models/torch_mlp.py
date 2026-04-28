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

    # If True (default), the trained net is wrapped with torch.compile after
    # training so ``forward_pass`` calls go through a JIT-traced graph. Falls
    # back to eager silently if compilation raises (e.g. no C++ compiler in
    # the runtime environment). Only the inference path is compiled — training
    # runs eager. ``dynamic=True`` so variable batch dims (S=1 single-candidate,
    # S=popsize vectorised DE) share one compiled graph instead of triggering
    # recompilation per shape. Subclasses can set ``COMPILE = False`` to opt
    # out (e.g. for environments where the first-call overhead exceeds the
    # inference savings, like short-lived test runs).
    COMPILE: bool = True

    # Class-level cache of whether torch.compile can succeed in this runtime.
    # ``None`` = unchecked, ``True`` = available, ``False`` = unavailable
    # (compiler missing, inductor unsupported, etc.). Set on first model
    # train via a one-shot probe; subsequent models skip the probe and
    # honour the cached result.
    _compile_available: bool | None = None

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)
        self._model: nn.Module | None = None
        self._compiled_forward: Any = None
        self._is_trained: bool = False

    @classmethod
    def _probe_compile_available(cls, logger: PfabLogger | None = None) -> bool:
        """One-shot env probe: can torch.compile produce a working graph here?

        Compiles a tiny stub net and runs one forward; caches the result on
        the class so the cost is paid once per process. Real models then
        compile/probe individually only if the env probe succeeded.
        """
        if cls._compile_available is not None:
            return cls._compile_available
        try:
            stub = nn.Linear(2, 2)
            stub_compiled = torch.compile(stub, dynamic=True)
            with torch.no_grad():
                _ = stub_compiled(torch.zeros(1, 2))
            cls._compile_available = True
        except Exception as e:
            cls._compile_available = False
            if logger is not None:
                logger.warning(
                    f"torch.compile unavailable in this runtime; falling back "
                    f"to eager forward for all TorchMLPModel subclasses. "
                    f"Reason: {e!r}"
                )
        return cls._compile_available

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
        if self.COMPILE and self._probe_compile_available(self.logger):
            try:
                compiled = torch.compile(net, dynamic=True)
                # Eager probe: torch.compile is lazy; force compilation now
                # so any failure surfaces here (with fallback) rather than
                # mid-acquisition where it would crash inference.
                with torch.no_grad():
                    _ = compiled(torch.zeros(1, X.shape[1], dtype=X.dtype))
                self._compiled_forward = compiled
            except Exception as e:
                self.logger.warning(
                    f"torch.compile failed for {self.__class__.__name__}, "
                    f"falling back to eager forward: {e!r}"
                )
                self._compiled_forward = None
        self._is_trained = True

    def forward_pass(self, X: torch.Tensor) -> torch.Tensor:
        n_outputs = len(self.outputs)
        if self._model is None or not self._is_trained:
            return torch.zeros((X.shape[0], n_outputs), dtype=X.dtype)
        net = self._compiled_forward if self._compiled_forward is not None else self._model
        with torch.no_grad():
            return net(X).reshape(-1, n_outputs)

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
