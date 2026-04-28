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

**Scale-aware training (Strategy D commit 6):** at ``n_rows > MINIBATCH_THRESHOLD``
the train loop uses ``DataLoader(TensorDataset, batch_size=...)`` for shuffled
minibatches. Below that, single-batch full-GD stays — DataLoader overhead
isn't justified at mock-scale (~50-200 rows). Threshold is class-level so
subclasses can override.
"""

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..interfaces import IPredictionModel
from ..utils import PfabLogger


class TorchMLPModel(IPredictionModel):
    """Feed-forward MLP base. Subclasses set ``HIDDEN`` and the IPredictionModel properties."""

    HIDDEN: tuple[int, ...] = (32, 16)

    EPOCHS: int = 1500
    LR: float = 5e-3
    WEIGHT_DECAY: float = 1e-3
    SEED: int = 0

    # Scale-aware minibatching threshold (Strategy D commit 6). At or below
    # this many training rows, the train loop uses single full-batch GD —
    # DataLoader/shuffle overhead doesn't pay at mock scale. Above, a
    # ``DataLoader`` runs ``MINIBATCH_SIZE`` shuffled minibatches per epoch.
    # ``EPOCHS`` is reinterpreted as "passes over the dataset" in either case.
    MINIBATCH_THRESHOLD: int = 1000
    MINIBATCH_SIZE: int = 256

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
        n_rows = int(X.shape[0])

        torch.manual_seed(self.SEED)
        net = self._build_network(X.shape[1], n_outputs)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)
        loss_fn = nn.MSELoss()
        net.train()

        # Scale-aware loop: above the threshold, shuffle into minibatches per
        # epoch via DataLoader; otherwise full-batch GD (mock-scale path).
        if n_rows > self.MINIBATCH_THRESHOLD:
            batch_size = min(self.MINIBATCH_SIZE, max(n_rows // 4, 1))
            loader = DataLoader(
                TensorDataset(X, y), batch_size=batch_size, shuffle=True,
            )
            for _ in range(self.EPOCHS):
                for X_b, y_b in loader:
                    optimizer.zero_grad()
                    loss = loss_fn(net(X_b), y_b)
                    loss.backward()
                    optimizer.step()
        else:
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

    def forward_pass(self, X: torch.Tensor, gradient_pass: bool = False) -> torch.Tensor:
        """Inference forward.

        ``gradient_pass=False`` (default): wraps in ``torch.no_grad()`` —
        autograd tape not built. Used by the existing batched DE path which
        is gradient-free.

        ``gradient_pass=True``: skips the no_grad context so gradients flow
        through the network to its inputs. Used by Strategy D's gradient-based
        acquisition where the optimiser drives params from a leaf tensor and
        needs ``∂predictions/∂params``. Note: ``self._compiled_forward`` is
        bypassed in gradient mode — torch.compile's traced graph doesn't
        compose cleanly with autograd today; eager forward is correct +
        efficient enough at our model sizes.
        """
        n_outputs = len(self.outputs)
        if self._model is None or not self._is_trained:
            return torch.zeros((X.shape[0], n_outputs), dtype=X.dtype)
        if gradient_pass:
            return self._model(X).reshape(-1, n_outputs)
        net = self._compiled_forward if self._compiled_forward is not None else self._model
        with torch.no_grad():
            return net(X).reshape(-1, n_outputs)

    def encode(self, X: torch.Tensor, gradient_pass: bool = False) -> torch.Tensor:
        if self._model is None or not self._is_trained:
            return X
        layers = list(self._model.children())
        if gradient_pass:
            h = X
            for layer in layers[:-1]:
                h = layer(h)
            return h
        with torch.no_grad():
            h = X
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
