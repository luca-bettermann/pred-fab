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

from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..interfaces import IPredictionModel
from ..utils import PfabLogger


def _embedding_dim(cardinality: int) -> int:
    """FastAI tabular heuristic for categorical embedding dimension.

    Balanced bottleneck for any C — modest compression at small C, capped
    at 50 for very large C so memory stays bounded.
    """
    return min(50, (cardinality + 1) // 2)


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

    # Strategy D commit 15: number of equally-spaced SS refresh checkpoints
    # during a training run. Effective only when train() is called with an
    # ``epoch_callback``; otherwise ignored. Default 4 matches the legacy
    # K-refit cadence of 4 rounds.
    SS_N_REFRESHES: int = 4

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
        # Strategy D commit 14: ``nn.Embedding`` per categorical input column,
        # keyed by model-relative col index. Set via
        # ``set_categorical_context`` before training. Each cat column
        # encodes into a learned latent of size ``_embedding_dim(C)`` (FastAI
        # heuristic), then concatenates with non-categorical columns ahead
        # of the Linear stack. Embeddings are part of ``state_dict()`` and
        # transfer via ``.to(device)`` automatically.
        self._cat_embeddings: nn.ModuleDict = nn.ModuleDict()
        self._cat_cardinalities: dict[int, int] = {}

    def set_categorical_context(self, col_to_cardinality: dict[int, int]) -> None:
        """Build embeddings for categorical input columns (Strategy D commit 14).

        Called by ``PredictionSystem`` before training; keys are
        model-relative column indices. One ``nn.Embedding(C, d)`` per cat
        column with ``d = _embedding_dim(C)`` (FastAI tabular heuristic).
        """
        self._cat_cardinalities = dict(col_to_cardinality)
        # ModuleDict keys must be strings.
        self._cat_embeddings = nn.ModuleDict({
            str(col_idx): nn.Embedding(C, _embedding_dim(C))
            for col_idx, C in col_to_cardinality.items()
        })

    def _embed_cats(self, X: torch.Tensor) -> torch.Tensor:
        """Embed categorical cat-index columns; pass others through.

        Returns a tensor whose categorical columns have been replaced by
        their learned ``d``-dim embedding, concatenated in column order.
        """
        if not self._cat_embeddings:
            return X
        n_cols = int(X.shape[-1])
        cols: list[torch.Tensor] = []
        for j in range(n_cols):
            key = str(j)
            if key in self._cat_embeddings:
                C = self._cat_cardinalities[j]
                idx = X[..., j].long().clamp(0, C - 1)
                # nn.Embedding output: (..., d). Cast to X's dtype for concat.
                cols.append(self._cat_embeddings[key](idx).to(dtype=X.dtype))
            else:
                cols.append(X[..., j:j + 1])
        return torch.cat(cols, dim=-1)

    def _expanded_input_size(self, n_raw: int) -> int:
        """Input size after categorical embedding expansion.

        Each cat column contributes ``embedding_dim`` (typically << C);
        non-cat columns contribute 1 each.
        """
        if not self._cat_cardinalities:
            return n_raw
        size = 0
        for j in range(n_raw):
            if j in self._cat_cardinalities:
                size += _embedding_dim(self._cat_cardinalities[j])
            else:
                size += 1
        return size

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
        *,
        epoch_callback: Callable[[float], list[tuple[torch.Tensor, torch.Tensor]] | None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Train the network. Optional ``epoch_callback`` enables per-epoch SS refresh.

        When ``epoch_callback`` is provided, it is invoked at
        ``SS_N_REFRESHES`` equally-spaced points during training. The
        callback receives the current progress in ``[0, 1]`` and returns
        fresh ``train_batches`` with updated SS substitution applied (or
        ``None`` to keep the current batches). Strategy D commit 15.
        """
        if not train_batches:
            return
        n_outputs = len(self.outputs)

        X = torch.cat([b[0] for b in train_batches], dim=0)
        y = torch.cat([b[1] for b in train_batches], dim=0)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_rows = int(X.shape[0])
        n_raw = int(X.shape[1])
        n_expanded = self._expanded_input_size(n_raw)

        torch.manual_seed(self.SEED)
        net = self._build_network(n_expanded, n_outputs)

        # Include embedding params alongside network params — both are learned
        # jointly. _cat_embeddings is empty when no categoricals, so chain
        # gracefully.
        params = list(net.parameters()) + list(self._cat_embeddings.parameters())
        optimizer = torch.optim.Adam(params, lr=self.LR, weight_decay=self.WEIGHT_DECAY)
        loss_fn = nn.MSELoss()
        net.train()
        self._cat_embeddings.train()

        # Strategy D commit 15: per-epoch SS refresh (replaces K-refit). Refresh
        # at SS_N_REFRESHES equally-spaced checkpoints during training; at each,
        # the callback returns fresh batches with current SS state.
        refresh_period = (
            max(self.EPOCHS // max(self.SS_N_REFRESHES, 1), 1)
            if epoch_callback else self.EPOCHS + 1
        )

        # Scale-aware loop: above the threshold, shuffle into minibatches per
        # epoch via DataLoader; otherwise full-batch GD (mock-scale path).
        # NOTE: must recompute _embed_cats per step — embeddings are learning,
        # so a cached pre-loop expansion would be stale after the first update.
        use_minibatch = n_rows > self.MINIBATCH_THRESHOLD

        if use_minibatch:
            batch_size = min(self.MINIBATCH_SIZE, max(n_rows // 4, 1))
            loader = DataLoader(
                TensorDataset(X, y), batch_size=batch_size, shuffle=True,
            )

        for epoch in range(self.EPOCHS):
            if epoch_callback is not None and epoch > 0 and epoch % refresh_period == 0:
                new_batches = epoch_callback(epoch / max(self.EPOCHS - 1, 1))
                if new_batches:
                    X = torch.cat([b[0] for b in new_batches], dim=0)
                    y = torch.cat([b[1] for b in new_batches], dim=0)
                    if y.ndim == 1:
                        y = y.reshape(-1, 1)
                    if use_minibatch:
                        loader = DataLoader(
                            TensorDataset(X, y), batch_size=batch_size, shuffle=True,
                        )
            if use_minibatch:
                for X_b, y_b in loader:
                    optimizer.zero_grad()
                    loss = loss_fn(net(self._embed_cats(X_b)), y_b)
                    loss.backward()
                    optimizer.step()
            else:
                optimizer.zero_grad()
                loss = loss_fn(net(self._embed_cats(X)), y)
                loss.backward()
                optimizer.step()
        net.eval()
        self._cat_embeddings.eval()

        self._model = net
        if self.COMPILE and self._probe_compile_available(self.logger):
            try:
                compiled = torch.compile(net, dynamic=True)
                # Eager probe: torch.compile is lazy; force compilation now
                # so any failure surfaces here (with fallback) rather than
                # mid-acquisition where it would crash inference.
                with torch.no_grad():
                    _ = compiled(torch.zeros(1, n_expanded, dtype=X.dtype))
                self._compiled_forward = compiled
            except Exception as e:
                self.logger.warning(
                    f"torch.compile failed for {self.__class__.__name__}, "
                    f"falling back to eager forward: {e!r}"
                )
                self._compiled_forward = None
        self._is_trained = True

    def forward_pass(self, X: torch.Tensor, gradient_pass: bool = False) -> torch.Tensor:
        """Inference forward — applies categorical one-hot expansion first.

        ``gradient_pass=False`` (default): wraps in ``torch.no_grad()``.
        ``gradient_pass=True``: keeps the autograd tape live (Strategy D
        gradient acquisition). Categorical expansion via ``_embed_cats``
        is non-differentiable (discrete index → one-hot float), but it
        doesn't break gradient flow on the *non-categorical* columns.
        """
        n_outputs = len(self.outputs)
        if self._model is None or not self._is_trained:
            return torch.zeros((X.shape[0], n_outputs), dtype=X.dtype)
        X_expanded = self._embed_cats(X)
        if gradient_pass:
            return self._model(X_expanded).reshape(-1, n_outputs)
        net = self._compiled_forward if self._compiled_forward is not None else self._model
        with torch.no_grad():
            return net(X_expanded).reshape(-1, n_outputs)

    def encode(self, X: torch.Tensor, gradient_pass: bool = False) -> torch.Tensor:
        if self._model is None or not self._is_trained:
            return X
        X_expanded = self._embed_cats(X)
        layers = list(self._model.children())
        if gradient_pass:
            h = X_expanded
            for layer in layers[:-1]:
                h = layer(h)
            return h
        with torch.no_grad():
            h = X_expanded
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
