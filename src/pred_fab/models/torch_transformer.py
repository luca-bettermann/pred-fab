"""Convenience base for sequence-aware (transformer) prediction models.

DRAFT — not yet wired into PredictionSystem dispatch. The contract is
captured here; the migration that drops the cell-loop autoreg path and
routes recursive problems through `predict_sequence` is the next step.

Usage (post-migration):
    class MyRecursiveModel(TorchTransformerModel):
        D_MODEL = 64
        N_HEADS = 4
        N_LAYERS = 2

        @property
        def input_parameters(self) -> list[str]: return ["param_1", ...]
        @property
        def input_features(self) -> list[str]: return []   # no explicit prev_x cols
        @property
        def outputs(self) -> list[str]: return ["feat_a"]
        @property
        def sequence_axis_code(self) -> str: return "n_layers"

The transformer sees the full sequence (all positions along ``sequence_axis_code``)
in one forward; causal attention enforces the autoregressive prediction
contract — position k can attend to positions 0..k-1 but not k+1..L-1.

Recursion is **structural**, not column-by-column. The schema's
`Feature.recursive(...)` declarations describe which axis is the
sequence axis (consumed by the framework when building per-candidate
sequences); the model itself does not see explicit `prev_x` columns.
"""

from typing import Any, Callable

import math

import torch
import torch.nn as nn

from ..interfaces import IPredictionModel
from ..utils import PfabLogger


class TorchTransformerModel(IPredictionModel):
    """Sequence-aware encoder-only transformer with causal attention.

    Subclasses set ``D_MODEL``, ``N_HEADS``, ``N_LAYERS``, plus the standard
    ``input_parameters`` / ``input_features`` / ``outputs`` properties and a
    ``sequence_axis_code`` property naming the schema axis to sequence over.
    """

    HAS_NATIVE_SEQUENCE = True

    D_MODEL: int = 64
    N_HEADS: int = 4
    N_LAYERS: int = 2
    DROPOUT: float = 0.1
    DIM_FEEDFORWARD: int = 128
    MAX_SEQ_LEN: int = 256       # caps the learned position embedding

    EPOCHS: int = 500
    LR: float = 5e-4
    WEIGHT_DECAY: float = 1e-3
    SEED: int = 0

    SS_N_REFRESHES: int = 4

    @property
    def sequence_axis_code(self) -> str:
        """Schema axis name to sequence over (e.g. ``"n_layers"``).

        Subclasses must override. The framework uses this to build per-candidate
        ``(L, n_input)`` sequences along the named axis when calling
        ``predict_sequence``.
        """
        raise NotImplementedError(
            "Subclasses must declare the sequence axis via the "
            "sequence_axis_code property.",
        )

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)
        self._model: nn.Module | None = None
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    def _build_network(self, n_input: int, n_output: int) -> nn.Module:
        """Encoder-only transformer: input proj → positional encoding → encoder → output proj.

        Override in subclasses to swap encoder type, add dropout schedules,
        or replace the projections with custom heads.
        """
        torch.manual_seed(self.SEED)
        return _TransformerNet(
            n_input=n_input,
            n_output=n_output,
            d_model=self.D_MODEL,
            n_heads=self.N_HEADS,
            n_layers=self.N_LAYERS,
            dim_ff=self.DIM_FEEDFORWARD,
            dropout=self.DROPOUT,
            max_seq_len=self.MAX_SEQ_LEN,
        )

    # ------------------------------------------------------------------
    # IPredictionModel contract — sequence-aware
    # ------------------------------------------------------------------

    def predict_sequence(
        self,
        x_norm: torch.Tensor,
        sequence_length: int,
        gradient_pass: bool = False,
    ) -> torch.Tensor:
        """Per-candidate sequence forward → ``(S, L, n_outputs)``.

        ``x_norm`` is the per-candidate static input ``(S, n_input)``;
        the framework tiles it across L positions and adds a position
        embedding before the encoder.

        Causal attention enforces the autoregressive contract — output at
        position k depends only on positions 0..k.
        """
        if not self._is_trained or self._model is None:
            S = int(x_norm.shape[0])
            return torch.zeros((S, sequence_length, len(self.outputs)), dtype=torch.float32)

        ctx: Any = torch.no_grad() if not gradient_pass else _NullContext()
        with ctx:
            # Tile static input across L positions: (S, n_input) → (S, L, n_input).
            S, n_input = x_norm.shape
            x_seq = x_norm.unsqueeze(1).expand(S, sequence_length, n_input).to(dtype=torch.float32)
            return self._model(x_seq)

    def forward_pass(self, X: torch.Tensor, gradient_pass: bool = False) -> torch.Tensor:
        """Per-cell forward — calls predict_sequence with L=1.

        Provided for compatibility with non-sequence call sites (e.g. the
        evidence encoder). Real sequence prediction goes through
        ``predict_sequence`` directly.
        """
        out = self.predict_sequence(X, sequence_length=1, gradient_pass=gradient_pass)
        return out[:, 0, :]

    def encode(self, X: torch.Tensor, gradient_pass: bool = False) -> torch.Tensor:
        """Penultimate-layer activations for KDE.

        For sequence-aware models, "encode" returns the input projection's
        output at the first sequence position — a static-only embedding
        suitable for KDE which doesn't see the sequence axis.
        """
        if not self._is_trained or self._model is None:
            return X
        ctx: Any = torch.no_grad() if not gradient_pass else _NullContext()
        with ctx:
            return self._model.input_proj(X.to(dtype=torch.float32))  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Training (placeholder — fleshed out in the migration)
    # ------------------------------------------------------------------

    def train(
        self,
        train_batches: list[tuple[torch.Tensor, ...]],
        val_batches: list[tuple[torch.Tensor, ...]],
        epoch_callback: Callable[[float], list[tuple[torch.Tensor, torch.Tensor]] | None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Train the transformer on sequence-shaped batches.

        Currently a stub — the migration that wires sequence-batch construction
        through DataModule and PredictionSystem will fill this in. The
        epoch_callback path matches TorchMLPModel: invoked at SS_N_REFRESHES
        equally-spaced points; returns refreshed batches with per-step Bernoulli
        substitution applied (sequence-aware scheduled sampling).
        """
        raise NotImplementedError(
            "TorchTransformerModel.train is a stub — wire in sequence-batch "
            "training when migrating PredictionSystem dispatch.",
        )


# ---------------------------------------------------------------------------
# Internal nn.Module
# ---------------------------------------------------------------------------


class _TransformerNet(nn.Module):
    """Encoder-only transformer with causal mask + learned position embedding."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_ff: int,
        dropout: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_input, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, n_output)
        self._max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(S, L, n_input)`` → ``(S, L, n_output)`` with causal attention."""
        S, L, _ = x.shape
        if L > self._max_seq_len:
            raise ValueError(
                f"sequence length {L} exceeds MAX_SEQ_LEN={self._max_seq_len}; "
                f"increase the class attribute on the model.",
            )
        h = self.input_proj(x)
        positions = torch.arange(L, device=x.device)
        h = h + self.pos_embed(positions)[None, :, :]
        causal_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=x.device), diagonal=1,
        )
        h = self.encoder(h, mask=causal_mask, is_causal=True)
        return self.output_proj(h)


class _NullContext:
    """Context manager that does nothing — used when gradient_pass=True."""
    def __enter__(self): return self
    def __exit__(self, *args): return False
