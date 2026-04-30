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

    def predict(
        self,
        params_list: list[dict[str, Any]],
        dm: Any,
        dim_info_list: list[dict[str, Any]],
        predictions_so_far: dict[str, dict[int, torch.Tensor]],
    ) -> list[dict[str, torch.Tensor]]:
        """Per-candidate sequence-batched prediction → ``list[dict[feat, (*feat_shape) tensor]]``.

        Builds ``(S, L, n_input)`` via ``dm.build_sequence_batch`` along
        ``self.sequence_axis_code``, runs the encoder with causal attention,
        denormalises, and reshapes per-(s, feat) to ``dim_info['shape']``.

        ``predictions_so_far`` is unused — causal attention sees prior
        positions' hidden states natively, so cross-model recursive lookups
        are handled internally by the encoder, not via column substitution.
        """
        del predictions_so_far
        S = len(params_list)
        if S == 0:
            return []
        if len(dim_info_list) != S:
            raise ValueError(
                f"predict: dim_info_list length {len(dim_info_list)} "
                f"does not match params_list length {S}.",
            )

        X_seq = dm.build_sequence_batch(self, params_list, dim_info_list)
        if X_seq.shape[1] == 0:
            return [{feat: torch.zeros(()) for feat in self.outputs} for _ in range(S)]

        input_indices = dm.get_input_indices(self.input_parameters + self.input_features)
        input_indices_t = torch.as_tensor(input_indices, dtype=torch.long)
        X_model = X_seq.index_select(2, input_indices_t)  # (S, L, n_model_input)

        if not self._is_trained or self._model is None:
            L = X_seq.shape[1]
            y_norm_seq = torch.zeros((S, L, len(self.outputs)), dtype=torch.float32)
        else:
            y_norm_seq = self._model(X_model.to(dtype=torch.float32))

        # Denormalise: (S, L, n_out) → flatten → denorm → reshape back. The
        # affine norm preserves gradients across the round-trip.
        L = int(y_norm_seq.shape[1])
        n_out = int(y_norm_seq.shape[2])
        y_norm_flat = y_norm_seq.reshape(S * L, n_out)
        y_denorm_flat = dm.denormalize_values(y_norm_flat, self.outputs)
        y_denorm_seq = y_denorm_flat.reshape(S, L, n_out)

        out: list[dict[str, torch.Tensor]] = [{} for _ in range(S)]
        for s in range(S):
            feat_shape = dim_info_list[s]['shape']
            for f_idx, feat in enumerate(self.outputs):
                t = y_denorm_seq[s, :, f_idx]
                out[s][feat] = t.reshape(feat_shape) if feat_shape else t.reshape(())
        return out

    def forward_pass(self, X: torch.Tensor, gradient_pass: bool = False) -> torch.Tensor:
        """Per-row forward — calls the encoder with L=1.

        Provided for the non-sequence call sites (KDE encode probe, etc.).
        Real sequence prediction goes through ``predict``.
        """
        if not self._is_trained or self._model is None:
            return torch.zeros((X.shape[0], len(self.outputs)), dtype=torch.float32)
        ctx: Any = torch.no_grad() if not gradient_pass else _NullContext()
        with ctx:
            x_seq = X.unsqueeze(1).to(dtype=torch.float32)  # (B, 1, n_input)
            return self._model(x_seq)[:, 0, :]

    def encode(self, X: torch.Tensor, gradient_pass: bool = False) -> torch.Tensor:
        """Penultimate-layer activations for KDE — input projection at position 0.

        The transformer doesn't expose a meaningful per-row latent because
        its hidden states are sequence-positional. We use the input
        projection's output as a static-only embedding for KDE evidence.
        """
        if not self._is_trained or self._model is None:
            return X
        ctx: Any = torch.no_grad() if not gradient_pass else _NullContext()
        with ctx:
            return self._model.input_proj(X.to(dtype=torch.float32))  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_batches: list[tuple[torch.Tensor, ...]],
        val_batches: list[tuple[torch.Tensor, ...]],
        epoch_callback: Callable[[float], list[tuple[torch.Tensor, torch.Tensor]] | None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Train the transformer on sequence-shaped batches: ``(B, L, n_input)`` X / ``(B, L, n_out)`` y.

        Adam + MSE loop. Causal attention enforces the autoregressive
        contract; no scheduled sampling is required because this encoder
        does not consume y as input during training — the train and
        inference forward passes are identical, so there's no exposure-bias
        train/inference mismatch to correct.

        ``epoch_callback`` is accepted for IPredictionModel-train signature
        compatibility but not invoked — sequence-form SS would require y to
        feed back as input, which this encoder design avoids by construction.
        """
        del epoch_callback
        if not train_batches:
            return

        X_full = torch.cat([b[0] for b in train_batches], dim=0)
        y_full = torch.cat([b[1] for b in train_batches], dim=0)
        if X_full.ndim != 3 or y_full.ndim != 3:
            raise ValueError(
                f"TorchTransformerModel.train expects sequence-shaped batches "
                f"(B, L, n_input)/(B, L, n_output); got X.ndim={X_full.ndim}, "
                f"y.ndim={y_full.ndim}. PredictionSystem must build sequence "
                f"batches (not flat batches) for transformer models.",
            )
        n_input = int(X_full.shape[-1])
        n_output = int(y_full.shape[-1])

        torch.manual_seed(self.SEED)
        self._model = self._build_network(n_input, n_output)
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY,
        )
        loss_fn = nn.MSELoss()
        self._model.train()

        X_full_f = X_full.to(dtype=torch.float32)
        y_full_f = y_full.to(dtype=torch.float32)
        for _ in range(self.EPOCHS):
            optimizer.zero_grad()
            loss = loss_fn(self._model(X_full_f), y_full_f)
            loss.backward()
            optimizer.step()

        self._model.eval()
        self._is_trained = True

    # ------------------------------------------------------------------
    # Schema check
    # ------------------------------------------------------------------

    def _validate_schema_compatibility(self, schema: Any) -> None:
        """Require ``sequence_axis_code`` to resolve to a real domain axis on the model's outputs.

        Recursive input features are allowed on a transformer (the new design
        no longer uses explicit prev_x columns; causal attention sees prior
        positions natively). Override ``IPredictionModel`` default which would
        reject them.
        """
        try:
            seq_axis_code = self.sequence_axis_code
        except NotImplementedError as e:
            raise ValueError(
                f"{self.__class__.__name__} must implement the `sequence_axis_code` "
                f"property to declare which schema axis to sequence over.",
            ) from e

        for feat_code in self.outputs:
            feat_obj = getattr(schema.features, "data_objects", {}).get(feat_code)
            if feat_obj is None:
                continue
            domain_code = getattr(feat_obj, "domain_code", None)
            if domain_code is None or not schema.domains.has(domain_code):
                continue
            domain = schema.domains.get(domain_code)
            for ax in domain.axes:
                if ax.code == seq_axis_code:
                    return  # found

        raise ValueError(
            f"{self.__class__.__name__}.sequence_axis_code='{seq_axis_code}' does not "
            f"resolve to any domain axis on this model's outputs ({list(self.outputs)}). "
            f"Verify the schema has a Dimension with this code.",
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
