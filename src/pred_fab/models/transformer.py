"""Convenience base for sequence-aware (transformer) prediction models.

Usage:
    class MyRecursiveModel(TransformerModel):
        D_MODEL = 64
        N_HEADS = 4
        N_LAYERS = 2

        @property
        def sequence_axis_code(self) -> tuple[str, ...]:
            return ("n_layers",)               # single-axis (length-1 tuple)
            # return ("n_layers", "n_segments")  # multi-axis flattened sequence

        @property
        def input_parameters(self) -> list[str]: return ["param_1", ...]
        @property
        def input_features(self) -> list[str]: return []
        @property
        def outputs(self) -> list[str]: return ["feat_a"]

Single-axis vs multi-axis trade-off:

- **Single axis** ``("n_layers",)``: non-listed axes (segments) become parallel
  batches. Sequence length L = n_layers; attention cost O(L²) per parallel
  batch. Causal attention propagates only along the listed axis. Cross-axis
  dependencies are NOT seen by attention.
- **Multi-axis** ``("n_layers", "n_segments")``: listed axes flatten into one
  sequence in declared order (C-order). L_total = product of axis sizes;
  attention cost O(L_total²) per candidate. Cross-axis dependencies ARE seen
  — every cell can attend to any prior cell across all listed axes. Per-axis
  additive position embeddings.

Pick multi-axis when grid-wide attention is wanted and the total cell
count is manageable. Pick single-axis when one axis is the dominant
recursion direction and cross-axis cells are expected to be roughly
independent (cheaper, more data-efficient under that assumption).

Recursion is **structural**, not column-by-column. The user declares which
schema axes are the sequence axes via ``sequence_axis_code``; causal
attention sees prior positions' hidden states natively, so no explicit
prev-position columns are required (and none are supported).
"""

from typing import Any, Callable

import math

import numpy as np
import torch
import torch.nn as nn

from ..interfaces import IPredictionModel
from ..utils import PfabLogger


class TransformerModel(IPredictionModel):
    """Sequence-aware encoder-only transformer with causal attention.

    Subclasses set ``D_MODEL``, ``N_HEADS``, ``N_LAYERS``, plus the standard
    ``input_parameters`` / ``input_features`` / ``outputs`` properties and a
    ``sequence_axis_code`` property naming the schema axis to sequence over.
    """

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

    @property
    def sequence_axis_code(self) -> tuple[str, ...]:
        """Schema axes to sequence over, in flatten order (C-order).

        Length-1 tuple = single-axis sequence (non-listed axes become parallel
        batches). Length-N tuple = multi-axis flattened sequence; listed axes
        join into one sequence of length ``prod(axis_sizes)`` with per-axis
        additive position embeddings; remaining axes (if any) remain parallel.

        Subclasses must override.
        """
        raise NotImplementedError(
            "Subclasses must declare the sequence axes via the "
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
            n_axes=len(self.sequence_axis_code),
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

        Single-axis: ``dm.build_sequence_batch`` produces ``(S × n_other, L, n_input)``
        where ``n_other`` is the product of non-sequence-axis sizes. Multi-axis:
        all listed axes are flattened into ``L = product(seq_axis_sizes)``;
        non-listed axes (if any) remain parallel-batched as ``n_other``.

        Per-axis additive position embeddings encode each cell's coord along
        every listed sequence axis; causal attention propagates across the
        joint flattened sequence.

        ``predictions_so_far`` is unused — causal attention sees prior
        positions' hidden states natively.
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
        X_model = X_seq.index_select(2, input_indices_t)  # (S_eff, L, n_model_input)

        # Compute per-position axis indices (L, n_seq_axes) — same for all
        # batch rows since all candidates in a calibration call share shape.
        di_first = dim_info_list[0]
        dim_codes = di_first['dim_codes_ordered']
        shape = di_first['shape']
        seq_axis_codes = self.sequence_axis_code
        seq_axis_indices = [dim_codes.index(c) for c in seq_axis_codes if c in dim_codes]
        seq_axis_sizes = [shape[i] for i in seq_axis_indices]
        L = int(np.prod(seq_axis_sizes)) if seq_axis_sizes else 1
        if seq_axis_sizes:
            axis_indices = torch.empty((L, len(seq_axis_sizes)), dtype=torch.long)
            for k in range(L):
                coord = np.unravel_index(k, seq_axis_sizes)
                for i in range(len(seq_axis_sizes)):
                    axis_indices[k, i] = int(coord[i])
        else:
            axis_indices = torch.zeros((1, 1), dtype=torch.long)

        if not self._is_trained or self._model is None:
            S_eff = int(X_seq.shape[0])
            y_norm_seq = torch.zeros((S_eff, L, len(self.outputs)), dtype=torch.float32)
        else:
            y_norm_seq = self._model(X_model.to(dtype=torch.float32), axis_indices)

        # Denormalise via flatten/reshape — affine, gradient-clean.
        S_eff = int(y_norm_seq.shape[0])
        L = int(y_norm_seq.shape[1])
        n_out = int(y_norm_seq.shape[2])
        y_norm_flat = y_norm_seq.reshape(S_eff * L, n_out)
        y_denorm_flat = dm.denormalize_values(y_norm_flat, self.outputs)
        y_denorm_seq = y_denorm_flat.reshape(S_eff, L, n_out)

        # Reshape per-(s, feat) → feat_shape. Cell (multi-axis coord) maps to
        # (batch_row=s*n_other+other_idx, seq_pos=ravel(seq_coords)).
        other_axis_indices = [i for i in range(len(shape)) if i not in seq_axis_indices]
        other_sizes = [shape[i] for i in other_axis_indices]
        n_other = int(np.prod(other_sizes)) if other_sizes else 1

        out: list[dict[str, torch.Tensor]] = [{} for _ in range(S)]
        for s in range(S):
            feat_shape = dim_info_list[s]['shape']
            if not feat_shape:
                for f_idx, feat in enumerate(self.outputs):
                    out[s][feat] = y_denorm_seq[s, 0, f_idx].reshape(())
                continue

            n_total = int(np.prod(feat_shape))
            for f_idx, feat in enumerate(self.outputs):
                slots: list[torch.Tensor] = []
                for cell_flat in range(n_total):
                    coord = np.unravel_index(cell_flat, feat_shape)
                    seq_coord = tuple(int(coord[i]) for i in seq_axis_indices)
                    seq_pos = int(np.ravel_multi_index(seq_coord, seq_axis_sizes)) if seq_axis_sizes else 0
                    if other_sizes:
                        other_coord = tuple(int(coord[i]) for i in other_axis_indices)
                        other_idx = int(np.ravel_multi_index(other_coord, other_sizes))
                    else:
                        other_idx = 0
                    batch_idx = s * n_other + other_idx
                    slots.append(y_denorm_seq[batch_idx, seq_pos, f_idx])
                out[s][feat] = torch.stack(slots, dim=0).reshape(feat_shape)
        return out

    def forward_pass(self, X: torch.Tensor, gradient_pass: bool = False) -> torch.Tensor:
        """Per-row forward — calls the encoder with L=1.

        Provided for the non-sequence call sites (KDE encode probe, etc.).
        Real sequence prediction goes through ``predict``. With L=1 we use
        position-0 embeddings on every axis (axis_indices = zeros).
        """
        if not self._is_trained or self._model is None:
            return torch.zeros((X.shape[0], len(self.outputs)), dtype=torch.float32)
        ctx: Any = torch.no_grad() if not gradient_pass else _NullContext()
        with ctx:
            x_seq = X.unsqueeze(1).to(dtype=torch.float32)  # (B, 1, n_input)
            n_axes = len(self.sequence_axis_code)
            axis_indices = torch.zeros((1, n_axes), dtype=torch.long)
            return self._model(x_seq, axis_indices)[:, 0, :]

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

        ``seq_axis_sizes`` (kwarg) is the per-axis decomposition of L —
        required for multi-axis (``n_axes >= 2``) so axis_indices can be
        computed unambiguously; defaults to ``(L,)`` for single-axis.

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
                f"TransformerModel.train expects sequence-shaped batches "
                f"(B, L, n_input)/(B, L, n_output); got X.ndim={X_full.ndim}, "
                f"y.ndim={y_full.ndim}. PredictionSystem must build sequence "
                f"batches (not flat batches) for transformer models.",
            )
        n_input = int(X_full.shape[-1])
        n_output = int(y_full.shape[-1])
        L = int(X_full.shape[1])
        n_axes = len(self.sequence_axis_code)

        seq_axis_sizes = kwargs.get("seq_axis_sizes")
        if seq_axis_sizes is None:
            if n_axes == 1:
                seq_axis_sizes = (L,)
            else:
                raise ValueError(
                    f"TransformerModel.train: multi-axis transformer "
                    f"(n_axes={n_axes}) requires `seq_axis_sizes` kwarg "
                    f"so axis_indices can be unambiguously decomposed.",
                )
        seq_axis_sizes = tuple(int(s) for s in seq_axis_sizes)
        if len(seq_axis_sizes) != n_axes:
            raise ValueError(
                f"seq_axis_sizes length {len(seq_axis_sizes)} does not "
                f"match sequence_axis_code length {n_axes}.",
            )
        if int(np.prod(seq_axis_sizes)) != L:
            raise ValueError(
                f"seq_axis_sizes {seq_axis_sizes} product != batch L={L}.",
            )

        # Per-position axis coords: (L, n_axes), shared across all rows.
        axis_indices = torch.empty((L, n_axes), dtype=torch.long)
        for k in range(L):
            coord = np.unravel_index(k, seq_axis_sizes)
            for i in range(n_axes):
                axis_indices[k, i] = int(coord[i])

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
            loss = loss_fn(self._model(X_full_f, axis_indices), y_full_f)
            loss.backward()
            optimizer.step()

        self._model.eval()
        self._is_trained = True

    # ------------------------------------------------------------------
    # Schema check
    # ------------------------------------------------------------------

    def _validate_schema_compatibility(self, schema: Any) -> None:
        """Require every ``sequence_axis_code`` entry to resolve to a real domain axis."""
        try:
            seq_axis_codes = self.sequence_axis_code
        except NotImplementedError as e:
            raise ValueError(
                f"{self.__class__.__name__} must implement the `sequence_axis_code` "
                f"property to declare which schema axes to sequence over.",
            ) from e

        # Collect all axis codes available across the model's outputs' domains.
        available_axes: set[str] = set()
        for feat_code in self.outputs:
            feat_obj = getattr(schema.features, "data_objects", {}).get(feat_code)
            if feat_obj is None:
                continue
            domain_code = getattr(feat_obj, "domain_code", None)
            if domain_code is None or not schema.domains.has(domain_code):
                continue
            domain = schema.domains.get(domain_code)
            for ax in domain.axes:
                available_axes.add(ax.code)

        unresolved = [c for c in seq_axis_codes if c not in available_axes]
        if unresolved:
            raise ValueError(
                f"{self.__class__.__name__}.sequence_axis_code entries {unresolved} "
                f"do not resolve to any domain axis on this model's outputs "
                f"({list(self.outputs)}). Available axes: {sorted(available_axes)}.",
            )


# ---------------------------------------------------------------------------
# Internal nn.Module
# ---------------------------------------------------------------------------


class _TransformerNet(nn.Module):
    """Encoder-only transformer with causal mask + per-axis additive position embeddings.

    For an N-axis flattened sequence, one ``nn.Embedding(max_seq_len, d_model)``
    table per axis; per-position vectors are summed before the encoder. Single-
    axis case (``n_axes=1``) is recovered with one embedding table.
    """

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
        n_axes: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_input, d_model)
        # One embedding table per sequence axis. All sized at max_seq_len —
        # subclasses bump the class attribute if any single axis exceeds it.
        self.pos_embeds = nn.ModuleList([
            nn.Embedding(max_seq_len, d_model) for _ in range(n_axes)
        ])
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
        self._n_axes = n_axes

    def forward(
        self,
        x: torch.Tensor,
        axis_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B, L, n_input)`` → ``(B, L, n_output)`` with causal attention.

        ``axis_indices`` is ``(L, n_axes)`` long-tensor of per-position per-axis
        coords. When ``n_axes == 1`` and ``axis_indices`` is ``None``, defaults
        to ``arange(L)`` (single-axis case, back-compat).
        """
        B, L, _ = x.shape
        if axis_indices is None:
            if self._n_axes != 1:
                raise ValueError(
                    f"axis_indices required for multi-axis transformer "
                    f"(n_axes={self._n_axes}).",
                )
            axis_indices = torch.arange(L, device=x.device).unsqueeze(1)
        if axis_indices.shape != (L, self._n_axes):
            raise ValueError(
                f"axis_indices shape {tuple(axis_indices.shape)} does not match "
                f"(L={L}, n_axes={self._n_axes}).",
            )
        if int(axis_indices.max().item()) >= self._max_seq_len:
            raise ValueError(
                f"per-axis position index {int(axis_indices.max().item())} "
                f"exceeds MAX_SEQ_LEN={self._max_seq_len}; "
                f"increase the class attribute on the model.",
            )

        h = self.input_proj(x)
        # Sum per-axis positional embeddings.
        for i, embed in enumerate(self.pos_embeds):
            h = h + embed(axis_indices[:, i])[None, :, :]
        causal_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=x.device), diagonal=1,
        )
        h = self.encoder(h, mask=causal_mask, is_causal=True)
        return self.output_proj(h)


class _NullContext:
    """Context manager that does nothing — used when gradient_pass=True."""
    def __enter__(self): return self
    def __exit__(self, *args): return False
