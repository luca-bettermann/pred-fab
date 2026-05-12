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

import numpy as np
import torch
import torch.nn as nn

from ..interfaces import IPredictionModel
from ..utils import PfabLogger
from .depth_decoders import PerNodeMLPDecoder


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

    # Per-depth loss weights. Default 1.0 each; override to balance multi-
    # depth losses (e.g. ``LOSS_WEIGHTS = {1: 1.0, 2: 0.5}`` halves depth-2's
    # contribution).
    LOSS_WEIGHTS: dict[int, float] = {}

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
        # Populated at train time. Maps output depth → ordered list of features
        # at that depth. Decoders are stored on ``self._model.decoders`` keyed by
        # str(depth); the order of features here matches the per-depth decoder's
        # output channel order.
        self._depth_to_features: dict[int, list[str]] = {}

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    def _feat_depth(self, feat_code: str) -> int:
        """Output depth of a feature from its registered DataArray columns."""
        feat_obj = self._ref_features.get(feat_code)
        cols = feat_obj.columns if (feat_obj is not None and hasattr(feat_obj, "columns")) else []  # type: ignore[union-attr]
        return (len(cols) - 1) if cols else 0

    def _group_outputs_by_depth(self) -> dict[int, list[str]]:
        """Group ``self.outputs`` by feature depth, preserving declaration order within each depth."""
        groups: dict[int, list[str]] = {}
        for feat in self.outputs:
            d = self._feat_depth(feat)
            groups.setdefault(d, []).append(feat)
        return groups

    def _build_network(
        self,
        n_input: int,
        domain_axis_sizes: tuple[int, ...],
    ) -> nn.Module:
        """Build encoder + per-depth decoders into one ``_EncDecNet``.

        ``domain_axis_sizes`` is the schema-derived size per domain axis (max_val
        of each ``Dimension``); the decoder uses these as the upper bound for
        positional embeddings on un-sequenced axes. Override in subclasses to
        swap the encoder type or wrap decoders with custom heads.
        """
        torch.manual_seed(self.SEED)
        n_axes = len(self.sequence_axis_code)

        encoder = _TransformerEncoder(
            n_input=n_input,
            d_model=self.D_MODEL,
            n_heads=self.N_HEADS,
            n_layers=self.N_LAYERS,
            dim_ff=self.DIM_FEEDFORWARD,
            dropout=self.DROPOUT,
            max_seq_len=self.MAX_SEQ_LEN,
            n_axes=n_axes,
        )

        self._depth_to_features = self._group_outputs_by_depth()
        decoders: dict[int, nn.Module] = {}
        for depth, feats in self._depth_to_features.items():
            extra_axis_sizes = tuple(domain_axis_sizes[n_axes:depth])
            decoders[depth] = self._build_decoder(
                depth=depth,
                n_features=len(feats),
                extra_axis_sizes=extra_axis_sizes,
            )
        return _EncDecNet(encoder, decoders)

    def _build_decoder(
        self,
        depth: int,
        n_features: int,
        extra_axis_sizes: tuple[int, ...],
    ) -> nn.Module:
        """Build the decoder for a single output depth.

        Default: ``PerNodeMLPDecoder`` — symmetric-prior shared MLP applied at
        every cell with positional embeddings on un-sequenced axes. Override
        in subclasses to swap heads (e.g. for asymmetric-prior cases).
        """
        del depth  # not used by the default decoder
        return PerNodeMLPDecoder(
            d_model=self.D_MODEL,
            n_features=n_features,
            extra_axis_sizes=extra_axis_sizes,
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
        """Per-candidate sequence-batched prediction → ``list[dict[feat, (*feat_native_shape) tensor]]``.

        Encoder operates at axis-depth granularity — input is collapsed to
        ``(S, L, n_input)``. For each output feature, the per-depth decoder
        expands hidden state ``(S, L, D)`` to the feature's native shape.

        With the validation rule ``input_feature_depth ≤ axis_depth``, all
        ``n_other`` parallel-batched rows from ``dm.build_sequence_batch`` are
        equivalent for a given candidate; we take the first row per candidate.

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

        X_seq_full = dm.build_sequence_batch(self, params_list, dim_info_list)
        if X_seq_full.shape[1] == 0:
            return [
                {feat: torch.zeros(self._feat_native_shape(feat, dim_info_list[s])) for feat in self.outputs}
                for s in range(S)
            ]

        # Compute axis layout for this call.
        di_first = dim_info_list[0]
        dim_codes = di_first['dim_codes_ordered']
        shape = di_first['shape']
        seq_axis_codes = self.sequence_axis_code
        n_axes = len(seq_axis_codes)
        seq_axis_indices = [dim_codes.index(c) for c in seq_axis_codes if c in dim_codes]
        seq_axis_sizes = tuple(shape[i] for i in seq_axis_indices)
        L = int(np.prod(seq_axis_sizes)) if seq_axis_sizes else 1
        other_axis_indices = [i for i in range(len(shape)) if i not in seq_axis_indices]
        other_sizes = tuple(shape[i] for i in other_axis_indices)
        n_other = int(np.prod(other_sizes)) if other_sizes else 1

        # Collapse n_other to 1 (rows are equivalent under input-depth ≤ axis-depth).
        X_seq = X_seq_full[::n_other]  # (S, L, n_input_full)
        input_indices = dm.get_input_indices(self.input_parameters + self.input_features)
        input_indices_t = torch.as_tensor(input_indices, dtype=torch.long)
        X_model = X_seq.index_select(2, input_indices_t)  # (S, L, n_model_input)

        # Per-position axis indices (L, n_axes).
        if seq_axis_sizes:
            axis_indices = torch.empty((L, n_axes), dtype=torch.long)
            for k in range(L):
                coord = np.unravel_index(k, seq_axis_sizes)
                for i in range(n_axes):
                    axis_indices[k, i] = int(coord[i])
        else:
            axis_indices = torch.zeros((1, n_axes), dtype=torch.long)

        # Untrained → return zeros at each feature's native shape.
        if not self._is_trained or self._model is None:
            return [
                {feat: torch.zeros(self._feat_native_shape(feat, dim_info_list[s])) for feat in self.outputs}
                for s in range(S)
            ]

        # Actual extra-axis sizes per depth from this call's dim_info: shape[axis_depth..depth).
        actual_extra_per_depth: dict[int, tuple[int, ...]] = {}
        for depth in self._depth_to_features:
            actual_extra_per_depth[depth] = tuple(shape[axis_idx] for axis_idx in range(n_axes, depth))

        per_depth_outputs: dict[int, torch.Tensor] = self._model(
            X_model.to(dtype=torch.float32), axis_indices, actual_extra_per_depth,
        )

        # Per-feature reshape + denormalise.
        out: list[dict[str, torch.Tensor]] = [{} for _ in range(S)]
        for depth, feats in self._depth_to_features.items():
            out_d = per_depth_outputs[depth]  # (S, L, *extra_d, n_features_d)
            for f_idx, feat in enumerate(feats):
                y_pred_norm = out_d[..., f_idx]  # (S, L, *extra_d)
                pred_shape = y_pred_norm.shape
                y_flat = y_pred_norm.reshape(-1, 1)
                y_denorm = dm.denormalize_values(y_flat, [feat]).reshape(pred_shape)
                # Per-candidate reshape: (L, *extra_d) → feat_native_shape.
                for s in range(S):
                    feat_native_shape = self._feat_native_shape(feat, dim_info_list[s])
                    out[s][feat] = y_denorm[s].reshape(feat_native_shape)
        return out

    def _feat_native_shape(self, feat: str, dim_info: dict[str, Any]) -> tuple[int, ...]:
        """Native per-cell shape of ``feat`` for a given dim_info: first ``feat_depth`` axes."""
        feat_depth = self._feat_depth(feat)
        if feat_depth == 0:
            return ()
        return tuple(dim_info['shape'][:feat_depth])

    def forward_pass(
        self,
        X: torch.Tensor,
        gradient_pass: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Per-row forward → ``dict[feat_code, (batch,) tensor]``. L=1 encoder + decoder dispatch.

        Only valid when every output's depth equals the model's axis depth —
        otherwise the per-cell scalar interpretation breaks (per-segment
        positions have no canonical collapse). Multi-depth or
        ``feat_depth > axis_depth`` cases must use ``predict``.
        """
        n_axes = len(self.sequence_axis_code)
        if self._is_trained and self._depth_to_features:
            depths = list(self._depth_to_features.keys())
            non_axis_depth = [d for d in depths if d != n_axes]
            if non_axis_depth:
                raise NotImplementedError(
                    f"{self.__class__.__name__}.forward_pass requires every output's "
                    f"depth to equal axis_depth ({n_axes}); got depths {depths}. "
                    f"Use predict() for sequence-aware multi-depth dispatch.",
                )

        if not self._is_trained or self._model is None:
            zero = torch.zeros((X.shape[0],), dtype=torch.float32)
            return {feat: zero.clone() for feat in self.outputs}
        ctx: Any = torch.no_grad() if not gradient_pass else _NullContext()
        with ctx:
            x_seq = X.unsqueeze(1).to(dtype=torch.float32)  # (B, 1, n_input)
            axis_indices = torch.zeros((1, n_axes), dtype=torch.long)
            per_depth_outputs: dict[int, torch.Tensor] = self._model(x_seq, axis_indices)
        result: dict[str, torch.Tensor] = {}
        for depth, feats in self._depth_to_features.items():
            out_d = per_depth_outputs[depth]  # (B, 1, n_features_d) — no extras since depth == axis_depth
            for f_idx, feat in enumerate(feats):
                result[feat] = out_d[:, 0, f_idx]
        return result

    def encode(self, X: torch.Tensor, gradient_pass: bool = False) -> torch.Tensor:
        """Penultimate-layer activations for KDE — input projection of the encoder.

        The transformer doesn't expose a meaningful per-row latent because
        its hidden states are sequence-positional. We use the encoder's input
        projection output as a static-only embedding for KDE evidence.
        """
        if not self._is_trained or self._model is None:
            return X
        ctx: Any = torch.no_grad() if not gradient_pass else _NullContext()
        with ctx:
            return self._model.encoder.input_proj(X.to(dtype=torch.float32))  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_batches: list[tuple[torch.Tensor, Any]],
        val_batches: list[tuple[torch.Tensor, Any]],
        epoch_callback: Callable[[float], list[tuple[torch.Tensor, torch.Tensor]] | None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Train the encoder + per-depth decoders on per-feature targets.

        Each training batch is ``(X_seq, y_dict)`` where:

        - ``X_seq`` has shape ``(S, L, n_input)`` (encoder-only granularity;
          ``n_other`` already collapsed by the orchestration).
        - ``y_dict`` maps feat_code → tensor at native shape
          ``(S, L, *extra_axis_sizes_d)`` for that feature's depth.

        Loss is per-feature mean MSE summed across features, with optional
        ``LOSS_WEIGHTS[depth]`` per-depth multipliers (default 1.0 each).

        Required kwargs:

        - ``seq_axis_sizes: tuple[int, ...]`` — per-axis decomposition of L.
        - ``domain_axis_sizes: tuple[int, ...]`` — schema upper bounds for the
          model's full domain (used to size decoder positional embeddings).

        ``epoch_callback`` is accepted for ``IPredictionModel.train`` signature
        compatibility but not invoked — the encoder doesn't consume y as input,
        so there's no exposure-bias train/inference mismatch to correct.
        """
        del epoch_callback
        if not train_batches:
            return

        # Validate batch shape: list[(X_seq, y_dict)].
        if not isinstance(train_batches[0][1], dict):
            raise ValueError(
                f"TransformerModel.train expects batches as ``(X_seq, y_dict)`` "
                f"with y_dict mapping feat_code → tensor; got y of type "
                f"{type(train_batches[0][1]).__name__}.",
            )

        X_full = torch.cat([b[0] for b in train_batches], dim=0)  # (B, L, n_input)
        if X_full.ndim != 3:
            raise ValueError(
                f"TransformerModel.train: X_seq must be 3-D (B, L, n_input); "
                f"got ndim={X_full.ndim}.",
            )
        y_full: dict[str, torch.Tensor] = {
            feat: torch.cat([b[1][feat] for b in train_batches], dim=0)
            for feat in self.outputs
        }

        n_input = int(X_full.shape[-1])
        L = int(X_full.shape[1])
        n_axes = len(self.sequence_axis_code)

        seq_axis_sizes = kwargs.get("seq_axis_sizes")
        domain_axis_sizes = kwargs.get("domain_axis_sizes")
        if seq_axis_sizes is None:
            if n_axes == 1:
                seq_axis_sizes = (L,)
            else:
                raise ValueError(
                    f"TransformerModel.train: multi-axis transformer "
                    f"(n_axes={n_axes}) requires `seq_axis_sizes` kwarg.",
                )
        if domain_axis_sizes is None:
            raise ValueError(
                f"TransformerModel.train: `domain_axis_sizes` kwarg required "
                f"to size decoder positional embeddings.",
            )
        seq_axis_sizes = tuple(int(s) for s in seq_axis_sizes)
        domain_axis_sizes = tuple(int(s) for s in domain_axis_sizes)
        if len(seq_axis_sizes) != n_axes:
            raise ValueError(
                f"seq_axis_sizes length {len(seq_axis_sizes)} does not "
                f"match sequence_axis_code length {n_axes}.",
            )
        if int(np.prod(seq_axis_sizes)) != L:
            raise ValueError(
                f"seq_axis_sizes {seq_axis_sizes} product != batch L={L}.",
            )

        # Per-position axis coords: (L, n_axes).
        axis_indices = torch.empty((L, n_axes), dtype=torch.long)
        for k in range(L):
            coord = np.unravel_index(k, seq_axis_sizes)
            for i in range(n_axes):
                axis_indices[k, i] = int(coord[i])

        torch.manual_seed(self.SEED)
        self._model = self._build_network(n_input, domain_axis_sizes)
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY,
        )
        self._model.train()

        X_full_f = X_full.to(dtype=torch.float32)
        y_full_f = {feat: t.to(dtype=torch.float32) for feat, t in y_full.items()}

        # Per-depth actual extra-axis sizes from y shape (uniform across batches).
        actual_extra_per_depth: dict[int, tuple[int, ...]] = {}
        for depth, feats in self._depth_to_features.items():
            sample = y_full_f[feats[0]]  # (B, L, *extra_d)
            actual_extra_per_depth[depth] = tuple(int(s) for s in sample.shape[2:])

        epoch_logger = kwargs.get("epoch_logger")

        for epoch in range(self.EPOCHS):
            optimizer.zero_grad()
            per_depth_outputs: dict[int, torch.Tensor] = self._model(
                X_full_f, axis_indices, actual_extra_per_depth,
            )
            total_loss = X_full_f.new_zeros(())
            per_depth_loss: dict[str, float] = {}
            for depth, feats in self._depth_to_features.items():
                out_d = per_depth_outputs[depth]
                weight = float(self.LOSS_WEIGHTS.get(depth, 1.0))
                depth_loss = 0.0
                for f_idx, feat in enumerate(feats):
                    y_pred_f = out_d[..., f_idx]
                    loss_f = nn.functional.mse_loss(y_pred_f, y_full_f[feat])
                    total_loss = total_loss + weight * loss_f
                    depth_loss += float(loss_f.detach())
                per_depth_loss[f"loss/depth_{depth}"] = depth_loss
            total_loss.backward()
            optimizer.step()

            if epoch_logger is not None:
                epoch_logger.log_epoch(epoch, {
                    "loss/total": float(total_loss.detach()),
                    **per_depth_loss,
                })

        self._model.eval()
        self._is_trained = True

    # ------------------------------------------------------------------
    # Schema check
    # ------------------------------------------------------------------

    def validate_dimensional_coherence(self, schema: Any) -> str | None:
        """Transformer rules layered on the base: axis prefix-of-domain +
        axis-depth ≤ min(accepted_depths) + input-depth ≤ axis-depth.

        Mixed accepted depths are allowed — depth decoders handle expansion.
        Axis must be a *prefix* of ``domain.axes`` so depth-d feature shape is
        unambiguously the first ``d`` axes. Input depth ≤ axis depth ensures
        all inputs reach the encoder (deeper inputs would be lost when
        n_other is collapsed).
        """
        name = self.__class__.__name__
        domain_code = super().validate_dimensional_coherence(schema)
        seq_axis_codes = self.sequence_axis_code
        axis_depth = len(seq_axis_codes)

        # Axis is a prefix of the domain.
        if domain_code is not None:
            domain = schema.domains.get(domain_code)
            domain_axis_codes = [ax.code for ax in domain.axes]
            if list(seq_axis_codes) != domain_axis_codes[:axis_depth]:
                raise ValueError(
                    f"{name}: sequence_axis_code {list(seq_axis_codes)} must be a "
                    f"prefix of the domain's axes {domain_axis_codes}. "
                    f"Reorder either the schema's Domain axes or the model's "
                    f"sequence_axis_code so they match in order.",
                )

        # Axis depth ≤ shallowest accepted depth.
        accepted = self._accepted_depths()
        if accepted and axis_depth > min(accepted):
            raise ValueError(
                f"{name}: sequence_axis_code has length {axis_depth} but the "
                f"shallowest accepted depth is {min(accepted)}. Axis must not be "
                f"deeper than any output (no pooling decoder).",
            )

        # Input depth ≤ axis depth (deeper inputs would not reach the encoder).
        for code in self.input_features:
            input_depth = self._schema_feature_depth(schema, code)
            if input_depth > axis_depth:
                raise ValueError(
                    f"{name}: input feature '{code}' has depth {input_depth}, "
                    f"which exceeds the model's sequence axis depth {axis_depth}. "
                    f"The encoder operates at axis-depth granularity; deeper "
                    f"inputs would be lost. Either lift the input to depth ≤ "
                    f"{axis_depth} or extend sequence_axis_code.",
                )
        return domain_code

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


class _TransformerEncoder(nn.Module):
    """Encoder-only transformer: ``(B, L, n_input)`` → ``(B, L, d_model)`` hidden state.

    Per-axis additive position embeddings (one ``nn.Embedding(max_seq_len, d_model)``
    per sequence axis, summed). No output projection — that lives in the depth
    decoders so the encoder can be shared across multi-depth output heads.
    """

    def __init__(
        self,
        n_input: int,
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
        self._max_seq_len = max_seq_len
        self._n_axes = n_axes

    def forward(
        self,
        x: torch.Tensor,
        axis_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B, L, n_input)`` → ``(B, L, d_model)`` with causal attention."""
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
        for i, embed in enumerate(self.pos_embeds):
            h = h + embed(axis_indices[:, i])[None, :, :]
        causal_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=x.device), diagonal=1,
        )
        return self.encoder(h, mask=causal_mask, is_causal=True)


class _EncDecNet(nn.Module):
    """Encoder + per-depth decoder dispatch.

    Combines the shared encoder with one decoder ``nn.Module`` per output
    depth. Forward returns ``dict[depth, decoder_output]`` — each decoder's
    output has shape ``(B, L, *extra_axis_sizes_d, n_features_at_d)`` per the
    decoder contract.
    """

    def __init__(
        self,
        encoder: _TransformerEncoder,
        decoders: dict[int, nn.Module],
    ):
        super().__init__()
        self.encoder = encoder
        # Keys must be strings for nn.ModuleDict; cast back at access time.
        self.decoders = nn.ModuleDict({str(d): m for d, m in decoders.items()})

    def hidden(self, x: torch.Tensor, axis_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Encoder pass only — returns ``(B, L, d_model)``."""
        return self.encoder(x, axis_indices)

    def forward(
        self,
        x: torch.Tensor,
        axis_indices: torch.Tensor | None = None,
        actual_extra_per_depth: dict[int, tuple[int, ...]] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Encoder + every decoder. Returns ``dict[depth, decoder_output]``.

        ``actual_extra_per_depth`` overrides the decoder's schema-max extra-axis
        sizes per depth (e.g. ``{2: (3,)}`` for a depth-2 head with runtime
        n_segments=3 < schema max). Defaults to schema-max for missing depths.
        """
        hidden = self.encoder(x, axis_indices)
        actual_extra_per_depth = actual_extra_per_depth or {}
        out: dict[int, torch.Tensor] = {}
        for d_str, module in self.decoders.items():
            depth = int(d_str)
            actual = actual_extra_per_depth.get(depth)
            out[depth] = module(hidden, actual)
        return out


class _NullContext:
    """Context manager that does nothing — used when gradient_pass=True."""
    def __enter__(self): return self
    def __exit__(self, *args): return False
