"""Depth decoder for ``TransformerModel`` — per-cell output head.

The encoder produces a single latent representation per sequence position:
``(B, L, D)``, shared across all output features regardless of depth. Each
output feature lives at a specific depth (number of domain axes it spans).
The decoder maps the encoder's latent to the feature's native shape.

The framework ships exactly one decoder, ``PerNodeMLPDecoder`` — a shared
2-layer MLP applied at every (sequence position, extra-axis position) with
learned positional embeddings on the un-sequenced axes. Symmetric prior:
the same physical mapping at each cell, distinguished only by position.
Subclasses can override ``TransformerModel._build_decoder`` if a different
head is ever needed.

Operating regimes:

- Axis depth == feature depth → MLP only, no extra position embedding.
- Axis depth < feature depth  → MLP + per-extra-axis position embedding
  broadcast across the un-sequenced axes.
- Axis depth > feature depth  → rejected at validation; no pooling decoder.

Caveat: per-extra-axis position embeddings are sized at the schema's
``max_val`` for that axis. Positions beyond what training actually saw
remain randomly initialised. Make sure training experiments cover the full
range of expected positions, or document degraded extrapolation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PerNodeMLPDecoder(nn.Module):
    """Shared 2-layer MLP applied per output cell, with per-extra-axis position embeddings.

    Maps ``(B, L, d_model) → (B, L, *extra_axis_sizes, n_features)``.

    ``extra_axis_sizes`` is the tuple of axis sizes the decoder expands over —
    axes the encoder did NOT sequence over. With zero extra axes the decoder
    degenerates to a per-position MLP.
    """

    def __init__(
        self,
        d_model: int,
        n_features: int,
        extra_axis_sizes: tuple[int, ...],
        hidden: int = 32,
        embed_dim: int = 16,
    ):
        super().__init__()
        self.extra_axis_sizes = tuple(int(s) for s in extra_axis_sizes)
        self.n_extra = len(self.extra_axis_sizes)

        # One embedding table per extra axis. Sized at the axis's schema max;
        # only the first n_actual rows are read at forward time.
        self.pos_embeds = nn.ModuleList([
            nn.Embedding(int(size), embed_dim) for size in self.extra_axis_sizes
        ])

        in_dim = d_model + (embed_dim * self.n_extra)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        actual_axis_sizes: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """``(B, L, D)`` → ``(B, L, *actual_axis_sizes, n_features)``.

        ``actual_axis_sizes`` is the per-call runtime size per extra axis (must
        be ≤ schema-max ``extra_axis_sizes`` used at decoder construction). If
        ``None``, defaults to schema-max. Each axis's positional embedding is
        sliced to the actual size, so the same decoder serves variable-shaped
        runtimes (e.g. experiments with smaller n_segments than schema max).
        """
        if self.n_extra == 0:
            return self.mlp(hidden)

        extra = (
            tuple(int(s) for s in actual_axis_sizes)
            if actual_axis_sizes is not None else self.extra_axis_sizes
        )
        if len(extra) != self.n_extra:
            raise ValueError(
                f"actual_axis_sizes length {len(extra)} != n_extra {self.n_extra}.",
            )
        for i, size in enumerate(extra):
            if size > self.extra_axis_sizes[i]:
                raise ValueError(
                    f"actual_axis_sizes[{i}] = {size} exceeds embedded size "
                    f"{self.extra_axis_sizes[i]}; bump the schema axis max_val.",
                )

        B, L, _D = hidden.shape
        coords = [torch.arange(s, device=hidden.device) for s in extra]
        embed_per_axis = [emb(c) for emb, c in zip(self.pos_embeds, coords)]

        broadcast_shapes = []
        for i, emb_i in enumerate(embed_per_axis):
            view_shape = [1] * self.n_extra + [emb_i.shape[-1]]
            view_shape[i] = emb_i.shape[0]
            broadcast_shapes.append(emb_i.view(*view_shape).expand(*extra, emb_i.shape[-1]))
        pos_concat = torch.cat(broadcast_shapes, dim=-1)

        hidden_b = hidden.view(B, L, *([1] * self.n_extra), -1).expand(B, L, *extra, hidden.shape[-1])
        pos_b = pos_concat.view(*([1, 1] + list(extra) + [pos_concat.shape[-1]]))
        pos_b = pos_b.expand(B, L, *extra, pos_concat.shape[-1])

        x = torch.cat([hidden_b, pos_b], dim=-1)
        return self.mlp(x)
