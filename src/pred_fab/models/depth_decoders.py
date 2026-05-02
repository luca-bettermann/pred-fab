"""Depth decoders for ``TransformerModel`` — per-feature output heads.

Encoders produce a single latent representation per sequence position:
``(B, L, D)``. The encoder is shared across all output features regardless of
depth. Each output feature, however, lives at a specific depth (number of
domain axes it spans). The decoder's job is to map the encoder's latent to
the feature's native shape.

The framework auto-selects a decoder per output feature based on the
relationship between ``len(sequence_axis_code)`` (axis depth) and the
feature's depth:

- Axis depth == feature depth → 2-layer MLP, no extra position embedding.
- Axis depth < feature depth  → 2-layer MLP + per-extra-axis position
  embedding broadcast across the un-sequenced axes.
- Axis depth > feature depth  → rejected at validation (no pooling decoder).

``PerNodeMLPDepthDecoder`` is the universal default. Users can override on a
per-depth basis by setting ``DEPTH_DECODERS = {2: CustomDecoder()}`` on a
TransformerModel subclass; the framework only auto-selects when no override
is provided.

Caveat: the per-extra-axis position embedding is sized at the schema's
``max_val`` for that axis. Positions beyond what training actually saw remain
randomly initialised. Make sure training experiments cover the full range of
expected positions, or document degraded extrapolation for unseen counts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


class IDepthDecoder(ABC):
    """Abstract decoder mapping encoder hidden state to a per-feature output tensor.

    Decoders are *factories*: ``build(...)`` returns an ``nn.Module`` sized for
    a specific (d_model, n_features, extra_axis_sizes) configuration. The
    factory pattern keeps the decoder declaration parameter-free at class
    declaration time on the model.
    """

    @abstractmethod
    def build(
        self,
        d_model: int,
        n_features: int,
        extra_axis_sizes: tuple[int, ...],
    ) -> nn.Module:
        """Return an ``nn.Module`` that maps ``(B, L, d_model)`` → ``(B, L, *extra_axis_sizes, n_features)``.

        ``extra_axis_sizes`` is the tuple of axis sizes the decoder must
        expand over (i.e. axes the encoder did NOT sequence over). When
        empty, the decoder simply maps ``(B, L, D) → (B, L, n_features)``.
        """


class PerNodeMLPDepthDecoder(IDepthDecoder):
    """Universal symmetric-prior decoder.

    Same shared MLP applied at every (sequence position, extra-axis position).
    Per-extra-axis learned position embeddings encode where in the
    un-sequenced grid each output cell sits; the encoder's own positional
    embeddings handle the sequence axes. With zero extra axes the decoder
    degenerates to a per-position MLP.
    """

    def __init__(self, hidden: int = 32, embed_dim: int = 16):
        self.hidden = hidden
        self.embed_dim = embed_dim

    def build(
        self,
        d_model: int,
        n_features: int,
        extra_axis_sizes: tuple[int, ...],
    ) -> nn.Module:
        return _PerNodeMLPDecoderModule(
            d_model=d_model,
            n_features=n_features,
            extra_axis_sizes=extra_axis_sizes,
            hidden=self.hidden,
            embed_dim=self.embed_dim,
        )


class _PerNodeMLPDecoderModule(nn.Module):
    """Shared 2-layer MLP applied per output cell, with per-extra-axis position embeddings."""

    def __init__(
        self,
        d_model: int,
        n_features: int,
        extra_axis_sizes: tuple[int, ...],
        hidden: int,
        embed_dim: int,
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

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """``(B, L, D)`` → ``(B, L, *extra_axis_sizes, n_features)``."""
        if self.n_extra == 0:
            return self.mlp(hidden)

        B, L, _D = hidden.shape
        extra = self.extra_axis_sizes
        # Build position embeddings on the extra grid: (*extra, n_extra · embed_dim).
        coords = [torch.arange(size, device=hidden.device) for size in extra]
        embed_per_axis = [emb(c) for emb, c in zip(self.pos_embeds, coords)]  # each (size_i, embed_dim)

        # Outer-broadcast each axis's embedding to the full extra grid.
        broadcast_shapes = []
        for i, emb_i in enumerate(embed_per_axis):
            view_shape = [1] * self.n_extra + [emb_i.shape[-1]]
            view_shape[i] = emb_i.shape[0]
            broadcast_shapes.append(emb_i.view(*view_shape).expand(*extra, emb_i.shape[-1]))
        pos_concat = torch.cat(broadcast_shapes, dim=-1)  # (*extra, n_extra · embed_dim)

        # Tile hidden over the extra grid: (B, L, *extra, D).
        hidden_b = hidden.view(B, L, *([1] * self.n_extra), -1).expand(B, L, *extra, hidden.shape[-1])
        # Tile pos_concat over (B, L): (B, L, *extra, n_extra · embed_dim).
        pos_b = pos_concat.view(*([1, 1] + list(extra) + [pos_concat.shape[-1]]))
        pos_b = pos_b.expand(B, L, *extra, pos_concat.shape[-1])

        x = torch.cat([hidden_b, pos_b], dim=-1)
        return self.mlp(x)


def select_default_decoder(
    axis_depth: int,
    feature_depth: int,
) -> IDepthDecoder:
    """Pick the framework-default decoder for an (axis_depth, feature_depth) combo.

    Validation has already enforced ``axis_depth ≤ feature_depth``; this just
    returns ``PerNodeMLPDepthDecoder`` (which handles both same-depth and
    feature-deeper-than-axis via its ``extra_axis_sizes`` argument).
    """
    if axis_depth > feature_depth:
        raise ValueError(
            f"axis_depth={axis_depth} > feature_depth={feature_depth}. "
            f"Pooling decoders are not supported; redesign the model.",
        )
    return PerNodeMLPDepthDecoder()
