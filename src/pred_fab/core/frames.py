"""Pure coordinate-frame transforms — one definition per frame transition.

The frames and why each is load-bearing are documented in the KB note
*PFAB - Coordinate Spaces*. This module owns the *transforms* so each
frame→frame rule lives in exactly one place; callers in calibration and
prediction delegate here instead of re-deriving them inline (the class of
drift that produced the σ-frame split and the categorical-inert bug).

Already-centralised transitions live elsewhere and are referenced, not moved:
``raw ↔ z-score`` is :mod:`pred_fab.core.normalisers`; ``z-score ↔ latent`` is
``model.encode``.
"""
from __future__ import annotations

from typing import Any, overload

import numpy as np
import torch


@overload
def to_unit_frame(points: np.ndarray, domain_bounds: np.ndarray | torch.Tensor | None) -> np.ndarray: ...
@overload
def to_unit_frame(points: torch.Tensor, domain_bounds: np.ndarray | torch.Tensor | None) -> torch.Tensor: ...
def to_unit_frame(points, domain_bounds):
    """``latent → [0,1]`` by ``domain_bounds`` so σ is a *fraction of range*.

    Shared by ``density_at`` (visualisation) and the acquisition Δ∫E path so both
    read σ in the same coordinate frame — otherwise the optimiser sees a different
    effective kernel width than the plots. Accepts numpy or torch and returns the
    same type; broadcasts ``domain_bounds`` (D, 2) over the trailing dim. No-op
    when ``domain_bounds`` is None (raw-frame fallback).
    """
    if domain_bounds is None:
        return points
    lo = domain_bounds[:, 0]
    span = domain_bounds[:, 1] - domain_bounds[:, 0]
    if isinstance(points, torch.Tensor):
        lo = torch.as_tensor(lo, dtype=points.dtype, device=points.device)
        span = torch.as_tensor(span, dtype=points.dtype, device=points.device)
        span = torch.where(span > 1e-10, span, torch.ones_like(span))
        return (points - lo) / span
    span = np.where(span > 1e-10, span, 1.0)
    return (points - np.asarray(lo)) / span


def param_value_to_fill(
    value: Any,
    *,
    categories: list[Any] | None = None,
    bounds: tuple[float, float] | None = None,
) -> float:
    """Encode one frozen-parameter value into the prior-fill (decode-input) frame.

    Categorical → the raw cat-index (the column has no normaliser stats, so the
    decode frame passes it through unchanged). Numeric → [0,1] by ``bounds``.
    Inverse of :func:`raw_scalar_to_param` for the value→frame→value round-trip.
    Raises ``ValueError`` for an unknown category (caller decides the fallback).
    """
    if categories is not None:
        return float(categories.index(value))
    if bounds is None:
        raise ValueError("param_value_to_fill needs either categories or bounds")
    lo, hi = bounds
    span = hi - lo
    return (float(value) - lo) / span if span > 0 else 0.5


def raw_scalar_to_param(
    raw_scalar: torch.Tensor,
    *,
    categories: list[Any] | None = None,
    is_integer: bool = False,
) -> Any:
    """Decode one physical-frame scalar to its parameter value.

    Categorical → label (``cats[round(raw)]``, clamped); integer → Python int;
    else the raw tensor element unchanged (kept grad-bearing for continuous
    params). Inverse of :func:`param_value_to_fill`.
    """
    if categories is not None:
        idx = int(round(float(raw_scalar.item())))
        return categories[max(0, min(idx, len(categories) - 1))]
    if is_integer:
        return int(round(float(raw_scalar.item())))
    return raw_scalar
