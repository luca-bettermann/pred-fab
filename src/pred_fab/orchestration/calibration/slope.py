"""Sigmoid-slope trajectory decode.

Trajectory values for layer k:
    z(k) = z_mid + (k - L//2) · z_slope
    value(k) = lo + (hi - lo) · σ(z(k))

Linear in the center, asymptotic at param boundaries.
Two variables per trajectory dimension per experiment: midpoint + slope.
"""
from __future__ import annotations

from typing import Any

import torch
import numpy as np


def decode_slope_trajectory(
    midpoint_norm: torch.Tensor,
    z_slope: torch.Tensor,
    L: int,
) -> torch.Tensor:
    """Decode midpoint + slope into per-layer normalised values.

    Args:
        midpoint_norm: (..., D_traj) in [0, 1] — trajectory param midpoint
        z_slope: (..., D_traj) — slope in logit space per layer step
        L: number of layers

    Returns:
        (..., L, D_traj) normalised values in (0, 1)
    """
    mid_idx = L // 2
    z_mid = torch.logit(midpoint_norm.clamp(1e-4, 1 - 1e-4))
    offsets = torch.arange(L, dtype=z_mid.dtype, device=z_mid.device) - mid_idx
    # Broadcast: (..., 1, D_traj) + (L, 1) * (..., 1, D_traj) → (..., L, D_traj)
    z_all = z_mid.unsqueeze(-2) + offsets.reshape(*([1] * (z_mid.ndim - 1)), L, 1) * z_slope.unsqueeze(-2)
    return torch.sigmoid(z_all)


def default_slope_max(dim_code: str, data_objects: dict[str, Any]) -> float:
    """Default slope bound = 1/max_dim_value from the domain axis definition."""
    obj = data_objects.get(dim_code)
    if obj is not None and hasattr(obj, "max_val"):
        return 1.0 / max(int(obj.max_val), 1)
    return 0.1
