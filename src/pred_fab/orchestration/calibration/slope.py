"""Tanh-slope trajectory decode.

Trajectory values for layer k:
    z(k) = z_mid + (k - L//2) · z_slope
    value(k) = 0.5 + 0.5 · tanh(z(k))

Tanh has a wider linear region than sigmoid (linear for |z| < ~1 vs ~0.5).
tanh'(0) = 1, so z_slope ≈ real-space normalised step near center.
Two variables per trajectory dimension per experiment: midpoint + slope.
"""
from __future__ import annotations

from typing import Any

import torch


def decode_slope_trajectory(
    midpoint_norm: torch.Tensor,
    z_slope: torch.Tensor,
    L: int,
) -> torch.Tensor:
    """Decode midpoint + slope into per-layer normalised values.

    Args:
        midpoint_norm: (..., D_traj) in [0, 1] — trajectory param midpoint
        z_slope: (..., D_traj) — slope per layer step
        L: number of layers

    Returns:
        (..., L, D_traj) normalised values in (0, 1)
    """
    mid_idx = L // 2
    # atanh: inverse of 0.5 + 0.5*tanh → z = atanh(2*x - 1)
    x_centered = (2.0 * midpoint_norm - 1.0).clamp(-1 + 1e-4, 1 - 1e-4)
    z_mid = torch.atanh(x_centered)
    offsets = torch.arange(L, dtype=z_mid.dtype, device=z_mid.device) - mid_idx
    z_all = z_mid.unsqueeze(-2) + offsets.reshape(*([1] * (z_mid.ndim - 1)), L, 1) * z_slope.unsqueeze(-2)
    return 0.5 + 0.5 * torch.tanh(z_all)


def default_slope_max(dim_code: str, data_objects: dict[str, Any]) -> float:
    """Default slope bound = param_bounds / 10 in normalised space.

    With tanh, slope ≈ real-space normalised step near center (tanh'(0) = 1).
    Default trust region = (hi - lo) / 10. Normalised: 0.1.
    """
    return 0.1
