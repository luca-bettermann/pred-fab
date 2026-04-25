"""Shared constants + math helpers for concept figures.

Where possible the figures pull constants directly from `pred_fab` so they
visualise what the production code actually does. Only values with no
production equivalent (the test-point layout used to draw an interesting
saturation pattern) live here as plain constants.
"""
from __future__ import annotations

import numpy as np

# Production σ — same SIGMA_DEFAULT the PredictionSystem uses when no
# override is configured. Changing it in prediction.py propagates here.
from pred_fab.orchestration.prediction import SIGMA_DEFAULT as SIGMA

# 8 existing kernel centres laid out for visually interesting topology:
# two clustered pairs (saturation), four scattered singletons (clean peaks),
# leaving a moderate-coverage gap around the bottom-centre where Z_NEW lands.
EXISTING_POINTS: np.ndarray = np.array([
    [0.18, 0.22],
    [0.30, 0.20],   # close to previous → overlap, saturated D
    [0.78, 0.18],
    [0.86, 0.40],
    [0.60, 0.55],
    [0.22, 0.62],
    [0.40, 0.85],
    [0.78, 0.82],
])

# Proposal point — partially covered by the 0.18,0.22 + 0.30,0.20 cluster
# on the left and the 0.78,0.18 kernel on the right, so ΔE is non-trivial.
Z_NEW: np.ndarray = np.array([0.50, 0.30])


def gaussian_density(grid: np.ndarray, center: np.ndarray, sigma: float) -> np.ndarray:
    """Mass-1 Gaussian density — matches production (`∫ρ dz = 1`).

    `ρ(z) = (σ√2π)^−D · exp(−‖z−c‖²/2σ²)`. Concept figures use the same
    normalisation as `KernelIndex.density_at` so what they show is exactly
    what the optimiser sees, including the strong saturation that follows
    from production-σ peaks (`E ≈ 0.97` for a single isolated kernel).
    """
    D = grid.shape[-1]
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi)) ** D
    d2 = np.sum((grid - np.asarray(center)) ** 2, axis=-1)
    return norm * np.exp(-d2 / (2.0 * sigma ** 2))
