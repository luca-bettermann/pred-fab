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


def gaussian_unit_peak(grid: np.ndarray, center: np.ndarray, sigma: float) -> np.ndarray:
    """Peak-1 Gaussian for the saturation figures.

    Production uses the mass-1 normalisation `(σ√2π)^−D · exp(−d²/2σ²)` so that
    `∫ρ dz = 1`. At our σ that puts a single isolated kernel's peak ρ ≈ 28 in
    2-D, so `E = D/(1+D) ≈ 0.97` — saturation happens immediately and the
    visual story collapses. Concept figures therefore plot the peak-rescaled
    version `exp(−d²/2σ²)`, which keeps the same kernel *shape* and σ but
    makes the saturation transform visible (single kernel → D=1 → E=0.5).
    The shape of D, E, and ΔE is identical up to a constant factor.
    """
    d2 = np.sum((grid - np.asarray(center)) ** 2, axis=-1)
    return np.exp(-d2 / (2.0 * sigma ** 2))
