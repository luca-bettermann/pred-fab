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
from pred_fab.orchestration.prediction import RADIUS_DEFAULT as SIGMA

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

# 3 concept points for the integration/computation figures.
# Used by both evidence and performance concept plots.
CONCEPT_POINTS: np.ndarray = np.array([
    [0.25, 0.18],
    [0.28, 0.82],
    [0.78, 0.48],
])
CONCEPT_LABELS: list[str] = ["A", "B", "C"]
SIGMA_VIS: float = 0.08


def make_topology_marginal_layout(fig_width: float = 11.0, fig_height: float = 5.0):
    """Shared layout: full-height topology left, two half-height marginals right.

    Used by both evidence integration and performance concept figures
    for consistent proportions.
    Returns (fig, ax_main, ax_top, ax_bottom).
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.5, wspace=0.15,
                          left=0.06, right=0.95, top=0.92, bottom=0.12)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bottom = fig.add_subplot(gs[1, 1])
    return fig, ax_main, ax_top, ax_bottom


def gaussian_density(grid: np.ndarray, center: np.ndarray, sigma: float) -> np.ndarray:
    """Peak-1 Gaussian density — matches production (`ρ(c) = 1`).

    `ρ(z) = exp(−‖z−c‖²/2σ²)`. Concept figures use the same normalisation
    as `KernelIndex.density_at` so what they show is exactly what the
    optimiser sees: D bounded by the sum of overlapping kernel weights,
    E = D/(1+D) keeping a usable gradient instead of saturating ≈ 1 from a
    single kernel.
    """
    d2 = np.sum((grid - np.asarray(center)) ** 2, axis=-1)
    return np.exp(-d2 / (2.0 * sigma ** 2))
