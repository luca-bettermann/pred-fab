"""Performance concept: feature prediction → scoring → system performance.

  Panel 1: Feature prediction f̂(x,y) — emerald sequential
  Panel 2: Scoring curve P(f̂) — emerald fill (centered, half height)
  Panel 3: System performance P_sys(x,y) — RdYlGn

Uses shared EXISTING_POINTS from _config.py (same as evidence figures).
Quadratic scoring function with target at 0.5.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _style import apply_style
from _config import EXISTING_POINTS
from pred_fab.plotting._style import save_fig, EMERALD_500, ZINC_600, FONT
from panels import (
    draw_experiments, feature_topology, performance_topology,
    marginal_performance, setup_axes, subplot_label, clean_spines,
)

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def main(
    save_path: str | None = None,
    predict_fn: Callable[[dict[str, Any]], dict[str, float]] | None = None,
    score_fn: Callable[[dict[str, Any]], float] | None = None,
    feature_code: str = "feature",
    target_value: float = 0.5,
    x_key: str = "x",
    y_key: str = "y",
    x_bounds: tuple[float, float] = (0.0, 1.0),
    y_bounds: tuple[float, float] = (0.0, 1.0),
    x_label: str = "$x$",
    y_label: str = "$y$",
    resolution: int = 80,
):
    x_lo, x_hi = x_bounds
    y_lo, y_hi = y_bounds

    if predict_fn is None:
        def predict_fn(params):
            xn = (params.get(x_key, 0.5) - x_lo) / (x_hi - x_lo)
            yn = (params.get(y_key, 0.5) - y_lo) / (y_hi - y_lo)
            # Multi-modal landscape for visual interest
            f = 0.6 * np.exp(-((xn - 0.35) ** 2 + (yn - 0.6) ** 2) / 0.08)
            f += 0.4 * np.exp(-((xn - 0.75) ** 2 + (yn - 0.3) ** 2) / 0.12)
            f += 0.25 * np.exp(-((xn - 0.2) ** 2 + (yn - 0.2) ** 2) / 0.06)
            f += 0.15 * np.sin(3.0 * np.pi * xn) * np.cos(2.5 * np.pi * yn) * 0.15
            return {feature_code: float(np.clip(f, 0, 1))}

    score_range = 0.5
    if score_fn is None:
        # Raised cosine: smooth bell centered at target
        def score_fn(params):
            features = predict_fn(params)
            f = features.get(feature_code, 0.0)
            if abs(f - target_value) > score_range:
                return 0.0
            return float(0.5 * (1.0 + np.cos(np.pi * (f - target_value) / score_range)))

    xs = np.linspace(x_lo, x_hi, resolution)
    ys = np.linspace(y_lo, y_hi, resolution)

    feat_grid = np.zeros((resolution, resolution))
    perf_grid = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            params = {x_key: float(xs[i]), y_key: float(ys[j])}
            features = predict_fn(params)
            feat_grid[j, i] = features.get(feature_code, 0.0)
            perf_grid[j, i] = score_fn(params)

    # Marginal: average P across y for each x
    perf_along_x = np.mean(perf_grid, axis=0)

    # Scoring curve: sweep feature value through the scoring function
    feat_range = np.linspace(0, 1, 200)
    score_curve = np.array([
        0.5 * (1.0 + np.cos(np.pi * (f - target_value) / score_range))
        if abs(f - target_value) <= score_range else 0.0
        for f in feat_range
    ])

    # Data points from shared config
    exp_x = EXISTING_POINTS[:, 0].tolist()
    exp_y = EXISTING_POINTS[:, 1].tolist()

    # ── Figure: 3-column layout, scoring curve centered in middle ──
    apply_style()
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 0.7, 1.3],
                          hspace=0.4, wspace=0.25,
                          left=0.05, right=0.96, top=0.92, bottom=0.10)

    # Panel 1: Feature topology (full height)
    ax_feat = fig.add_subplot(gs[:, 0])
    feature_topology(fig, ax_feat, xs, ys, feat_grid,
                     x_label, y_label, x_bounds, y_bounds,
                     target_value=target_value,
                     label=r"$\hat{f}(x, y)$")
    draw_experiments(ax_feat, exp_x, exp_y)

    # Panel 2: Scoring curve (centered, half height)
    ax_score = fig.add_subplot(gs[0, 1])
    ax_score.fill_between(feat_range, score_curve, alpha=0.15, color=EMERALD_500)
    ax_score.plot(feat_range, score_curve, color=EMERALD_500, linewidth=2)
    ax_score.axvline(target_value, color=EMERALD_500, ls='--', lw=1, alpha=0.5)
    ax_score.set_xlim(0, 1)
    ax_score.set_ylim(-0.05, 1.1)
    ax_score.set_xlabel(r"$\hat{f}$", fontsize=FONT["axis_label"], color=ZINC_600)
    subplot_label(ax_score, r"$P(\hat{f})$")
    clean_spines(ax_score)

    # Hide bottom-middle slot
    ax_empty = fig.add_subplot(gs[1, 1])
    ax_empty.set_visible(False)

    # Panel 3: Performance topology (full height)
    ax_perf = fig.add_subplot(gs[:, 2])
    performance_topology(fig, ax_perf, xs, ys, perf_grid,
                         x_label, y_label, x_bounds, y_bounds,
                         label="$P_{\\mathrm{sys}}(x, y)$",
                         fit_colorbar=True)
    draw_experiments(ax_perf, exp_x, exp_y)

    path = save_path or str(PLOTS_DIR / "performance_concept.png")
    save_fig(path, dpi=200)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
