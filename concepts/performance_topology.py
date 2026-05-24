"""Performance concept: two figures mirroring evidence integration + gain.

  Figure 1 — Performance computation:
    Left: Feature prediction f̂(x,y) — emerald sequential
    Top right: Scoring curve P(f̂) — emerald fill
    Bottom right: empty (single scoring function, no second marginal)

  Figure 2 — Performance topology:
    P_sys(x,y) — RdYlGn, standalone

Uses shared CONCEPT_POINTS and layout from _config.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _style import apply_style
from _config import CONCEPT_POINTS, CONCEPT_LABELS, SIGMA_VIS, make_topology_marginal_layout
from pred_fab.plotting._style import (
    save_fig, EMERALD_500, ZINC_600, ZINC_700, FONT,
)
from panels import (
    draw_experiments, feature_topology, performance_topology,
    setup_axes, subplot_label, clean_spines,
)
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def main(
    save_path_computation: str | None = None,
    save_path_topology: str | None = None,
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
    score_range = 0.5

    if predict_fn is None:
        def predict_fn(params):
            xn = (params.get(x_key, 0.5) - x_lo) / (x_hi - x_lo)
            yn = (params.get(y_key, 0.5) - y_lo) / (y_hi - y_lo)
            f = 0.9 * np.exp(-((xn - 0.55) ** 2 + (yn - 0.6) ** 2) / 0.18)
            f += 0.15 * yn
            return {feature_code: float(np.clip(f, 0, 1))}

    if score_fn is None:
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

    exp_x = CONCEPT_POINTS[:, 0].tolist()
    exp_y = CONCEPT_POINTS[:, 1].tolist()

    # Scoring curve
    feat_range = np.linspace(0, 1, 200)
    score_curve = np.array([
        0.5 * (1.0 + np.cos(np.pi * (f - target_value) / score_range))
        if abs(f - target_value) <= score_range else 0.0
        for f in feat_range
    ])

    # ================================================================
    # Figure 1: Performance computation — how we score
    # ================================================================
    apply_style()
    fig1, ax_feat, ax_score, ax_empty = make_topology_marginal_layout()

    feature_topology(fig1, ax_feat, xs, ys, feat_grid,
                     x_label, y_label, x_bounds, y_bounds,
                     target_value=target_value,
                     label=r"Predicted feature  $\hat{f}(x, y)$")
    ax_feat.set_aspect("equal")
    draw_experiments(ax_feat, exp_x, exp_y, sigma=SIGMA_VIS, labels=CONCEPT_LABELS)

    ax_score.fill_between(feat_range, score_curve, alpha=0.15, color=EMERALD_500)
    ax_score.plot(feat_range, score_curve, color=EMERALD_500, linewidth=2)
    ax_score.axvline(target_value, color=EMERALD_500, ls='--', lw=1, alpha=0.5)
    from pred_fab.plotting._style import ACCENT_RED
    for i, (px, py) in enumerate(zip(exp_x, exp_y)):
        f_val = predict_fn({x_key: px, y_key: py}).get(feature_code, 0.0)
        p_val = score_fn({x_key: px, y_key: py})
        ax_score.scatter([f_val], [p_val], c=ACCENT_RED, s=38,
                         edgecolors="white", linewidth=0.8, zorder=10)
        if i < len(CONCEPT_LABELS):
            ax_score.annotate(CONCEPT_LABELS[i], (f_val, p_val), xytext=(6, 6),
                              textcoords="offset points", fontsize=8,
                              color=ZINC_700)
    ax_score.set_xlim(0, 1)
    ax_score.set_ylim(-0.05, 1.1)
    ax_score.set_xlabel(r"$\hat{f}$", fontsize=FONT["axis_label"], color=ZINC_600)
    subplot_label(ax_score, r"Scoring function  $P(\hat{f})$")
    clean_spines(ax_score)

    ax_empty.set_visible(False)

    path1 = save_path_computation or str(PLOTS_DIR / "04_performance_computation.png")
    fig1.savefig(path1, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {path1}")

    # ================================================================
    # Figure 2: Performance topology — the result
    # ================================================================
    apply_style()
    fig2, ax2 = plt.subplots(figsize=(6, 5.5))
    ax2.set_aspect("equal")

    performance_topology(fig2, ax2, xs, ys, perf_grid,
                         x_label, y_label, x_bounds, y_bounds,
                         label=r"Predicted system performance  $P_{\mathrm{sys}}(x, y)$",
                         fit_colorbar=True)
    draw_experiments(ax2, exp_x, exp_y, sigma=SIGMA_VIS, labels=CONCEPT_LABELS)

    path2 = save_path_topology or str(PLOTS_DIR / "05_performance_topology.png")
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {path2}")


if __name__ == "__main__":
    main()
