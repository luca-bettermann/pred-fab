"""Performance concept: predicted feature topology + marginal scoring + performance topology.

Layout mirrors evidence_integration.py figure 2:
  Plot 1: Joint feature topology (left) + marginal performance P(x), P(y) (right, stacked)
  Plot 2: Performance topology (standalone, same size as evidence gain plot)

Uses mock data by default. Wire in real predict_fn / score_fn for paper figures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from _style import (
    apply_style, clean_spines, subplot_label, style_colorbar, cmap,
    ZINC_300, ZINC_400, ZINC_500, ZINC_600, ZINC_700,
    ACCENT_YELLOW, RED,
)
from pred_fab.plotting._style import (
    SURFACES, FILL_ALPHA, FONT, MARKERS, LINES,
    surface as get_surface,
)

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def _marginal_performance_panels(
    ax_x, ax_y,
    xs, ys,
    perf_along_x, perf_along_y,
    x_label: str, y_label: str,
    experiments_x: list[float] | None = None,
    experiments_y: list[float] | None = None,
):
    """Draw marginal performance curves with gradient fill, mirroring _draw_marginal_panels."""
    surf = get_surface("performance")
    cm_p = cmap("performance")
    line_color = cm_p(0.7)
    y_max = 1.0

    def _gradient_fill(ax, xs_arr, curve, y_max_val, cm_obj):
        res_y = 100
        extent = [float(xs_arr[0]), float(xs_arr[-1]), 0, y_max_val]
        gradient = np.linspace(0, 1, res_y).reshape(-1, 1) * np.ones((1, len(xs_arr)))
        curve_norm = curve / y_max_val if y_max_val > 0 else curve
        gradient = gradient * curve_norm[None, :]
        norm_fill = Normalize(vmin=0, vmax=1)
        ax.imshow(gradient, aspect="auto", origin="lower", extent=extent,
                  cmap=cm_obj, norm=norm_fill, alpha=0.7, zorder=0)
        ax.fill_between(xs_arr, curve, y_max_val, color="white", zorder=1)

    _gradient_fill(ax_x, xs, perf_along_x, y_max, cm_p)
    ax_x.plot(xs, perf_along_x, color=line_color, linewidth=1.5)
    ax_x.set_xlim(float(xs[0]), float(xs[-1]))
    ax_x.set_ylim(0, y_max)
    ax_x.set_xlabel(x_label, fontsize=FONT["axis_label"], color=ZINC_600)
    clean_spines(ax_x)

    _gradient_fill(ax_y, ys, perf_along_y, y_max, cm_p)
    ax_y.plot(ys, perf_along_y, color=line_color, linewidth=1.5)
    ax_y.set_xlim(float(ys[0]), float(ys[-1]))
    ax_y.set_ylim(0, y_max)
    ax_y.set_xlabel(y_label, fontsize=FONT["axis_label"], color=ZINC_600)
    clean_spines(ax_y)


def main(
    predict_fn: Callable[[dict[str, Any]], dict[str, float]] | None = None,
    score_fn: Callable[[dict[str, float], dict[str, Any]], float] | None = None,
    feature_code: str = "extrusion_consistency",
    target_value: float = 0.65,
    scaling: float = 0.7,
    x_key: str = "V_fab",
    y_key: str = "calibrationFactor",
    x_bounds: tuple[float, float] = (0.05, 0.1),
    y_bounds: tuple[float, float] = (1.8, 2.2),
    x_label: str = "Print Speed [m/s]",
    y_label: str = "Calibration Factor",
    fixed_params: dict[str, Any] | None = None,
    experiments: list[dict[str, float]] | None = None,
):
    """Generate both plots. Pass predict_fn/score_fn for real model; None uses mock."""

    fixed = fixed_params or {}
    x_lo, x_hi = x_bounds
    y_lo, y_hi = y_bounds
    res = 80

    # Mock model if not provided
    if predict_fn is None:
        def predict_fn(params):
            xn = (params.get(x_key, 0.075) - x_lo) / (x_hi - x_lo)
            yn = (params.get(y_key, 2.0) - y_lo) / (y_hi - y_lo)
            f = 0.7 * np.exp(-((xn-0.4)**2 + (yn-0.55)**2) / 0.15)
            f += 0.35 * np.exp(-((xn-0.8)**2 + (yn-0.25)**2) / 0.2)
            f += 0.1 * np.sin(2.5*np.pi*xn) * np.cos(1.8*np.pi*yn) * 0.2
            return {feature_code: float(np.clip(f, 0, 1))}

    if score_fn is None:
        def score_fn(features, params):
            f = features.get(feature_code, 0.0)
            return float(np.clip(1.0 - abs(f - target_value) / scaling, 0, 1))

    xs = np.linspace(x_lo, x_hi, res)
    ys = np.linspace(y_lo, y_hi, res)
    XX, YY = np.meshgrid(xs, ys)

    feat_grid = np.zeros((res, res))
    perf_grid = np.zeros((res, res))
    for i in range(res):
        for j in range(res):
            params = dict(fixed)
            params[x_key] = float(xs[i])
            params[y_key] = float(ys[j])
            features = predict_fn(params)
            feat_grid[j, i] = features.get(feature_code, 0.0)
            perf_grid[j, i] = score_fn(features, params)

    perf_along_x = np.mean(perf_grid, axis=0)
    perf_along_y = np.mean(perf_grid, axis=1)

    # Mock experiments if not provided
    if experiments is None:
        np.random.seed(42)
        experiments = [
            {x_key: v, y_key: c}
            for v, c in zip(np.random.uniform(x_lo+0.005, x_hi-0.005, 8),
                            np.random.uniform(y_lo+0.05, y_hi-0.05, 8))
        ]
    exp_x = [e[x_key] for e in experiments]
    exp_y = [e[y_key] for e in experiments]

    opt_idx = np.unravel_index(np.argmax(perf_grid), perf_grid.shape)
    opt_xv, opt_yv = xs[opt_idx[1]], ys[opt_idx[0]]

    # ================================================================
    # Plot 1: Feature topology + marginal performance
    # Matches evidence_integration.py fig2 layout exactly
    # ================================================================
    apply_style()
    fig1 = plt.figure(figsize=(11, 5))
    gs = fig1.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.5, wspace=0.15,
                           left=0.06, right=0.95, top=0.92, bottom=0.12)
    ax_joint = fig1.add_subplot(gs[:, 0])
    ax_mx = fig1.add_subplot(gs[0, 1])
    ax_my = fig1.add_subplot(gs[1, 1])

    # Joint feature topology — Greys (raw data, no semantic judgment)
    cm_f = cmap("density")
    norm_f = Normalize(vmin=0.0, vmax=float(feat_grid.max()))
    levels_f = np.linspace(0.02, float(feat_grid.max()), 18)
    im = ax_joint.contourf(xs, ys, feat_grid, levels=levels_f, cmap=cm_f, norm=norm_f)
    ax_joint.contour(xs, ys, feat_grid, levels=8, colors=[ZINC_300], linewidths=0.4)

    # Target contour
    tc = ax_joint.contour(xs, ys, feat_grid, levels=[target_value],
                          colors=[ZINC_300], linewidths=1.2, linestyles="--")
    ax_joint.clabel(tc, fmt=f"t={target_value:.2f}", fontsize=7, colors=ZINC_400)

    # Experiments — white dots, Zinc-700 edge (baseline style)
    for ex, ey in zip(exp_x, exp_y):
        ax_joint.scatter([ex], [ey], c="white", s=30, edgecolors=ZINC_700,
                         linewidth=0.5, zorder=10)

    ax_joint.set_xlim(x_lo, x_hi)
    ax_joint.set_ylim(y_lo, y_hi)
    ax_joint.set_xlabel(x_label, fontsize=FONT["axis_label"], color=ZINC_600)
    ax_joint.set_ylabel(y_label, fontsize=FONT["axis_label"], color=ZINC_600)
    subplot_label(ax_joint, f"$\\hat{{f}}(x, y)$")
    clean_spines(ax_joint)
    sm = ScalarMappable(norm=norm_f, cmap=cm_f)
    cb = fig1.colorbar(sm, ax=ax_joint, shrink=0.85, pad=0.06)
    style_colorbar(cb)

    # Marginal performance panels
    subplot_label(ax_mx, "$P(x)$")
    subplot_label(ax_my, "$P(y)$")
    _marginal_performance_panels(ax_mx, ax_my, xs, ys,
                                  perf_along_x, perf_along_y,
                                  x_label, y_label)

    path1 = PLOTS_DIR / "performance_prediction.png"
    fig1.savefig(path1, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {path1}")

    # ================================================================
    # Plot 2: Performance topology (standalone)
    # Matches evidence_integration.py fig3 (evidence gain) layout
    # ================================================================
    apply_style()
    fig2, ax2 = plt.subplots(figsize=(6, 5.5))

    cm_perf = cmap("performance")
    norm_p = Normalize(vmin=0, vmax=1)
    ax2.contourf(xs, ys, perf_grid, levels=18, cmap=cm_perf, norm=norm_p)
    ax2.contour(xs, ys, perf_grid, levels=8, colors="white", linewidths=0.3, alpha=0.5)

    for ex, ey in zip(exp_x, exp_y):
        ax2.scatter([ex], [ey], c="white", s=30, edgecolors=ZINC_700,
                    linewidth=0.5, zorder=10)

    # Optimum
    ax2.scatter([opt_xv], [opt_yv], marker="x", c=ACCENT_YELLOW, s=55,
                linewidths=1.0, zorder=10)
    ax2.annotate("$x^*$", (opt_xv, opt_yv), xytext=(8, 8), textcoords="offset points",
                 fontsize=8, color=ZINC_700,
                 arrowprops=dict(arrowstyle="->", color=ZINC_400, lw=0.8))

    ax2.set_xlim(x_lo, x_hi)
    ax2.set_ylim(y_lo, y_hi)
    ax2.set_xlabel(x_label, fontsize=FONT["axis_label"], color=ZINC_600)
    ax2.set_ylabel(y_label, fontsize=FONT["axis_label"], color=ZINC_600)
    subplot_label(ax2, "$P(x, y)$ topology")
    clean_spines(ax2)
    sm2 = ScalarMappable(norm=norm_p, cmap=cm_perf)
    cb2 = fig2.colorbar(sm2, ax=ax2, shrink=0.85, pad=0.06)
    style_colorbar(cb2)

    path2 = PLOTS_DIR / "performance_topology.png"
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {path2}")


if __name__ == "__main__":
    main()
