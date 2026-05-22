"""Inference concept: performance topology + predicted attribute radar.

Two panels:
  1. Performance topology with inference proposal marked
  2. Radar chart showing predicted performance attributes at the proposal

Uses mock data by default. Pass predict_fn, score_fn, attribute_fns for real figures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _style import apply_style, clean_spines, subplot_label, style_colorbar, cmap
from pred_fab.plotting._style import (
    FONT, ZINC_300, ZINC_400, ZINC_500, ZINC_600, ZINC_700,
    ACCENT_YELLOW, STEEL_500, EMERALD_500, save_fig,
)
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from panels import draw_experiments, performance_topology, setup_axes

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def _radar_panel(
    ax,
    attribute_names: list[str],
    attribute_scores: list[float],
    label: str = "Predicted attributes",
):
    """Radar chart for predicted performance attributes at the inference proposal."""
    n = len(attribute_names)
    if n < 3:
        return

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    scores = list(attribute_scores) + [attribute_scores[0]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    ax.plot(angles, scores, color=EMERALD_500, linewidth=1.8)
    ax.fill(angles, scores, color=EMERALD_500, alpha=0.15)

    ax.set_xticks(angles[:-1])
    short_names = [n[:12] for n in attribute_names]
    ax.set_xticklabels(short_names, fontsize=FONT["tick"], color=ZINC_600)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                       fontsize=6, color=ZINC_400)
    ax.spines["polar"].set_color(ZINC_300)
    ax.grid(color=ZINC_300, linewidth=0.4, alpha=0.5)

    for angle, score, name in zip(angles[:-1], attribute_scores, attribute_names):
        ax.scatter([angle], [score], c=EMERALD_500, s=20, zorder=5,
                   edgecolors="white", linewidth=0.5)


def main(
    predict_fn: Callable[[dict[str, Any]], dict[str, float]] | None = None,
    score_fn: Callable[[dict[str, float], dict[str, Any]], float] | None = None,
    attribute_fns: dict[str, Callable[[dict[str, float], dict[str, Any]], float]] | None = None,
    x_key: str = "V_fab",
    y_key: str = "calibrationFactor",
    x_bounds: tuple[float, float] = (0.05, 0.1),
    y_bounds: tuple[float, float] = (1.8, 2.2),
    x_label: str = "Print Speed [m/s]",
    y_label: str = "Calibration Factor",
    fixed_params: dict[str, Any] | None = None,
    experiments: list[dict[str, float]] | None = None,
    resolution: int = 60,
):
    fixed = fixed_params or {}
    x_lo, x_hi = x_bounds
    y_lo, y_hi = y_bounds

    if predict_fn is None:
        def predict_fn(params):
            xn = (params.get(x_key, 0.075) - x_lo) / (x_hi - x_lo)
            yn = (params.get(y_key, 2.0) - y_lo) / (y_hi - y_lo)
            f = 0.7 * np.exp(-((xn-0.4)**2 + (yn-0.55)**2) / 0.15)
            f += 0.35 * np.exp(-((xn-0.8)**2 + (yn-0.25)**2) / 0.2)
            return {"feature": float(np.clip(f, 0, 1))}

    if score_fn is None:
        def score_fn(features, params):
            f = features.get("feature", 0.0)
            return float(np.clip(1.0 - abs(f - 0.65) / 0.7, 0, 1))

    if attribute_fns is None:
        attribute_fns = {
            "structural\nintegrity": lambda feat, p: float(np.clip(feat.get("feature", 0) * 1.1, 0, 1)),
            "material\ndeposition": lambda feat, p: float(np.clip(0.9 - abs(feat.get("feature", 0) - 0.5) * 0.8, 0, 1)),
            "extrusion\nstability": lambda feat, p: float(np.clip(feat.get("feature", 0) * 0.95 + 0.05, 0, 1)),
            "energy\nfootprint": lambda feat, p: float(np.clip(1.0 - (p.get(x_key, 0.075) - x_lo) / (x_hi - x_lo) * 0.4, 0, 1)),
            "fabrication\ntime": lambda feat, p: float(np.clip(0.85 - (p.get(x_key, 0.075) - x_lo) / (x_hi - x_lo) * 0.3, 0, 1)),
        }

    xs = np.linspace(x_lo, x_hi, resolution)
    ys = np.linspace(y_lo, y_hi, resolution)

    perf_grid = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            params = dict(fixed)
            params[x_key] = float(xs[i])
            params[y_key] = float(ys[j])
            features = predict_fn(params)
            perf_grid[j, i] = score_fn(features, params)

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

    opt_params = dict(fixed)
    opt_params[x_key] = float(opt_xv)
    opt_params[y_key] = float(opt_yv)
    opt_features = predict_fn(opt_params)

    attr_names = list(attribute_fns.keys())
    attr_scores = [attribute_fns[name](opt_features, opt_params) for name in attr_names]

    # ── Figure: topology + radar ──
    apply_style()
    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.25,
                          left=0.06, right=0.95, top=0.92, bottom=0.08)

    ax_topo = fig.add_subplot(gs[0, 0])
    performance_topology(fig, ax_topo, xs, ys, perf_grid,
                         x_label, y_label, x_bounds, y_bounds,
                         show_optimum=False)
    draw_experiments(ax_topo, exp_x, exp_y)
    ax_topo.scatter([opt_xv], [opt_yv], marker="x", c=ACCENT_YELLOW, s=65,
                    linewidths=1.2, zorder=12)
    ax_topo.annotate("inference\nproposal", (opt_xv, opt_yv),
                     xytext=(10, 10), textcoords="offset points",
                     fontsize=7, color=ZINC_700,
                     arrowprops=dict(arrowstyle="->", color=ZINC_400, lw=0.8))

    ax_radar = fig.add_subplot(gs[0, 1], polar=True)
    _radar_panel(ax_radar, attr_names, attr_scores)

    path = PLOTS_DIR / "inference_result.png"
    save_fig(str(path), dpi=200)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
