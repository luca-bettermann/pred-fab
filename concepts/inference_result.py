"""Inference concept: performance topology + predicted attribute radar.

Two panels:
  1. Performance topology with inference proposal marked
  2. Radar chart showing predicted performance attributes at the proposal

Usage with real model::

    cal = agent.calibration_system
    main(
        score_fn=cal._compute_normalised_perf_for_params,  # params → combined [0,1]
        attribute_fn=cal._compute_perf_dict_for_params,    # params → {perf: float}
        ...
    )

All callables accept a raw params dict. No wrapping needed.
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
from panels import draw_experiments, draw_proposal, performance_topology, setup_axes

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def _radar_panel(
    ax,
    attribute_names: list[str],
    attribute_scores: list[float],
    score: float | None = None,
    score_std: float | None = None,
):
    """Radar chart — delegates to the shared radar_chart function."""
    from pred_fab.plotting.performance import radar_chart
    radar_chart(ax, attribute_names, attribute_scores,
                score=score, score_std=score_std)


def main(
    proposed_params: dict[str, Any],
    proposed_score: float,
    save_path: str | None = None,
    score_fn: Callable[[dict[str, Any]], float] | None = None,
    attribute_fn: Callable[[dict[str, Any]], dict[str, float | None]] | None = None,
    x_key: str = "V_fab",
    y_key: str = "calibrationFactor",
    x_bounds: tuple[float, float] = (0.05, 0.1),
    y_bounds: tuple[float, float] = (1.8, 2.2),
    x_label: str = "Print Speed [m/s]",
    y_label: str = "Calibration Factor",
    fixed_params: dict[str, Any] | None = None,
    datapoints: list[dict[str, float]] | None = None,
    fit_colorbar: bool = True,
    resolution: int = 60,
):
    fixed = fixed_params or {}
    x_lo, x_hi = x_bounds
    y_lo, y_hi = y_bounds

    if score_fn is None:
        def score_fn(params):
            xn = (params.get(x_key, 0.075) - x_lo) / (x_hi - x_lo)
            yn = (params.get(y_key, 2.0) - y_lo) / (y_hi - y_lo)
            f = 0.7 * np.exp(-((xn-0.4)**2 + (yn-0.55)**2) / 0.15)
            f += 0.35 * np.exp(-((xn-0.8)**2 + (yn-0.25)**2) / 0.2)
            return float(np.clip(1.0 - abs(f - 0.65) / 0.7, 0, 1))

    if attribute_fn is None:
        def attribute_fn(params):
            xn = (params.get(x_key, 0.075) - x_lo) / (x_hi - x_lo)
            yn = (params.get(y_key, 2.0) - y_lo) / (y_hi - y_lo)
            f = 0.7 * np.exp(-((xn-0.4)**2 + (yn-0.55)**2) / 0.15)
            return {
                "structural\nintegrity": float(np.clip(f * 1.1, 0, 1)),
                "material\ndeposition": float(np.clip(0.9 - abs(f - 0.5) * 0.8, 0, 1)),
                "extrusion\nstability": float(np.clip(f * 0.95 + 0.05, 0, 1)),
                "energy\nfootprint": float(np.clip(1.0 - xn * 0.4, 0, 1)),
                "fabrication\ntime": float(np.clip(0.85 - xn * 0.3, 0, 1)),
            }

    xs = np.linspace(x_lo, x_hi, resolution)
    ys = np.linspace(y_lo, y_hi, resolution)

    perf_grid = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            params = dict(fixed)
            params[x_key] = float(xs[i])
            params[y_key] = float(ys[j])
            perf_grid[j, i] = score_fn(params)

    exp_x = [d[x_key] for d in datapoints] if datapoints else []
    exp_y = [d[y_key] for d in datapoints] if datapoints else []

    opt_xv = float(proposed_params[x_key])
    opt_yv = float(proposed_params[y_key])
    eval_params = dict(fixed)
    eval_params.update(proposed_params)
    attr_dict = attribute_fn(eval_params)
    attr_names = list(attr_dict.keys())
    attr_scores = [float(attr_dict[k] or 0) for k in attr_names]

    # ── Figure: topology + radar ──
    apply_style()
    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.25,
                          left=0.06, right=0.95, top=0.92, bottom=0.08)

    ax_topo = fig.add_subplot(gs[0, 0])
    performance_topology(fig, ax_topo, xs, ys, perf_grid,
                         x_label, y_label, x_bounds, y_bounds,
                         show_optimum=False,
                         label="Predicted $S(\\mathbf{x})$",
                         fit_colorbar=fit_colorbar)
    if exp_x:
        draw_experiments(ax_topo, exp_x, exp_y)
    draw_proposal(ax_topo, opt_xv, opt_yv)

    ax_radar = fig.add_subplot(gs[0, 1], polar=True)
    _radar_panel(ax_radar, attr_names, attr_scores, score=proposed_score)

    path = save_path or str(PLOTS_DIR / "inference_result.png")
    save_fig(path, dpi=200)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
