"""Performance concept: single 3-panel figure.

  Panel 1: Feature topology f̂(x,y) — Greys
  Panel 2: Marginal performance P(x) + P(y) stacked — RdYlGn gradient fill
  Panel 3: Performance topology P(x,y) — RdYlGn

Usage with real model::

    cal = agent.calibration_system
    main(
        predict_fn=cal._compute_perf_dict_for_params,   # params → {perf: float}
        score_fn=cal._compute_normalised_perf_for_params, # params → combined [0,1]
        ...
    )

Both accept a raw params dict — no normalization or wrapping needed.
The plot fixes non-plotted params and sweeps the two axis params.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _style import apply_style
from pred_fab.plotting._style import save_fig
from panels import (
    draw_experiments, feature_topology, performance_topology,
    marginal_performance,
)

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def main(
    save_path: str | None = None,
    predict_fn: Callable[[dict[str, Any]], dict[str, float]] | None = None,
    score_fn: Callable[[dict[str, Any]], float] | None = None,
    feature_code: str = "extrusion_consistency",
    target_value: float = 0.65,
    x_key: str = "V_fab",
    y_key: str = "calibrationFactor",
    x_bounds: tuple[float, float] = (0.05, 0.1),
    y_bounds: tuple[float, float] = (1.8, 2.2),
    x_label: str = "Print Speed [m/s]",
    y_label: str = "Calibration Factor",
    fixed_params: dict[str, Any] | None = None,
    experiments: list[dict[str, float]] | None = None,
    resolution: int = 80,
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
            f += 0.1 * np.sin(2.5*np.pi*xn) * np.cos(1.8*np.pi*yn) * 0.2
            return {feature_code: float(np.clip(f, 0, 1))}

    if score_fn is None:
        def score_fn(params):
            features = predict_fn(params)
            f = features.get(feature_code, 0.0)
            return float(np.clip(1.0 - abs(f - target_value) / 0.7, 0, 1))

    xs = np.linspace(x_lo, x_hi, resolution)
    ys = np.linspace(y_lo, y_hi, resolution)

    feat_grid = np.zeros((resolution, resolution))
    perf_grid = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            params = dict(fixed)
            params[x_key] = float(xs[i])
            params[y_key] = float(ys[j])
            features = predict_fn(params)
            feat_grid[j, i] = features.get(feature_code, 0.0)
            perf_grid[j, i] = score_fn(params)

    perf_along_x = np.mean(perf_grid, axis=0)
    perf_along_y = np.mean(perf_grid, axis=1)

    if experiments is None:
        np.random.seed(42)
        experiments = [
            {x_key: v, y_key: c}
            for v, c in zip(np.random.uniform(x_lo+0.005, x_hi-0.005, 8),
                            np.random.uniform(y_lo+0.05, y_hi-0.05, 8))
        ]
    exp_x = [e[x_key] for e in experiments]
    exp_y = [e[y_key] for e in experiments]

    # ── Single figure: 3 panels ──
    apply_style()
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 0.7, 1.3],
                          hspace=0.5, wspace=0.25,
                          left=0.05, right=0.96, top=0.92, bottom=0.10)

    # Panel 1: Feature topology (spans both rows)
    ax_feat = fig.add_subplot(gs[:, 0])
    feature_topology(fig, ax_feat, xs, ys, feat_grid,
                     x_label, y_label, x_bounds, y_bounds,
                     target_value=target_value)
    draw_experiments(ax_feat, exp_x, exp_y)

    # Panel 2 top: Marginal P(x)
    ax_mx = fig.add_subplot(gs[0, 1])
    marginal_performance(ax_mx, xs, perf_along_x, x_label, "$P(x)$")

    # Panel 2 bottom: TBD
    ax_placeholder = fig.add_subplot(gs[1, 1])
    ax_placeholder.set_visible(False)

    # Panel 3: Performance topology (spans both rows)
    ax_perf = fig.add_subplot(gs[:, 2])
    performance_topology(fig, ax_perf, xs, ys, perf_grid,
                         x_label, y_label, x_bounds, y_bounds)
    draw_experiments(ax_perf, exp_x, exp_y)

    path = save_path or str(PLOTS_DIR / "performance_concept.png")
    save_fig(path, dpi=200)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
