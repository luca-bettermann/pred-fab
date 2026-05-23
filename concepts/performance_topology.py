"""Performance concept: three equal panels explaining the performance pipeline.

  Panel 1: Feature prediction f̂(x,y) — what the model predicts (Greys)
  Panel 2: Marginal performance P(x) — how predictions become scores
  Panel 3: System performance P_sys(x,y) — the combined objective (RdYlGn)

All synthetic data — explains the pipeline, not domain-specific results.
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
    feature_code: str = "feature",
    target_value: float = 0.65,
    x_key: str = "x",
    y_key: str = "y",
    x_bounds: tuple[float, float] = (0.0, 1.0),
    y_bounds: tuple[float, float] = (0.0, 1.0),
    x_label: str = "$x$",
    y_label: str = "$y$",
    fixed_params: dict[str, Any] | None = None,
    datapoints: list[dict[str, float]] | None = None,
    fit_colorbar: bool = True,
    resolution: int = 80,
):
    fixed = fixed_params or {}
    x_lo, x_hi = x_bounds
    y_lo, y_hi = y_bounds

    if predict_fn is None:
        def predict_fn(params):
            xn = (params.get(x_key, 0.5) - x_lo) / (x_hi - x_lo)
            yn = (params.get(y_key, 0.5) - y_lo) / (y_hi - y_lo)
            f = 0.7 * np.exp(-((xn - 0.4) ** 2 + (yn - 0.55) ** 2) / 0.15)
            f += 0.35 * np.exp(-((xn - 0.8) ** 2 + (yn - 0.25) ** 2) / 0.2)
            f += 0.1 * np.sin(2.5 * np.pi * xn) * np.cos(1.8 * np.pi * yn) * 0.2
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

    exp_x = [d[x_key] for d in datapoints] if datapoints else []
    exp_y = [d[y_key] for d in datapoints] if datapoints else []

    apply_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5))
    fig.subplots_adjust(wspace=0.30, left=0.04, right=0.97, bottom=0.12, top=0.90)

    feature_topology(fig, ax1, xs, ys, feat_grid,
                     x_label, y_label, x_bounds, y_bounds,
                     target_value=target_value,
                     label=r"$\hat{f}(x, y)$")
    if exp_x:
        draw_experiments(ax1, exp_x, exp_y)

    marginal_performance(ax2, xs, perf_along_x, x_label,
                         r"$\bar{P}(x)$", fit_colorbar=fit_colorbar)

    performance_topology(fig, ax3, xs, ys, perf_grid,
                         x_label, y_label, x_bounds, y_bounds,
                         label="$P_{\\mathrm{sys}}(x, y)$",
                         fit_colorbar=fit_colorbar)
    if exp_x:
        draw_experiments(ax3, exp_x, exp_y)

    path = save_path or str(PLOTS_DIR / "performance_concept.png")
    save_fig(path, dpi=200)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
