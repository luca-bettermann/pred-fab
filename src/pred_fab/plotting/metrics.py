"""Metric-level plot: per-metric topology breakdown."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from ._style import AxisSpec, save_fig, _add_fixed_subtitle, apply_style, subplot_topology


def plot_metric_topology(
    save_path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    x_values: np.ndarray,
    y_values: np.ndarray,
    metric_grids: dict[str, np.ndarray],
    combined_grid: np.ndarray,
    combined_label: str = "Combined",
    *,
    weights: dict[str, float] | None = None,
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """1x(N+1) heatmap: individual metrics (YlGn) + combined (RdYlGn)."""
    apply_style()
    n_panels = len(metric_grids) + 1

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))
    _add_fixed_subtitle(fig, fixed_params)

    optima: dict[str, tuple[float, float]] = {}

    for ax, (name, data) in zip(axes[:-1], metric_grids.items()):
        label = f"{name} (w={weights[name]:g})" if weights and name in weights else name
        subplot_topology(ax, x_axis, y_axis, x_values, y_values, data,
                         cmap_name="YlGn", label=label)

        best_idx = np.unravel_index(np.argmax(data), data.shape)
        opt_x, opt_y = float(x_values[best_idx[1]]), float(y_values[best_idx[0]])
        optima[name] = (opt_x, opt_y)
        ax.plot(opt_x, opt_y, "*", color="white", ms=14,
                markeredgecolor="black", markeredgewidth=0.8, zorder=8)

    ax_c = axes[-1]
    subplot_topology(ax_c, x_axis, y_axis, x_values, y_values, combined_grid,
                     cmap_name="performance", label=combined_label)

    for ox, oy in optima.values():
        ax_c.plot(ox, oy, "o", color="white", ms=6,
                  markeredgecolor="black", markeredgewidth=0.6, zorder=8)

    best_idx = np.unravel_index(np.argmax(combined_grid), combined_grid.shape)
    cx, cy = float(x_values[best_idx[1]]), float(y_values[best_idx[0]])
    ax_c.plot(cx, cy, "*", color="white", ms=16,
              markeredgecolor="black", markeredgewidth=0.8, zorder=9)

    save_fig(save_path)
