"""Exploration phase plots: acquisition objective."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from ._style import (
    AxisSpec, save_fig, _add_fixed_subtitle,
    apply_style, subplot_topology,
    ACCENT_YELLOW,
)


def plot_acquisition(
    save_path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    x_values: np.ndarray,
    y_values: np.ndarray,
    perf_grid: np.ndarray,
    unc_grid: np.ndarray,
    combined_grid: np.ndarray,
    *,
    points: list[dict[str, Any]] | None = None,
    proposed: dict[str, Any] | None = None,
    schedules: dict[str, list[dict[str, Any]]] | None = None,
    codes: list[str] | None = None,
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """3-panel: performance | evidence | combined acquisition."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _add_fixed_subtitle(fig, fixed_params)

    panels = [
        (axes[0], perf_grid, "Performance", "performance"),
        (axes[1], unc_grid, "Evidence", "evidence"),
        (axes[2], combined_grid, "Combined", "mixed"),
    ]
    for ax, grid, label, cmap_name in panels:
        subplot_topology(ax, x_axis, y_axis, x_values, y_values, grid,
                         cmap_name=cmap_name, label=label,
                         points=points, schedules=schedules, codes=codes,
                         point_size=18)

    if proposed is not None:
        axes[2].plot(proposed[x_axis.key], proposed[y_axis.key],
                     "x", color=ACCENT_YELLOW, ms=10,
                     markeredgewidth=2, zorder=8, label="Proposed")
        axes[2].legend(fontsize=7, loc="upper left", framealpha=0.8)

    save_fig(save_path)
