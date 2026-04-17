"""Baseline phase plots: parameter space coverage and initial model comparison."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from ._style import AxisSpec, save_fig, _extract_xy, _apply_axes, _add_fixed_subtitle, STEEL_500


def plot_parameter_space(
    save_path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    x_values: np.ndarray,
    y_values: np.ndarray,
    points: list[dict[str, Any]],
    true_grid: np.ndarray,
    pred_grid: np.ndarray,
    *,
    title: str = "Baseline",
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """1x3: parameter space scatter + ground truth topology + initial model topology."""
    px, py = _extract_xy(points, x_axis, y_axis)
    n = len(px)
    vmin, vmax = true_grid.min(), true_grid.max()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.5))
    fig.suptitle(f"{title} ({n} experiments)", fontsize=14, fontweight="bold", y=1.02)
    _add_fixed_subtitle(fig, fixed_params)

    # Panel 1: parameter space
    ax1.scatter(px, py, s=60, c=STEEL_500, edgecolors="white", linewidth=0.8, zorder=5)
    for i, (x, y) in enumerate(zip(px, py)):
        ax1.annotate(f"{i+1}", (x, y), fontsize=6, ha="center", va="bottom",
                     xytext=(0, 5), textcoords="offset points", color="#666")
    _apply_axes(ax1, x_axis, y_axis)
    ax1.set_title("Parameter Space", fontsize=10)
    ax1.grid(True, alpha=0.2)

    # Panel 2: ground truth
    im2 = ax2.contourf(x_values, y_values, true_grid, levels=20, cmap="RdYlGn",
                        vmin=vmin, vmax=vmax)
    ax2.contour(x_values, y_values, true_grid, levels=10, colors="white",
                linewidths=0.3, alpha=0.5)
    ax2.scatter(px, py, s=20, c="white", edgecolors="black", linewidth=0.5, zorder=5)
    _apply_axes(ax2, x_axis, y_axis)
    ax2.set_title("Ground Truth", fontsize=10)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Panel 3: initial model
    im3 = ax3.contourf(x_values, y_values, pred_grid, levels=20, cmap="RdYlGn",
                        vmin=vmin, vmax=vmax)
    ax3.contour(x_values, y_values, pred_grid, levels=10, colors="white",
                linewidths=0.3, alpha=0.5)
    ax3.scatter(px, py, s=20, c="white", edgecolors="black", linewidth=0.5, zorder=5)
    _apply_axes(ax3, x_axis, y_axis)
    ax3.set_title("Initial Model", fontsize=10)
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    save_fig(save_path)
