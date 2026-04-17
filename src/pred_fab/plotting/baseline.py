"""Baseline phase plots: parameter space coverage and initial model comparison."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

from ._style import (
    AxisSpec, save_fig, _extract_xy, _apply_axes, _add_fixed_subtitle,
    STEEL_500,
)


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


# Steel gradient colormap for z-axis encoding (data points use Steel spectrum)
_STEEL_CMAP = LinearSegmentedColormap.from_list(
    "steel", ["#D6E4F0", "#8BB0CC", "#4A7FA5", "#2D5F85", "#1A3A5C"]
)


def plot_parameter_space_3d(
    save_path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    z_axis: AxisSpec,
    points: list[dict[str, Any]],
    *,
    title: str = "Parameter Space",
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """3D scatter where point color encodes the z-axis value on a Zinc gradient."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    px = [float(p[x_axis.key]) for p in points]
    py = [float(p[y_axis.key]) for p in points]
    pz = [float(p[z_axis.key]) for p in points]
    n = len(px)

    z_arr = np.array(pz)
    if z_axis.bounds:
        vmin, vmax = z_axis.bounds
    else:
        vmin, vmax = float(z_arr.min()), float(z_arr.max())
    # Avoid zero-span when all z values are identical
    if vmax - vmin < 1e-12:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle(f"{title} ({n} experiments)", fontsize=13, fontweight="bold")
    _add_fixed_subtitle(fig, fixed_params)

    sc = ax.scatter(
        px, py, pz,  # type: ignore[arg-type]
        c=pz, cmap=_STEEL_CMAP, norm=norm,
        s=50, edgecolors="white", linewidth=0.6, zorder=5, depthshade=False,
    )

    for i, (x, y, z) in enumerate(zip(px, py, pz)):
        ax.text(x, y, z, f" {i+1}", fontsize=6, color="#666", zorder=6)

    ax.set_xlabel(x_axis.display_label, labelpad=8, fontsize=9)
    ax.set_ylabel(y_axis.display_label, labelpad=8, fontsize=9)
    ax.set_zlabel(z_axis.display_label, labelpad=8, fontsize=9)  # type: ignore[attr-defined]

    if x_axis.bounds:
        ax.set_xlim(*x_axis.bounds)
    if y_axis.bounds:
        ax.set_ylim(*y_axis.bounds)
    if z_axis.bounds:
        ax.set_zlim(*z_axis.bounds)  # type: ignore[attr-defined]

    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label(z_axis.display_label, fontsize=9)

    ax.view_init(elev=25, azim=-50)

    save_fig(save_path)
