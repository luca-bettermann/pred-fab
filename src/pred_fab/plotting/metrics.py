"""Metric-level plots: topology breakdown, cross-sections, sensitivity."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from ._style import (
    AxisSpec, save_fig, _apply_axes, _add_fixed_subtitle,
    STEEL_500, ZINC_600,
)


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
    title: str = "Performance Topology",
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """1x(N+1) heatmap: individual metrics (YlGn) + combined (RdYlGn)."""
    n_panels = len(metric_grids) + 1

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    _add_fixed_subtitle(fig, fixed_params)

    optima: dict[str, tuple[float, float]] = {}

    for ax, (name, data) in zip(axes[:-1], metric_grids.items()):
        im = ax.contourf(x_values, y_values, data, levels=20, cmap="YlGn")
        ax.contour(x_values, y_values, data, levels=10, colors="white",
                    linewidths=0.3, alpha=0.5)

        best_idx = np.unravel_index(np.argmax(data), data.shape)
        opt_x, opt_y = float(x_values[best_idx[1]]), float(y_values[best_idx[0]])
        optima[name] = (opt_x, opt_y)
        ax.plot(opt_x, opt_y, "*", color="white", ms=14,
                markeredgecolor="black", markeredgewidth=0.8, zorder=8)

        label = name
        if weights and name in weights:
            label = f"{name} (w={weights[name]:g})"
        ax.set_title(label, fontsize=10)
        _apply_axes(ax, x_axis, y_axis)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Combined panel
    ax_c = axes[-1]
    im_c = ax_c.contourf(x_values, y_values, combined_grid, levels=20, cmap="RdYlGn")
    ax_c.contour(x_values, y_values, combined_grid, levels=10, colors="white",
                  linewidths=0.3, alpha=0.5)

    for ox, oy in optima.values():
        ax_c.plot(ox, oy, "o", color="white", ms=6,
                  markeredgecolor="black", markeredgewidth=0.6, zorder=8)

    best_idx = np.unravel_index(np.argmax(combined_grid), combined_grid.shape)
    cx, cy = float(x_values[best_idx[1]]), float(y_values[best_idx[0]])
    ax_c.plot(cx, cy, "*", color="white", ms=16,
              markeredgecolor="black", markeredgewidth=0.8, zorder=9)

    ax_c.set_title(combined_label, fontsize=10)
    _apply_axes(ax_c, x_axis, y_axis)
    plt.colorbar(im_c, ax=ax_c, shrink=0.8)

    save_fig(save_path)


def plot_cross_sections(
    save_path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    x_values: np.ndarray,
    y_values: np.ndarray,
    grids: dict[str, np.ndarray],
    slice_point: dict[str, float],
    *,
    title: str = "Cross-Sections",
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """1x2: cross-sections through a point, varying each axis while fixing the other."""
    x_val = slice_point[x_axis.key]
    y_val = slice_point[y_axis.key]
    x_idx = int(np.argmin(np.abs(x_values - x_val)))
    y_idx = int(np.argmin(np.abs(y_values - y_val)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    _add_fixed_subtitle(fig, fixed_params)

    for name, data in grids.items():
        ax1.plot(y_values, data[:, x_idx], label=name, lw=2)
        ax2.plot(x_values, data[y_idx, :], label=name, lw=2)

    ax1.axvline(y_val, color="gray", ls="--", lw=1,
                label=f"Slice ({y_val:.1f})")
    ax1.set_xlabel(y_axis.display_label)
    ax1.set_ylabel("Score [0\u20131]")
    ax1.set_title(f"{x_axis.label} = {x_val:.3f} (fixed)")
    ax1.legend(fontsize=7, loc="lower left")
    ax1.grid(True, alpha=0.2)

    ax2.axvline(x_val, color="gray", ls="--", lw=1,
                label=f"Slice ({x_val:.3f})")
    ax2.set_xlabel(x_axis.display_label)
    ax2.set_ylabel("Score [0\u20131]")
    ax2.set_title(f"{y_axis.label} = {y_val:.1f} (fixed)")
    ax2.legend(fontsize=7, loc="lower left")
    ax2.grid(True, alpha=0.2)

    save_fig(save_path)


def plot_sensitivity(
    save_path: str,
    sensitivities: dict[str, float],
    *,
    title: str = "Local Sensitivity Analysis",
) -> None:
    """Horizontal bar chart of parameter sensitivity magnitudes."""
    codes = list(sensitivities.keys())
    values = [sensitivities[c] for c in codes]

    sorted_pairs = sorted(zip(codes, values), key=lambda x: x[1], reverse=True)
    codes = [p[0] for p in sorted_pairs]
    values = [p[1] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(codes))))
    y_pos = np.arange(len(codes))
    ax.barh(y_pos, values, height=0.6, color=STEEL_500,
            edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(codes, fontsize=9, color=ZINC_600)
    ax.invert_yaxis()
    ax.set_xlabel("|\u2202combined/\u2202param|", fontsize=9, color=ZINC_600)
    ax.set_title(title, fontsize=12, fontweight="bold", color="#3F3F46")
    ax.tick_params(colors="#71717A", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D4D4D8")
    ax.spines["bottom"].set_color("#D4D4D8")
    ax.grid(True, alpha=0.15, color="#A1A1AA", axis="x")

    save_fig(save_path)
