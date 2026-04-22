"""Exploration phase plots: uncertainty, acquisition, optimizer comparison."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from ._style import (
    AxisSpec, save_fig, _extract_xy, _apply_axes, _add_fixed_subtitle,
    _plot_schedule_ranges,
    ACCENT_YELLOW,
)


def plot_uncertainty_map(
    save_path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    x_values: np.ndarray,
    y_values: np.ndarray,
    unc_grid: np.ndarray,
    bf_grid: np.ndarray,
    *,
    points: list[dict[str, Any]] | None = None,
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """3-panel: raw uncertainty | boundary factor | buffered uncertainty."""
    unc_buffered = unc_grid * bf_grid

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _add_fixed_subtitle(fig, fixed_params)

    for ax, data, subtitle in [
        (axes[0], unc_grid, "Raw Uncertainty"),
        (axes[1], bf_grid, "Boundary Factor"),
        (axes[2], unc_buffered, "Uncertainty \u00d7 Boundary"),
    ]:
        im = ax.contourf(x_values, y_values, data, levels=20, cmap="Blues")
        if points:
            px, py = _extract_xy(points, x_axis, y_axis)
            ax.scatter(px, py, s=25, c="white", edgecolors="black", linewidth=0.5,
                       zorder=5, label="Baseline")
            ax.legend(fontsize=7, loc="upper right")
        _apply_axes(ax, x_axis, y_axis)
        ax.set_title(subtitle, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    save_fig(save_path)


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
    """3-panel: performance | uncertainty | combined acquisition."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _add_fixed_subtitle(fig, fixed_params)

    for ax, data, subtitle, cmap in [
        (axes[0], perf_grid, "Performance", "YlGn"),
        (axes[1], unc_grid, "Uncertainty", "Blues"),
        (axes[2], combined_grid, "Combined", "RdYlGn"),
    ]:
        im = ax.contourf(x_values, y_values, data, levels=20, cmap=cmap)
        _apply_axes(ax, x_axis, y_axis)
        ax.set_title(subtitle, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Schedule ranges (white on heatmaps)
        if points:
            _plot_schedule_ranges(ax, points, x_axis, y_axis, schedules, codes,
                                  color="white", alpha=0.6)
            px, py = _extract_xy(points, x_axis, y_axis)
            ax.scatter(px, py, s=18, c="white", edgecolors="#3F3F46",
                       linewidth=0.5, zorder=5, label="Evaluated")

    if proposed is not None:
        axes[2].plot(proposed[x_axis.key], proposed[y_axis.key],
                     "x", color=ACCENT_YELLOW, ms=10,
                     markeredgewidth=2, zorder=8, label="Proposed")

    axes[2].legend(fontsize=7, loc="upper left", framealpha=0.8)

    save_fig(save_path)


def plot_optimizer_comparison(
    save_path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    results: dict[str, list[dict[str, Any]]],
    baseline_pts: dict[str, list[dict[str, Any]]],
    *,
    nfev_key: str = "nfev",
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """Side-by-side scatter of optimizer proposals per optimizer tag."""
    tags = list(results.keys())
    n = len(tags)
    colors = ["#DD8452", "#D65F5F", "#4878CF", "#55A868"]

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    _add_fixed_subtitle(fig, fixed_params)

    for ax, tag, color in zip(axes, tags, colors):
        rounds = results[tag]
        bp = baseline_pts[tag]
        bx, by = _extract_xy(bp, x_axis, y_axis)
        ax.scatter(bx, by, s=40, c="#cccccc", edgecolors="gray", linewidth=0.5,
                   zorder=3, label="Baseline")
        for i, r in enumerate(rounds):
            ax.scatter(r[x_axis.key], r[y_axis.key], s=60, c=color,
                       edgecolors="white", linewidth=0.8, zorder=5)
            ax.annotate(f"{i+1}", (r[x_axis.key], r[y_axis.key]), fontsize=7,
                        ha="center", va="bottom", xytext=(0, 5),
                        textcoords="offset points")
        total_nfev = sum(r.get(nfev_key, 0) for r in rounds)
        ax.set_title(f"{tag}\n{total_nfev} total evals", fontsize=10)
        _apply_axes(ax, x_axis, y_axis)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    save_fig(save_path)
