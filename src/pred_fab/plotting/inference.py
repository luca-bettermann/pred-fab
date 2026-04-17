"""Inference phase plots: single-shot result and convergence trajectory."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from ._style import (
    AxisSpec, save_fig, _extract_xy, _apply_axes, _add_fixed_subtitle,
    STEEL_500, ACCENT_YELLOW,
)


def plot_inference_result(
    save_path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    x_values: np.ndarray,
    y_values: np.ndarray,
    pred_grid: np.ndarray,
    proposed: dict[str, float],
    proposed_score: float,
    *,
    optimum: dict[str, float] | None = None,
    optimum_score: float | None = None,
    points: list[dict[str, Any]] | None = None,
    title: str = "Inference Result",
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """Single-shot inference result on the predicted performance topology."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    _add_fixed_subtitle(fig, fixed_params)

    im = ax.contourf(x_values, y_values, pred_grid, levels=20, cmap="RdYlGn")
    ax.contour(x_values, y_values, pred_grid, levels=10, colors="white",
               linewidths=0.3, alpha=0.5)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Predicted Combined Score")

    if points:
        px, py = _extract_xy(points, x_axis, y_axis)
        ax.scatter(px, py, s=15, c="white", edgecolors="black",
                   linewidth=0.4, zorder=4, alpha=0.6)

    if optimum is not None:
        label = f"Optimum ({optimum_score:.3f})" if optimum_score is not None else "Optimum"
        ax.plot(optimum[x_axis.key], optimum[y_axis.key], "*", color="white", ms=16,
                markeredgecolor="black", markeredgewidth=1, zorder=8, label=label)

    prop_label = f"Proposed ({proposed_score:.3f})"
    ax.plot(proposed[x_axis.key], proposed[y_axis.key], "x", color=ACCENT_YELLOW,
            ms=14, markeredgewidth=2.5, zorder=9, label=prop_label)

    _apply_axes(ax, x_axis, y_axis)
    ax.legend(fontsize=8, loc="upper left")

    save_fig(save_path)


def plot_inference_convergence(
    save_path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    x_values: np.ndarray,
    y_values: np.ndarray,
    topology_grid: np.ndarray,
    trajectory: list[dict[str, Any]],
    score_key: str = "score",
    *,
    optimum: dict[str, float] | None = None,
    optimum_score: float | None = None,
    title: str = "Inference Convergence",
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """1x2: convergence trajectory on topology + score per round."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    _add_fixed_subtitle(fig, fixed_params)

    im = ax1.contourf(x_values, y_values, topology_grid, levels=20, cmap="RdYlGn")
    ax1.contour(x_values, y_values, topology_grid, levels=10, colors="white",
                linewidths=0.3, alpha=0.5)
    plt.colorbar(im, ax=ax1, shrink=0.8, label="Combined Score")

    tx = [r[x_axis.key] for r in trajectory]
    ty = [r[y_axis.key] for r in trajectory]
    if len(trajectory) > 1:
        ax1.plot(tx, ty, "w-", lw=2, alpha=0.8)
    for j, r in enumerate(trajectory):
        ax1.plot(r[x_axis.key], r[y_axis.key], "wo", ms=8,
                 markeredgecolor="black", markeredgewidth=0.8)
        ax1.annotate(f"{j+1}", (r[x_axis.key], r[y_axis.key]), fontsize=7,
                     ha="center", va="center", color="black", fontweight="bold")

    if optimum is not None:
        ox, oy = optimum[x_axis.key], optimum[y_axis.key]
        ax1.plot(ox, oy, "w*", ms=14, markeredgecolor="black", markeredgewidth=1,
                 label=f"Optimum ({ox:.2f}, {oy:.1f})")
        ax1.legend(fontsize=7, loc="upper left")

    _apply_axes(ax1, x_axis, y_axis)
    ax1.set_title("Trajectory on Performance Topology")

    scores = [r[score_key] for r in trajectory]
    ax2.plot(range(1, len(scores) + 1), scores, "o-", color=STEEL_500, lw=2, ms=8)
    if optimum_score is not None:
        ax2.axhline(optimum_score, color="#6ACC65", ls="--", lw=1.5,
                     label="Optimum score")
        ax2.legend(fontsize=8)
    ax2.set_xlabel("Inference Round")
    ax2.set_ylabel("Combined Score")
    ax2.set_title("Performance per Round")
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(0, 1)

    save_fig(save_path)
