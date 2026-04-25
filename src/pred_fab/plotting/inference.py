"""Inference phase plot: single-shot result on predicted topology."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from ._style import (
    AxisSpec, save_fig, _add_fixed_subtitle,
    apply_style, subplot_topology,
    ACCENT_YELLOW,
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
    schedules: dict[str, list[dict[str, Any]]] | None = None,
    codes: list[str] | None = None,
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """Single-shot inference result on the predicted performance topology."""
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    _add_fixed_subtitle(fig, fixed_params)

    subplot_topology(ax, x_axis, y_axis, x_values, y_values, pred_grid,
                     cmap_name="performance",
                     points=points, schedules=schedules, codes=codes,
                     point_size=15, point_alpha=0.6,
                     cbar_label="Predicted Combined Score")

    if optimum is not None:
        label = f"Optimum ({optimum_score:.3f})" if optimum_score is not None else "Optimum"
        ax.plot(optimum[x_axis.key], optimum[y_axis.key], "*", color="white", ms=16,
                markeredgecolor="black", markeredgewidth=1, zorder=8, label=label)

    prop_label = f"Proposed ({proposed_score:.3f})"
    ax.plot(proposed[x_axis.key], proposed[y_axis.key], "x", color=ACCENT_YELLOW,
            ms=14, markeredgewidth=2.5, zorder=9, label=prop_label)

    ax.legend(fontsize=8, loc="upper left")

    save_fig(save_path)
