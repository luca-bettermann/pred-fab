"""Inference phase plot: single-shot result on predicted topology."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from ._style import (
    AxisSpec, save_fig, _add_fixed_subtitle, annotate_point,
    apply_style, subplot_topology,
    ACCENT_YELLOW, ZINC_900,
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
    trajectories: dict[str, list[dict[str, Any]]] | None = None,
    codes: list[str] | None = None,
    fixed_params: dict[str, Any] | None = None,
    evidence_grid: np.ndarray | None = None,
) -> None:
    """Single-shot inference result on the predicted performance topology.

    ``evidence_grid`` fades the prediction where evidence is low — the
    proposal should visibly sit in trusted territory.
    """
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    _add_fixed_subtitle(fig, fixed_params)

    subplot_topology(ax, x_axis, y_axis, x_values, y_values, pred_grid,
                     cmap_name="performance",
                     evidence_grid=evidence_grid,
                     points=points, trajectories=trajectories, codes=codes,
                     point_size=15, point_alpha=0.6,
                     cbar_label="Predicted Combined Score")

    if optimum is not None:
        ox, oy = float(optimum[x_axis.key]), float(optimum[y_axis.key])
        ax.plot(ox, oy, "*", color="white", ms=16,
                markeredgecolor=ZINC_900, markeredgewidth=1, zorder=8)
        text = (f"optimum · {optimum_score:.3f}" if optimum_score is not None
                else "optimum")
        annotate_point(ax, ox, oy, text)

    px, py = float(proposed[x_axis.key]), float(proposed[y_axis.key])
    ax.plot(px, py, "x", color=ACCENT_YELLOW,
            ms=14, markeredgewidth=2.5, zorder=9)
    annotate_point(ax, px, py, f"proposed · {proposed_score:.3f}")

    save_fig(save_path)
