"""Phase validation plots: what each optimizer saw."""

from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._style import (
    AxisSpec, save_fig,
    STEEL_500, ZINC_200, ZINC_400, ZINC_600,
)


def plot_phase_validation(
    save_path: str,
    panels: list[tuple[str, AxisSpec, AxisSpec, list[dict[str, Any]] | np.ndarray, list[int] | None]],
) -> None:
    """Generic multi-panel scatter for phase validation diagnostics.

    Each panel is (title, x_axis, y_axis, points, exp_ids_or_None).
    Points can be list[dict] (access via axis.key) or np.ndarray (columns 0=x, 1=y).
    If exp_ids is provided, connect dots per experiment with grey lines and label first point.
    For ndarray points with AxisSpec bounds, values are denormalized using bounds.
    """
    if not panels:
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    for ax, (panel_title, x_axis, y_axis, points, exp_ids) in zip(axes, panels):
        _draw_panel(ax, panel_title, x_axis, y_axis, points, exp_ids)

    save_fig(save_path)


def _draw_panel(
    ax: plt.Axes,  # type: ignore[name-defined]
    panel_title: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    points: list[dict[str, Any]] | np.ndarray,
    exp_ids: list[int] | None,
) -> None:
    """Draw a single validation panel: scatter with optional per-experiment lines."""
    ax.set_title(panel_title, fontsize=10, color=ZINC_600)
    ax.grid(True, alpha=0.2, color=ZINC_200)

    # Extract x/y coordinates
    if isinstance(points, np.ndarray):
        n_total = points.shape[0]
        # ndarray: column 0 = x, column 1 = y; denormalize if bounds available
        if x_axis.bounds:
            lo, hi = x_axis.bounds
            px = [float(lo + points[j, 0] * (hi - lo)) for j in range(n_total)]
        else:
            px = [float(points[j, 0]) for j in range(n_total)]
        if y_axis.bounds:
            lo, hi = y_axis.bounds
            py = [float(lo + points[j, 1] * (hi - lo)) for j in range(n_total)]
        else:
            py = [float(points[j, 1]) for j in range(n_total)]
    else:
        n_total = len(points)
        px = [float(p.get(x_axis.key, 0)) for p in points]
        py = [float(p.get(y_axis.key, 0)) for p in points]

    # Connect dots per experiment and label first point
    if exp_ids is not None:
        n_exp = max(exp_ids) + 1
        for eid in range(n_exp):
            mask = [j for j, e in enumerate(exp_ids) if e == eid]
            if len(mask) > 1:
                ex = [px[j] for j in mask]
                ey = [py[j] for j in mask]
                ax.plot(ex, ey, color=ZINC_400, linewidth=0.6, alpha=0.4, zorder=1)
            if mask:
                ax.annotate(f"{eid+1}", (px[mask[0]], py[mask[0]]), fontsize=7,
                           ha="center", va="bottom", xytext=(0, 5),
                           textcoords="offset points", color=ZINC_400)

    ax.scatter(px, py, s=60, c=STEEL_500, edgecolors="white",
               linewidth=0.8, zorder=5)

    # Labels without experiment annotations when no exp_ids
    if exp_ids is None:
        for i, (x, y) in enumerate(zip(px, py)):
            ax.annotate(f"{i+1}", (x, y), fontsize=7, ha="center", va="bottom",
                       xytext=(0, 5), textcoords="offset points", color=ZINC_400)

    ax.set_xlabel(x_axis.display_label, fontsize=9, color=ZINC_600)
    ax.set_ylabel(y_axis.display_label, fontsize=9, color=ZINC_600)
    if x_axis.bounds:
        ax.set_xlim(*x_axis.bounds)
    if y_axis.bounds:
        ax.set_ylim(*y_axis.bounds)
