"""Exploration phase plots: acquisition objective."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from ._style import (
    AxisSpec, fig_size, save_fig, _add_fixed_subtitle, annotate_point,
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
    gain_grid: np.ndarray,
    combined_grid: np.ndarray,
    *,
    points: list[dict[str, Any]] | None = None,
    proposed: dict[str, Any] | None = None,
    trajectories: dict[str, list[dict[str, Any]]] | None = None,
    codes: list[str] | None = None,
    fixed_params: dict[str, Any] | None = None,
    evidence_grid: np.ndarray | None = None,
) -> None:
    """3-panel: performance | evidence gain | combined acquisition.

    ``gain_grid`` is the evidence-*gain* field from
    ``compute_acquisition_grids`` — small magnitudes, so it renders
    fit-to-data (anchored at 0) rather than on the [0,1] scale.

    ``evidence_grid`` (the per-point E field from ``compute_evidence_grids``)
    fades the model-derived performance panel where evidence is low. The gain
    and combined panels stay unfaded — gain already encodes coverage, and the
    acquisition surface is a decision surface, not a model claim.
    """
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=fig_size(3, panel_w=5.0, panel_h=5.0))
    _add_fixed_subtitle(fig, fixed_params)

    panels = [
        (axes[0], perf_grid, "Performance", "performance", False, evidence_grid),
        (axes[1], gain_grid, "Evidence Gain", "evidence_gain", True, None),
        (axes[2], combined_grid, "Combined", "acquisition", False, None),
    ]
    for ax, grid, label, cmap_name, fit, ev in panels:
        subplot_topology(ax, x_axis, y_axis, x_values, y_values, grid,
                         cmap_name=cmap_name, label=label,
                         vmin=0.0 if fit else None, fit_to_data=fit,
                         evidence_grid=ev,
                         points=points, trajectories=trajectories, codes=codes,
                         point_size=18)

    if proposed is not None:
        px, py = float(proposed[x_axis.key]), float(proposed[y_axis.key])
        axes[2].plot(px, py, "x", color=ACCENT_YELLOW, ms=10,
                     markeredgewidth=2, zorder=8)
        ix = int(np.abs(np.asarray(x_values) - px).argmin())
        iy = int(np.abs(np.asarray(y_values) - py).argmin())
        annotate_point(axes[2], px, py,
                       f"proposed · {float(combined_grid[iy, ix]):.2f}")

    save_fig(save_path)
