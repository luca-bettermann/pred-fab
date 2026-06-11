"""Round-evolution strips: the learning arc as small multiples."""

from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._style import (
    AxisSpec, _add_fixed_subtitle, apply_style, row_colorbar,
    save_fig, subplot_topology,
)


def plot_topology_evolution(
    save_path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    x_values: np.ndarray,
    y_values: np.ndarray,
    grids: list[np.ndarray],
    *,
    cmap_name: str = "performance",
    round_labels: list[str] | None = None,
    truth_grid: np.ndarray | None = None,
    evidence_grids: list[np.ndarray] | None = None,
    points_per_round: list[list[dict[str, Any]]] | None = None,
    fit_to_data: bool = False,
    fixed_params: dict[str, Any] | None = None,
    cbar_label: str | None = None,
) -> None:
    """One row, one panel per round, shared scale, one colorbar — the
    progression (model sharpening, evidence filling in) in a single figure.

    Generic over any semantic surface: pass per-round performance grids,
    evidence grids, whatever tells the arc. ``truth_grid`` prepends a
    reference panel; ``evidence_grids`` (one per round) fades each round's
    panel by its evidence at the time; ``points_per_round`` overlays the
    experiments known at each round (cumulative lists tell the story best).
    """
    if not grids:
        return

    apply_style()
    labels = round_labels or [f"round {i}" for i in range(len(grids))]
    panels: list[tuple[str, np.ndarray, np.ndarray | None, list[dict[str, Any]] | None]] = []
    if truth_grid is not None:
        panels.append(("truth", truth_grid, None, None))
    for i, g in enumerate(grids):
        ev = evidence_grids[i] if evidence_grids is not None else None
        pts = points_per_round[i] if points_per_round is not None else None
        panels.append((labels[i], g, ev, pts))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n + 1.2, 3.8),
                             layout="constrained", squeeze=False)
    axes = axes[0]
    _add_fixed_subtitle(fig, fixed_params)

    vmin = vmax = None
    if fit_to_data:
        all_vals = np.concatenate([p[1].ravel() for p in panels])
        vmin, vmax = float(all_vals.min()), float(all_vals.max())

    im = None
    for j, (ax, (label, grid, ev, pts)) in enumerate(zip(axes, panels)):
        im = subplot_topology(ax, x_axis, y_axis, x_values, y_values, grid,
                              cmap_name=cmap_name, label=label,
                              vmin=vmin, vmax=vmax, fit_to_data=fit_to_data,
                              evidence_grid=ev, points=pts,
                              contour_labels=False, point_size=14,
                              show_colorbar=False)
        if j > 0:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    row_colorbar(fig, axes, im, label=cbar_label)
    save_fig(save_path)
