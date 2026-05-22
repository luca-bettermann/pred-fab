"""Evidence panels — density and gain visualization via KernelField.

All grid computation delegates to ``pred_fab.orchestration.evidence``
(same KernelField ANOVA pipeline the optimizer uses). This module
handles experiment expansion, overlay rendering, and panel composition.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt

from ._style import (
    AxisSpec,
    apply_style,
    clean_spines,
    subplot_label,
    subplot_topology,
    save_fig,
    STEEL_500,
    ZINC_400,
)


# ======================================================================
# Experiment expansion — specs → flat points with trajectory weights
# ======================================================================

def expand_experiments(
    experiments: list[Any],
    param_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[int], list[float]]:
    """Expand experiments into flat point lists with per-layer 1/L weights.

    Accepts:
      - ``ExperimentSpec`` — expands via ``.trajectories``
      - ``ExperimentData`` — expands via ``.parameter_updates``
      - ``dict`` — single point, no expansion
    """
    transform = param_transform or (lambda d: d)
    all_pts: list[dict[str, Any]] = []
    exp_ids: list[int] = []
    weights: list[float] = []

    for i, exp in enumerate(experiments):
        if hasattr(exp, "initial_params"):
            base = dict(exp.initial_params.to_dict())
            layers: list[dict[str, Any]] = [transform(base)]
            for _dim, traj in exp.trajectories.items():
                for _step, proposal in traj.entries:
                    layer_p = dict(base)
                    layer_p.update(transform(proposal.to_dict()))
                    layers.append(layer_p)
        elif hasattr(exp, "parameter_updates") and hasattr(exp, "parameters"):
            base_params = exp.parameters.get_values_dict()
            if exp.parameter_updates:
                dim_names = exp.parameters.get_dim_names()
                n_steps = int(exp.parameters.get_value(dim_names[0])) if dim_names else 1
                layers = []
                for step in range(n_steps):
                    ctx = {exp.parameters.get_dim_iterator_codes(dim_names[:1])[0]: step} if dim_names else {}
                    layers.append(transform(exp.get_effective_parameters_for_context(ctx)))
            else:
                layers = [transform(base_params)]
        else:
            layers = [transform(dict(exp))]

        w = 1.0 / len(layers)
        for lp in layers:
            all_pts.append(lp)
            exp_ids.append(i)
            weights.append(w)

    return all_pts, exp_ids, weights


# ======================================================================
# Scatter + trajectory overlay
# ======================================================================

def _overlay_points(
    ax: plt.Axes,  # type: ignore[name-defined]
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    points: list[dict[str, Any]],
    exp_ids: list[int],
) -> None:
    """Draw scatter points with trajectory lines, colored by layer index."""
    from matplotlib.colors import Normalize as MplNorm
    px = [float(p.get(x_axis.key, 0)) for p in points]
    py = [float(p.get(y_axis.key, 0)) for p in points]

    # Compute per-point layer fraction: 0 = first layer, 1 = last
    layer_frac = np.zeros(len(points))
    n_exp = max(exp_ids) + 1 if exp_ids else 0
    for eid in range(n_exp):
        mask = [j for j, e in enumerate(exp_ids) if e == eid]
        if len(mask) > 1:
            for k, j in enumerate(mask):
                layer_frac[j] = k / (len(mask) - 1)
            ex = [px[j] for j in mask]
            ey = [py[j] for j in mask]
            ax.plot(ex, ey, color="#E8913A", linewidth=0.6, alpha=0.4, zorder=1)
        if mask:
            ax.annotate(f"{eid+1}", (px[mask[0]], py[mask[0]]), fontsize=7,
                       ha="center", va="bottom", xytext=(0, 5),
                       textcoords="offset points", color="#B06A1E")

    cm = plt.get_cmap("Oranges")
    colors = [cm(0.3 + 0.55 * layer_frac[j]) for j in range(len(points))]
    ax.scatter(px, py, s=60, c=colors, edgecolors="white",
               linewidth=0.8, zorder=5)


# ======================================================================
# Panel functions — delegate grid computation to orchestration.evidence
# ======================================================================

def plot_evidence_panel(
    ax: plt.Axes,  # type: ignore[name-defined]
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    experiments: list[Any],
    sigma: float,
    *,
    all_axes: list[AxisSpec] | None = None,
    label: str | None = None,
    param_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    resolution: int = 40,
) -> None:
    """Evidence density panel using KernelField ANOVA integration."""
    from ..orchestration.evidence import compute_evidence_density_grid

    axes = all_axes or [x_axis, y_axis]
    xs, ys, grid = compute_evidence_density_grid(
        experiments, axes, x_axis.key, y_axis.key, sigma,
        resolution=resolution, param_transform=param_transform,
    )
    grid_max = float(grid.max()) if grid.size > 0 else 1.0
    subplot_topology(ax, x_axis, y_axis, xs, ys, grid,
                     cmap_name="evidence", contour_overlay=False, show_colorbar=True,
                     vmin=0.0, vmax=1.0, cbar_lim=max(grid_max, 1e-6))
    if label:
        subplot_label(ax, label)
    pts, exp_ids, _ = expand_experiments(experiments, param_transform)
    _overlay_points(ax, x_axis, y_axis, pts, exp_ids)


def plot_evidence_gain_panel(
    ax: plt.Axes,  # type: ignore[name-defined]
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    experiments_before: list[Any],
    experiments_after: list[Any],
    sigma: float,
    *,
    all_axes: list[AxisSpec] | None = None,
    label: str | None = None,
    param_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    resolution: int = 40,
) -> None:
    """Evidence gain ΔE panel — difference between after and before.

    Uses the KernelField ANOVA pipeline for both grids.
    """
    from ..orchestration.evidence import compute_evidence_density_grid

    axes_b = all_axes or [x_axis, y_axis]
    axes_a = all_axes or [x_axis, y_axis]
    _, _, grid_before = compute_evidence_density_grid(
        experiments_before, axes_b, x_axis.key, y_axis.key, sigma,
        resolution=resolution, param_transform=param_transform,
    )
    xs, ys, grid_after = compute_evidence_density_grid(
        experiments_after, axes_a, x_axis.key, y_axis.key, sigma,
        resolution=resolution, param_transform=param_transform,
    )
    grid = grid_after - grid_before

    subplot_topology(ax, x_axis, y_axis, xs, ys, grid,
                     cmap_name="evidence_gain", contour_overlay=False, show_colorbar=True)
    if label:
        subplot_label(ax, label)
    pts_a, exp_ids_a, _ = expand_experiments(experiments_after, param_transform)
    _overlay_points(ax, x_axis, y_axis, pts_a, exp_ids_a)


# ======================================================================
# Multi-angle composition
# ======================================================================

def plot_multi_angle(
    focus_axis: AxisSpec,
    other_axes: list[AxisSpec],
    experiments: list[Any],
    sigma: float,
    path: str,
    *,
    panel_fn: Callable | None = None,
    param_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    resolution: int = 40,
    title: str | None = None,
) -> None:
    """Plot ``focus_axis`` against each axis in ``other_axes`` in a single row.

    Uses the KernelField ANOVA pipeline for evidence computation.
    """
    fn = panel_fn or plot_evidence_panel
    all_axes = [focus_axis] + list(other_axes)
    n = len(other_axes)
    if n == 0:
        return

    apply_style()
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for i, other in enumerate(other_axes):
        label = f"{focus_axis.label} × {other.label}"
        kwargs: dict[str, Any] = dict(
            label=label, param_transform=param_transform, resolution=resolution,
        )
        if fn is plot_evidence_panel:
            kwargs["all_axes"] = all_axes
        fn(axes[i], focus_axis, other, experiments, sigma, **kwargs)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
