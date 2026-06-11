"""Parameter space plots: 2D coverage, 3D scatter, and dimensional trajectories."""

from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

from ._style import (
    AxisSpec, FONT, fig_size, save_fig, _extract_xy, _add_fixed_subtitle,
    apply_style, row_colorbar, subplot_topology,
    STEEL_500, ZINC_400, ZINC_600, ZINC_900,
)

_STEEL_CMAP = matplotlib.colormaps["steel_progression"]
_EMERALD_CMAP = matplotlib.colormaps["emerald_progression"]
_ZINC_CMAP = matplotlib.colormaps["zinc_progression"]


# ── Shared 3D helpers ──

def _setup_3d_figure(
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    z_label: str,
    fixed_params: dict[str, Any] | None = None,
) -> tuple[plt.Figure, Any]:  # type: ignore[name-defined]
    """Create a 3D figure with standard axis labels, bounds, and subtitle."""
    apply_style()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    _add_fixed_subtitle(fig, fixed_params)

    ax.set_xlabel(x_axis.display_label, labelpad=8, fontsize=FONT["axis_label"])
    ax.set_ylabel(y_axis.display_label, labelpad=8, fontsize=FONT["axis_label"])
    ax.set_zlabel(z_label, labelpad=8, fontsize=FONT["axis_label"])  # type: ignore[attr-defined]

    if x_axis.bounds:
        ax.set_xlim(*x_axis.bounds)
    if y_axis.bounds:
        ax.set_ylim(*y_axis.bounds)

    ax.view_init(elev=25, azim=-50)
    return fig, ax


def _highlight_style(
    code: str | None,
    highlight: str | None,
) -> tuple[Any, float, float, float, float]:
    """Return (cmap, line_alpha, dot_alpha, linewidth, markersize) for a trajectory."""
    if highlight is None:
        return _STEEL_CMAP, 0.5, 0.7, 1.2, 25
    if code == highlight:
        return _EMERALD_CMAP, 0.9, 1.0, 2.5, 40
    return _ZINC_CMAP, 0.25, 0.35, 1.0, 18


# ── 2D parameter space ──

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
    trajectories: dict[str, list[dict[str, Any]]] | None = None,
    codes: list[str] | None = None,
    fixed_params: dict[str, Any] | None = None,
    fit_to_data: bool = False,
    evidence_grid: np.ndarray | None = None,
) -> None:
    """1x2: ground truth topology + initial model topology.

    Panels share one color scale: the bounded [0,1] default, or — with
    ``fit_to_data`` — bounds computed across *both* grids. ``evidence_grid``
    fades the model panel where evidence is low; the truth panel is never
    faded (it makes no epistemic claim).
    """
    apply_style()
    vmin = vmax = None
    if fit_to_data:
        all_vals = np.concatenate([true_grid.ravel(), pred_grid.ravel()])
        vmin, vmax = float(all_vals.min()), float(all_vals.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size(2, panel_w=5.0, panel_h=4.5),
                                   layout="constrained")
    _add_fixed_subtitle(fig, fixed_params)

    im = None
    for ax, grid, label, ev in [
        (ax1, true_grid, "Ground Truth", None),
        (ax2, pred_grid, "Initial Model", evidence_grid),
    ]:
        im = subplot_topology(ax, x_axis, y_axis, x_values, y_values, grid,
                              cmap_name="performance", label=label,
                              vmin=vmin, vmax=vmax, fit_to_data=fit_to_data,
                              evidence_grid=ev,
                              points=points, trajectories=trajectories,
                              codes=codes, show_colorbar=False,
                              point_size=20, point_edge=ZINC_900)

    row_colorbar(fig, (ax1, ax2), im)
    save_fig(save_path)


def plot_parameter_space_per_cell(
    save_path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    x_values: np.ndarray,
    y_values: np.ndarray,
    points: list[dict[str, Any]],
    true_grid: np.ndarray,
    pred_grid: np.ndarray,
    *,
    cell_label: str = "",
    trajectories: dict[str, list[dict[str, Any]]] | None = None,
    codes: list[str] | None = None,
    fixed_params: dict[str, Any] | None = None,
    fit_to_data: bool = False,
) -> None:
    """1x3: ground truth, model prediction, and absolute difference at one cell.

    Bypasses eval-aggregation so the model's per-cell behaviour is visible
    directly. The third panel uses ``|truth - pred|`` so the diff is sign-free
    and can be visually compared across cells.
    """
    apply_style()
    diff_grid = np.abs(true_grid - pred_grid)

    if fit_to_data:
        val_vmin = float(min(true_grid.min(), pred_grid.min()))
        val_vmax = float(max(true_grid.max(), pred_grid.max()))
    else:
        val_vmin = val_vmax = None
    diff_vmax = float(diff_grid.max()) if diff_grid.size > 0 else 1.0

    cell_suffix = f"  ·  {cell_label}" if cell_label else ""
    panels = [
        (true_grid, f"Ground Truth{cell_suffix}", "performance", val_vmin, val_vmax),
        (pred_grid, f"Model Prediction{cell_suffix}", "performance", val_vmin, val_vmax),
        (diff_grid, f"|Truth − Pred|{cell_suffix}", "error", 0.0, max(diff_vmax, 1e-9)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    _add_fixed_subtitle(fig, fixed_params)

    for ax, (grid, label, cmap, vmin, vmax) in zip(axes, panels):
        subplot_topology(ax, x_axis, y_axis, x_values, y_values, grid,
                         cmap_name=cmap, label=label,
                         vmin=vmin, vmax=vmax,
                         points=points, trajectories=trajectories, codes=codes,
                         point_size=20, point_edge=ZINC_900)

    save_fig(save_path)


def plot_mean_error_topology(
    save_path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    x_values: np.ndarray,
    y_values: np.ndarray,
    points: list[dict[str, Any]],
    mean_diff_grid: np.ndarray,
    *,
    label: str = "Mean |Truth − Pred| (all cells)",
    trajectories: dict[str, list[dict[str, Any]]] | None = None,
    codes: list[str] | None = None,
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """Single-panel heatmap of the mean absolute error across all cells.

    Surfaces parameter regions where the model is systematically weak
    regardless of which (layer, segment) cell you look at.
    """
    apply_style()
    vmax = float(mean_diff_grid.max()) if mean_diff_grid.size > 0 else 1.0

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    _add_fixed_subtitle(fig, fixed_params)
    subplot_topology(ax, x_axis, y_axis, x_values, y_values, mean_diff_grid,
                     cmap_name="error", label=label,
                     vmin=0.0, vmax=max(vmax, 1e-9),
                     points=points, trajectories=trajectories, codes=codes,
                     point_size=20, point_edge=ZINC_900)

    save_fig(save_path)


# ── 3D parameter space scatter ──

def plot_parameter_space_3d(
    save_path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    z_axis: AxisSpec,
    points: list[dict[str, Any]],
    *,
    codes: list[str] | None = None,
    highlight: str | None = None,
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """3D scatter where point color encodes the z-axis value."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig, ax = _setup_3d_figure(x_axis, y_axis, z_axis.display_label, fixed_params)

    px = [float(p[x_axis.key]) for p in points]
    py = [float(p[y_axis.key]) for p in points]
    pz = [float(p[z_axis.key]) for p in points]

    z_arr = np.array(pz)
    vmin = float(z_axis.bounds[0]) if z_axis.bounds else float(z_arr.min())
    vmax = float(z_axis.bounds[1]) if z_axis.bounds else float(z_arr.max())
    if vmax - vmin < 1e-12:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    norm = Normalize(vmin=vmin, vmax=vmax)

    if highlight and codes:
        # Highlighted mode: one point in Emerald, rest in Zinc
        for i, (x, y, z) in enumerate(zip(px, py, pz)):
            code = codes[i] if i < len(codes) else None
            cmap, _, dot_alpha, _, ms = _highlight_style(code, highlight)
            ax.scatter(
                [x], [y], [z],  # type: ignore[arg-type]
                c=[cmap(norm(z))], s=ms, edgecolors="white", linewidth=0.5,
                alpha=dot_alpha, depthshade=False,
            )
    else:
        sc = ax.scatter(
            px, py, pz,  # type: ignore[arg-type]
            c=pz, cmap=_STEEL_CMAP, norm=norm,
            s=50, edgecolors="white", linewidth=0.6, zorder=5, depthshade=False,
        )
        for i, (x, y, z) in enumerate(zip(px, py, pz)):
            ax.text(x, y, z, f" {i+1}", fontsize=6, color=ZINC_600, zorder=6)
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
        cb.set_label(z_axis.display_label, fontsize=FONT["axis_label"])

    if z_axis.bounds:
        ax.set_zlim(*z_axis.bounds)  # type: ignore[attr-defined]
    if all(z == int(z) for z in pz):
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))  # type: ignore[attr-defined]

    save_fig(save_path)


# ── 3D dimensional trajectories ──

def plot_dimensional_trajectories(
    save_path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    z_dim: str,
    points: list[dict[str, Any]],
    *,
    trajectories: dict[str, list[dict[str, Any]]] | None = None,
    codes: list[str] | None = None,
    highlight: str | None = None,
    z_label: str = "Layer",
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """3D trajectories along a dimensional axis (e.g., per-layer parameter values).

    z-axis = iterator of z_dim (0, 1, ..., n_steps-1).
    Without trajectories: straight vertical lines (constant params per step).
    With trajectories: tilted/curved trajectories (params vary per step).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig, ax = _setup_3d_figure(x_axis, y_axis, z_label, fixed_params)

    for i, params in enumerate(points):
        n_steps = int(params.get(z_dim, 1))
        code = codes[i] if codes and i < len(codes) else None
        cmap, line_alpha, dot_alpha, lw, ms = _highlight_style(code, highlight)

        # Build per-step coordinates
        xs, ys, zs = [], [], []
        for k in range(n_steps):
            x = float(params.get(x_axis.key, 0))
            y = float(params.get(y_axis.key, 0))
            # Override from trajectory if available
            if code and trajectories and code in trajectories and k < len(trajectories[code]):
                step = trajectories[code][k]
                x = float(step.get(x_axis.key, x))
                y = float(step.get(y_axis.key, y))
            xs.append(x)
            ys.append(y)
            zs.append(k)

        # Trajectory line
        ax.plot(xs, ys, zs, color=cmap(0.5), alpha=line_alpha,
                linewidth=lw, zorder=3 if code == highlight else 1)

        # Per-step dots colored by progression
        step_norm = Normalize(vmin=0, vmax=max(n_steps - 1, 1))
        colors = [cmap(step_norm(k)) for k in range(n_steps)]
        is_highlighted = highlight is not None and code == highlight
        is_faded = highlight is not None and not is_highlighted
        ax.scatter(
            xs, ys, zs,  # type: ignore[arg-type]
            c=colors, s=ms, edgecolors="white" if not is_faded else "none",
            linewidth=0.5, alpha=dot_alpha,
            zorder=4 if is_highlighted else 2, depthshade=False,
        )

        # Value labels on highlighted experiment's dots
        if is_highlighted:
            for k in range(n_steps):
                label = f" {ys[k]:.1f}" if len(set(ys)) > 1 else f" {xs[k]:.3f}"
                ax.text(xs[k], ys[k], zs[k], label, fontsize=FONT["annotation"],
                        color=ZINC_600, zorder=6)

    ax.zaxis.set_major_locator(MaxNLocator(integer=True))  # type: ignore[attr-defined]

    save_fig(save_path)
