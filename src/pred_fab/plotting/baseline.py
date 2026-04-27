"""Parameter space plots: 2D coverage, 3D scatter, and dimensional trajectories."""

from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator

from ._style import (
    AxisSpec, save_fig, _extract_xy, _add_fixed_subtitle,
    apply_style, subplot_topology,
    STEEL_500, ZINC_400, ZINC_600,
)

# Colormaps
_STEEL_CMAP = LinearSegmentedColormap.from_list(
    "steel", ["#D6E4F0", "#8BB0CC", "#4A7FA5", "#2D5F85", "#1A3A5C"]
)
_EMERALD_CMAP = LinearSegmentedColormap.from_list(
    "emerald", ["#D1FAE5", "#6EE7B7", "#10B981", "#047857", "#064E3B"]
)
_ZINC_CMAP = LinearSegmentedColormap.from_list(
    "zinc", ["#E4E4E7", "#A1A1AA", "#71717A", "#52525B", "#3F3F46"]
)


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

    ax.set_xlabel(x_axis.display_label, labelpad=8, fontsize=9)
    ax.set_ylabel(y_axis.display_label, labelpad=8, fontsize=9)
    ax.set_zlabel(z_label, labelpad=8, fontsize=9)  # type: ignore[attr-defined]

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
    schedules: dict[str, list[dict[str, Any]]] | None = None,
    codes: list[str] | None = None,
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """1x2: ground truth topology + initial model topology."""
    apply_style()
    vmin, vmax = float(true_grid.min()), float(true_grid.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    _add_fixed_subtitle(fig, fixed_params)

    for ax, grid, label in [
        (ax1, true_grid, "Ground Truth"),
        (ax2, pred_grid, "Initial Model"),
    ]:
        subplot_topology(ax, x_axis, y_axis, x_values, y_values, grid,
                         cmap_name="performance", label=label,
                         vmin=vmin, vmax=vmax,
                         points=points, schedules=schedules, codes=codes,
                         point_size=20, point_edge="black")

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
    schedules: dict[str, list[dict[str, Any]]] | None = None,
    codes: list[str] | None = None,
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """1x3: ground truth, model prediction, and absolute difference at one cell.

    Bypasses eval-aggregation so the model's per-cell behaviour is visible
    directly. The third panel uses ``|truth - pred|`` so the diff is sign-free
    and can be visually compared across cells.
    """
    apply_style()
    diff_grid = np.abs(true_grid - pred_grid)

    val_vmin = float(min(true_grid.min(), pred_grid.min()))
    val_vmax = float(max(true_grid.max(), pred_grid.max()))
    diff_vmax = float(diff_grid.max()) if diff_grid.size > 0 else 1.0

    cell_suffix = f"  ·  {cell_label}" if cell_label else ""
    panels = [
        (true_grid, f"Ground Truth{cell_suffix}", "performance", val_vmin, val_vmax),
        (pred_grid, f"Model Prediction{cell_suffix}", "performance", val_vmin, val_vmax),
        (diff_grid, f"|Truth − Pred|{cell_suffix}", "Reds", 0.0, max(diff_vmax, 1e-9)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    _add_fixed_subtitle(fig, fixed_params)

    for ax, (grid, label, cmap, vmin, vmax) in zip(axes, panels):
        subplot_topology(ax, x_axis, y_axis, x_values, y_values, grid,
                         cmap_name=cmap, label=label,
                         vmin=vmin, vmax=vmax,
                         points=points, schedules=schedules, codes=codes,
                         point_size=20, point_edge="black")

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
    schedules: dict[str, list[dict[str, Any]]] | None = None,
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
                     cmap_name="Reds", label=label,
                     vmin=0.0, vmax=max(vmax, 1e-9),
                     points=points, schedules=schedules, codes=codes,
                     point_size=20, point_edge="black")

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
            ax.text(x, y, z, f" {i+1}", fontsize=6, color="#666", zorder=6)
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
        cb.set_label(z_axis.display_label, fontsize=9)

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
    schedules: dict[str, list[dict[str, Any]]] | None = None,
    codes: list[str] | None = None,
    highlight: str | None = None,
    z_label: str = "Layer",
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """3D trajectories along a dimensional axis (e.g., per-layer parameter values).

    z-axis = iterator of z_dim (0, 1, ..., n_steps-1).
    Without schedules: straight vertical lines (constant params per step).
    With schedules: tilted/curved trajectories (params vary per step).
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
            # Override from schedule if available
            if code and schedules and code in schedules and k < len(schedules[code]):
                step = schedules[code][k]
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
                ax.text(xs[k], ys[k], zs[k], label, fontsize=7,
                        color=ZINC_600, zorder=6)

    ax.zaxis.set_major_locator(MaxNLocator(integer=True))  # type: ignore[attr-defined]

    save_fig(save_path)
