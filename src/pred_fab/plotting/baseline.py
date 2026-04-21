"""Parameter space plots: 2D coverage, 3D scatter, and dimensional trajectories."""

from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator

from ._style import (
    AxisSpec, save_fig, _extract_xy, _apply_axes, _add_fixed_subtitle,
    _plot_schedule_ranges,
    STEEL_500, ZINC_400, ZINC_600, ZINC_700,
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
    title: str,
    fixed_params: dict[str, Any] | None = None,
) -> tuple[plt.Figure, Any]:  # type: ignore[name-defined]
    """Create a 3D figure with standard axis labels, bounds, and title."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle(title, fontsize=13, fontweight="bold", color=ZINC_700)
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
    title: str = "Baseline",
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """1x3: parameter space scatter + ground truth topology + initial model topology."""
    px, py = _extract_xy(points, x_axis, y_axis)
    n = len(px)
    vmin, vmax = true_grid.min(), true_grid.max()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.5))
    fig.suptitle(f"{title} ({n} experiments)", fontsize=14, fontweight="bold", y=1.02)
    _add_fixed_subtitle(fig, fixed_params)

    # Schedule range lines (before dots so dots are on top)
    _plot_schedule_ranges(ax1, points, x_axis, y_axis, schedules, codes)

    ax1.scatter(px, py, s=60, c=STEEL_500, edgecolors="white", linewidth=0.8, zorder=5)
    for i, (x, y) in enumerate(zip(px, py)):
        ax1.annotate(f"{i+1}", (x, y), fontsize=6, ha="center", va="bottom",
                     xytext=(0, 5), textcoords="offset points", color="#666")
    _apply_axes(ax1, x_axis, y_axis)
    ax1.set_title("Parameter Space", fontsize=10)
    ax1.grid(True, alpha=0.2)

    # Schedule ranges on heatmaps too
    _plot_schedule_ranges(ax2, points, x_axis, y_axis, schedules, codes,
                          color="white", alpha=0.4)
    im2 = ax2.contourf(x_values, y_values, true_grid, levels=20, cmap="RdYlGn",
                        vmin=vmin, vmax=vmax)
    ax2.contour(x_values, y_values, true_grid, levels=10, colors="white",
                linewidths=0.3, alpha=0.5)
    ax2.scatter(px, py, s=20, c="white", edgecolors="black", linewidth=0.5, zorder=5)
    _apply_axes(ax2, x_axis, y_axis)
    ax2.set_title("Ground Truth", fontsize=10)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    _plot_schedule_ranges(ax3, points, x_axis, y_axis, schedules, codes,
                          color="white", alpha=0.4)
    im3 = ax3.contourf(x_values, y_values, pred_grid, levels=20, cmap="RdYlGn",
                        vmin=vmin, vmax=vmax)
    ax3.contour(x_values, y_values, pred_grid, levels=10, colors="white",
                linewidths=0.3, alpha=0.5)
    ax3.scatter(px, py, s=20, c="white", edgecolors="black", linewidth=0.5, zorder=5)
    _apply_axes(ax3, x_axis, y_axis)
    ax3.set_title("Initial Model", fontsize=10)
    plt.colorbar(im3, ax=ax3, shrink=0.8)

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
    title: str = "Parameter Space",
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """3D scatter where point color encodes the z-axis value."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n = len(points)
    title_full = f"{title}  ·  {highlight}" if highlight else f"{title} ({n} experiments)"
    fig, ax = _setup_3d_figure(x_axis, y_axis, z_axis.display_label, title_full, fixed_params)

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
    title: str = "Parameter Space",
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """3D trajectories along a dimensional axis (e.g., per-layer parameter values).

    z-axis = iterator of z_dim (0, 1, ..., n_steps-1).
    Without schedules: straight vertical lines (constant params per step).
    With schedules: tilted/curved trajectories (params vary per step).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n = len(points)
    title_full = f"{title}  ·  {highlight}" if highlight else f"{title} ({n} experiments)"
    fig, ax = _setup_3d_figure(x_axis, y_axis, z_label, title_full, fixed_params)

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
