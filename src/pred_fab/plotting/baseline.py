"""Baseline phase plots: parameter space coverage and initial model comparison."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

from ._style import (
    AxisSpec, save_fig, _extract_xy, _apply_axes, _add_fixed_subtitle,
    STEEL_500,
)


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

    # Panel 1: parameter space
    ax1.scatter(px, py, s=60, c=STEEL_500, edgecolors="white", linewidth=0.8, zorder=5)
    for i, (x, y) in enumerate(zip(px, py)):
        ax1.annotate(f"{i+1}", (x, y), fontsize=6, ha="center", va="bottom",
                     xytext=(0, 5), textcoords="offset points", color="#666")
    _apply_axes(ax1, x_axis, y_axis)
    ax1.set_title("Parameter Space", fontsize=10)
    ax1.grid(True, alpha=0.2)

    # Panel 2: ground truth
    im2 = ax2.contourf(x_values, y_values, true_grid, levels=20, cmap="RdYlGn",
                        vmin=vmin, vmax=vmax)
    ax2.contour(x_values, y_values, true_grid, levels=10, colors="white",
                linewidths=0.3, alpha=0.5)
    ax2.scatter(px, py, s=20, c="white", edgecolors="black", linewidth=0.5, zorder=5)
    _apply_axes(ax2, x_axis, y_axis)
    ax2.set_title("Ground Truth", fontsize=10)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Panel 3: initial model
    im3 = ax3.contourf(x_values, y_values, pred_grid, levels=20, cmap="RdYlGn",
                        vmin=vmin, vmax=vmax)
    ax3.contour(x_values, y_values, pred_grid, levels=10, colors="white",
                linewidths=0.3, alpha=0.5)
    ax3.scatter(px, py, s=20, c="white", edgecolors="black", linewidth=0.5, zorder=5)
    _apply_axes(ax3, x_axis, y_axis)
    ax3.set_title("Initial Model", fontsize=10)
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    save_fig(save_path)


# Steel gradient colormap for z-axis encoding (data points use Steel spectrum)
_STEEL_CMAP = LinearSegmentedColormap.from_list(
    "steel", ["#D6E4F0", "#8BB0CC", "#4A7FA5", "#2D5F85", "#1A3A5C"]
)


_EMERALD_CMAP = LinearSegmentedColormap.from_list(
    "emerald", ["#D1FAE5", "#6EE7B7", "#10B981", "#047857", "#064E3B"]
)

_ZINC_CMAP = LinearSegmentedColormap.from_list(
    "zinc", ["#E4E4E7", "#A1A1AA", "#71717A", "#52525B", "#3F3F46"]
)


def plot_parameter_space_3d(
    save_path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    z_axis: AxisSpec,
    points: list[dict[str, Any]],
    *,
    schedules: dict[str, list[dict[str, Any]]] | None = None,
    codes: list[str] | None = None,
    highlight: str | None = None,
    title: str = "Parameter Space",
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """3D parameter space with optional schedule trajectories.

    Without schedules: scatter with z-color encoding.
    With schedules: connected trajectories per experiment (z = layer index).
    With highlight: one experiment in Emerald, rest in Zinc.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.ticker import MaxNLocator

    n = len(points)
    has_schedules = schedules is not None and codes is not None and any(
        c in schedules for c in (codes or [])
    )

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    title_suffix = f" ({n} experiments)" if not highlight else f"  ·  {highlight}"
    fig.suptitle(f"{title}{title_suffix}", fontsize=13, fontweight="bold",
                 color="#3F3F46")
    _add_fixed_subtitle(fig, fixed_params)

    # z-axis normalization for color
    if has_schedules:
        # Collect all z values from schedules for color range
        all_z: list[float] = []
        for c in (codes or []):
            if schedules and c in schedules:
                for step in schedules[c]:
                    all_z.append(float(step.get(z_axis.key, 0)))
            else:
                all_z.append(float(points[(codes or []).index(c)].get(z_axis.key, 0)))
        z_min = min(all_z) if all_z else 0
        z_max = max(all_z) if all_z else 1
    else:
        pz = [float(p[z_axis.key]) for p in points]
        z_min = float(min(pz)) if pz else 0
        z_max = float(max(pz)) if pz else 1

    if z_max - z_min < 1e-12:
        z_min, z_max = z_min - 0.5, z_max + 0.5
    norm = Normalize(vmin=z_min, vmax=z_max)

    if has_schedules and codes is not None and schedules is not None:
        # ── Trajectory mode ──
        for i, code in enumerate(codes):
            is_highlighted = highlight is not None and code == highlight
            is_faded = highlight is not None and code != highlight

            if code in schedules:
                steps = schedules[code]
                xs = [float(s.get(x_axis.key, points[i].get(x_axis.key, 0))) for s in steps]
                ys = [float(s.get(y_axis.key, points[i].get(y_axis.key, 0))) for s in steps]
                zs = list(range(len(steps)))

                if is_highlighted:
                    cmap = _EMERALD_CMAP
                    line_alpha, dot_alpha, lw, ms = 0.9, 1.0, 2.5, 40
                elif is_faded:
                    cmap = _ZINC_CMAP
                    line_alpha, dot_alpha, lw, ms = 0.15, 0.2, 0.8, 15
                else:
                    cmap = _STEEL_CMAP
                    line_alpha, dot_alpha, lw, ms = 0.5, 0.7, 1.2, 25

                # Trajectory line
                ax.plot(xs, ys, zs, color=cmap(0.5), alpha=line_alpha,
                        linewidth=lw, zorder=3 if is_highlighted else 1)

                # Per-step dots colored by z (layer index)
                step_norm = Normalize(vmin=0, vmax=max(len(steps) - 1, 1))
                colors = [cmap(step_norm(k)) for k in range(len(steps))]
                ax.scatter(
                    xs, ys, zs,  # type: ignore[arg-type]
                    c=colors, s=ms, edgecolors="white" if not is_faded else "none",
                    linewidth=0.5, alpha=dot_alpha, zorder=4 if is_highlighted else 2,
                    depthshade=False,
                )
            else:
                # Single point (no schedule)
                x = float(points[i][x_axis.key])
                y = float(points[i][y_axis.key])
                z = 0
                if is_faded:
                    color, alpha, ms = "#A1A1AA", 0.2, 15
                elif is_highlighted:
                    color, alpha, ms = "#10B981", 1.0, 40
                else:
                    color, alpha, ms = _STEEL_CMAP(norm(z)), 0.7, 25  # type: ignore[assignment]
                ax.scatter(
                    [x], [y], [z],  # type: ignore[arg-type]
                    c=[color], s=ms, edgecolors="white", linewidth=0.5,
                    alpha=alpha, depthshade=False,
                )

        # z-axis = layer index (integer)
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))  # type: ignore[attr-defined]

    else:
        # ── Scatter mode (no schedules) ──
        px = [float(p[x_axis.key]) for p in points]
        py = [float(p[y_axis.key]) for p in points]
        pz = [float(p[z_axis.key]) for p in points]

        sc = ax.scatter(
            px, py, pz,  # type: ignore[arg-type]
            c=pz, cmap=_STEEL_CMAP, norm=norm,
            s=50, edgecolors="white", linewidth=0.6, zorder=5, depthshade=False,
        )

        for j, (x, y, z) in enumerate(zip(px, py, pz)):
            ax.text(x, y, z, f" {j+1}", fontsize=6, color="#666", zorder=6)

        if all(z == int(z) for z in pz):
            ax.zaxis.set_major_locator(MaxNLocator(integer=True))  # type: ignore[attr-defined]

        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
        cb.set_label(z_axis.display_label, fontsize=9)

    ax.set_xlabel(x_axis.display_label, labelpad=8, fontsize=9)
    ax.set_ylabel(y_axis.display_label, labelpad=8, fontsize=9)
    ax.set_zlabel("Layer" if has_schedules else z_axis.display_label,  # type: ignore[attr-defined]
                  labelpad=8, fontsize=9)

    if x_axis.bounds:
        ax.set_xlim(*x_axis.bounds)
    if y_axis.bounds:
        ax.set_ylim(*y_axis.bounds)
    if not has_schedules and z_axis.bounds:
        ax.set_zlim(*z_axis.bounds)  # type: ignore[attr-defined]

    ax.view_init(elev=25, azim=-50)

    save_fig(save_path)
