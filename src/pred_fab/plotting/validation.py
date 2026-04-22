"""Phase validation plots: what each optimizer saw."""

from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ._style import (
    AxisSpec, save_fig,
    STEEL_500, ZINC_200, ZINC_400, ZINC_600, ZINC_700,
)


def plot_phase_validation(
    save_path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    *,
    domain_values: list[dict[str, int]] | None = None,
    process_points: list[dict[str, Any]] | None = None,
    schedule_points: np.ndarray | None = None,
    schedule_exp_ids: list[int] | None = None,
    title: str = "Phase Validation",
) -> None:
    """Per-phase diagnostic: what each optimizer's repulsion space looked like.

    Panels adapt to available data: Domain (if provided), Process, Schedule (if provided).
    """
    panels: list[tuple[str, str]] = []
    if domain_values is not None:
        panels.append(("domain", "Domain"))
    if process_points is not None:
        panels.append(("process", "Process"))
    if schedule_points is not None:
        panels.append(("schedule", "Schedule"))

    if not panels:
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold", color=ZINC_700, y=1.02)

    for ax, (panel_type, panel_title) in zip(axes, panels):
        ax.set_title(panel_title, fontsize=10, color=ZINC_600)
        ax.grid(True, alpha=0.2, color=ZINC_200)

        if panel_type == "domain":
            _draw_domain(ax, domain_values)  # type: ignore[arg-type]
        elif panel_type == "process":
            _draw_process(ax, process_points, x_axis, y_axis)  # type: ignore[arg-type]
        elif panel_type == "schedule":
            _draw_schedule(ax, schedule_points, schedule_exp_ids,  # type: ignore[arg-type]
                          x_axis, y_axis, process_points)

    save_fig(save_path)


def _draw_domain(
    ax: plt.Axes,  # type: ignore[name-defined]
    domain_values: list[dict[str, int]],
) -> None:
    """2D scatter of domain parameter assignments (always show all domain dims)."""
    if not domain_values:
        return

    codes = sorted(domain_values[0].keys())
    n = len(domain_values)

    if len(codes) < 2:
        # Only one domain dim — add a dummy y=0 axis
        code = codes[0]
        vals = [dv[code] for dv in domain_values]
        ax.scatter(vals, [0] * n, s=60, c=STEEL_500, edgecolors="white",
                   linewidth=0.8, zorder=5)
        for i, v in enumerate(vals):
            ax.annotate(f"{i+1}", (v, 0), fontsize=7, ha="center", va="bottom",
                       xytext=(0, 8), textcoords="offset points", color=ZINC_400)
        ax.set_xlabel(code, fontsize=9, color=ZINC_600)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        # 2D scatter — all domain dims
        xs = [dv[codes[0]] for dv in domain_values]
        ys = [dv[codes[1]] for dv in domain_values]
        ax.scatter(xs, ys, s=60, c=STEEL_500, edgecolors="white",
                   linewidth=0.8, zorder=5)
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax.annotate(f"{i+1}", (x, y), fontsize=7, ha="center", va="bottom",
                       xytext=(0, 5), textcoords="offset points", color=ZINC_400)
        ax.set_xlabel(codes[0], fontsize=9, color=ZINC_600)
        ax.set_ylabel(codes[1], fontsize=9, color=ZINC_600)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def _draw_process(
    ax: plt.Axes,  # type: ignore[name-defined]
    process_points: list[dict[str, Any]],
    x_axis: AxisSpec,
    y_axis: AxisSpec,
) -> None:
    """2D scatter of flat process parameters."""
    px = [float(p.get(x_axis.key, 0)) for p in process_points]
    py = [float(p.get(y_axis.key, 0)) for p in process_points]

    ax.scatter(px, py, s=60, c=STEEL_500, edgecolors="white",
               linewidth=0.8, zorder=5)
    for i, (x, y) in enumerate(zip(px, py)):
        ax.annotate(f"{i+1}", (x, y), fontsize=7, ha="center", va="bottom",
                   xytext=(0, 5), textcoords="offset points", color=ZINC_400)

    ax.set_xlabel(x_axis.display_label, fontsize=9, color=ZINC_600)
    ax.set_ylabel(y_axis.display_label, fontsize=9, color=ZINC_600)
    if x_axis.bounds:
        ax.set_xlim(*x_axis.bounds)
    if y_axis.bounds:
        ax.set_ylim(*y_axis.bounds)


def _draw_schedule(
    ax: plt.Axes,  # type: ignore[name-defined]
    schedule_points: np.ndarray,
    schedule_exp_ids: list[int] | None,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    process_points: list[dict[str, Any]] | None,
) -> None:
    """2D scatter of all N_total layer-points — what the schedule optimizer saw."""
    n_total = schedule_points.shape[0]

    # Color by experiment ID if available
    if schedule_exp_ids is not None:
        n_exp = max(schedule_exp_ids) + 1
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap
        steel_cmap = LinearSegmentedColormap.from_list(
            "steel_v", ["#D6E4F0", "#8BB0CC", "#4A7FA5", "#2D5F85", "#1A3A5C"]
        )
        norm = Normalize(vmin=0, vmax=max(n_exp - 1, 1))
        colors = [steel_cmap(norm(eid)) for eid in schedule_exp_ids]
    else:
        colors = [STEEL_500] * n_total  # type: ignore[list-item]

    # The schedule optimizer only sees sched dims — map to display axes
    # schedule_points columns are sched params in normalized [0,1] space
    # For display, we need the actual axis values
    # If process_points provided, x-axis values come from there (shared per experiment)
    if process_points and schedule_exp_ids:
        px = [float(process_points[eid].get(x_axis.key, 0)) for eid in schedule_exp_ids]
    else:
        px = [float(x_axis.bounds[0] + schedule_points[j, 0] * (x_axis.bounds[1] - x_axis.bounds[0]))
              if x_axis.bounds else float(schedule_points[j, 0])
              for j in range(n_total)]

    # y-axis = the scheduled param (speed), from schedule_points
    # schedule_points are in normalized [0,1], denormalize using y_axis bounds
    if y_axis.bounds:
        lo, hi = y_axis.bounds
        py = [float(lo + schedule_points[j, 0] * (hi - lo)) for j in range(n_total)]
    else:
        py = [float(schedule_points[j, 0]) for j in range(n_total)]

    # Connect dots per experiment with grey lines and add experiment number at step0
    if schedule_exp_ids is not None:
        n_exp = max(schedule_exp_ids) + 1
        for eid in range(n_exp):
            mask = [j for j, e in enumerate(schedule_exp_ids) if e == eid]
            if len(mask) > 1:
                ex = [px[j] for j in mask]
                ey = [py[j] for j in mask]
                ax.plot(ex, ey, color=ZINC_400, linewidth=0.6, alpha=0.4, zorder=1)
            if mask:
                ax.annotate(f"{eid+1}", (px[mask[0]], py[mask[0]]), fontsize=7,
                           ha="center", va="bottom", xytext=(0, 5),
                           textcoords="offset points", color=ZINC_400)

    ax.scatter(px, py, s=30, c=colors, edgecolors="white",
               linewidth=0.4, zorder=5, alpha=0.8)

    ax.set_xlabel(x_axis.display_label, fontsize=9, color=ZINC_600)
    ax.set_ylabel(y_axis.display_label, fontsize=9, color=ZINC_600)
    if x_axis.bounds:
        ax.set_xlim(*x_axis.bounds)
    if y_axis.bounds:
        ax.set_ylim(*y_axis.bounds)
