"""Shared style utilities and AxisSpec for schema-agnostic PFAB plots."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Visual Identity — core palette
STEEL_100 = "#D6E4F0"
STEEL_300 = "#8BB0CC"
STEEL_500 = "#4A7FA5"
STEEL_700 = "#2D5F85"
STEEL_900 = "#1A3A5C"
EMERALD_100 = "#D1FAE5"
EMERALD_300 = "#6EE7B7"
EMERALD_500 = "#10B981"
ZINC_50 = "#FAFAFA"
ZINC_100 = "#F4F4F5"
ZINC_200 = "#E4E4E7"
ZINC_300 = "#D4D4D8"
ZINC_400 = "#A1A1AA"
ZINC_500 = "#71717A"
ZINC_600 = "#52525B"
ZINC_700 = "#3F3F46"
ZINC_800 = "#27272A"
ZINC_900 = "#18181B"
ACCENT_YELLOW = "#EAB308"
ACCENT_RED = "#DC2626"


@dataclass(frozen=True)
class AxisSpec:
    """Definition of a plot axis tied to a schema parameter."""

    key: str
    label: str
    unit: str = ""
    bounds: tuple[float, float] | None = None

    @property
    def display_label(self) -> str:
        """Label with unit suffix for axis display."""
        return f"{self.label} [{self.unit}]" if self.unit else self.label


def save_fig(path: str, dpi: int = 150) -> None:
    """Save current figure and close."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _extract_xy(
    points: list[dict[str, Any]],
    x_axis: AxisSpec,
    y_axis: AxisSpec,
) -> tuple[list[float], list[float]]:
    """Extract x and y coordinate lists from point dicts."""
    return (
        [float(p[x_axis.key]) for p in points],
        [float(p[y_axis.key]) for p in points],
    )


def _add_fixed_subtitle(
    fig: plt.Figure,  # type: ignore[name-defined]
    fixed_params: dict[str, Any] | None,
) -> None:
    """Add a small gray subtitle showing fixed parameter values."""
    if not fixed_params:
        return
    parts = [f"{k} = {v}" for k, v in fixed_params.items()]
    text = "fixed: " + ", ".join(parts)
    fig.text(0.5, 0.98, text, ha="center", va="top",
             fontsize=7, color=ZINC_400, style="italic")


def _plot_schedule_ranges(
    ax: "plt.Axes",  # type: ignore[name-defined]
    points: list[dict[str, Any]],
    x_axis: "AxisSpec",
    y_axis: "AxisSpec",
    schedules: dict[str, list[dict[str, Any]]] | None,
    codes: list[str] | None,
    *,
    color: str = ZINC_400,
    alpha: float = 0.5,
    linewidth: float = 1.2,
    cap_size: float = 3.0,
) -> None:
    """Draw T-ended range lines for scheduled parameters on a 2D scatter.

    For each experiment with a schedule, draws a vertical or horizontal line
    showing the min-max range of the scheduled parameter across layers.
    """
    if not schedules or not codes:
        return

    for i, code in enumerate(codes):
        if code not in schedules or len(schedules[code]) <= 1:
            continue

        steps = schedules[code]
        x_vals = [float(s.get(x_axis.key, points[i].get(x_axis.key, 0))) for s in steps]
        y_vals = [float(s.get(y_axis.key, points[i].get(y_axis.key, 0))) for s in steps]

        x_varies = max(x_vals) - min(x_vals) > 1e-8
        y_varies = max(y_vals) - min(y_vals) > 1e-8

        if y_varies:
            # Vertical range line (speed varies across layers)
            x_center = float(points[i].get(x_axis.key, 0))
            ax.plot([x_center, x_center], [min(y_vals), max(y_vals)],
                    color=color, alpha=alpha, linewidth=linewidth, zorder=1)
            # T caps
            x_span = (x_axis.bounds[1] - x_axis.bounds[0]) if x_axis.bounds else 1.0
            cap = x_span * 0.008 * cap_size
            for y_end in [min(y_vals), max(y_vals)]:
                ax.plot([x_center - cap, x_center + cap], [y_end, y_end],
                        color=color, alpha=alpha, linewidth=linewidth, zorder=1)

        if x_varies:
            # Horizontal range line (water varies across layers)
            y_center = float(points[i].get(y_axis.key, 0))
            ax.plot([min(x_vals), max(x_vals)], [y_center, y_center],
                    color=color, alpha=alpha, linewidth=linewidth, zorder=1)
            y_span = (y_axis.bounds[1] - y_axis.bounds[0]) if y_axis.bounds else 1.0
            cap = y_span * 0.008 * cap_size
            for x_end in [min(x_vals), max(x_vals)]:
                ax.plot([x_end, x_end], [y_center - cap, y_center + cap],
                        color=color, alpha=alpha, linewidth=linewidth, zorder=1)


def _apply_axes(
    ax: plt.Axes,  # type: ignore[name-defined]
    x_axis: AxisSpec,
    y_axis: AxisSpec,
) -> None:
    """Set axis labels and optional bounds."""
    ax.set_xlabel(x_axis.display_label)
    ax.set_ylabel(y_axis.display_label)
    if x_axis.bounds:
        ax.set_xlim(*x_axis.bounds)
    if y_axis.bounds:
        ax.set_ylim(*y_axis.bounds)
