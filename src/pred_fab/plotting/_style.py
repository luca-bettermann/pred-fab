"""Shared style utilities and AxisSpec for schema-agnostic PFAB plots."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Visual Identity — core palette
STEEL_500 = "#4A7FA5"
EMERALD_500 = "#10B981"
ZINC_400 = "#A1A1AA"
ZINC_500 = "#71717A"
ZINC_600 = "#52525B"
ZINC_700 = "#3F3F46"
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
