"""Shared visual identity helpers — palette and semantic colormaps.

Reference: knowledge-base/SKILLS - Visual Identity.md.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


# ---------- Palette (Tailwind-style stops) ----------

ZINC_100 = "#F4F4F5"
ZINC_200 = "#E4E4E7"
ZINC_300 = "#D4D4D8"
ZINC_400 = "#A1A1AA"
ZINC_500 = "#71717A"
ZINC_600 = "#52525B"
ZINC_700 = "#3F3F46"
ZINC_800 = "#27272A"
ZINC_900 = "#18181B"

STEEL_100 = "#D6E4F0"
STEEL_300 = "#8BB0CC"
STEEL_500 = "#4A7FA5"   # primary data
STEEL_700 = "#2D5F85"
STEEL_900 = "#1A3A5C"

EMERALD_100 = "#D1FAE5"
EMERALD_300 = "#6EE7B7"
EMERALD_500 = "#10B981"  # secondary data
EMERALD_700 = "#047857"
EMERALD_900 = "#064E3B"

RED = "#DC2626"
YELLOW = "#EAB308"


# ---------- Semantic colormaps ----------

def evidence_cmap() -> LinearSegmentedColormap:
    """Evidence `E(z) ∈ [0, 1]` — very light grey → light yellow → strong orange.

    Complements the `Blues` uncertainty colormap: uncertainty is the absence
    of evidence, so the two are semantically inverse. The heat sequence
    (grey → yellow → orange) reads as "activity / saturation" without
    competing with the blue uncertainty palette.

    Stops:
        0.00  very light grey   (absence)
        0.25  very light yellow (faint activity)
        0.50  yellow            (present)
        0.75  orange            (saturating)
        1.00  strong orange     (saturated)
    """
    return LinearSegmentedColormap.from_list(
        "evidence",
        ["#F4F4F5", "#FEF9C3", "#FDE047", "#F97316", "#C2410C"],
        N=256,
    )


# ---------- Matplotlib defaults ----------

def apply_style() -> None:
    """Set matplotlib rcParams to the visual-identity defaults."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titlecolor": ZINC_700,
        "axes.labelsize": 9,
        "axes.labelcolor": ZINC_600,
        "axes.edgecolor": ZINC_300,
        "axes.linewidth": 0.8,
        "xtick.color": ZINC_500,
        "ytick.color": ZINC_500,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def clean_spines(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(ZINC_300)
    ax.spines["bottom"].set_color(ZINC_300)


def clean_3d_panes(ax) -> None:
    """Transparent panes, light edges, muted ticks — 3blue1brown-style 3-D."""
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.0)
        pane.set_edgecolor(ZINC_300)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_tick_params(colors=ZINC_500, labelsize=7, pad=1)
        axis.label.set_color(ZINC_600)
        axis.line.set_color(ZINC_300)
    ax.grid(False)
