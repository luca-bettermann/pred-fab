"""Concept-script re-export of pred-fab's central style module.

Single source of truth: `pred_fab.plotting._style`. Concepts use the same
palette, colormaps, and helpers as production figures.
"""
from __future__ import annotations

from pred_fab.plotting._style import (  # noqa: F401
    # palette
    STEEL_100, STEEL_300, STEEL_500, STEEL_700, STEEL_900,
    EMERALD_100, EMERALD_300, EMERALD_500, EMERALD_700, EMERALD_900,
    ZINC_50, ZINC_100, ZINC_200, ZINC_300, ZINC_400,
    ZINC_500, ZINC_600, ZINC_700, ZINC_800, ZINC_900,
    RED, YELLOW, ACCENT_RED, ACCENT_YELLOW,
    # colormap registry
    cmap,
    # style helpers
    apply_style, clean_spines, clean_3d_panes, subplot_label, figure_subtitle,
    # geometric overlays
    add_kernel_radii_2d, add_kernel_radii_3d,
    cube_wireframe, square_wireframe,
    style_colorbar,
)
