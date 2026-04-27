"""Schema-agnostic plotting for the PFAB framework.

All plot functions accept ``AxisSpec`` objects that define which schema
parameters map to x/y axes, keeping the plotting layer decoupled from
any specific schema or domain.
"""

from ._style import (
    AxisSpec,
    save_fig,
    cmap,
    apply_style,
    clean_spines,
    clean_3d_panes,
    subplot_label,
    subplot_topology,
    add_kernel_radii_2d,
    add_kernel_radii_3d,
    cube_wireframe,
    square_wireframe,
    style_colorbar,
    STEEL_100, STEEL_300, STEEL_500, STEEL_700, STEEL_900,
    EMERALD_100, EMERALD_300, EMERALD_500, EMERALD_700, EMERALD_900,
    ZINC_50, ZINC_100, ZINC_200, ZINC_300, ZINC_400,
    ZINC_500, ZINC_600, ZINC_700, ZINC_800, ZINC_900,
    RED, YELLOW, ACCENT_RED, ACCENT_YELLOW,
)

from .baseline import plot_parameter_space, plot_parameter_space_per_cell, plot_parameter_space_3d, plot_dimensional_trajectories
from .prediction import plot_topology_comparison, plot_importance_weights
from .exploration import plot_acquisition
from .inference import plot_inference_result
from .schedule import plot_schedule_comparison
from .metrics import plot_metric_topology
from .performance import plot_performance_radar
from .convergence import plot_convergence
from .validation import plot_phase_proposals

__all__ = [
    "AxisSpec",
    "save_fig",
    "cmap",
    "apply_style",
    "clean_spines",
    "clean_3d_panes",
    "subplot_label",
    "subplot_topology",
    "add_kernel_radii_2d",
    "add_kernel_radii_3d",
    "cube_wireframe",
    "square_wireframe",
    "style_colorbar",
    "plot_parameter_space",
    "plot_parameter_space_per_cell",
    "plot_parameter_space_3d",
    "plot_dimensional_trajectories",
    "plot_topology_comparison",
    "plot_importance_weights",
    "plot_acquisition",
    "plot_inference_result",
    "plot_schedule_comparison",
    "plot_metric_topology",
    "plot_performance_radar",
    "plot_convergence",
    "plot_phase_proposals",
]
