"""Schema-agnostic plotting for the PFAB framework.

All plot functions accept ``AxisSpec`` objects that define which schema
parameters map to x/y axes, keeping the plotting layer decoupled from
any specific schema or domain.
"""

from ._style import AxisSpec, save_fig

from .baseline import plot_parameter_space, plot_parameter_space_3d, plot_dimensional_trajectories
from .prediction import (
    plot_prediction_scatter,
    plot_topology_comparison,
    plot_importance_weights,
)
from .exploration import (
    plot_uncertainty_map,
    plot_acquisition,
    plot_optimizer_comparison,
)
from .inference import plot_inference_result, plot_inference_convergence
from .schedule import plot_schedule_comparison, plot_adaptation, plot_schedule_detail
from .metrics import plot_metric_topology, plot_cross_sections, plot_sensitivity
from .performance import plot_performance_radar
from .convergence import plot_convergence
from .validation import plot_phase_validation

__all__ = [
    "AxisSpec",
    "save_fig",
    "plot_parameter_space",
    "plot_parameter_space_3d",
    "plot_dimensional_trajectories",
    "plot_prediction_scatter",
    "plot_topology_comparison",
    "plot_importance_weights",
    "plot_uncertainty_map",
    "plot_acquisition",
    "plot_optimizer_comparison",
    "plot_inference_result",
    "plot_inference_convergence",
    "plot_schedule_comparison",
    "plot_adaptation",
    "plot_schedule_detail",
    "plot_metric_topology",
    "plot_cross_sections",
    "plot_sensitivity",
    "plot_performance_radar",
    "plot_convergence",
    "plot_phase_validation",
]
