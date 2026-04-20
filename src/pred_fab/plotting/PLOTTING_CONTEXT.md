# Plotting — Module Context

## Purpose

Schema-agnostic plotting for every PFAB phase. Users pass `AxisSpec` objects that map schema parameters to plot axes — no hardcoded field names. The mock (or any application) supplies data and axis definitions; this module renders the figures.

## Modules

| Module | Functions | Description |
|--------|-----------|-------------|
| `_style.py` | `AxisSpec`, `save_fig`, helpers | Shared dataclass, palette constants, axis/subtitle utilities |
| `baseline.py` | `plot_parameter_space`, `plot_parameter_space_3d` | 1×3: scatter + truth + model; 3D scatter with Zinc z-color |
| `prediction.py` | `plot_prediction_scatter`, `plot_topology_comparison`, `plot_importance_weights` | Pred vs actual, side-by-side topologies, R²_adj sigmoid |
| `exploration.py` | `plot_uncertainty_map`, `plot_acquisition`, `plot_optimizer_comparison` | Uncertainty 3-panel, acquisition 3-panel, per-optimizer scatter |
| `inference.py` | `plot_inference_result`, `plot_inference_convergence` | Single-shot result, trajectory + score convergence |
| `schedule.py` | `plot_schedule_comparison`, `plot_adaptation`, `plot_schedule_detail` | Fixed vs schedule bars, layer-by-layer adaptation, per-param schedule lines |
| `metrics.py` | `plot_metric_topology`, `plot_cross_sections`, `plot_sensitivity` | Per-metric + combined heatmaps, 1D slices, sensitivity bars |

## Key Concepts

- **`AxisSpec(key, label, unit, bounds)`** — frozen dataclass tying a dict key to display properties
- **`fixed_params`** — optional dict rendered as a small gray subtitle below the title
- **`save_fig(path)`** — saves with tight layout, closes figure, creates dirs as needed
- All 2D heatmaps use `contourf` + white contour overlay; colormaps follow SKILLS - Visual Identity

## Usage Pattern

```python
from pred_fab.plotting import AxisSpec, plot_acquisition

x = AxisSpec("water_ratio", "Water Ratio", bounds=(0.30, 0.50))
y = AxisSpec("print_speed", "Print Speed", unit="mm/s", bounds=(20, 60))

plot_acquisition(path, x, y, waters, speeds, perf, unc, combined,
                 points=experiments, proposed=proposal,
                 fixed_params={"n_layers": 5})
```

## What Does NOT Belong Here

- Domain-specific data generation (e.g., physics grid evaluation) → stays in the application
- 3D process visualizations tied to sensor objects → stays in the application
- Study-specific multi-figure compositions → orchestrated by the application
