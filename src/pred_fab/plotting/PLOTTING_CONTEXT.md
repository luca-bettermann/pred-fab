# Plotting — Module Context

## Purpose

Schema-agnostic plotting for every PFAB phase. Users pass `AxisSpec` objects that map schema parameters to plot axes — no hardcoded field names. The mock (or any application) supplies data and axis definitions; this module renders the figures.

## Modules

| Module | Functions | Description |
|--------|-----------|-------------|
| `_style.py` | `AxisSpec`, `save_fig`, helpers | Shared dataclass, palette constants, axis/subtitle utilities |
| `discovery.py` | `plot_parameter_space`, `plot_parameter_space_per_cell`, `plot_mean_error_topology`, `plot_parameter_space_3d`, `plot_dimensional_trajectories` | parameter-space scatter + truth + model; per-cell & mean-error topology; 3D scatter with Zinc z-color |
| `prediction.py` | `plot_topology_comparison`, `plot_importance_weights` | Side-by-side topologies, R²_inf sigmoid |
| `exploration.py` | `plot_acquisition` | 3-panel: performance + evidence + combined acquisition |
| `inference.py` | `plot_inference_result` | Single-shot result on predicted topology |
| `schedule.py` | `plot_schedule_comparison` | Fixed vs schedule bars + per-step parameter schedules |
| `performance.py` | `plot_performance_radar` | Radar/spider per-attribute plot with dataset avg overlay, score colors |
| `metrics.py` | `plot_metric_topology` | Per-metric + combined heatmaps |
| `convergence.py` | `plot_convergence` | Optimizer convergence curves per phase |
| `validation.py` | `plot_phase_proposals` | Per-phase scatter overlay on uncertainty topology |

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
