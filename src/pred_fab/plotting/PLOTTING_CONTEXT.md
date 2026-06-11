# Plotting — Module Context

## Purpose

Schema-agnostic plotting for every PFAB phase. Users pass `AxisSpec` objects that map schema parameters to plot axes — no hardcoded field names. The mock (or any application) supplies data and axis definitions; this module renders the figures.

## Modules

| Module | Functions | Description |
|--------|-----------|-------------|
| `_style.py` | `AxisSpec`, `save_fig`, `subplot_topology`, `add_evidence_fade`, `annotate_point`, `row_colorbar`, `marginal_layout`/`draw_marginal_slices`, `fig_size`, `set_publication_mode`, helpers | Shared style SSOT: palette, semantic surfaces + progression ramps, evidence-aware rendering, publication mode |
| `discovery.py` | `plot_parameter_space`, `plot_parameter_space_per_cell`, `plot_mean_error_topology`, `plot_parameter_space_3d`, `plot_dimensional_trajectories` | parameter-space scatter + truth + model; per-cell & mean-error topology; 3D scatter |
| `prediction.py` | `plot_topology_comparison`, `plot_importance_weights`, `overlay_diagnosed_points` | Side-by-side topologies (shared colorbar), R²_inf sigmoid, CV error-vs-coverage point overlay |
| `evolution.py` | `plot_topology_evolution` | Small-multiples round strip: shared scale, one colorbar, optional truth panel / per-round evidence fades / cumulative points |
| `exploration.py` | `plot_acquisition` | 3-panel: performance (evidence-faded) + evidence gain (fit-to-data) + combined acquisition |
| `inference.py` | `plot_inference_result` | Single-shot result on predicted topology; opt-in marginal slices through the proposal |
| `trajectory.py` | `plot_trajectory_comparison` | Fixed vs trajectory bars + per-step parameter trajectories |
| `performance.py` | `plot_performance_radar` | Radar/spider per-attribute plot with dataset avg overlay |
| `metrics.py` | `plot_metric_topology` | Per-metric + combined heatmaps (all on the performance scale) |
| `convergence.py` | `plot_convergence` | Optimizer convergence curves per phase (progression ramps) |
| `validation.py` | `plot_phase_proposals` | Per-phase scatter overlay on uncertainty topology |
| `evidence.py` | `plot_evidence_panel`, `plot_multi_angle`, `expand_experiments` | Pre-computed ΔE grid panels with experiment/trajectory overlays |
| `parallel.py` | `plot_parallel_coordinates` | One vertical axis per parameter, proposals in Yellow |
| `sensitivity.py` | `plot_sensitivity_matrix` | Sobol S_T annotated heatmap |

## Key Concepts

- **`AxisSpec(key, label, unit, bounds)`** — frozen dataclass tying a dict key to display properties
- **Bounded scales by default** — semantic surfaces with `bounded=True` render on [0,1] (fixed level edges) so identical values look identical across figures; `fit_to_data=True` per call fits the range instead (e.g. small ΔE gain fields). Explicit `vmin`/`vmax` always win.
- **Evidence-aware rendering** — `evidence_grid=` on `subplot_topology` (and threaded through the figure functions) fades low-evidence regions toward the paper and draws the dashed `E = TRUST_THRESHOLD` trust boundary. Model-derived surfaces only; truth panels and decision surfaces stay unfaded.
- **Direct labeling** — sparse inline contour labels (haloed), `annotate_point` value labels at proposal/optimum markers (auto-flip near edges), `cbar_mark` for a reference tick on colorbars. Legends only where direct labels can't carry it.
- **Publication mode** — `set_publication_mode(True)`: typography-table fonts, `fig_size` scales rows to column/page width, `save_fig` writes vector PDF + 300-dpi PNG (dev default: 150-dpi PNG). `save_fig(metadata=...)` embeds provenance into PNG text chunks / PDF Subject.
- **`save_fig(path)`** — the only save path (all figures route through it); handles tight vs constrained layout, closes the figure, creates dirs.
- All 2D heatmaps go through `subplot_topology`; colormaps follow [[SKILLS - Visual Identity]] via the `SURFACES` registry — never hardcode cmap names.

## Usage Pattern

```python
from pred_fab.plotting import AxisSpec, plot_acquisition

x = AxisSpec("water_ratio", "Water Ratio", bounds=(0.30, 0.50))
y = AxisSpec("print_speed", "Print Speed", unit="mm/s", bounds=(20, 60))

plot_acquisition(path, x, y, waters, speeds, perf, gain, combined,
                 points=experiments, proposed=proposal,
                 evidence_grid=evidence,           # from compute_evidence_grids
                 fixed_params={"n_layers": 5})
```

## What Does NOT Belong Here

- Domain-specific data generation (e.g., physics grid evaluation) → stays in the application
- 3D process visualizations tied to sensor objects → stays in the application
- Study-specific multi-figure compositions → orchestrated by the application
