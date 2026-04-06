# Orchestration — Context

## Purpose
Coordinates all subsystems. `PfabAgent` is the user-facing API; the four sub-systems handle specific concerns.

## Systems

| Class | File | Role |
|-------|------|------|
| `PfabAgent` | `agent.py` | Registration, initialization, step methods, single `configure()` entry point |
| `FeatureSystem` | `features.py` | Runs feature models; writes tensors into ExperimentData |
| `EvaluationSystem` | `evaluation.py` | Runs evaluation models; writes performance into ExperimentData |
| `PredictionSystem` | `prediction.py` | Trains/infers prediction models; per-model KDE uncertainty + similarity |
| `CalibrationSystem` | `calibration.py` | Optimization engine: UCB, inference, baseline, MPC |
| `Optimizer` | `calibration.py` | Enum: `LBFGSB` (gradient multi-start, fast) / `DE` (differential evolution, global) |

## Configuration — `agent.configure()`
All calibration and strategy configuration goes through a single call on `PfabAgent`:
```python
agent.configure(
    bounds={"param": (lo, hi)},          # parameter search bounds
    performance_weights={"perf": 2.0},   # calibration objective weights
    fixed_params={"param_3": "B"},       # locked values (e.g. categorical)
    adaptation_delta={"speed": 10.0},    # trust-region half-width (runtime params only)
    step_parameters={"speed": "n_layers"},  # param → dimension for trajectory stepping
    ofat_strategy=["speed"],             # OFAT cycling order (requires trust regions)
    exploration_radius=0.15,             # UCB exploration fraction
    optimizer=Optimizer.DE,              # optimization backend
    force=False,                         # re-configure after training if True
)
```
Users interact only with `PfabAgent` — no direct access to subsystems needed.

## Telemetry Properties on PfabAgent
After each calibration call, readable via:
- `agent.last_opt_score` — best objective value found
- `agent.last_opt_nfev` — total function evaluations
- `agent.last_opt_n_starts` — number of restarts / population evaluations

Convenience prediction methods (no calibration run):
- `agent.predict_performance(params)` → `Dict[str, float]`
- `agent.predict_uncertainty(params, datamodule)` → `float`

## Calibration Architecture
`run_calibration(mode, current_params, target_indices, mpc_lookahead, …)` is the single optimization entry point.

- **Offline** (no `current_params`/`target_indices`): global bounds + random restarts
- **Online** (`current_params` + `target_indices`): trust-region bounds, single-shot
- **Step-grid** (`step_parameters` configured): iterates over Cartesian product of dimension steps; only params whose mapped dimension transitions are free at each step
- **MPC** (`mpc_lookahead > 0`): wraps objective with N-step discounted lookahead via `_wrap_mpc_objective`; default 0 = greedy

`run_baseline(n)` is a separate entry point for LHS space-filling proposals (no model required).

### OFAT (One-Factor-At-a-Time) Strategy
Enabled via `agent.configure(ofat_strategy=[...])`. Only the currently active param is freed within
its trust region; all others are fixed to current. Index auto-advances after each online call.
Requires trust regions (adaptation_delta) to be set first.

### Context Features
`PfabAgent._context_snapshot` holds current measured context feature values (e.g. temperature, humidity).
Call `agent.update_context_snapshot(values)` before exploration/inference steps; the `_perf_fn` closure
merges these values into every parameter proposal before calling the prediction model — so context
covariates are correctly injected at calibration time without being part of the optimization space.

## Step Methods on PfabAgent

| Method | Mode | Notes |
|--------|------|-------|
| `baseline_step(n)` | BASELINE | Latin Hypercube Sampling, no trained model needed |
| `exploration_step(…)` | EXPLORATION | UCB acquisition |
| `inference_step(…)` | INFERENCE | Feature extraction + perf-max |
| `adaptation_step(…)` | INFERENCE | Online tuning + trust-region calibration; batch_size via `**kwargs` |

## Return Type
All step methods return `ExperimentSpec(initial_params, schedules)`.
`schedules` is non-empty only when trajectory dimensions are configured.

## perf_fn Closure
CalibrationSystem never calls PredictionSystem or EvaluationSystem directly.
PfabAgent wires them together via a `_perf_fn` closure at initialization time.
