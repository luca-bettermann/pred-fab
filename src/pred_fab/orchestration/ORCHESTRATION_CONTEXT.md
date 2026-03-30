# Orchestration — Context

## Purpose
Coordinates all subsystems. `PfabAgent` is the user-facing API; the four sub-systems handle specific concerns.

## Systems

| Class | File | Role |
|-------|------|------|
| `PfabAgent` | `agent.py` | Registration, initialization, step methods |
| `FeatureSystem` | `features.py` | Runs feature models; writes tensors into ExperimentData |
| `EvaluationSystem` | `evaluation.py` | Runs evaluation models; writes performance into ExperimentData |
| `PredictionSystem` | `prediction.py` | Trains/infers prediction models |
| `CalibrationSystem` | `calibration.py` | Optimization engine: UCB exploration, inference, greedy-maximin baseline |

## Uncertainty Estimation (GP Surrogate)

`PfabAgent` owns a `GaussianProcessSurrogate` instance (`self._gp_surrogate`).
After each `train()` call, it fits the GP on experiment-level `(params → performance)` data from the training split.
The `uncertainty_fn` injected into `CalibrationSystem` is a closure over the GP:
- Returns `1.0` (maximum uncertainty) before the GP is fitted
- Returns `clip(mean(GP std), 0, 1)` after fitting

## Calibration Architecture

`run_calibration(mode, current_params, target_indices, …)` is the single optimization entry point.

- **Offline** (no `current_params`/`target_indices`): global bounds + random restarts
- **Online** (`current_params` + `target_indices`): trust-region bounds, single-shot

`run_baseline(n)` is a separate entry point for space-filling proposals (no model required).

### OFAT (One-Factor-At-a-Time) Strategy
`CalibrationSystem.configure_ofat_strategy(codes)` enables layer-level OFAT cycling for online steps:
only the currently active param (cycled in order through `codes`) is freed within its trust region;
all other trust-region params are fixed to current. Index auto-advances after each online call.
Configure via `PfabAgent.configure_ofat_strategy(codes)` (requires trust regions already set).

### Context Features
`PfabAgent._context_snapshot` holds current measured context feature values (e.g. temperature, humidity).
Call `agent.update_context_snapshot(values)` before exploration/inference steps; the `_perf_fn` closure
merges these values into every parameter proposal before calling the prediction model — so context
covariates are correctly injected at calibration time without being part of the optimization space.

## Step Methods on PfabAgent

| Method | Mode | Notes |
|--------|------|-------|
| `baseline_step(n)` | BASELINE | Greedy maximin, no trained model needed |
| `exploration_step(…)` | EXPLORATION | UCB acquisition using GP uncertainty |
| `inference_step(…)` | INFERENCE | Feature extraction + perf-max |

## Return Type
All step methods return `ExperimentSpec(initial_params, schedules={})`.
`schedules` is always empty in this release (no trajectory/MPC).

## perf_fn / uncertainty_fn / similarity_fn Closures
`CalibrationSystem` never calls subsystems directly.
`PfabAgent` wires them together via closures at initialization time.
