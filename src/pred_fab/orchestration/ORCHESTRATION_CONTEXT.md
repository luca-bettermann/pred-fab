# Orchestration — Context

## Purpose
Coordinates all subsystems. `PfabAgent` is the user-facing API; the four sub-systems handle specific concerns.

## Systems

| Class | File | Role |
|-------|------|------|
| `PfabAgent` | `agent.py` | Registration, initialization, step methods |
| `FeatureSystem` | `features.py` | Runs feature models; writes tensors into ExperimentData |
| `EvaluationSystem` | `evaluation.py` | Runs evaluation models; writes performance into ExperimentData |
| `PredictionSystem` | `prediction.py` | Trains/infers prediction models; provides uncertainty + similarity |
| `CalibrationSystem` | `calibration.py` | Optimization engine: UCB, inference, baseline, MPC |

## Calibration Architecture
`run_calibration(mode, current_params, target_indices, mpc_lookahead, …)` is the single optimization entry point.

- **Offline** (no `current_params`/`target_indices`): global bounds + random restarts
- **Online** (`current_params` + `target_indices`): trust-region bounds, single-shot
- **Step-grid** (`configure_step_parameter` + `current_params`): iterates over Cartesian product of dimension steps; only params whose mapped dimension transitions are free at each step
- **MPC** (`mpc_lookahead > 0`): wraps objective with N-step discounted lookahead via `_wrap_mpc_objective`; default 0 = greedy

`run_baseline(n)` is a separate entry point for LHS space-filling proposals (no model required).

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
| `baseline_step(n)` | BASELINE | Latin Hypercube Sampling, no trained model needed |
| `exploration_step(…)` | EXPLORATION | UCB acquisition |
| `inference_step(…)` | INFERENCE | Feature extraction + perf-max |
| `adaptation_step(…)` | INFERENCE | Online tuning + trust-region calibration; batch_size via `**kwargs` |
| `configure_step_parameter(code, dimension_code)` | — | Delegates to CalibrationSystem; declares runtime param to re-optimise per dimension step |

## Return Type
All step methods return `ExperimentSpec(initial_params, schedules)`.
`schedules` is non-empty only when trajectory dimensions are configured.

## perf_fn Closure
CalibrationSystem never calls PredictionSystem or EvaluationSystem directly.
PfabAgent wires them together via a `_perf_fn` closure at initialization time.
