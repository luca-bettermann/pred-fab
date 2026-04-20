# Orchestration — Context

## Purpose
Coordinates all subsystems. `PfabAgent` is the user-facing API; the four sub-systems handle specific concerns.

## Systems

| Class | File | Role |
|-------|------|------|
| `PfabAgent` | `agent.py` | Registration, initialization, step methods, explicit `configure_*()` methods |
| `FeatureSystem` | `features.py` | Runs feature models; writes tensors into ExperimentData |
| `EvaluationSystem` | `evaluation.py` | Runs evaluation models; writes performance into ExperimentData |
| `PredictionSystem` | `prediction.py` | Trains/infers prediction models; per-model KDE uncertainty + similarity |
| `CalibrationSystem` | `calibration.py` | Optimization engine: UCB, inference, baseline, MPC (online only) |
| `Optimizer` | `calibration.py` | Enum: `DE` (differential evolution + L-BFGS-B polish, default offline) / `LBFGSB` (gradient multi-start, default online) |

## Configuration
All calibration and strategy configuration goes through explicit methods on `PfabAgent`:
```python
agent.configure_performance(weights={"perf": 2.0})
agent.configure_exploration(radius=0.20)
agent.configure_optimizer(backend=Optimizer.DE, de_maxiter=100)
agent.configure_schedule("speed", "n_layers", delta=5.0, smoothing=0.25)
```

## Unified Acquisition Function
One optimizer, one equation, three modes controlled by a single parameter κ (kappa):
```
score = (1 - κ) · performance + κ · evidence
```
| κ | Mode | Purpose |
|---|------|---------|
| 1.0 | Baseline | Pure evidence — maximally-spaced coverage via virtual KDE points |
| 0 < κ < 1 | Exploration | Balance coverage + performance (default κ=0.5) |
| 0.0 | Inference | Pure performance — first-time-right manufacturing |

- **Performance** is normalized to its running observed range from training data.
- **Evidence** (KDE uncertainty) is inherently [0, 1] and not renormalized.
- **Baseline** uses `fit_empty_kde()` + iterative virtual point injection — no separate sampling algorithm (LHS/Sobol) needed.

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
- **Step-grid** (`schedule_configs` configured): iterates over Cartesian product of dimension steps; only params whose mapped dimension transitions are free at each step
- **MPC** (`mpc_lookahead > 0`, online only): wraps objective with N-step discounted lookahead via `_wrap_mpc_objective`; ignored in offline mode

`run_baseline(n)` is a separate entry point for space-filling proposals (no model required).

### Schedule Configuration
`configure_schedule(parameter, dimension, delta, smoothing)` declares that a runtime-adjustable parameter should vary per step of a dimension. Auto-delta (1/10 of param range) is applied when no explicit delta is provided. Schedule smoothing penalizes large jumps between adjacent steps (default λ=0.25).

### Context Features
`PfabAgent._context_snapshot` holds current measured context feature values (e.g. temperature, humidity).
Call `agent.update_context_snapshot(values)` before exploration/inference steps; the `_perf_fn` closure
merges these values into every parameter proposal before calling the prediction model — so context
covariates are correctly injected at calibration time without being part of the optimization space.

## Step Methods on PfabAgent

| Method | Mode | Notes |
|--------|------|-------|
| `baseline_step(n)` | BASELINE | κ=1 pure evidence, no trained model needed |
| `exploration_step(…)` | EXPLORATION | UCB acquisition |
| `inference_step(…)` | INFERENCE | Feature extraction + perf-max |
| `adaptation_step(…)` | INFERENCE | Online tuning + trust-region calibration; batch_size via `**kwargs` |

## Return Type
All step methods return `ExperimentSpec(initial_params, schedules)`.
`schedules` is non-empty only when schedule dimensions are configured.

## perf_fn Closure
CalibrationSystem never calls PredictionSystem or EvaluationSystem directly.
PfabAgent wires them together via a `_perf_fn` closure at initialization time.
