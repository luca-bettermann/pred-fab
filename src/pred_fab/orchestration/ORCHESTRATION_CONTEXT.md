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
| `CalibrationSystem` | `calibration.py` | Orchestrator composing OptimizationEngine + BoundsManager + SolutionSpace |
| `OptimizationEngine` | `calibration.py` | DE, L-BFGS-B, smoothing, MPC — pure numerical optimization |
| `BoundsManager` | `calibration.py` | Schema-aware bounds, trust regions, schedule configs |
| `SolutionSpace` | `calibration.py` | Decision vector layout, bounds, decode/encode for DE optimization |
| `BaseOrchestrationSystem` | `base_system.py` | Shared base: logger, models, random_seed/rng |

## Unified Acquisition Function
One equation, two modes controlled by κ (kappa):
```
score = (1 - κ) · performance + κ · uncertainty
```
| κ | Mode | Purpose |
|---|------|---------|
| 0 < κ < 1 | Exploration | Balance coverage + performance (default κ=0.5) |
| 0.0 | Inference | Pure performance — first-time-right manufacturing |

Baseline uses Riesz energy particle repulsion (separate objective, same engine).

## Three-Layer Optimization Architecture
1. **SolutionSpace** — defines decision vector layout, bounds, decode/encode
2. **Objective** — evaluates decoded points (riesz energy / acquisition) — just a callable
3. **Optimizer** — minimizes objective (DE / L-BFGS-B) — `OptimizationEngine._run_de()` / `._run_lbfgsb()`

`SolutionSpace` manages:
- **Vector layout per experiment:** `[static, sched_step0, offset_1, ..., offset_{L-1}]`
- **Hard delta constraints** via offset bounds `[-δ, +δ]`
- **Decode** to normalized point arrays
- **Smoothing penalty** (additive, self-scaling)
- **Init population** (warm-started DE)
- **Decode to specs** (convert back to ExperimentSpec)

| Caller | N | L | Objective | Optimizer |
|--------|---|---|-----------|-----------|
| `run_baseline` (no schedule) | n_exp | 1 | Riesz energy | DE |
| `run_baseline` (schedule, unfixed dim) | n_exp | varies | Two-phase: structural → process | DE |
| `run_baseline` (schedule, fixed dim) | n_exp | L | Riesz energy | DE |
| `run_calibration` (schedule) | 1 | L | Acquisition via SolutionSpace | DE |
| `run_calibration` (offline, single) | 1 | 1 | Acquisition | DE |
| `run_calibration` (online, single) | 1 | 1 | Acquisition + MPC | L-BFGS-B |

## Two-Phase Baseline
When schedule dimensions are `DataDomainAxis` and not fixed:
1. **Structural** — repulsion over all dims (assigns domain axis values naturally)
2. **Process** — group by structural values, run schedule optimization per group

## Configuration
```python
agent.configure_performance(weights={"perf": 2.0})
agent.configure_exploration(radius=0.20)
agent.configure_optimizer(backend=Optimizer.DE, de_maxiter=1000)
agent.configure_schedule("speed", "n_layers", delta=5.0, smoothing=0.25)
```

## Step Methods on PfabAgent

| Method | Mode | Notes |
|--------|------|-------|
| `baseline_step(n)` | BASELINE | Riesz energy, no trained model needed |
| `exploration_step(…)` | EXPLORATION | UCB acquisition (κ > 0) |
| `inference_step(…)` | INFERENCE | κ=0 pure performance |
| `adaptation_step(…)` | INFERENCE | Online tuning + trust-region; optional MPC lookahead |

## Return Type
All step methods return `ExperimentSpec(initial_params, schedules)`.

## perf_fn Closure
CalibrationSystem never calls PredictionSystem or EvaluationSystem directly.
PfabAgent wires them together via a `_perf_fn` closure at initialization time.
