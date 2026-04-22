# Orchestration — Context

## Purpose
Coordinates all subsystems. `PfabAgent` is the user-facing API; the four sub-systems handle specific concerns.

## Systems

| Class | File | Role |
|-------|------|------|
| `PfabAgent` | `agent.py` | Registration, initialization, step methods, explicit `configure_*()` methods |
| `FeatureSystem` | `features.py` | Runs feature models; writes tensors into ExperimentData |
| `EvaluationSystem` | `evaluation.py` | Runs evaluation models; writes performance into ExperimentData |
| `PredictionSystem` | `prediction.py` | Trains/infers prediction models; evidence-based uncertainty + similarity |
| `CalibrationSystem` | `calibration.py` | Orchestrator composing OptimizationEngine + BoundsManager + SolutionSpace |
| `OptimizationEngine` | `calibration.py` | DE, L-BFGS-B, MPC — pure numerical optimization |
| `BoundsManager` | `calibration.py` | Schema-aware bounds, trust regions, schedule configs |
| `SolutionSpace` | `calibration.py` | Decision vector layout, bounds, decode/encode for DE optimization |
| `BaseOrchestrationSystem` | `base_system.py` | Shared base: logger, models, random_seed/rng |

## Unified Acquisition Function
One equation, three configurations:
```
score = (1 - κ) · performance + κ · u(x)
u(x) = 1 / (1 + E(x))
E(x) = Σ_real K(||x-z||/σ) + Σ_siblings K(||x-s||/σ) + E_boundary(x)
```

| κ | Mode | Purpose |
|---|------|---------|
| 1.0 | Baseline | Pure uncertainty — space-filling via evidence model |
| 0 < κ < 1 | Exploration | Balance coverage + performance (default κ=0.5) |
| 0.0 | Inference | Pure performance — first-time-right manufacturing |

One objective. Three configurations. Same code path.

### Evidence Model
- **Kernel:** Cauchy (default) or Gaussian, configurable via `kernel_type`
- **σ per model:** `σ = exploration_radius · √(n_active_dims)` — fixed, does NOT shrink with data
- **Data points:** Each segment = 1 evidence unit (no per-experiment normalization)
- **Boundary evidence:** `0.5 · K(d_lo/σ) + 0.5 · K(d_hi/σ)` per dimension

### Batch-aware UCB for Schedule phase
`_ucb_scores(X_batch, κ)` computes per-row UCB where each row sees the other L-1 rows as sibling evidence points. At L=1, reduces exactly to single-point uncertainty.

## Three-Layer Optimization Architecture
1. **SolutionSpace** — defines decision vector layout, bounds, decode/encode
2. **Objective** — evaluates decoded points (UCB / acquisition) — just a callable
3. **Optimizer** — minimizes objective (DE / L-BFGS-B) — `OptimizationEngine._run_de()` / `._run_lbfgsb()`

| Caller | N | L | Objective | Optimizer |
|--------|---|---|-----------|-----------|
| `run_baseline` (domain) | n_exp | 1 | Riesz energy (integer repulsion) | DE |
| `run_baseline` (process) | n_exp | 1 | UCB κ=1 (batch uncertainty) | DE |
| `run_baseline` (schedule) | n_exp | varies | UCB κ=1 + smoothing | DE |
| `run_calibration` (schedule) | 1 | L | UCB κ + smoothing | DE |
| `run_calibration` (offline, single) | 1 | 1 | UCB κ | DE |
| `run_calibration` (online, single) | 1 | 1 | UCB κ + MPC | L-BFGS-B |

## Three-Phase Baseline
When schedule dimensions are `DataDomainAxis` and not fixed:
1. **Domain** (Phase 1) — Riesz energy over domain axis params only (assigns structural values)
2. **Process** (Phase 2) — UCB κ=1 batch uncertainty maximization (space-filling)
3. **Schedule** (Phase 3) — fix initial params from Phase 2, UCB κ=1 over layer offsets

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
| `baseline_step(n)` | BASELINE | UCB κ=1 evidence model, no trained model needed |
| `exploration_step(…)` | EXPLORATION | UCB acquisition (κ > 0) |
| `inference_step(…)` | INFERENCE | κ=0 pure performance |
| `adaptation_step(…)` | INFERENCE | Online tuning + trust-region; optional MPC lookahead |

## Return Type
All step methods return `ExperimentSpec(initial_params, schedules)`.

## perf_fn Closure
CalibrationSystem never calls PredictionSystem or EvaluationSystem directly.
PfabAgent wires them together via a `_perf_fn` closure at initialization time.
