# Orchestration — Context

## Purpose
Coordinates all subsystems. `PfabAgent` is the user-facing API; the four sub-systems handle specific concerns.

## Systems

| Class | File | Role |
|-------|------|------|
| `PfabAgent` | `agent.py` | Registration, initialization, step methods, explicit `configure_*()` methods |
| `FeatureSystem` | `features.py` | Runs feature models; writes tensors into ExperimentData |
| `EvaluationSystem` | `evaluation.py` | Runs evaluation models; writes performance into ExperimentData |
| `PredictionSystem` | `prediction.py` | Trains/infers prediction models; integrated-evidence math (Gaussian density, Sobol MC) |
| `CalibrationSystem` | `calibration/system.py` | Orchestrator composing OptimizationEngine + BoundsManager + SolutionSpace |
| `OptimizationEngine` | `calibration/engine.py` | DE, L-BFGS-B — pure numerical optimization |
| `BoundsManager` | `calibration/bounds.py` | Schema-aware bounds, trust regions, schedule configs |
| `SolutionSpace` | `calibration/space.py` | Decision vector layout, bounds, decode/encode, smoothing tie-breaker |
| `BaseOrchestrationSystem` | `base_system.py` | Shared base: logger, models, random_seed/rng |

## Unified Acquisition — Integrated Evidence

Single score function for all phases. Higher is better.

```
score(batch) = (1 − κ) · mean_k combined_score(perf(z_k), w_perf)
             + κ · Δ∫E(batch | data_old)

Δ∫E  = ∫_[0,1]^D [ E(z | data_old ∪ batch) − E(z | data_old) ] dz
E(z) = D(z) / (1 + D(z))                  ∈ [0, 1)   — actual evidence
D(z) = Σ_j w_j · ρ_j(z)                   ≥ 0       — raw density
ρ_j  = N(z; z_j, σ²I), normalized (mass w_j in ℝ^D)
```

- `Δ∫E` estimated by fixed Sobol quasi-MC: `M = round(2^(n_active_dims + mc_exponent_offset))`, default offset 3.0.
- σ length scale: `exploration_radius · √(n_active_dims)` by default; `sigma` override is accepted directly when the `√D` scaling is not desired.
- **No boundary term, no `+1` hack, no α correction.** Boundary behaviour emerges from the integration bounds `[0,1]^D`: a kernel centered near the wall has most of its mass outside the integrand's support, so its Δ∫E contribution is naturally small.

| κ | Mode | Active term(s) | Batch |
|---|------|-----------------|-------|
| 1.0 | Baseline | Δ∫E only (perf skipped) | N joint |
| 0 < κ < 1 | Exploration | both | 1 |
| 0.0 | Inference | perf only (Δ∫E skipped) | 1 |

### Per-candidate vs. stacked evidence
Datapoint weight `w_j` defaults to 1. Sequential-mode Phase 2 (fallback for very large problems) uses `w = L_j` at an unscheduled experiment's step0 to represent its anticipated total mass — preserving total-evidence accounting across placement order.

### Smoothing tie-breaker
`SolutionSpace.smoothing_penalty` adds a small step-to-step `Σ |step_s − step_(s−1)|²` term to the Phase-2 objective. Under the integrated objective the Δ∫E landscape is flat across degenerate placements; smoothing picks the monotonic/minimal-change trajectory among them. Engineering bias, not a statistical correction. Default weight 0.05.

## Three-Layer Optimization Architecture
1. **SolutionSpace** — defines decision vector layout, bounds, decode/encode, smoothing
2. **Objective** — evaluates decoded points via `CalibrationSystem._acquisition`
3. **Optimizer** — minimizes objective (DE / L-BFGS-B)

| Caller | Phase | N | L | κ | Variables |
|--------|-------|---|---|---|-----------|
| `run_baseline` | Process | n_exp | 1 | 1 | n_exp × D_flat (all params joint) |
| `run_baseline` | Schedule | n_exp | variable | 1 | n_exp × (L−1) × D_sched (all experiment×step joint) |
| `run_calibration` | Schedule | 1 | L | κ | L × D_sched (joint across layers) |
| `run_calibration` | Single | 1 | 1 | κ | D_flat (single candidate) |

## Two-Phase Baseline
Domain is merged into Process; there is no separate Riesz-energy phase.
1. **Process** (Phase 1) — joint N-point placement maximizing `Δ∫E` over `[0,1]^D` (continuous + integer + domain params).
2. **Schedule** (Phase 2) — joint across all experiments and all scheduled steps. Step-0 values fixed from Phase 1. Sequential-stacked fallback available for problems above ~200 joint variables.

## Configuration
```python
agent.configure_performance(weights={"perf": 2.0})
agent.configure_exploration(
    radius=0.09,              # σ = radius · √(n_active_dims)
    sigma=None,               # direct σ override, bypasses √D scaling
    mc_exponent_offset=3.0,   # M = round(2^(D + offset)) Sobol samples
)
agent.configure_optimizer(backend=Optimizer.DE, de_maxiter=1000)
agent.configure_schedule("speed", "n_layers", delta=5.0, smoothing=0.05)
```

## Step Methods on PfabAgent

| Method | Mode | Notes |
|--------|------|-------|
| `baseline_step(n)` | BASELINE | Joint Δ∫E maximization, empty evidence initialization |
| `exploration_step(…)` | EXPLORATION | Mixed perf + Δ∫E (κ ∈ (0,1)) |
| `inference_step(…)` | INFERENCE | κ=0 pure performance |
| `adaptation_step(…)` | INFERENCE | Online tuning + trust-region; optional MPC lookahead |

All step methods return `ExperimentSpec(initial_params, schedules)`.

## perf_fn Closure
CalibrationSystem never calls PredictionSystem or EvaluationSystem directly.
PfabAgent wires them together via a `_perf_fn` closure at initialization time.
Uncertainty path (visualization only) goes through `PredictionSystem.uncertainty`;
the acquisition path uses `PredictionSystem.delta_integrated_evidence_aggregated`.
