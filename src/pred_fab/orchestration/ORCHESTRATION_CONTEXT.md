# Orchestration — Context

## Purpose
Coordinates all subsystems. `PfabAgent` is the user-facing API; the four sub-systems handle specific concerns.

## Systems

| Class | File | Role |
|-------|------|------|
| `PfabAgent` | `agent.py` | Registration, initialization, step methods, `configure_*` API |
| `FeatureSystem` | `features.py` | Runs feature models; writes tensors into ExperimentData |
| `EvaluationSystem` | `evaluation.py` | Runs evaluation models; tensor-typed batched eval (`_evaluate_feature_dict_tensor`) |
| `PredictionSystem` | `prediction.py` | Trains/infers prediction models; canonical autoreg is `predict_for_calibration_tensor` (gradient-traversable). Owns per-model evidence (KDE) state. |
| `EvidenceEstimator` | `evidence.py` | Pluggable ∫D/(1+D) dz: `KernelFieldEstimator` (deterministic shells, gradient-traversable) or `SobolLocalEstimator` (QMC cube, numpy-only) |
| `KernelIndex` | `evidence.py` | `torch.cdist + 5σ mask` density lookup over kernel centres |
| `_choose_kde_regime` | `evidence.py` | σ/D-aware regime dispatcher (dense / knn / cluster); only dense implemented, others log INFO and fall back |
| `CalibrationSystem` | `calibration/system.py` | Orchestrator composing OptimizationEngine + BoundsManager + SolutionSpace |
| `OptimizationEngine` | `calibration/engine.py` | Torch-native DE (integer-aware) + multi-start gradient (Adam / LBFGS) with sigmoid bound reparam |
| `BoundsManager` | `calibration/bounds.py` | Schema-aware bounds + trust regions |
| `SolutionSpace` | `calibration/space.py` | Decision vector layout + bounds (static-only; gradient owns the schedule path) |
| `EvidenceBackend` | `calibration/system.py` | Bundles the five Δ∫E callbacks (scalar / batched / joint_batched / batched_tensor / joint_batched_tensor) |

## Unified acquisition

Single score for all phases — higher is better.

```
score(batch) = (1 − κ) · mean_k combined_score(perf(z_k), w_perf) + κ · Δ∫E(batch | data_old)
Δ∫E(z) = ∫_[0,1]^D [E(z | old ∪ batch) − E(z | old)] dz
E(z)   = D(z) / (1 + D(z))                ∈ [0, 1)   — actual evidence
D(z)   = Σ_j w_j · ρ_j(z)                  ≥ 0       — raw density
```

| κ | Mode | Active terms |
|---|------|---------------|
| 1.0 | Baseline | Δ∫E only |
| 0 < κ < 1 | Exploration | both |
| 0.0 | Inference | perf only |

## Optimizer dispatch

`Optimizer.GRADIENT` (default) — `torch.optim` multi-start with sigmoid bound reparam (`x = σ(z)·(hi−lo)+lo`); strict bounds, autograd-traversable. Smart inits via Sobol + Boltzmann selection.

`Optimizer.DE` — torch-native DE for integer-only phases (Domain phase categoricals / domain axes). Integer-aware mutation, no-improvement-window halt.

The schedule path (`_phase3_schedule` → `_optimise_schedule_for_experiment`) is gradient-only with absolute-step encoding (each step in `[0, 1]` z-space, sigmoid-reparam'd; soft delta-constraint penalty).

## Step methods on PfabAgent

| Method | Mode | Notes |
|--------|------|-------|
| `baseline_step(n)` | BASELINE | κ=1; joint Δ∫E maximisation; empty-KDE init |
| `exploration_step(…)` | EXPLORATION | κ ∈ (0, 1) |
| `inference_step(…)` | INFERENCE | κ=0 |
| `adaptation_step(…)` | INFERENCE | Online tuning + trust region; optional MPC lookahead |

All return `ExperimentSpec(initial_params, schedules)`.

## Perf closure wiring

`PfabAgent` wires `_perf_fn_tensor` (predict + eval, gradient-traversable) into `CalibrationSystem` at initialisation. There is one perf path. Single-candidate reporting goes via `_compute_perf_dict_for_params` (calls the tensor closure with `S=1`).

## GPU + compile

`agent.to(device)` moves all `nn.Module` state (model networks, embeddings, normalisers). `engine.run_acquisition_gradient(compile_objective=True)` wraps the acquisition objective with `torch.compile(dynamic=True)`; eager fallback on failure. KDE storage stays CPU-bound until needed.

## Configuration

```python
agent.configure_performance(weights={"perf_1": 2.0})
agent.configure_exploration(sigma=0.075)
agent.configure_optimizer(backend=Optimizer.GRADIENT, gradient_n_starts=4, gradient_n_iters=60)
agent.configure_schedule("speed", "n_layers", delta=5.0)
```
