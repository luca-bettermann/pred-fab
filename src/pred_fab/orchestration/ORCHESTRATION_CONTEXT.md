# Orchestration — Context

## Purpose
Coordinates all subsystems. `PfabAgent` is the user-facing API; the four sub-systems handle specific concerns.

## Systems

| Class | File | Role |
|-------|------|------|
| `PfabAgent` | `agent.py` | Registration, initialization, step methods, `configure_*` API |
| `FeatureSystem` | `features.py` | Runs feature models; writes tensors into ExperimentData |
| `EvaluationSystem` | `evaluation.py` | Tensor-typed batched eval (`_evaluate_feature_dict_tensor`) |
| `PredictionSystem` | `prediction.py` | Trains/infers prediction models. Canonical autoreg is `predict_for_calibration_tensor` (gradient-traversable). Owns per-model evidence (KDE) state. |
| `EvidenceEstimator` | `evidence.py` | Pluggable Δ∫E: `KernelFieldEstimator` (deterministic shells, gradient-traversable; probe count grows with D) or `SobolLocalEstimator` (QMC cube, fixed `n_samples` regardless of D — high-D escape hatch, also gradient-traversable). |
| `_choose_kde_regime` / `_resolve_kde_regime` | `evidence.py` | σ/D-aware regime dispatcher (dense / knn / cluster); only dense implemented, others log INFO and fall back |
| `CalibrationSystem` | `calibration/system.py` | Single-path optimisation orchestrator: Variables → SolutionSpace → Engine |
| `OptimizationEngine` | `calibration/engine.py` | Sobol → top-N → independent LBFGS with sigmoid bound reparam |
| `BoundsManager` | `calibration/bounds.py` | Schema-aware bounds + trust regions |
| `SolutionSpace` | `calibration/space.py` | Decision-vector layout from Variable objects; single `decode()` for all call sites |
| `Variable` / `StaticVariable` / `TrajectoryVariable` | `calibration/variables.py` | Optimization variable types — contract between step methods and optimizer |
| `EvidenceBackend` | `calibration/system.py` | Bundles the Δ∫E callbacks (batched_tensor / joint_batched_tensor) |

## Unified acquisition

Single κ-blend score for all modes — higher is better.

```
score(batch) = (1 − κ) · mean_k combined_score(perf(z_k), w_perf) + κ · Δ∫E(batch | data_old)
Δ∫E(z) = ∫_[0,1]^D [E(z | old ∪ batch) − E(z | old)] dz
E(z)   = D(z) / (1 + D(z))                ∈ [0, 1)   — actual evidence
D(z)   = Σ_j w_j · ρ_j(z)                  ≥ 0       — raw density
```

| κ | Mode | Active terms |
|---|------|---------------|
| 1.0 | Discovery | Δ∫E only |
| 0 < κ < 1 | Exploration | both |
| 0.0 | Inference | perf only |

Persistent κ default settable via `configure_exploration(kappa=...)`; `inference_step` hardcodes κ=0.

## Optimisation strategy — Sobol → LBFGS (single path)

Step methods compose `Variable` objects → `_optimize` builds `SolutionSpace` → engine runs.

| Component | Responsibility |
|---|---|
| Step method (`run_discovery`, etc.) | Compose `list[Variable]` from schema + config |
| `SolutionSpace` | Bounds, prior fill, `decode()` (single definition), `decode_to_specs()` |
| `OptimizationEngine` | Sobol → top-N → independent LBFGS → pick best |

Variable types: `StaticVariable` (1 value per experiment, optional integer with STE), `TrajectoryVariable` (midpoint + slope, tanh decode to L layers with 1/L weights). Both wrap the schema `DataObject` directly — no duplicated info.

Sigmoid bound reparam (`x = σ(z)·(hi−lo) + lo`) enforces strict bounds; no clipping.

## Step methods on PfabAgent

| Method | Mode | Notes |
|--------|------|-------|
| `discovery_step(n)` | DISCOVERY | κ=1; joint Δ∫E maximisation; empty-KDE init |
| `exploration_step(…)` | EXPLORATION | κ ∈ (0, 1); resolves to `configure_exploration(kappa=…)` default if not given |
| `inference_step(…)` | INFERENCE | κ=0 (hardcoded) |
| `adaptation_step(…)` | INFERENCE | Online tuning + trust region; optional MPC lookahead |

All return `ExperimentSpec(initial_params, trajectories)`.

## Perf closure wiring

`PfabAgent` wires `_perf_fn_tensor` (predict + eval, gradient-traversable) into `CalibrationSystem` at initialisation. Single perf path. Single-candidate reporting goes via `_compute_perf_dict_for_params` (calls the tensor closure with S=1).

## GPU + compile

`agent.to(device)` moves all `nn.Module` state (model networks, embeddings, normalisers). `engine.optimize(compile_objective=True)` wraps the acquisition objective with `torch.compile(dynamic=True)`; eager fallback on failure. KDE storage stays CPU-bound until needed.

## Configuration

```python
agent.configure_performance(weights={"perf_1": 2.0})
agent.configure_exploration(sigma=0.075, kappa=0.5)
agent.configure_evidence(estimator="kernel_field")
agent.configure_optimizer(n_starts=16, n_sobol=512, lr=0.05)
agent.configure_trajectory("speed", "n_layers")
agent.to("cuda")
```
