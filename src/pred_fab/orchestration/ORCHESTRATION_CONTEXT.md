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
| `CalibrationSystem` | `calibration/system.py` | Two-phase optimisation orchestrator |
| `OptimizationEngine` | `calibration/engine.py` | Torch-native DE (Phase 1) + multi-start gradient (LBFGS default) with sigmoid bound reparam (Phase 2/3) |
| `BoundsManager` | `calibration/bounds.py` | Schema-aware bounds + trust regions |
| `SolutionSpace` | `calibration/space.py` | Static-only decision-vector layout + bounds (gradient owns the trajectory path) |
| `EvidenceBackend` | `calibration/system.py` | Bundles the five Δ∫E callbacks (scalar / batched / joint_batched / batched_tensor / joint_batched_tensor) |

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
| 1.0 | Baseline | Δ∫E only |
| 0 < κ < 1 | Exploration | both |
| 0.0 | Inference | perf only |

Persistent κ default settable via `configure_exploration(kappa=...)`; `inference_step` hardcodes κ=0.

## Optimisation strategy — 2 phases

Internal dispatch; users do not pick backends.

| Phase | Method | When |
|---|---|---|
| **Phase 1 — Joint global** | DE on all dims with `integrality_mask`, fixed `maxiter=30, popsize=64`. Output: joint `(discrete*, continuous*)`. | Every step except Adaptation |
| **Phase 2 — Continuous refine** | Single LBFGS from Phase 1's continuous best; discretes fixed. Same path as Adaptation (with `x0 = current_params`). | When no trajectory |
| **Phase 3 — Trajectory** | Joint multi-start LBFGS over `(D_static + L_i × D_sched) × N` dims; Sobol+Boltzmann diverse starts; per-experiment static drift trust regions. Replaces the iterative-pass Gauss-Seidel scheme. | When `trajectory_configs` set |
| **Adaptation** | Single LBFGS, `x0 = current_params`, trust-region bounds. | Online refinement |

Sigmoid bound reparam (`x = σ(z)·(hi−lo) + lo`) enforces strict bounds in z-space; no clipping. Soft delta-constraint penalty for trajectory adjacency. Default `gradient_method = "lbfgs"` (Adam available).

## Step methods on PfabAgent

| Method | Mode | Notes |
|--------|------|-------|
| `baseline_step(n)` | BASELINE | κ=1; joint Δ∫E maximisation; empty-KDE init |
| `exploration_step(…)` | EXPLORATION | κ ∈ (0, 1); resolves to `configure_exploration(kappa=…)` default if not given |
| `inference_step(…)` | INFERENCE | κ=0 (hardcoded) |
| `adaptation_step(…)` | INFERENCE | Online tuning + trust region; optional MPC lookahead |

All return `ExperimentSpec(initial_params, trajectories)`.

## Perf closure wiring

`PfabAgent` wires `_perf_fn_tensor` (predict + eval, gradient-traversable) into `CalibrationSystem` at initialisation. Single perf path. Single-candidate reporting goes via `_compute_perf_dict_for_params` (calls the tensor closure with S=1).

## GPU + compile

`agent.to(device)` moves all `nn.Module` state (model networks, embeddings, normalisers). `engine.run_acquisition_gradient(compile_objective=True)` wraps the acquisition objective with `torch.compile(dynamic=True)`; eager fallback on failure. KDE storage stays CPU-bound until needed.

## Configuration

```python
agent.configure_performance(weights={"perf_1": 2.0})
agent.configure_exploration(sigma=0.075, kappa=0.5)
agent.configure_evidence(estimator="kernel_field")          # or "sobol_local" for high-D
agent.configure_optimizer(de_maxiter=30, de_popsize=64,
                          n_starts=8, n_iters=60, lr=0.05)
agent.configure_trajectory("speed", "n_layers", delta=5.0)
agent.to("cuda")
```
