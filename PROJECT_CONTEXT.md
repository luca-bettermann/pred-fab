# PFAB — Project Context

## Purpose
Predictive Fabrication (PFAB) framework: active-learning loop for manufacturing process calibration.
Combines ML-based prediction, evaluation scoring, and acquisition-driven optimization to propose
experiment parameters that balance exploration (evidence gain) and exploitation (performance).

## High-Level Flow
1. **Evaluate** — run feature and evaluation models on physical experiment data
2. **Train** — fit prediction models (MLP/Transformer) on historical dataset
3. **Propose** — acquisition optimizer proposes next experiments via κ-blended objective:
   - `A = (1-κ)·P_sys + κ·ΔE` where P_sys = weighted system performance, ΔE = evidence gain
   - κ=1.0 → discovery (pure space-filling via evidence), κ=0.0 → inference (pure performance)
4. **Execute** — apply proposed parameters; record results; repeat

## Repo Structure

| Path | Role |
|------|------|
| `src/pred_fab/core/` | Data model, schema, DataModule, normalization |
| `src/pred_fab/interfaces/` | Model contracts (feature, evaluation, prediction) |
| `src/pred_fab/models/` | MLP, Transformer, DeterministicModel bases |
| `src/pred_fab/orchestration/` | System coordination (PfabAgent, CalibrationSystem, PredictionSystem, EvaluationSystem) |
| `src/pred_fab/orchestration/calibration/` | Acquisition optimizer: SolutionSpace (sigmoid variables), Engine (Sobol→LBFGS), BoundsManager |
| `src/pred_fab/orchestration/evidence.py` | KernelField ANOVA evidence estimator (marginal + joint integration) |
| `src/pred_fab/utils/` | Logging, console output (ProgressBar), metrics, local persistence |
| `src/pred_fab/plotting/` | Schema-agnostic visualization (see `PLOTTING_CONTEXT.md`) |
| `src/pred_fab/diagnostics/` | Sobol/Morris global sensitivity analysis |
| `concepts/` | Concept figures for paper — synthetic data demonstrations |
| `tests/` | Pytest suite |

## Entry Point
`PfabAgent` in `orchestration/agent.py` is the single integration surface for users.

## Key Concepts

### Acquisition Pipeline
- **SolutionSpace** — sigmoid-based variable encoding (z-space → [0,1] → DataModule z-score)
- **Engine** — Sobol global sampling → LBFGS local refinement
- **κ-blend** — `A = (1-κ)·P_sys + κ·ΔE`, negated and scaled for minimization
- **compute_acquisition_grids()** — single function for all plotting (slices through the real pipeline)

### Evidence System
- **KernelField ANOVA** — marginal (D independent 1D integrals, weight D/(D+1)) + joint (D-dim shell probes, weight 1/(D+1))
- **KDE weights** — 1/L per trajectory layer (each experiment contributes 1 total evidence unit)
- **Domain bounds** — actual latent-space bounds stored on KernelIndex, not hardcoded [0,1]
- **dimension_derivations** — per-axis derivation functions for domain axes (e.g., N_layers from layer_height)

### Performance System
- **predict_features()** → per-feature predictions via perf_fn_tensor closure
- **system_performance()** → weighted P_sys via combined_score (single source of truth)
- **Gradient flow** — stats.reverse() for differentiable z-score → raw conversion in _reattach_tensor_continuous

### Normalization
- **DataModule** — z-score normalization (StandardScalerModule) for ML training
- **SolutionSpace** — sigmoid decode to [0,1] bounds, then _bounds_to_dm_norm converts to z-score
- **Context features** — default to z-score 0.0 (training mean) when not in optimization vector

### Metrics
- **R²** — standard coefficient of determination
- **R²_inf** — importance-weighted R², per-feature (weighted by that feature's performance scores)
- **MAE** — mean absolute error
- These are the only three metrics used everywhere (console, wandb, validation)

## Knowledge Base
Full project context, decisions, and research notes live in `../knowledge-base/` (Obsidian vault).
