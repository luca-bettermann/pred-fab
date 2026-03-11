# Orchestration Context

High-level architecture and long-horizon risks for `src/pred_fab/orchestration`.
Keep this concise and synchronized with implementation.

## Purpose

`orchestration` coordinates models and data flow across feature extraction, evaluation,
training, inference, and calibration:
- top-level workflow API (`agent.py`)
- system wrappers (`features.py`, `evaluation.py`, `prediction.py`, `calibration.py`)
- exportable runtime inference wrapper (`inference_bundle.py`)

## Structure

1. `agent.py`
- Main integration surface (`PfabAgent`).
- Owns model registration, system initialization, workflow steps, and calibration config.
- Step methods return typed `ParameterProposal` outputs for parameter suggestions.
- API taxonomy:
  - step methods: `exploration_step`, `inference_step`, `adaptation_step`
  - operation methods: `evaluate`, `train`, `predict`, `configure_calibration`, `sample_baseline_experiments`

2. `base_system.py`
- Shared helpers for model specs and schema reference wiring.

3. `features.py`
- Runs feature models and writes canonical tensors into experiment data.

4. `evaluation.py`
- Runs evaluation models from computed features to performance attributes.

5. `prediction.py`
- Handles prediction model training/tuning/validation/inference and inference bundle export.
- Online tuning now respects requested slice boundaries (`start:end`) for step-local adaptation.
- After training, fits a weighted KDE on unique latent training configs (NatPN-light).
- Exposes `encode()`, `uncertainty()`, `kernel_similarity()`, `predict_for_calibration()`.
- Degenerate-dimension guard: zero-variance latent dims are dropped before KDE fitting
  (`_kde_active_mask` applied consistently in `uncertainty()` and `kernel_similarity()`).

6. `calibration.py`
- Acquisition/inference objectives, sampling, and optimization.
- **No longer owns a surrogate model.** Instead receives three injected callables:
  - `perf_fn(params_dict) → {perf_code: value_or_None}` — encapsulates predict + evaluate
  - `uncertainty_fn(X_norm) → float` — epistemic uncertainty from `PredictionSystem`
  - `similarity_fn(X1, X2) → float` — optional; drives Level 2 trajectory diversity discounting
- `_active_datamodule` is set before each optimization run to enable `array_to_params`.
- Adaptation uses current effective parameters (initial + recorded updates) as optimization center.

7. `inference_bundle.py`
- Lightweight deployed inference wrapper (prediction + normalization + schema validation).

## Calibration and Tuning Semantics

1. Exploration vs inference objective split
- `Mode.EXPLORATION`: UCB acquisition objective — `(1-κ)·perf_fn(x) + κ·uncertainty_fn(x)`.
- `Mode.INFERENCE`: direct `perf_fn(x)` objective (no uncertainty term).

2. Offline vs online optimization domains
- Offline calibration (`run_calibration`) uses global bounds/fixed params.
- Online adaptation (`run_adaptation`) uses trust-region bounds around current effective parameters.

3. Online residual adaptation lifecycle
- `PredictionSystem.tune(...)` trains residual correction on selected row slices only.
- Residual model uses base inputs + base predictions as tuning input.
- Applied online parameter changes are persisted as `ParameterUpdateEvent` records, then replayed in export/datamodule flows.

## Open Refactor Risks (Large-Scope Only)

1. Use of private DataModule internals across systems
- Orchestration systems still rely on internal DataModule methods/fields (normalization internals).
- Risk: tight coupling makes independent evolution/testing of modules harder.

2. Inference bundle schema/input validation depth
- Bundle validates unknown columns but does not yet enforce full schema constraint checks.
- Risk: invalid but schema-shaped requests can pass until deeper model/runtime stages.

## Agent Update Rule

When changing orchestration APIs or workflow semantics:
- update this file for cross-system behavior changes.
- fix small bugs immediately; keep only larger migration/refactor risks here.
