# Interfaces Context

High-level architecture and long-horizon risks for `src/pred_fab/interfaces`.
Keep this concise and aligned with code.

## Purpose

`interfaces` defines contracts for model and data adapters used by orchestration:
- base model contract and schema references (`base_interface.py`)
- feature/evaluation/prediction model interfaces (`features.py`, `evaluation.py`, `prediction.py`)
- residual tuning interface (`tuning.py`)
- external data gateway contract (`external_data.py`)

Note: `calibration.py` (which previously defined `ISurrogateModel` and `GaussianProcessSurrogate`)
has been removed. Uncertainty estimation is now handled by `PredictionSystem` via NatPN-light KDE.

## Structure

1. `base_interface.py`
- Shared abstract base for model contracts.
- Defines `input_parameters`, `input_features`, `outputs`, and schema reference wiring.

2. `features.py`
- `IFeatureModel`: dimensional iteration + feature extraction contract.
- Produces tabular arrays that are transformed to canonical tensors by orchestration/core.

3. `evaluation.py`
- `IEvaluationModel`: target/scaling/performance computation contract.
- `compute_performance()` returns a 3-tuple `(avg, per_row_list, std_list)`; accepts optional
  `feature_std` for uncertainty propagation.

4. `prediction.py`
- `IPredictionModel`: training/inference/export/import contract.
- Optional online tuning hook.
- Optional `encode(X)` method — default identity; override to expose learned latent representations
  for KDE-based uncertainty estimation in `PredictionSystem`.

5. `tuning.py`
- `IResidualModel` + default residual MLP implementation.

6. `external_data.py`
- `IExternalData` abstraction for pull/push operations to external stores.

## Open Refactor Risks (Large-Scope Only)

1. Contract strictness vs. ergonomics
- Interface reference binding (`set_reference`) is permissive and does not enforce completeness at binding time.
- Risk: missing bindings can surface later during orchestration execution, not at model registration.

2. Serialization contract variability
- Prediction export/import relies on custom model artifact methods per implementation.
- Risk: model portability quality varies by model implementation discipline.

3. Residual model output-shape assumptions
- Default residual MLP uses implicit shape behavior before full fit lifecycle is stabilized.
- Risk: adaptation behavior can diverge across multi-output prediction setups.

## Agent Update Rule

When changing interface contracts:
- update this file for signature/behavior changes that affect multiple systems.
- fix small bugs immediately; keep only larger cross-cutting refactor risks here.
