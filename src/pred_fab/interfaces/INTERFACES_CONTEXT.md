# Interfaces Context

High-level architecture and long-horizon risks for `src/pred_fab/interfaces`.
Keep this concise and aligned with code.

## Purpose

`interfaces` defines contracts for model and data adapters used by orchestration:
- base model contract and schema references (`base_interface.py`)
- feature/evaluation/prediction model interfaces (`features.py`, `evaluation.py`, `prediction.py`)
- calibration surrogate interface (`calibration.py`)
- residual tuning interface (`tuning.py`)
- external data gateway contract (`external_data.py`)

## Structure

1. `base_interface.py`
- Shared abstract base for model contracts.
- Defines `input_parameters`, `input_features`, `outputs`, and schema reference wiring.

2. `features.py`
- `IFeatureModel`: dimensional iteration + feature extraction contract.
- Produces tabular arrays that are transformed to canonical tensors by orchestration/core.

3. `evaluation.py`
- `IEvaluationModel`: target/scaling/performance computation contract.
- Computes per-point and averaged performance values.

4. `prediction.py`
- `IPredictionModel`: training/inference/export/import contract.
- Optional online tuning hook.

5. `calibration.py`
- `ISurrogateModel` + default GP implementation for exploration calibration.

6. `tuning.py`
- `IResidualModel` + default residual MLP implementation.

7. `external_data.py`
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
