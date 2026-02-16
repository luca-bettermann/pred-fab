# Test Suite Layout

This suite is rebuilt from scratch to track the current `pred_fab` architecture.

## Structure

- `tests/utils/`
  - Shared builders/fixtures/utilities for test setup.
  - Reusable interface implementations plus system builders for orchestration and integration tests.
- `tests/core/`
  - Core data model tests (`data_objects`, `data_blocks`, `dataset`, `datamodule`).
- `tests/orchestration/`
  - System/agent orchestration tests, including step-method contracts.
- `tests/interfaces/`
  - Interface contract tests for model implementations.
- `tests/integration/`
  - Stepwise workflow integration tests including local persistence checks.

## Conventions

- Prefer small, contract-focused unit tests.
- Reuse builders from `tests/utils/` instead of duplicating setup code.
- Reuse interface implementations from `tests/utils/interfaces.py`; do not define ad-hoc models in individual tests.
- Add new tests under the module area they validate.
- Do not stub orchestration systems (`FeatureSystem`, `EvaluationSystem`, `PredictionSystem`, `CalibrationSystem`).
- Keep simplification at interface/model level only (feature/evaluation/prediction/surrogate implementations).
