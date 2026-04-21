# Tests — Context

## Status
485 passing, 2 skipped (expected).

## Structure

| Folder | Focus |
|--------|-------|
| `tests/core/` | Data model: DataObjects, DataBlocks, Dataset, DataModule, Schema |
| `tests/interfaces/` | Interface contract compliance for feature/evaluation/prediction models |
| `tests/orchestration/` | System contracts: CalibrationSystem, PfabAgent step methods, joint schedule |
| `tests/integration/` | End-to-end workflow: initialize → load → train → calibrate |
| `tests/utils/` | Shared mock builders and test fixtures |

## Test Conventions
- `tests/utils/` contains reusable builders (schema, datamodule, mock models) — prefer extending these over duplicating
- Contract tests assert observable behaviour (return types, source_step tags, schedule structure), not internal implementation
- Joint schedule tests verify that multi-step optimization produces correct `ParameterSchedule` entries for configured dimensions
- Trajectory/step-grid tests verify that `ParameterSchedule` entries are produced for the correct dimensions

## Key Fixtures
- `build_schema()` — minimal schema with one real param and one performance attr
- `build_calibration_system()` — CalibrationSystem with injected mock perf/uncertainty fns
- `WorkflowPredictionModel` / `WorkflowFeatureModelA` — full mock model implementations for integration tests
