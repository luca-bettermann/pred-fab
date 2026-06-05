# Tests — Context

## Status
485 passing, 2 skipped (expected).

## Structure

| Folder | Focus |
|--------|-------|
| `tests/core/` | Data model: DataObjects, DataBlocks, Dataset, DataModule, Schema |
| `tests/interfaces/` | Interface contract compliance for feature/evaluation/prediction models |
| `tests/orchestration/` | System contracts: CalibrationSystem, PfabAgent step methods, joint trajectory |
| `tests/integration/` | End-to-end workflow: initialize → load → train → calibrate |
| `tests/utils/` | Shared mock builders and test fixtures |

## Test Conventions
- `tests/utils/` contains reusable builders (schema, datamodule, mock models) — prefer extending these over duplicating
- Contract tests assert observable behaviour (return types, source_step tags, trajectory structure), not internal implementation
- Joint trajectory tests verify that multi-step optimization produces correct `ParameterTrajectory` entries for configured dimensions
- Trajectory/step-grid tests verify that `ParameterTrajectory` entries are produced for the correct dimensions

## Key Fixtures
- `build_schema()` — minimal schema with one real param and one performance attr
- `build_calibration_system()` — CalibrationSystem with injected mock perf/uncertainty fns
- `WorkflowPredictionModel` / `WorkflowFeatureModelA` — full mock model implementations for integration tests
