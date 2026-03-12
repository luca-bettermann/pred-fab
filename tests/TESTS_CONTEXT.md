# Test Suite Context

This suite validates the current `pred_fab` architecture.  458 tests passing, 2 expected skips.

## Structure

- `tests/utils/`
  - `builders.py` — Shared builders for schemas, datasets, agents, datamodules, and orchestration stacks.
  - `interfaces.py` — Reusable interface implementations (`WorkflowFeatureModelA/B`, `WorkflowEvaluationModelA/B`, `WorkflowPredictionModel`, `MixedFeatureModel`, `ScalarEvaluationModel`, `MixedPredictionModel`, `ShapeCheckingPredictionModel`, `CapturingResidualModel`, etc.). Note: `CapturingSurrogateModel` removed (GP surrogate removed from architecture).
- `tests/core/`
  - `test_data_objects.py` — `DataReal`, `DataInt`, `DataBool`, `DataCategorical`, `DataDimension`, `DataArray`, and factory methods.
  - `test_data_blocks.py` — `Parameters`, `Features`, `PerformanceAttributes`: strides, iterator combinations, `value_at`, round-trip persistence.
  - `test_dataset_operations.py` — `Dataset` CRUD, `ExperimentData` lifecycle, `ParameterUpdateEvent` persistence.
  - `test_datamodule_normalization.py` — `DataModule` prepare/split/normalization, one-hot encoding, batch generation, serialization round-trip.
  - `test_datamodule_array_io.py` — `params_to_array`/`array_to_params` (continuous + categorical round-trip), `denormalize_values`, normalization overrides, `get_onehot_column_map`, `copy()`, `get_split_codes`, `__repr__`, degenerate std=0.
  - `test_datamodule_calibration_arrays.py` — Calibration-specific array IO (`get_calibration_array`, `calibration_array_to_params`).
  - `test_feature_representation.py` — Feature tensor/table conversion and dimensional indexing.
  - `test_parameter_sanitization.py` — Parameter constraint clamping and type coercion.
  - `test_parameter_updates.py` — `ParameterUpdateEvent` recording, rejection of dimension updates, no-change skip, save/load round-trip.
  - `test_schema_registry.py` — `DatasetSchema` hashing and `SchemaRegistry` lookup.
- `tests/orchestration/`
  - `test_agent_step_contracts.py` — `PfabAgent` step method (`exploration_step`, `inference_step`) output contracts: `ExperimentSpec` return type, source_step tagging, spy mocks verifying `Mode` propagation.
  - `test_adaptation_step.py` — Adaptation step: `ExperimentSpec` return type, `SourceStep.ADAPTATION` tagging, effective-param replay with prior updates.
  - `test_calibration_config.py` — `CalibrationSystem` configuration: `configure_param_bounds`, `set_performance_weights`, `configure_fixed_params`, `configure_adaptation_delta`, `_compute_system_performance`, bounds validation.
  - `test_calibration_sampling.py` — `generate_baseline_experiments` (count, empty, schema keys, fixed params, bounds, integer typing, infinite bounds skip, determinism), `run_calibration(domain=ONLINE)` behavior (returns `ExperimentSpec`, trust-region constraints, raises for missing trust region), `configure_trajectory` (new string API), trajectory-based baseline schedule generation.
  - `test_run_calibration_contracts.py` — **Unified `run_calibration()` contract tests:**
    - Offline exploration, experiment level (single step): return type, source_step, empty schedules, fixed params, dict-like access
    - Offline inference, experiment level (single step): return type, source_step, mode differentiation
    - Offline exploration, dimensional level (multi-step): schedule keyed by dimension, ParameterSchedule type, entries contain trajectory params, delta constraints
    - Offline inference, dimensional level (multi-step): schedule produced, delta constraints
    - Step-loop internals: `_build_step_grid`, `_get_eligible_params`, `_build_experiment_spec` unit tests
    - Multi-dimensional mixed scenario (`TestMultiDimMixedScenario`): two trajectory params on different dimension levels (layer_height→dim_1 level 1, speed→dim_2 level 2), verifying 6-step Cartesian product grid (coarsest-first), per-step eligibility logic, both dimension schedules produced, delta constraints per param, and direct `_build_experiment_spec` unit test with 6 proposals
    - `ExperimentSpec` integration: `apply_schedules`, dict delegation
    - Agent-level spy tests confirming Mode/Domain propagation from `exploration_step`/`inference_step`
    - `target_indices` (`TestTargetIndices`): collapsing full grid to a single targeted step, per-dim eligibility isolation, empty schedules invariant, source_step preservation, works without trajectory configs
  - `test_orchestration_edge_cases.py` — `PredictionSystem.validate` model-specific input slices, `CalibrationSystem._acquisition_func` integration (perf_fn + uncertainty_fn callables), `InferenceBundle` degenerate min-max, `tune` row-slice semantics.
  - `test_uncertainty_estimation.py` — NatPN-light: `encode()` identity/override, `uncertainty()` pre/post training, KDE OOD behavior, `kernel_similarity()`, `predict_for_calibration()`, `compute_performance()` with `feature_std`, UCB acquisition and inference function integration, trajectory step-loop with KDE delta constraints.
  - `test_prediction_system_errors.py` — `PredictionSystem.train` empty-split error, `validate` before/after training, `tune` boundary errors (start/end/negative), `predict_experiment` shape and partial-range NaN fill.
- `tests/interfaces/`
  - Interface contract tests for model implementations.
- `tests/integration/`
  - `test_stepwise_workflow.py` — End-to-end stepwise workflow including local persistence; `exploration_step` returns `ExperimentSpec`.

## Conventions

- Prefer small, contract-focused unit tests.
- Reuse builders from `tests/utils/builders.py` instead of duplicating setup code.
- Reuse interface implementations from `tests/utils/interfaces.py`; do not define ad-hoc models in individual tests.
- Add new tests under the module area they validate.
- Do not stub orchestration systems (`FeatureSystem`, `EvaluationSystem`, `PredictionSystem`, `CalibrationSystem`).
- Keep simplification at interface/model level only (feature/evaluation/prediction implementations).

## Bugs Discovered via Tests

The following source bugs were found and fixed while writing the new tests:

- **`DataModule._split_codes` KeyError on fresh instance** (`datamodule.py`): `_split_codes` was initialized as `{}`, causing `get_split_sizes()` / `__repr__` to raise `KeyError` when the datamodule had never been prepared. Fixed by initializing with all three `SplitType` keys.
- **`DataModule.set_parameter_normalize()` silently ignored after `initialize()`** (`datamodule.py`): `_set_input_columns()` pre-computes `_col_norm_methods` during `initialize()`. Subsequent calls to `set_parameter_normalize` stored to `_parameter_overrides` only, which `_fit_normalize` never read. Fixed by immediately propagating the override to `_col_norm_methods` when the column is already present.
- **`PredictionSystem.tune()` incomplete guard** (`prediction.py`): The guard `if self.datamodule is None` allowed an unfitted (not trained) datamodule through, resulting in a confusing error message from inside `get_normalization_state()`. Fixed to also check `not self.datamodule._is_fitted`.
