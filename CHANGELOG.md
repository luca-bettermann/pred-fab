# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **BaseOrchestrationSystem** (2025-11-28)
  - Shared base class for `EvaluationSystem` and `PredictionSystem`
  - Common initialization (`dataset`, `logger`)
  - Shared `get_model_specs()` implementation for DataObject extraction
  - Shared `_extract_params_from_exp_data()` helper method
  - Abstract `get_models()` method for implementation-specific model storage
  - ~90 lines of duplicate code eliminated across systems
  - Enables consistent patterns for future orchestration systems

### Changed
- **Schema Generation Optimization** (2025-11-28)
  - Moved spec extraction from agent to orchestration systems
  - `EvaluationSystem.get_model_specs()` and `PredictionSystem.get_model_specs()` extract DataObjects from model fields
  - `DatasetSchema.from_model_specs()` builds schema from system specs
  - Agent no longer instantiates models twice (removed temporary systems)
  - `_instantiate_models()` creates systems and model instances once, reused during initialization
  - Agent code reduced by 66 lines (450 → 384 lines, -14.7%)
  - Clearer separation: systems understand models, agent orchestrates workflow

### Breaking Changes
- **Phase 10: Dimensional Prediction Architecture** (2025-11-28)
  - Prediction models now predict at dimensional level (not aggregated scalars)
  - All training data extracted as dimensional positions (params + dimensional indices)
  - Removed `extract_all()` and `get_batches()` methods (use `get_split()` instead)
  - Simplified metric array storage: only feature values (not redundant dimensions/targets/scaling)
  - `PredictionSystem.predict()` replaced with `predict_experiment(exp_data)`
  - InferenceBundle removed (Phase 9 scalar inference not compatible with dimensional architecture)
  - 23 Phase 9 tests removed, 31 tests migrated to Phase 10 API
  - **Migration**: No backward compatibility - retrain models on dimensional data

### Added
- **Phase 10: Dimensional Prediction** (2025-11-27 to 2025-11-28)
  - `predicted_metric_arrays` in `ExperimentData` for dimensional predictions
  - `DataModule.get_split(split_name)` for dimensional data extraction
  - `PredictionSystem.predict_experiment(exp_data, predict_from, predict_to, batch_size, overlap)`
  - Vectorized batched prediction for memory efficiency (default: 1000 positions/batch)
  - Batch overlap parameter for context-aware models (transformers, RNNs)
  - NaN validation for training data
  - Model compatibility validation (dimensional requirements)
  - `required_features` property in `IPredictionModel` for explicit external dependencies
  - 32 comprehensive tests for dimensional prediction (including overlap validation)
  - Full documentation in IMPLEMENTATION_SUMMARY.md

### Fixed
- **Architecture Fix: DataObject Value Storage** (2025-11-26)
  - Removed `value` attribute and `set_value()` methods from DataObject classes
  - Values now stored exclusively in `DataBlock.values` dictionary
  - Moved rounding logic to `DataBlock.set_value()`
  - Prevents data corruption when DataObject instances are shared across experiments
  - Updated 9 tests and `interfaces/evaluation.py`
  - **Migration**: Use `data_block.get_value(name)` and `data_block.set_value(name, value)`

### Deprecated
- **Phase 9 Scalar Prediction API** (2025-11-28)
  - `DataModule.extract_all()` → use `get_split('all')` or `get_split('train')`
  - `DataModule.get_batches()` → not needed with vectorized dimensional prediction
  - `PredictionSystem.predict(X)` → use `predict_experiment(exp_data)`
  - `InferenceBundle` → dimensional inference patterns to be designed in future

## [1.0.0] - 2025-11-25

### Major Refactor - Dataset-Centric Architecture (Phase 7)

**Breaking Changes:**
- User owns `Dataset` instance (not agent)
- Agent is stateless orchestrator (no longer stores dataset)
- Hierarchical load/save patterns: Memory → Local → External
- New API: `dataset.add_experiment()` instead of `agent.add_experiment()`

**Phase 7 Subprojects:**
- **7A: Rounding Configuration**: Moved `round_digits` to DataObject level (declarative)
- **7B: External Schema Persistence**: Added schema push/pull to IExternalData interface
- **7C: Dataset Hierarchical Methods**: `populate()`, `save()`, hierarchical `add_experiment()`
- **7D: Agent Refactoring**: Removed dataset ownership, removed delegation methods
- **7E: Documentation & Tests**: Updated all docs and examples for new API

**Results:**
- 23% code reduction (agent.py: 1002 → 778 lines)
- Removed dead code and delegation layers
- All 157 tests passing

### Phase 1-6: Core Architecture

**Phase 1-3: Foundation (Core Data Model)**
- `DataObject`: 6 typed primitives (DataReal, DataInt, DataBool, DataCategorical, DataString, DataArray)
- `DataBlock`: Value containers with validation (Parameters, Dimensions, PerformanceAttributes, MetricArrays)
- `DatasetSchema`: Deterministic hashing and compatibility checks
- `SchemaRegistry`: Hash-to-ID mapping with auto-increment
- `Dataset`: Container with validation and hierarchical persistence
- `ExperimentData`: Single experiment structure

**Phase 4: Interface & Orchestration**
- `IEvaluationModel`: Evaluation interface with feature model support
- `IFeatureModel`: Feature computation interface with memoization
- `IPredictionModel`: Prediction interface with train/predict
- `ICalibrationModel`: Calibration interface with weighted optimization
- `EvaluationSystem`, `PredictionSystem`: Orchestration layers

**Phase 5: Agent & Schema Generation**
- `LBPAgent`: Orchestration layer with two-phase initialization
- Dataclass-based schema generation (replaced decorator introspection)
- Delayed model instantiation pattern
- Register → Initialize workflow

**Phase 6: Naming & Terminology**
- Unified terminology: `study_code` → `schema_id`
- Unified parameter blocks (removed static/dynamic split)
- Consistent method naming across codebase

### Core Features

- Type-safe data model with runtime validation
- Schema registry with deterministic hashing
- Interface-based design for evaluation, prediction, calibration
- Automatic feature memoization
- Dynamic dimensionality for multi-dimensional analysis
- Hierarchical data persistence (local JSON + optional external storage)
- Comprehensive test coverage (157 tests)
- Factory classes: `Parameter`, `Performance`, `Dimension`
