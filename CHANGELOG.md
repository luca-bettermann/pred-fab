# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Fixed
- **Architecture Fix: DataObject Value Storage** (2025-11-26)
  - Removed `value` attribute and `set_value()` methods from DataObject classes
  - Values now stored exclusively in `DataBlock.values` dictionary
  - Moved rounding logic to `DataBlock.set_value()`
  - Prevents data corruption when DataObject instances are shared across experiments
  - Updated 9 tests and `interfaces/evaluation.py`
  - **Migration**: Use `data_block.get_value(name)` and `data_block.set_value(name, value)`

### Added
- **Train/Val/Test Splits** (2025-11-25)
  - `DataModule` with configurable `test_size`, `val_size`, `random_seed` parameters
  - `extract_all(split='train'|'val'|'test')` for split-specific extraction
  - `get_split_sizes()` method and empty split validation
  - Reproducible splits with fixed random seed (default: 42)

- **Model Validation Metrics** (2025-11-25)
  - `PredictionSystem.validate(use_test=False)` computes MAE, RMSE, R² metrics
  - Returns per-feature metrics with sample counts
  - Error handling for empty splits and untrained models

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
