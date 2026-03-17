# Core — Context

## Purpose
Canonical data model and ML data-prep layer. All other packages depend on core; core depends on nothing internal.

## Modules

| Module | Role |
|--------|------|
| `data_objects.py` | Schema primitives: `DataReal`, `DataInt`, `DataBool`, `DataCategorical`, `DataDimension`, `DataArray` |
| `data_blocks.py` | Typed containers: `Parameters`, `Features`, `PerformanceAttributes` |
| `schema.py` | `DatasetSchema` (owns the three blocks) + `SchemaRegistry` |
| `dataset.py` | `ExperimentData` (one experiment's live data), `Dataset` (collection + persistence) |
| `datamodule.py` | ML data-prep: normalization, one-hot encoding, batching, train/val/test splits |

## Data Flow
Schema → Dataset → ExperimentData → DataModule (for training/calibration)

## Key Design Points
- `DataDimension` declares hierarchical levels (experiment / layer / segment); dimension sizes are stored as parameters
- `DataModule` is shared between PredictionSystem (training) and CalibrationSystem (bounds + encoding)
- Categorical params are one-hot encoded; normalization is per-column via `NormMethod`
- `fit_without_data()` sets up column mappings without any training rows (used by baseline generation)

## Open Risks
- Column-order coupling: `input_columns` order must be stable across fit/transform calls
- `fit_without_data` does not populate normalization stats — callers must use `NormMethod.NONE` or supply bounds explicitly
