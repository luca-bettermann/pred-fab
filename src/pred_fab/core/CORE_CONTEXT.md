# Core — Context

## Purpose
Canonical data model and ML data-prep layer. All other packages depend on core; core depends on nothing internal.

## Modules

| Module | Role |
|--------|------|
| `data_objects.py` | Schema primitives: `DataReal`, `DataInt`, `DataBool`, `DataCategorical`, `DataArray`, `DataDomainAxis`, `Domain` |
| `data_blocks.py` | Typed containers: `Parameters`, `Features`, `PerformanceAttributes`, `Domains` |
| `schema.py` | `DatasetSchema` (owns parameters + features + performance + domains) + `SchemaRegistry` |
| `dataset.py` | `ExperimentData` (one experiment's live data), `Dataset` (collection + persistence) |
| `datamodule.py` | ML data-prep: normalization, one-hot encoding, batching, train/val/test splits |

## Data Flow
Schema → Dataset → ExperimentData → DataModule (for training/calibration)

## Key Design Points
- `Domain` declares named ordered iteration axes (e.g. `layers → segments`); `DataDomainAxis` params are auto-created by the schema and stored in `Parameters`. Users never instantiate `DataDomainAxis` directly.
- Features declare `domain="<code>"` and optional `depth=<int>` — column structure is derived automatically from the domain axes at schema init time. The schema hash includes domain definitions.
- `DataModule` is shared between PredictionSystem (training) and CalibrationSystem (bounds + encoding)
- Categorical params are one-hot encoded; normalization is per-column via `NormMethod`
- `fit_without_data()` sets up column mappings without any training rows (used by baseline generation)

## Open Risks
- Column-order coupling: `input_columns` order must be stable across fit/transform calls
- `fit_without_data` does not populate normalization stats — callers must use `NormMethod.NONE` or supply bounds explicitly
