# Core — Context

## Purpose
Canonical data model and ML data-prep layer. All other packages depend on core; core depends on nothing internal.

## Modules

| Module | Role |
|--------|------|
| `data_objects.py` | Schema primitives: `DataReal`, `DataInt`, `DataBool`, `DataCategorical`, `DataArray`, `DataDomainAxis`, `Dimension`, `Domain` |
| `data_blocks.py` | Typed containers: `Parameters`, `Features`, `PerformanceAttributes`, `Domains` |
| `schema.py` | `DatasetSchema` (owns parameters + features + performance + domains) + `SchemaRegistry` |
| `dataset.py` | `ExperimentData` (one experiment's live data), `Dataset` (collection + persistence). `export_to_tensor_dict(codes, x_columns, y_columns, …)` is the canonical batch-export — returns `(X, y, cell_meta)` per-column tensors. |
| `datamodule.py` | ML data-prep: `nn.Module` normalisers, categorical-index encoding, batching, splits, stateless `substitute_recursive_features` for scheduled sampling |
| `normalisers.py` | `nn.Module` normalisers (`StandardScalerModule` / `MinMaxScalerModule` / `RobustScalerModule` / `IdentityNormaliser`); stats live in `state_dict()` |

## Data Flow
Schema → Dataset → ExperimentData → DataModule (for training/calibration)

## Key Design Points
- `Dimension` is the user-facing class for declaring a single domain axis; `Domain` takes a `List[Dimension]`. `DataDomainAxis` params are auto-created by the schema and stored in `Parameters` — users never touch `DataDomainAxis` directly.
- `DataBlock.add(obj)` uses `obj.code` as the key automatically — no explicit name argument needed.
- Features declare `domain=<Domain>` and optional `depth=<int>` — column structure is derived automatically from the domain axes at schema init time. The schema hash includes domain definitions.
- **Context features** (`Feature.context(...)` / `DataArray.context=True`): observable but uncontrollable covariates (e.g. temperature). Included in `DataModule.input_columns` but never in `output_columns`; their values are copied from `y_df` during training and injected via `PfabAgent._context_snapshot` at calibration time.
- `DataModule` is shared between PredictionSystem (training) and CalibrationSystem (bounds + encoding)
- **Categoricals** emit a single int-index `long` column per parent. Models learn `nn.Embedding(C, d)` per categorical (FastAI-sized); `cat_cardinalities` exposes `{col_idx: n_categories}` for `set_categorical_context`.
- **Normalisation** is per-column via `NormMethod`; the corresponding `nn.Module` instance lives in `_parameter_stats` / `_feature_stats`. `module(x)` forward, `module.reverse(x)` inverse.
- `fit_without_data()` sets up column mappings without any training rows (used by baseline generation)
- **Tensor-dict path:** `prepare_input_from_tensor_dict(X_dict)` is the canonical input encoder. Pandas no longer appears inside the framework hot path — only at user-facing I/O (`Dataset.export_to_dataframe`, parquet via `LocalData`).

## Open Risks
- Column-order coupling: `input_columns` order must be stable across fit/transform calls
- `fit_without_data` does not populate normalisation stats — callers must use `NormMethod.NONE` or supply bounds explicitly
