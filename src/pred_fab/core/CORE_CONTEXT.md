# Core Context

High-level architecture and risk inventory for `src/pred_fab/core`.
Keep this file short, current, and aligned with implementation changes.

## Purpose

`core` defines the canonical data model and ML data-prep path used by higher orchestration layers:
- schema/type system (`data_objects.py`, `schema.py`)
- runtime experiment containers and load/save/export flow (`data_blocks.py`, `dataset.py`)
- model-ready tabular/normalized views (`datamodule.py`)

## High-Level Structure

1. `data_objects.py`
- Typed schema primitives (`DataReal`, `DataInt`, `DataCategorical`, `DataDimension`, `DataArray`).
- Factory helpers (`Parameter`, `Feature`, `PerformanceAttribute`) used when constructing schema.

2. `data_blocks.py`
- Block containers (`Parameters`, `Features`, `PerformanceAttributes`) that hold schema objects + runtime values.
- `Features` is tensor-first in-memory:
  - canonical storage: N-D tensors keyed by feature code
  - boundary transforms: `table_to_tensor`, `tensor_to_table`, `value_at`

3. `schema.py`
- `DatasetSchema` combines parameter/feature/performance block definitions.
- `SchemaRegistry` manages stable schema name/hash mapping and collision checks.

4. `dataset.py`
- `ExperimentData`: one experiment’s parameter/feature/performance values.
- `Dataset`: experiment lifecycle and hierarchical load/save (`memory -> local -> external`).
- Converts canonical tensors to/from tabular boundaries for persistence and dataframe export.

5. `datamodule.py`
- ML preprocessing layer over `Dataset`.
- Train/val/test splitting, one-hot encoding, normalization stats, batch generation.

## Data Flow (Current)

1. Schema defines allowed objects and roles.
2. Experiments are created/loaded into `ExperimentData`.
3. Features are initialized as tensors from parameter dimensions.
4. Feature extraction writes tensors (canonical representation).
5. Persistence/export boundaries transform tensors <-> tabular arrays/dataframes.
6. `DataModule` consumes dataframe export for training workflows.

## Open Refactor Risks (Large-Scope Only)

1. Completeness check semantics are flatten-based
- `ExperimentData.is_complete` flattens tensors and checks `[evaluate_from:evaluate_to]` globally.
- Risk: step-range semantics can diverge from intended per-dimension iteration semantics for complex workflows.
- File: `src/pred_fab/core/dataset.py`.

2. Column-order coupling at IO boundaries
- Tensor/tabular transforms depend on consistent `DataArray.columns` and dimension iterator mapping.
- Risk: stale or mismatched column metadata can produce wrong value alignment.
- Files: `src/pred_fab/core/data_blocks.py`, `src/pred_fab/core/dataset.py`.

3. DataModule assumes clean dataframe outputs
- `DataModule` normalization/encoding path assumes expected columns and numeric coercion compatibility.
- Risk: missing outputs, sparse `y`, or mixed dtypes can cause fitting/runtime issues in edge cases.
- File: `src/pred_fab/core/datamodule.py`.

## Agent Update Rule

When touching any `core` code:
- update this file if architecture, canonical representation, or known risk points changed.
- do not list small bugs here; fix them immediately and only keep larger refactor-risk topics.
