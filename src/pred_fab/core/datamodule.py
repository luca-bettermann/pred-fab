"""DataModule — ML preprocessing (normalization, splitting, batching) for Dataset instances."""

from typing import Any, cast
import pandas as pd
import numpy as np
import torch
import copy
from .data_objects import DataArray, DataCategorical
from .dataset import Dataset
from .normalisers import (
    NormaliserModule,
    make_normaliser,
    normaliser_from_dict,
)
from ..utils import NormMethod, SplitType


class DataModule:
    """Preprocessing pipeline: normalization fitting, train/val/test splitting, and batching for ML workflows."""
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int | None = None,
        normalize: NormMethod = NormMethod.STANDARD,
        random_seed: int | None = 42
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self._default_normalize = normalize
        self.random_seed = random_seed
        self._initialized = False
        
        # Per-feature/parameter normalization overrides
        self._feature_overrides: dict[str, NormMethod] = {}
        self._parameter_overrides: dict[str, NormMethod] = {}
        
        # Fitted normalization parameters
        # stats dicts replaced by NormaliserModule
        # instances. Dict-like __getitem__ on the modules preserves the old
        # access pattern (e.g. ``stats[col]["mean"]`` still works).
        self._feature_stats: dict[str, NormaliserModule] = {}
        self._parameter_stats: dict[str, NormaliserModule] = {}
        self._is_fitted = False
        
        # Feature system metadata (no data storage)
        # input_columns lists each parameter / feature
        # ONCE (no expansion). Categoricals are emitted as a single int-index
        # column; models that want one-hot do ``F.one_hot`` themselves in
        # forward (and learnable cats use ``nn.Embedding``).
        self.input_columns: list[str] = []
        self.output_columns: list[str] = []
        # parent_col → ordered list of categories. Index in this list is the
        # cat-index encoding emitted in batches.
        self.categorical_mappings: dict[str, list[str]] = {}
        # Context features: observed but uncontrollable — input only, never in output_columns.
        self._context_feature_codes: list[str] = []
        # Cardinality lookup: {col_index_in_input_columns: n_categories}.
        # Empty for non-categorical columns. Models read this via
        # ``set_categorical_context`` to size their internal one-hot/embedding.
        self._cat_cardinalities: dict[int, int] = {}
        
        # Column normalization methods map (for X)
        self._col_norm_methods: dict[str, NormMethod] = {}

        # Lagged parameter config: {original_col: lag}. Populated by add_lagged_params().
        self._lagged_params: dict[str, int] = {}

        # Row exclusion: {axis_code: set of values to drop}. Applied in get_batches().
        self._excluded_rows: dict[str, set[int]] = {}

        # Domain depth limit: only iterate this many axes when building rows.
        self._max_depth: int | None = None

        # Create splits (stores experiment codes)
        self._split_codes: dict[str, list[str]] = {
            SplitType.TRAIN: [],
            SplitType.VAL: [],
            SplitType.TEST: [],
        }

    @property
    def context_feature_codes(self) -> list[str]:
        """Schema codes of context features (observable but uncontrollable)."""
        return list(self._context_feature_codes)

    def initialize(
            self,
            input_parameters: list[str],
            input_features: list[str],
            output_columns: list[str]
            ) -> None:
        self._set_input_columns(input_parameters, input_features)
        self.output_columns = output_columns
        self._initialized = True
        
    def _set_input_columns(self, input_parameters: list[str], input_features: list[str]):
        """Build ``input_columns`` (parent-level, no categorical expansion).

        each categorical parameter contributes ONE
        int-index column to the batch (its category index in
        ``categorical_mappings[parent]``). Cardinality is tracked in
        ``_cat_cardinalities`` so models can size their own
        ``F.one_hot`` / ``nn.Embedding`` internally.
        """
        # Store parameter methods
        for col in input_parameters:
            method = self._get_parameter_normalize_method(col)
            self._col_norm_methods[col] = method

            if method == NormMethod.CATEGORICAL:
                # Schema validation
                if not self.dataset.schema.parameters.has(col):
                    raise KeyError(f"Parameter '{col}' can not be retrieved from schema.")
                obj = self.dataset.schema.parameters.get(col)
                if not isinstance(obj, DataCategorical):
                    raise ValueError(f"Obj expected to be of type 'DataCategorical', got {obj.__class__} instead.")
                # Sorted category list — index in this list is the encoding.
                categories = sorted(obj.constraints["categories"])
                self.categorical_mappings[col] = categories
                # Single column for the parameter; cardinality recorded for models.
                col_idx = len(self.input_columns)
                self.input_columns.append(col)
                self._cat_cardinalities[col_idx] = len(categories)
            else:
                self.input_columns.append(col)

        # Store feature methods; track context features separately.
        for col in input_features:
            method = self._get_feature_normalize_method(col)
            self._col_norm_methods[col] = method
            self.input_columns.append(col)
            # Detect context features via schema lookup.
            if self.dataset.schema.features.has(col):
                feat_obj = self.dataset.schema.features.get(col)
                if isinstance(feat_obj, DataArray) and feat_obj.context:
                    self._context_feature_codes.append(col)

    def add_lagged_params(self, lag: int = 1) -> None:
        """Register lagged parameter columns. Call before ``prepare()``.

        For each non-categorical parameter in ``input_columns``, a
        ``prev_{code}`` column is added. Within each experiment, values
        are shifted by ``lag`` rows; the first ``lag`` rows copy their
        own values (no change signal).
        """
        if self._is_fitted:
            raise RuntimeError("add_lagged_params must be called before prepare().")
        for col in list(self.input_columns):
            if col in self.categorical_mappings:
                continue
            lagged_col = f"prev_{col}"
            if lagged_col not in self.input_columns:
                self._lagged_params[col] = lag
                self.input_columns.append(lagged_col)
                self._col_norm_methods[lagged_col] = self._col_norm_methods.get(col, NormMethod.NONE)

    def set_max_depth(self, depth: int) -> None:
        """Limit domain iteration to ``depth`` axes. Call before ``prepare()``.

        With ``depth=1`` on a (layers, nodes) domain, rows are built per
        layer only (12 per experiment, not 84). Depth-2 features are averaged
        over the extra axis. Fixes row duplication for depth-1 models and
        ensures ``add_lagged_params(lag=1)`` shifts by one layer, not one node.
        """
        if self._is_fitted:
            raise RuntimeError("set_max_depth must be called before prepare().")
        self._max_depth = depth

    def exclude_rows(self, axis: str, values: list[int]) -> None:
        """Exclude rows where ``axis`` matches any of ``values``. Call before ``prepare()``.

        Rows are dropped from batches before normalization fitting, so
        excluded positions don't affect normalization stats. Uses
        ``cell_meta`` axis columns for filtering — no cross-experiment leakage.
        """
        if self._is_fitted:
            raise RuntimeError("exclude_rows must be called before prepare().")
        self._excluded_rows.setdefault(axis, set()).update(values)

    @property
    def cat_cardinalities(self) -> dict[int, int]:
        """``{col_index: n_categories}`` — what models need for own-side one-hot."""
        return dict(self._cat_cardinalities)


    # === DATAMODULE OPERATIONS ===

    def prepare(self, val_size: float = 0.0, test_size: float = 0.0, recompute: bool = False) -> None:
        """Create train/val/test splits and fit normalization; raises if splits exist and recompute=False."""
        has_splits = any(len(c) > 0 for c in self._split_codes.values())
        
        if has_splits and not recompute:
             raise RuntimeError(
                 "DataModule already has defined splits. "
                 "Use update() to add new experiments to the training set, "
                 "or set recompute=True to reshuffle and overwrite validation/test sets."
             )
        
        self._create_splits(val_size, test_size)
        self._fit_normalize()

    def update(self) -> int:
        """Add newly populated experiments to the training set and refit normalization; returns count added."""
        # Get all valid codes in dataset
        valid_codes = set(self.dataset.get_populated_experiment_codes())
        
        # Get currently tracked codes
        current_codes = set()
        for codes in self._split_codes.values():
            current_codes.update(codes)
            
        # Find new codes
        new_codes = list(valid_codes - current_codes)
        
        if not new_codes:
            return 0
            
        # Add to training set
        new_codes.sort() # Deterministic order
        self._split_codes[SplitType.TRAIN].extend(new_codes)
        
        # Refit normalization since train set changed
        self._fit_normalize()
        
        return len(new_codes)

    def _create_splits(self, val_size: float, test_size: float) -> None:
        """Creates split codes based on dataset, preserving pre-set splits.

        Experiments already assigned to a split are kept. Only unassigned
        experiments are auto-split according to val_size / test_size.
        """
        all_codes = set(self.dataset.get_populated_experiment_codes())
        if not all_codes:
            return

        pre_assigned: set[str] = set()
        for codes in self._split_codes.values():
            pre_assigned.update(codes)

        unassigned = sorted(all_codes - pre_assigned)

        if not unassigned:
            return

        if test_size == 0.0 and val_size == 0.0:
            if not pre_assigned:
                self._split_codes[SplitType.TRAIN].extend(unassigned)
            return

        n = len(unassigned)
        indices = np.arange(n)
        rng = np.random.default_rng(self.random_seed)
        shuffled = rng.permutation(indices)

        if test_size > 0.0:
            n_test = int(round(len(shuffled) * test_size))
            test_idx = shuffled[:n_test]
            train_val_idx = shuffled[n_test:]
        else:
            train_val_idx = shuffled
            test_idx = np.array([], dtype=int)

        if val_size > 0.0 and len(train_val_idx) > 0:
            adjusted = val_size / (1.0 - test_size) if test_size < 1.0 else 0.0
            n_val = int(round(len(train_val_idx) * adjusted))
            val_idx = train_val_idx[:n_val]
            train_idx = train_val_idx[n_val:]
        else:
            train_idx = train_val_idx
            val_idx = np.array([], dtype=int)

        self._split_codes[SplitType.TRAIN].extend([unassigned[i] for i in train_idx])
        self._split_codes[SplitType.VAL].extend([unassigned[i] for i in val_idx])
        self._split_codes[SplitType.TEST].extend([unassigned[i] for i in test_idx])
    
    def _apply_row_exclusions(
        self,
        exported: "ExportedTensorDict",
    ) -> "ExportedTensorDict":
        """Drop rows where any excluded axis matches a configured value.

        Modifies X, y, and cell_meta in place. Called before normalization
        fitting and batch building so excluded rows never affect training.
        """
        if not self._excluded_rows:
            return exported

        from .dataset import ExportedTensorDict

        mask = torch.ones(exported.n_rows, dtype=torch.bool)
        for axis_code, excluded_values in self._excluded_rows.items():
            if axis_code in exported.X:
                col = exported.X[axis_code]
                for val in excluded_values:
                    mask &= col != val
            elif axis_code in exported.y:
                col = exported.y[axis_code]
                for val in excluded_values:
                    mask &= col != val

        if mask.all():
            return exported

        X_filtered = {k: v[mask] for k, v in exported.X.items()}
        y_filtered = {k: v[mask] for k, v in exported.y.items()}
        meta_filtered = exported.cell_meta[mask]
        return ExportedTensorDict(X_filtered, y_filtered, meta_filtered)

    def _inject_lagged_params(
        self,
        X_dict: dict[str, torch.Tensor],
        cell_meta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Add prev_{col} columns by shifting per experiment (no cross-experiment leakage).

        ``cell_meta`` has shape ``(n_rows, 2)`` with ``[exp_idx, cell_idx]``.
        Within each experiment, values are shifted by ``lag`` rows along
        cell_idx order. The first ``lag`` rows copy their own values.
        """
        out = dict(X_dict)
        exp_ids = cell_meta[:, 0]

        for col, lag in self._lagged_params.items():
            if col not in X_dict:
                continue
            vals = X_dict[col]
            shifted = vals.clone()

            for exp_id in exp_ids.unique():
                mask = exp_ids == exp_id
                exp_vals = vals[mask]
                n = exp_vals.shape[0]
                if n <= lag:
                    pass  # all rows copy themselves (no shift possible)
                else:
                    shifted[mask] = torch.cat([exp_vals[:lag], exp_vals[:-lag]])

            out[f"prev_{col}"] = shifted

        return out

    def _inject_context_features(self, X_df: pd.DataFrame, y_df: pd.DataFrame) -> pd.DataFrame:
        """Copy context feature columns from y_df into X_df so they appear in input during training."""
        if not self._context_feature_codes:
            return X_df
        X_df = X_df.copy()
        for code in self._context_feature_codes:
            if code in y_df.columns:
                X_df[code] = y_df[code].values
        return X_df

    def _inject_context_features_tensor(
        self,
        X_dict: dict[str, torch.Tensor],
        y_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Inject context-feature columns into a tensor-dict batch."""
        if not self._context_feature_codes:
            return X_dict
        out = dict(X_dict)
        for code in self._context_feature_codes:
            if code in y_dict:
                out[code] = y_dict[code]
        return out

    def _fit_normalize(self, split: SplitType = SplitType.TRAIN) -> None:
        """Fit normalization parameters on the specified split."""
        if not self._initialized:
            raise RuntimeError("Datamodule has not been initialized yet. Call agent_initialize_datamodule(datamodule).")

        if split not in self._split_codes:
            raise ValueError(f"Unknown split: {split}")

        codes = self._split_codes[split]
        if not codes:
            return

        # tensor-native fit path. No pandas roundtrip.
        exported = self.dataset.export_to_tensor_dict(
            codes,
            x_columns=self.input_columns,
            y_columns=self.output_columns,
            categorical_mappings=self.categorical_mappings,
            max_depth=self._max_depth,
        )
        if exported.is_empty():
            return
        exported = self._apply_row_exclusions(exported)
        if exported.is_empty():
            return

        X_dict = self._inject_context_features_tensor(exported.X, exported.y)

        # Fit X — skip categorical columns (cat-index passed through unnormalised).
        self._parameter_stats = {}
        for col in self.input_columns:
            method = self._col_norm_methods.get(col, NormMethod.NONE)
            if method == NormMethod.NONE or method == NormMethod.CATEGORICAL:
                continue
            if col not in X_dict:
                continue
            self._parameter_stats[col] = make_normaliser(method, X_dict[col])

        # Fit y — tensor-native: read each output col tensor from exported.y.
        # Skip cells where the value is NaN (missing measurement) when fitting.
        self._feature_stats = {}
        for col in self.output_columns:
            method = self._get_feature_normalize_method(col)
            if method == NormMethod.NONE:
                continue
            if col not in exported.y:
                continue
            col_t = exported.y[col]
            valid = col_t[~torch.isnan(col_t)]
            if valid.numel() == 0:
                continue
            self._feature_stats[col] = make_normaliser(method, valid)

        for orig_col in self._lagged_params:
            lagged_col = f"prev_{orig_col}"
            if orig_col in self._parameter_stats:
                self._parameter_stats[lagged_col] = self._parameter_stats[orig_col]

        self._is_fitted = True

    def fit_normalization(self, split: SplitType = SplitType.TRAIN) -> None:
        """Public wrapper for fitting normalization on a dataset split."""
        self._fit_normalize(split)

    def fit_without_data(self) -> None:
        """Mark as fitted with empty stats (identity normalization); enables encoding machinery without training data."""
        if not self._initialized:
            raise RuntimeError(
                "DataModule must be initialized before calling fit_without_data()."
            )
        self._parameter_stats = {}
        self._feature_stats = {}
        self._is_fitted = True
    
    def get_batches(
        self, split: SplitType = SplitType.TRAIN,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return ``(X, y, cell_meta)`` tensor batch tuples for the given split.

        tensor-native end-to-end. Uses
        ``Dataset.export_to_tensor_dict`` directly (no pandas) and returns
        ``cell_meta`` as the third tuple element for downstream callers
        (SS substitution, per-cell loss masking, etc).
        """
        codes = self._split_codes.get(split, [])
        if not codes:
            return []

        exported = self.dataset.export_to_tensor_dict(
            codes,
            x_columns=self.input_columns,
            y_columns=self.output_columns,
            categorical_mappings=self.categorical_mappings,
            max_depth=self._max_depth,
        )
        if exported.is_empty():
            return []
        exported = self._apply_row_exclusions(exported)
        if exported.is_empty():
            return []

        X_dict = self._inject_context_features_tensor(exported.X, exported.y)
        if self._lagged_params:
            X_dict = self._inject_lagged_params(X_dict, exported.cell_meta)
        X_t = self.prepare_input_from_tensor_dict(X_dict)

        # Build y as (n_rows, len(output_columns)) tensor; apply normalisation per col.
        n_rows = exported.n_rows
        y_cols = []
        for col in self.output_columns:
            col_t = exported.y.get(col, torch.zeros(n_rows, dtype=torch.float32))
            stats = self._feature_stats.get(col)
            if stats is not None:
                col_t = self._apply_normalization_tensor(col_t, stats)
            y_cols.append(col_t.reshape(-1))
        y_t = torch.stack(y_cols, dim=-1) if y_cols else torch.zeros((n_rows, 0))

        cell_meta = exported.cell_meta

        # Drop rows where any output is NaN — missing measurements must not train
        valid = ~torch.isnan(y_t).any(dim=-1)
        if not valid.all():
            X_t = X_t[valid]
            y_t = y_t[valid]
            cell_meta = cell_meta[valid]

        # Batch
        if self.batch_size is None:
            return [(X_t, y_t, cell_meta)]

        batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for i in range(0, X_t.shape[0], self.batch_size):
            batches.append((
                X_t[i:i + self.batch_size],
                y_t[i:i + self.batch_size],
                cell_meta[i:i + self.batch_size],
            ))

        return batches

    def prepare_input(self, X_df: pd.DataFrame) -> torch.Tensor:
        """Prepare input DataFrame for inference (encode + normalize), returning a tensor.

        thin shim over ``prepare_input_from_tensor_dict``.
        Pandas users keep this entry point; tensor-typed callers (autoreg
        loop, etc.) can call ``prepare_input_from_tensor_dict`` directly to
        skip the DataFrame→numpy→tensor roundtrip.
        """
        if not self._is_fitted:
            raise RuntimeError("DataModule not fitted.")

        X_arr = self._encode_inputs(X_df)
        # Normalize
        self._normalize_batch(X_arr, self.input_columns, self._parameter_stats)
        return torch.from_numpy(X_arr)

    def prepare_input_from_tensor_dict(
        self, X_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Tensor-native ``prepare_input``.

        Takes a tensor dict as produced by ``Dataset.export_to_tensor_dict``
        (or constructed directly by the autoreg loop) and returns a stacked
        ``(n_rows, n_input_cols)`` tensor with normalisation applied. No
        pandas, no numpy roundtrip.
        """
        if not self._is_fitted:
            raise RuntimeError("DataModule not fitted.")
        if not X_dict:
            return torch.zeros((0, len(self.input_columns)), dtype=torch.float32)

        # Determine n_rows from any column
        n_rows = next(iter(X_dict.values())).shape[0]
        cols: list[torch.Tensor] = []
        for col_name in self.input_columns:
            if col_name in X_dict:
                col_t = X_dict[col_name]
                # Categoricals are long-typed; numerics are float.
                if col_name in self.categorical_mappings:
                    col_f = col_t.to(dtype=torch.float32)
                else:
                    col_f = col_t.to(dtype=torch.float32)
                # Replace NaNs (boundary cells) with 0 — matches _encode_inputs.
                col_f = torch.nan_to_num(col_f, nan=0.0)
            else:
                col_f = torch.zeros(n_rows, dtype=torch.float32)

            stats = self._parameter_stats.get(col_name)
            if stats is not None:
                col_f = self._apply_normalization_tensor(col_f, stats)
            cols.append(col_f.reshape(-1))

        return torch.stack(cols, dim=-1)

    def denormalize_output(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Reverse normalization for target features (y)."""
        return self.denormalize_values(y_norm, self.output_columns)

    def denormalize_values(self, values: torch.Tensor, feature_names: list[str]) -> torch.Tensor:
        """Reverse normalization for specific features. Tensor in / tensor out."""
        if not self._is_fitted:
            return values.clone()

        out = values.clone()
        if out.ndim == 1:
            for i, name in enumerate(feature_names):
                if name in self._feature_stats:
                    out[i] = self._reverse_normalization_tensor(out[i], self._feature_stats[name])
        else:
            for i, name in enumerate(feature_names):
                if name in self._feature_stats:
                    out[:, i] = self._reverse_normalization_tensor(out[:, i], self._feature_stats[name])
        return out


    # === NORMALIZATION STATE ===
    
    def set_feature_normalize(self, feature_name: str, method: NormMethod) -> None:
        """Override normalization method for a specific feature."""
        self._feature_overrides[feature_name] = method
    
    def set_parameter_normalize(self, parameter_name: str, method: NormMethod) -> None:
        """Override normalization method for a specific parameter."""
        self._parameter_overrides[parameter_name] = method
        # Propagate immediately to _col_norm_methods if initialize() has already been called.
        if parameter_name in self._col_norm_methods:
            self._col_norm_methods[parameter_name] = method
    
    def _get_feature_normalize_method(self, feature_name: str) -> NormMethod:
        """Get normalization method for a feature (override or default)."""
        return cast(NormMethod, self._feature_overrides.get(feature_name, self._default_normalize))
    
    def _get_parameter_normalize_method(self, param_name: str) -> NormMethod:
        """Get normalization method for a parameter using schema metadata."""
        if param_name in self._parameter_overrides:
            return self._parameter_overrides[param_name]
        
        data_obj = self.dataset.schema.parameters.get(param_name)
        if data_obj.normalize_strategy == NormMethod.DEFAULT:
            return self._default_normalize
        else:
            return data_obj.normalize_strategy
    
    def get_normalization_state(self) -> dict[str, Any]:
        """Export normalisation state for inference bundle.

        stat values come from the NormaliserModule
        instances (still serialised as plain dicts for backwards compat
        with prior on-disk inference bundles).
        """
        if not self._is_fitted:
            raise RuntimeError("DataModule has not been fitted yet.")

        def _module_to_dict(m: NormaliserModule) -> dict[str, Any]:
            d: dict[str, Any] = {"method": m.method}
            if m.method == NormMethod.STANDARD:
                d["mean"] = m["mean"]
                d["std"] = m["std"]
            elif m.method == NormMethod.MIN_MAX:
                d["min"] = m["min"]
                d["max"] = m["max"]
            elif m.method == NormMethod.ROBUST:
                d["median"] = m["median"]
                d["q1"] = m["q1"]
                d["q3"] = m["q3"]
            return d

        return {
            'method': self._default_normalize,
            'is_fitted': True,
            'feature_stats': {k: _module_to_dict(v) for k, v in self._feature_stats.items()},
            'parameter_stats': {k: _module_to_dict(v) for k, v in self._parameter_stats.items()},
            'categorical_mappings': copy.deepcopy(self.categorical_mappings),
            'input_columns': copy.deepcopy(self.input_columns),
            'output_columns': copy.deepcopy(self.output_columns)
        }

    def set_normalization_state(self, state: dict[str, Any]) -> None:
        """Restore normalisation state from exported bundle.

        Accepts either the legacy stat-dict format or the new module-typed
        format; both rebuild fresh NormaliserModule instances.
        """
        self._default_normalize = state['method']
        self._is_fitted = state['is_fitted']

        def _coerce(item: Any) -> NormaliserModule:
            if isinstance(item, NormaliserModule):
                return item
            return normaliser_from_dict(item)

        self._feature_stats = {k: _coerce(v) for k, v in state.get('feature_stats', {}).items()}
        self._parameter_stats = {k: _coerce(v) for k, v in state.get('parameter_stats', {}).items()}
        self.categorical_mappings = copy.deepcopy(state.get('categorical_mappings', {}))
        self.input_columns = copy.deepcopy(state.get('input_columns', []))
        self.output_columns = copy.deepcopy(state.get('output_columns', []))
    
    def get_input_indices(self, codes: list[str], skip_missing: bool = False) -> list[int]:
        """Return input_columns indices for the given schema codes.

        each schema code maps to exactly ONE column
        index — categoricals are no longer expanded.
        """
        indices: list[int] = []
        for code in codes:
            if code in self.input_columns:
                indices.append(self.input_columns.index(code))
            elif not skip_missing:
                raise ValueError(f"Column '{code}' not found in input_columns.")
        return indices

    # === SHARED NORMALIZATION HELPERS ===
    
    def _compute_normalization_stats(self, data: np.ndarray, method: NormMethod) -> NormaliserModule:
        """Fit a NormaliserModule to ``data``.

        Replaces the legacy dict-of-stats output. The returned module exposes
        ``module["mean"]`` / ``module["min"]`` etc. via ``__getitem__`` so
        existing dict-style access patterns still work.
        """
        return make_normaliser(method, data)

    def _apply_normalization(self, data: np.ndarray, stats: NormaliserModule) -> np.ndarray:
        """Apply normalisation via the NormaliserModule."""
        return stats(data)  # type: ignore[return-value]

    def _apply_normalization_tensor(self, data: torch.Tensor, stats: NormaliserModule) -> torch.Tensor:
        """Apply normalisation via the NormaliserModule (tensor-typed)."""
        return stats(data)

    def normalize_parameter_bounds(self, col: str, low: float, high: float) -> tuple[float, float]:
        """Normalize raw parameter bounds if normalization stats exist for the column."""
        if col not in self._parameter_stats:
            return (low, high)
        stats = self._parameter_stats[col]
        n_low = self._apply_normalization(np.array([low]), stats)[0]
        n_high = self._apply_normalization(np.array([high]), stats)[0]
        return (min(n_low, n_high), max(n_low, n_high))
    
    def _reverse_normalization(self, data_norm: np.ndarray, stats: NormaliserModule) -> np.ndarray:
        """Reverse normalisation via the NormaliserModule."""
        return stats.reverse(data_norm)  # type: ignore[return-value]

    def _reverse_normalization_tensor(self, data_norm: torch.Tensor, stats: NormaliserModule) -> torch.Tensor:
        """Reverse normalisation via the NormaliserModule (tensor-typed)."""
        return stats.reverse(data_norm)

    def _normalize_batch(self, data: np.ndarray, columns: list[str], stats: dict[str, NormaliserModule]) -> None:
        """Apply normalization to a batch of data in-place."""
        if not self._is_fitted:
            return

        for i, col in enumerate(columns):
            if col in stats:
                data[:, i] = self._apply_normalization(data[:, i], stats[col])
    
    def _encode_inputs(self, X_df: pd.DataFrame) -> np.ndarray:
        """Encode a DataFrame to ``(n_rows, n_input_cols)`` float array.

        categoricals emit a single int-index column
        (encoded as float for shape uniformity); numerics emit float values
        directly. Models that want one-hot expansion do it themselves in
        ``forward`` via ``F.one_hot`` (cardinality available via
        ``DataModule.cat_cardinalities``).
        """
        n = len(X_df)
        result = np.zeros((n, len(self.input_columns)), dtype=np.float32)
        for j, col in enumerate(self.input_columns):
            if col not in X_df.columns:
                continue  # missing column stays 0.0
            vals = X_df[col].values
            if col in self.categorical_mappings:
                # Categorical → cat-index float (categories ordered by self.categorical_mappings[col]).
                cat_list = self.categorical_mappings[col]
                cat_to_idx = {c: i for i, c in enumerate(cat_list)}
                result[:, j] = np.array(
                    [cat_to_idx.get(v, 0) for v in vals], dtype=np.float32,
                )
            else:
                result[:, j] = vals.astype(np.float32)
        # Replace NaN with 0 — recursive features use NaN for boundary padding.
        np.nan_to_num(result, copy=False, nan=0.0)
        return result

    # === CALIBRATION HELPERS ===

    def params_to_array(self, params: dict[str, Any]) -> np.ndarray:
        """Convert a parameter dict to a normalized 1D input array (numpy, calibration-side use).

        thin wrapper around the tensor-native
        ``params_to_tensor`` — no pandas roundtrip.
        """
        if not self._is_fitted:
            raise RuntimeError("DataModule not fitted.")
        return self.params_to_tensor(params).detach().cpu().numpy()

    def array_to_params(self, array: np.ndarray) -> dict[str, Any]:
        """Reverse-normalize and decode a 1D input array back to a parameter dict.

        categorical columns hold a cat-index (rounded to
        nearest int); decoding maps it back to the category string via
        ``categorical_mappings``.
        """
        if not self._is_fitted:
            raise RuntimeError("DataModule not fitted.")

        if array.shape != (len(self.input_columns),):
            raise ValueError(f"Array shape {array.shape} does not match input columns {len(self.input_columns)}")

        # Denormalize
        denorm_array = array.copy()
        for i, col in enumerate(self.input_columns):
            if col in self._parameter_stats:
                denorm_array[i] = self._reverse_normalization(np.array([array[i]]), self._parameter_stats[col])[0]

        params: dict[str, Any] = {}
        ctx = set(self._context_feature_codes)
        for i, col in enumerate(self.input_columns):
            if col in ctx:
                continue  # context features are uncontrollable
            if col in self.categorical_mappings:
                cats = self.categorical_mappings[col]
                idx = int(round(float(denorm_array[i])))
                idx = max(0, min(idx, len(cats) - 1))
                params[col] = cats[idx]
            else:
                params[col] = float(denorm_array[i])

        # Apply canonical parameter coercion/rounding at the array->dict boundary.
        return self.dataset.schema.parameters.sanitize_values(
            params,
            ignore_unknown=True
        )

    def params_to_tensor(
        self,
        params: dict[str, Any],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Tensor-typed mirror of ``params_to_array``. Returns ``(len(input_columns),)``.

        categoricals emit a single int-index column
        (encoded as float for shape uniformity); models expand via
        ``F.one_hot`` / ``nn.Embedding`` in their forward.

        Gradient flow: if any continuous value is a ``torch.Tensor`` with
        ``requires_grad=True``, the gradient flows through normalisation
        (affine for all NormMethod variants) into the output tensor.
        Used by the gradient-based acquisition. Categorical
        positions are non-differentiable by construction (discrete choices).
        """
        if not self._is_fitted:
            raise RuntimeError("DataModule not fitted.")

        out: list[torch.Tensor] = []
        for j, col_name in enumerate(self.input_columns):
            if col_name in self.categorical_mappings:
                # Categorical column: emit cat-index as float (no grad).
                cats = self.categorical_mappings[col_name]
                raw = params.get(col_name, None)
                idx = cats.index(raw) if (raw is not None and raw in cats) else 0
                t: torch.Tensor = torch.tensor(float(idx), dtype=dtype)
            else:
                # Continuous: preserve grad if raw is a tensor; else lift.
                raw = params.get(col_name, 0.0)
                if isinstance(raw, torch.Tensor):
                    t = raw.to(dtype=dtype) if raw.dtype != dtype else raw
                else:
                    raw_f = 0.0 if raw is None else float(raw)
                    if raw_f != raw_f:  # NaN guard
                        raw_f = 0.0
                    t = torch.tensor(raw_f, dtype=dtype)

            stats = self._parameter_stats.get(col_name)
            if stats is not None:
                t = self._apply_normalization_tensor(t, stats)
            out.append(t)

        return torch.stack(out)

    def tensor_to_params(self, tensor: torch.Tensor) -> dict[str, Any]:
        """Tensor-typed inverse of ``array_to_params``.

        Categoricals at column j are decoded by rounding the (denormalised)
        cat-index float to nearest int and looking up
        ``categorical_mappings[col][idx]``. Gradient flow on the output dict
        isn't preserved — categorical decoding is inherently discrete.
        """
        if not self._is_fitted:
            raise RuntimeError("DataModule not fitted.")
        if tensor.shape != (len(self.input_columns),):
            raise ValueError(
                f"Tensor shape {tuple(tensor.shape)} does not match "
                f"input columns {len(self.input_columns)}"
            )

        # Denormalise per-column in tensor land.
        denorm_t = tensor.clone()
        for i, col in enumerate(self.input_columns):
            stats = self._parameter_stats.get(col)
            if stats is not None:
                denorm_t[i] = self._reverse_normalization_tensor(tensor[i], stats)

        denorm_array = denorm_t.detach().cpu().numpy()
        params: dict[str, Any] = {}
        ctx = set(self._context_feature_codes)
        for i, col in enumerate(self.input_columns):
            if col in ctx:
                continue
            if col in self.categorical_mappings:
                cats = self.categorical_mappings[col]
                idx = int(round(float(denorm_array[i])))
                idx = max(0, min(idx, len(cats) - 1))
                params[col] = cats[idx]
            else:
                params[col] = float(denorm_array[i])

        return self.dataset.schema.parameters.sanitize_values(params, ignore_unknown=True)

    def build_calibration_training_arrays(
        self,
        performance_order: list[str],
        split: SplitType = SplitType.TRAIN,
        strict: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build (X, y) arrays for calibration training; strict=True raises on missing performance values."""
        if not self._is_fitted:
            raise RuntimeError("DataModule not fitted.")
        if split not in self._split_codes:
            raise ValueError(f"Unknown split: {split}")
        if not performance_order:
            raise ValueError("performance_order must contain at least one performance code.")

        X_rows: list[np.ndarray] = []
        y_rows: list[list[float]] = []

        for code in self._split_codes[split]:
            exp = self.dataset.get_experiment(code)
            perf = exp.performance.get_values_dict()

            missing = [name for name in performance_order if name not in perf]
            if missing:
                if strict:
                    raise ValueError(
                        f"Missing performance values for experiment '{code}': {missing}"
                    )
                continue

            # Use latest effective parameters so recorded online updates are reflected.
            x_arr = self.params_to_array(exp.get_effective_parameters_for_row(exp.get_num_rows() - 1))
            y_arr = [float(perf[name]) for name in performance_order]

            X_rows.append(x_arr)
            y_rows.append(y_arr)

        if not X_rows:
            return (
                np.empty((0, len(self.input_columns)), dtype=np.float64),
                np.empty((0, len(performance_order)), dtype=np.float64),
            )

        return (
            np.asarray(X_rows, dtype=np.float64),
            np.asarray(y_rows, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # Per-model batch construction
    # ------------------------------------------------------------------
    #
    # ``build_flat_batch`` and ``build_sequence_batch`` are the two batch
    # builders concrete prediction models call from their ``predict``
    # method. The framework owns shape construction (iterator overrides,
    # tiling); the model owns the forward pass.
    #
    # Flat batch is for MLP / Deterministic — one input row per (candidate,
    # cell). Sequence batch is for Transformer — one row per candidate,
    # with the sequence axis as the second dim and causal attention seeing
    # prior positions internally.

    def build_flat_batch(
        self,
        params_list: list[dict[str, Any]],
        dim_info_list: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Build (sum_s n_cells_s, n_input_cols) flat batch + per-row (s, cell_flat) map.

        Per cell: clone the per-candidate (S, n_input) tensor, override
        iterator-feature columns with the cell's normalised position. Static
        parameters and context features are tiled across cells via
        ``params_to_tensor``. Recursive features are not supported on this
        path — they belong to ``TransformerModel`` (which uses
        ``build_sequence_batch``); the model's ``_validate_schema_compatibility``
        rejects them at training time.

        Gradient flow: continuous values in ``params_list`` propagate through
        ``params_to_tensor``'s affine normalisation into ``X_flat``; the row map
        is plain Python ints (no autograd implications).

        Candidates are grouped by ``dim_info['shape']`` to share the cell
        loop within each shape group. The resulting row order is
        ``[group_0_cell_0_s0, group_0_cell_0_s1, ..., group_0_cell_1_s0, ...]``.
        """
        if not self._is_fitted:
            raise RuntimeError("DataModule not fitted.")
        S = len(params_list)
        n_cols = len(self.input_columns)
        if S == 0:
            return torch.zeros((0, n_cols), dtype=torch.float32), []
        if len(dim_info_list) != S:
            raise ValueError(
                f"build_flat_batch: dim_info_list length {len(dim_info_list)} "
                f"does not match params_list length {S}."
            )

        x_norm_S = torch.stack([self.params_to_tensor(p) for p in params_list])
        code_to_idx = {c: i for i, c in enumerate(self.input_columns)}

        # Group candidates by shape — within a group, all candidates share the
        # iterator-override layout, so we batch across S inside the cell loop.
        shape_groups: dict[tuple, list[int]] = {}
        for s, di in enumerate(dim_info_list):
            shape_groups.setdefault(di['shape'], []).append(s)

        X_parts: list[torch.Tensor] = []
        row_map: list[tuple[int, int]] = []

        for shape, group_indices in shape_groups.items():
            n_cells = int(np.prod(shape)) if shape else 1
            x_group = x_norm_S[group_indices]
            di = dim_info_list[group_indices[0]]
            iterator_feats = di['iterator_feats']

            if shape:
                cell_indices_arr = np.empty((n_cells, len(shape)), dtype=np.int64)
                for cell_row in range(n_cells):
                    cell_indices_arr[cell_row] = np.unravel_index(cell_row, shape)
            else:
                cell_indices_arr = np.zeros((n_cells, 0), dtype=np.int64)

            # Pre-compute per-cell iterator-feature normalised values.
            iter_overrides_per_cell: list[list[tuple[int, float]]] = []
            for cell_row in range(n_cells):
                overrides: list[tuple[int, float]] = []
                for feat_code, axis_pos, size in iterator_feats:
                    if feat_code not in code_to_idx:
                        continue
                    col_idx = code_to_idx[feat_code]
                    stats = self._parameter_stats.get(feat_code)
                    raw_val = float(cell_indices_arr[cell_row, axis_pos]) / max(size - 1, 1)
                    if stats is not None:
                        normed = float(self._apply_normalization_tensor(
                            torch.tensor(raw_val, dtype=x_group.dtype), stats,
                        ).item())
                    else:
                        normed = raw_val
                    overrides.append((col_idx, normed))
                iter_overrides_per_cell.append(overrides)

            # Per cell: clone (S_g, n_input), apply iterator overrides, append.
            for cell_row in range(n_cells):
                x_cell = x_group.clone()
                for col_idx, normed_val in iter_overrides_per_cell[cell_row]:
                    x_cell[:, col_idx] = normed_val
                X_parts.append(x_cell)
                for s in group_indices:
                    row_map.append((s, cell_row))

        X_flat = (
            torch.cat(X_parts, dim=0)
            if X_parts else torch.zeros((0, n_cols), dtype=torch.float32)
        )
        return X_flat, row_map

    def build_sequence_batch(
        self,
        model: Any,
        params_list: list[dict[str, Any]],
        dim_info_list: list[dict[str, Any]],
    ) -> torch.Tensor:
        """Build (S × n_other, L, n_input_cols) sequence batch along ``model.sequence_axis_code``.

        Multi-axis grids supported: non-sequence axes are flattened into
        the batch dimension as parallel sequences. For shape
        ``(L=n_layers, n_segments)`` with ``sequence_axis_code='n_layers'``,
        each segment becomes its own sequence over layers — one causal
        attention path per (candidate, segment).

        Static inputs are tiled across L positions; non-sequence-axis
        iterator inputs are tiled per-other-coord; the sequence-axis
        iterator (if declared) is filled per-position.

        ``n_other = product of non-sequence-axis sizes``. Returned tensor
        is ``(S × n_other, L, n_input)``; batch index = ``s * n_other + other_idx``
        where ``other_idx`` ravels the non-sequence coords in
        ``dim_codes_ordered`` order.
        """
        if not self._is_fitted:
            raise RuntimeError("DataModule not fitted.")
        S = len(params_list)
        n_cols = len(self.input_columns)
        if S == 0:
            return torch.zeros((0, 0, n_cols), dtype=torch.float32)
        if len(dim_info_list) != S:
            raise ValueError(
                f"build_sequence_batch: dim_info_list length {len(dim_info_list)} "
                f"does not match params_list length {S}."
            )

        seq_axis_codes = getattr(model, "sequence_axis_code", None)
        if seq_axis_codes is None:
            raise ValueError(
                f"build_sequence_batch requires the model to declare "
                f"`sequence_axis_code`; {model.__class__.__name__} does not."
            )

        di_first = dim_info_list[0]
        dim_codes = di_first.get('dim_codes_ordered', [])
        unresolved = [c for c in seq_axis_codes if c not in dim_codes]
        if unresolved:
            raise ValueError(
                f"sequence_axis_code entries {unresolved} not in this model's "
                f"domain axes {dim_codes}."
            )
        seq_axis_indices = [dim_codes.index(c) for c in seq_axis_codes]
        seq_axis_idx_set = set(seq_axis_indices)

        shape = di_first['shape']
        seq_axis_sizes = [shape[i] for i in seq_axis_indices]
        L = int(np.prod(seq_axis_sizes)) if seq_axis_sizes else 1
        other_axis_indices = [i for i in range(len(shape)) if i not in seq_axis_idx_set]
        other_sizes = [shape[i] for i in other_axis_indices]
        n_other = int(np.prod(other_sizes)) if other_sizes else 1
        S_eff = S * n_other

        # Pre-compute per-position seq coords (L, n_seq_axes) and per-other-coord
        # indices (n_other, n_other_axes).
        if seq_axis_sizes:
            seq_coords_arr = np.empty((L, len(seq_axis_sizes)), dtype=np.int64)
            for k in range(L):
                seq_coords_arr[k] = np.unravel_index(k, seq_axis_sizes)
        else:
            seq_coords_arr = np.zeros((1, 0), dtype=np.int64)
        if other_sizes:
            other_coords_arr = np.empty((n_other, len(other_sizes)), dtype=np.int64)
            for j in range(n_other):
                other_coords_arr[j] = np.unravel_index(j, other_sizes)
        else:
            other_coords_arr = np.zeros((1, 0), dtype=np.int64)

        x_norm_S = torch.stack([self.params_to_tensor(p) for p in params_list])
        code_to_idx = {c: i for i, c in enumerate(self.input_columns)}

        # Tile candidates × n_other → (S_eff, n_input), then × L positions.
        x_repeat = x_norm_S.unsqueeze(1).expand(S, n_other, n_cols).reshape(S_eff, n_cols)
        X_seq = x_repeat.unsqueeze(1).expand(S_eff, L, n_cols).clone()

        # Iterator overrides.
        iterator_feats = di_first['iterator_feats']
        for feat_code, axis_pos, size in iterator_feats:
            if feat_code not in code_to_idx:
                continue
            col_idx = code_to_idx[feat_code]
            stats = self._parameter_stats.get(feat_code)
            if axis_pos in seq_axis_idx_set:
                # Sequence axis: per-position value depends on the seq_coord
                # at this axis_pos — find which seq-axis position it is.
                p_seq = seq_axis_indices.index(axis_pos)
                for pos in range(L):
                    raw_val = float(seq_coords_arr[pos, p_seq]) / max(size - 1, 1)
                    if stats is not None:
                        normed = float(self._apply_normalization_tensor(
                            torch.tensor(raw_val, dtype=X_seq.dtype), stats,
                        ).item())
                    else:
                        normed = raw_val
                    X_seq[:, pos, col_idx] = normed
            else:
                # Non-sequence axis: per-other-coord constant across L.
                p_other = other_axis_indices.index(axis_pos)
                for j in range(n_other):
                    raw_val = float(other_coords_arr[j, p_other]) / max(size - 1, 1)
                    if stats is not None:
                        normed = float(self._apply_normalization_tensor(
                            torch.tensor(raw_val, dtype=X_seq.dtype), stats,
                        ).item())
                    else:
                        normed = raw_val
                    # All batch rows whose other_idx == j: indices j, j+n_other, j+2*n_other, ...
                    X_seq[j::n_other, :, col_idx] = normed

        return X_seq

    def copy(self) -> 'DataModule':
        """Create a deep copy of this DataModule."""
        return copy.deepcopy(self)
    
    def get_split_codes(self, split: SplitType = SplitType.TRAIN) -> list[str]:
        return self._split_codes[split]
    
    def get_split_sizes(self) -> dict[str, int]:
        """Get the sizes of each split as dict with train/val/test keys."""
        return {
            SplitType.TRAIN: len(self._split_codes[SplitType.TRAIN]),
            SplitType.VAL: len(self._split_codes[SplitType.VAL]),
            SplitType.TEST: len(self._split_codes[SplitType.TEST])
        }

    def set_split_codes(
        self,
        train_codes: list[str],
        val_codes: list[str] | None = None,
        test_codes: list[str] | None = None,
    ) -> None:
        """Explicitly set split membership without triggering split recomputation or refit."""
        self._split_codes = {
            SplitType.TRAIN: list(train_codes),
            SplitType.VAL: list(val_codes or []),
            SplitType.TEST: list(test_codes or []),
        }

    def set_split_dataset(
        self,
        dataset_code: str,
        split: SplitType = SplitType.TRAIN,
    ) -> list[str]:
        """Set one split's membership from all experiments tagged with ``dataset_code``.

        Sugar over ``set_split_codes``: filters ``self.dataset.experiments``
        whose ``ExperimentData.dataset_code`` equals ``dataset_code`` and assigns
        the resulting code list to the named ``split``. Other splits are left
        untouched. Returns the matched experiment codes (in dataset iteration
        order) for inspection.

        Raises ``ValueError`` if no experiments match — usually a sign of a
        typo in the dataset code or experiments not being tagged at creation.
        """
        if split not in (SplitType.TRAIN, SplitType.VAL, SplitType.TEST):
            raise ValueError(f"set_split_dataset: unknown split {split!r}.")
        codes = [
            code for code, exp in self.dataset._experiments.items()
            if exp.dataset_code == dataset_code
        ]
        if not codes:
            raise ValueError(
                f"set_split_dataset: no experiments tagged with dataset_code "
                f"{dataset_code!r}.",
            )
        self._split_codes[split] = codes
        return codes

    def set_split_datasets(
        self,
        dataset_codes: list[str],
        split: SplitType = SplitType.TRAIN,
    ) -> list[str]:
        """Union multiple ``dataset_code`` values into one split."""
        if split not in (SplitType.TRAIN, SplitType.VAL, SplitType.TEST):
            raise ValueError(f"set_split_datasets: unknown split {split!r}.")
        codes: list[str] = []
        for dc in dataset_codes:
            matched = [
                code for code, exp in self.dataset._experiments.items()
                if exp.dataset_code == dc
            ]
            codes.extend(matched)
        if not codes:
            raise ValueError(
                f"set_split_datasets: no experiments tagged with any of "
                f"{dataset_codes!r}.",
            )
        self._split_codes[split] = codes
        return codes

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        batch_str = f"batch_size={self.batch_size}" if self.batch_size else "no batching"
        
        split_sizes = self.get_split_sizes()
        size_str = f"train={split_sizes[SplitType.TRAIN]}, val={split_sizes[SplitType.VAL]}, test={split_sizes[SplitType.TEST]}"
        
        return (
            f"DataModule(normalize='{self._default_normalize}', {batch_str}, splits=({size_str}), "
            f"{fitted_str}, overrides={len(self._feature_overrides)})"
        )
