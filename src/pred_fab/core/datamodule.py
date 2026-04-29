"""DataModule — ML preprocessing (normalization, splitting, batching) for Dataset instances."""

from typing import Any, cast
import pandas as pd
import numpy as np
import torch
import copy
from sklearn.model_selection import train_test_split

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
        # Strategy D commit 13: stats dicts replaced by NormaliserModule
        # instances. Dict-like __getitem__ on the modules preserves the old
        # access pattern (e.g. ``stats[col]["mean"]`` still works).
        self._feature_stats: dict[str, NormaliserModule] = {}
        self._parameter_stats: dict[str, NormaliserModule] = {}
        self._is_fitted = False
        
        # Feature system metadata (no data storage)
        # Strategy D commit 14: input_columns lists each parameter / feature
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

        Strategy D commit 14: each categorical parameter contributes ONE
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
        """Creates split codes based on dataset."""
        exp_codes = self.dataset.get_populated_experiment_codes()
        
        if not exp_codes:
            self._split_codes = {SplitType.TRAIN: [], SplitType.VAL: [], SplitType.TEST: []}
            return
            
        # Sort for deterministic split
        exp_codes.sort()
        n_samples = len(exp_codes)
        indices = np.arange(n_samples)
        
        if test_size == 0.0 and val_size == 0.0:
            self._split_codes = {
                SplitType.TRAIN: exp_codes,
                SplitType.VAL: [],
                SplitType.TEST: []
            }
            return
        
        # Split off test set
        if test_size > 0.0:
            train_val_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=self.random_seed
            )
        else:
            train_val_idx = indices
            test_idx = np.array([], dtype=int)
        
        # Split remaining into train/val
        if val_size > 0.0 and len(train_val_idx) > 0:
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_size / (1.0 - test_size) if test_size < 1.0 else 0.0, # adjust val_size relative to remaining
                random_state=self.random_seed
            )
        else:
            train_idx = train_val_idx
            val_idx = np.array([], dtype=int)
        
        self._split_codes = {
            SplitType.TRAIN: [exp_codes[i] for i in train_idx],
            SplitType.VAL: [exp_codes[i] for i in val_idx],
            SplitType.TEST: [exp_codes[i] for i in test_idx]
        }
    
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
        """Tensor-native equivalent of ``_inject_context_features`` (commit 16)."""
        if not self._context_feature_codes:
            return X_dict
        out = dict(X_dict)
        for code in self._context_feature_codes:
            if code in y_dict:
                out[code] = y_dict[code]
        return out

    def substitute_recursive_features(
        self,
        X: torch.Tensor,
        cell_meta: torch.Tensor,
        experiment_codes: list[str],
        predictions_by_exp: dict[str, dict[str, torch.Tensor]],
        p_student: float,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Replace recursive-feature column values with predictions at prior cells.

        Strategy D commit 15b — stateless, tensor-typed, no DM SS state. Pure
        gather/scatter using ``cell_meta`` and the schema's recursive-feature
        metadata. The caller (typically PredictionSystem) is responsible for
        producing fresh ``predictions_by_exp`` from the current network state.

        Args:
            X: ``(n_rows, n_input_cols)`` normalised input batch.
            cell_meta: ``(n_rows, 2)`` long tensor with ``[exp_idx, cell_idx]``
                per row (from ``Dataset.export_to_tensor_dict``).
            experiment_codes: list whose ``[exp_idx]`` indexing matches
                ``cell_meta[:, 0]``. Used to fetch each experiment's
                ``dim_sizes`` (for unravel/ravel) and to key into
                ``predictions_by_exp``.
            predictions_by_exp: ``{exp_code: {source_feat: predictions_tensor}}``.
                Each predictions_tensor matches the schema's domain shape
                for that feature (e.g. ``(n_layers, n_segments)``).
            p_student: per-row substitution probability. ``0`` → no substitution
                (return X unchanged). ``1`` → substitute every row's recursive
                feature columns.
            rng: optional torch RNG for deterministic per-row Bernoulli draws.

        Returns:
            New tensor of shape ``(n_rows, n_input_cols)``. Original X
            unmodified. Boundary cells (prior < 0) substitute with NaN —
            matches the legacy ``_perturb_recursive_features`` semantics.

        No-op fast paths: ``p_student <= 0``, no recursive features in
        schema, empty batch.
        """
        if p_student <= 0.0 or X.numel() == 0:
            return X

        schema = self.dataset.schema
        # (input_col_idx, source_code, iter_axis_code, depth) per recursive feature.
        recursive_feats: list[tuple[int, str, str, int]] = []
        for feat_code, feat_obj in schema.features.data_objects.items():
            if not getattr(feat_obj, "is_recursive", False):
                continue
            if feat_code not in self.input_columns:
                continue
            source_code = getattr(feat_obj, "recursive_source", None)
            if source_code is None:
                continue
            depth = getattr(feat_obj, "recursive_depth", None) or 1
            for iter_code in (getattr(feat_obj, "recursive_dimensions", None) or ()):
                col_idx = self.input_columns.index(feat_code)
                recursive_feats.append((col_idx, source_code, iter_code, depth))
        if not recursive_feats:
            return X

        # Working copy — never mutate caller's tensor.
        X_out = X.clone()
        n_rows = int(X.shape[0])

        # Per-experiment metadata cache (dim_sizes + dim_iterators).
        exp_meta: dict[int, tuple[list[int], list[str]]] = {}

        for row in range(n_rows):
            exp_idx = int(cell_meta[row, 0].item())
            cell_idx_flat = int(cell_meta[row, 1].item())

            if exp_idx not in exp_meta:
                exp = self.dataset.get_experiment(experiment_codes[exp_idx])
                dim_names = exp.parameters.get_dim_names()
                if not dim_names:
                    exp_meta[exp_idx] = ([], [])
                    continue
                dim_iterators = exp.parameters.get_dim_iterator_codes(codes=dim_names)
                dim_sizes = [int(exp.parameters.get_value(dn)) for dn in dim_names]
                exp_meta[exp_idx] = (dim_sizes, dim_iterators)

            dim_sizes, dim_iterators = exp_meta[exp_idx]
            if not dim_sizes:
                continue
            cell = list(np.unravel_index(cell_idx_flat, tuple(dim_sizes)))

            preds_for_exp = predictions_by_exp.get(experiment_codes[exp_idx], {})

            for col_idx, source_code, iter_code, depth in recursive_feats:
                # Per-row Bernoulli — substitute with prob p_student.
                draw = (
                    float(torch.rand(1, generator=rng).item())
                    if rng is not None else float(np.random.random())
                )
                if draw >= p_student:
                    continue

                axis_idx = next(
                    (i for i, ic in enumerate(dim_iterators) if ic == iter_code),
                    None,
                )
                if axis_idx is None:
                    continue

                prior = cell.copy()
                prior[axis_idx] -= depth
                if prior[axis_idx] < 0:
                    new_val = float("nan")
                else:
                    src = preds_for_exp.get(source_code)
                    if src is None:
                        continue
                    src_idx = tuple(prior[: src.ndim])
                    new_val = float(src[src_idx].item()) if torch.is_tensor(src) else float(src[src_idx])

                # Apply parameter normalisation if a stat exists for this column.
                col_name = self.input_columns[col_idx]
                stats = self._parameter_stats.get(col_name)
                if stats is not None:
                    new_val_t = stats(torch.tensor(new_val, dtype=X_out.dtype))
                    new_val = float(new_val_t.item())  # type: ignore[arg-type]
                X_out[row, col_idx] = new_val

        return X_out

    def _fit_normalize(self, split: SplitType = SplitType.TRAIN) -> None:
        """Fit normalization parameters on the specified split."""
        if not self._initialized:
            raise RuntimeError("Datamodule has not been initialized yet. Call agent_initialize_datamodule(datamodule).")

        if split not in self._split_codes:
            raise ValueError(f"Unknown split: {split}")

        codes = self._split_codes[split]
        if not codes:
            return

        # Strategy D commit 16: tensor-native fit path. No pandas roundtrip.
        exported = self.dataset.export_to_tensor_dict(
            codes,
            x_columns=self.input_columns,
            y_columns=self.output_columns,
            categorical_mappings=self.categorical_mappings,
        )
        if exported.is_empty():
            return

        X_dict = self._inject_context_features_tensor(exported.X, exported.y)

        # Fit X — skip categorical columns (cat-index passed through unnormalised; commit 14).
        self._parameter_stats = {}
        for col in self.input_columns:
            method = self._col_norm_methods.get(col, NormMethod.NONE)
            if method == NormMethod.NONE or method == NormMethod.CATEGORICAL:
                continue
            if col not in X_dict:
                continue
            col_arr = X_dict[col].cpu().numpy()
            self._parameter_stats[col] = make_normaliser(method, col_arr)

        # Fit y — tensor-native: read each output col tensor from exported.y.
        # Skip cells where the value is NaN (missing measurement) when fitting.
        self._feature_stats = {}
        for col in self.output_columns:
            method = self._get_feature_normalize_method(col)
            if method == NormMethod.NONE:
                continue
            if col not in exported.y:
                continue
            col_arr = exported.y[col].cpu().numpy()
            valid = col_arr[~np.isnan(col_arr)]
            if valid.size == 0:
                continue
            self._feature_stats[col] = make_normaliser(method, valid)

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

        Strategy D commit 16b: tensor-native end-to-end. Uses
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
        )
        if exported.is_empty():
            return []

        X_dict = self._inject_context_features_tensor(exported.X, exported.y)
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

        Strategy D commit 16: thin shim over ``prepare_input_from_tensor_dict``.
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
        """Tensor-native ``prepare_input`` (Strategy D commit 16).

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

        Strategy D commit 13: stat values come from the NormaliserModule
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

        Strategy D commit 14: each schema code maps to exactly ONE column
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
        """Fit a NormaliserModule to ``data`` (Strategy D commit 13).

        Replaces the legacy dict-of-stats output. The returned module exposes
        ``module["mean"]`` / ``module["min"]`` etc. via ``__getitem__`` so
        existing dict-style access patterns still work.
        """
        return make_normaliser(method, data)

    def _apply_normalization(self, data: np.ndarray, stats: NormaliserModule) -> np.ndarray:
        """Apply normalisation by delegating to the NormaliserModule (commit 13)."""
        return stats(data)  # type: ignore[return-value]

    def _apply_normalization_tensor(self, data: torch.Tensor, stats: NormaliserModule) -> torch.Tensor:
        """Tensor-native equivalent — same module dispatch."""
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
        """Reverse normalisation by delegating to the NormaliserModule (commit 13)."""
        return stats.reverse(data_norm)  # type: ignore[return-value]

    def _reverse_normalization_tensor(self, data_norm: torch.Tensor, stats: NormaliserModule) -> torch.Tensor:
        """Tensor-native equivalent — same module dispatch."""
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

        Strategy D commit 14: categoricals emit a single int-index column
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

        Strategy D commit 16b: thin wrapper around the tensor-native
        ``params_to_tensor`` (commit 1) — no pandas roundtrip.
        """
        if not self._is_fitted:
            raise RuntimeError("DataModule not fitted.")
        return self.params_to_tensor(params).detach().cpu().numpy()

    def array_to_params(self, array: np.ndarray) -> dict[str, Any]:
        """Reverse-normalize and decode a 1D input array back to a parameter dict.

        Strategy D commit 14: categorical columns hold a cat-index (rounded to
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

        Strategy D commit 14: categoricals emit a single int-index column
        (encoded as float for shape uniformity); models expand via
        ``F.one_hot`` / ``nn.Embedding`` in their forward.

        Gradient flow: if any continuous value is a ``torch.Tensor`` with
        ``requires_grad=True``, the gradient flows through normalisation
        (affine for all NormMethod variants) into the output tensor.
        Used by Strategy D's gradient-based acquisition. Categorical
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
        """Tensor-typed inverse of ``array_to_params`` (commit 14).

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
    
    def __repr__(self) -> str:
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        batch_str = f"batch_size={self.batch_size}" if self.batch_size else "no batching"
        
        split_sizes = self.get_split_sizes()
        size_str = f"train={split_sizes[SplitType.TRAIN]}, val={split_sizes[SplitType.VAL]}, test={split_sizes[SplitType.TEST]}"
        
        return (
            f"DataModule(normalize='{self._default_normalize}', {batch_str}, splits=({size_str}), "
            f"{fitted_str}, overrides={len(self._feature_overrides)})"
        )
