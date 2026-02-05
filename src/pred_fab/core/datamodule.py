"""
DataModule - ML preprocessing configuration for LBP datasets.

Manages batching and normalization for machine learning workflows.
Stores normalization parameters, not normalized data (memory efficient).
"""

from typing import Optional, Dict, List, Tuple, Any, cast
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split

from .data_objects import DataDimension, DataCategorical
from .dataset import Dataset
from ..utils import NormMethod, SplitType


class DataModule:
    """
    Preprocessing configuration for ML workflows.
    
    - Extract and batch features from Dataset experiments
    - Fit/apply normalization to features (y) and parameters (X)
    - Stores data as Numpy arrays for efficiency
    - Handles one-hot encoding and batching internally
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = None,
        normalize: NormMethod = NormMethod.STANDARD,
        random_seed: Optional[int] = 42
    ):
        """
        Initialize DataModule.
        
        Args:
            dataset: Dataset instance
            batch_size: Number of experiments per batch (None = single batch)
            normalize: Default normalization method for features and parameters
            random_seed: Random seed for reproducible splits (None = random)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._default_normalize = normalize
        self.random_seed = random_seed
        self._initialized = False
        
        # Per-feature/parameter normalization overrides
        self._feature_overrides: Dict[str, NormMethod] = {}
        self._parameter_overrides: Dict[str, NormMethod] = {}
        
        # Fitted normalization parameters
        self._feature_stats: Dict[str, Dict[str, Any]] = {}
        self._parameter_stats: Dict[str, Dict[str, Any]] = {}
        self._categorical_mappings: Dict[str, List[str]] = {}
        self._is_fitted = False
        
        # Feature system metadata (no data storage)
        self.input_columns: List[str] = []  # Processed columns (after one-hot)
        self.output_columns: List[str] = []
        # self.original_input_columns: List[str] = []  # Before one-hot
        
        # Column normalization methods map (for X)
        self._col_norm_methods: Dict[str, NormMethod] = {}
        
        # Create splits (stores experiment codes)
        self._split_codes: Dict[str, List[str]] = {}

    def initialize(
            self, 
            input_parameters: List[str], 
            input_features: List[str],
            output_columns: List[str]
            ) -> None:
        self._set_input_columns(input_parameters, input_features)
        self.output_columns = output_columns
        self._initialized = True
        
    def _set_input_columns(self, input_parameters: List[str], input_features: List[str]):
        # Store parameter methods
        for col in input_parameters:
            method = self._get_parameter_normalize_method(col)
            self._col_norm_methods[col] = method

            # If categorical, store categories as well
            if method == NormMethod.CATEGORICAL:
                # Try to find categories in schema constraints
                categories = []
                
                # Check regular parameters
                if not self.dataset.schema.parameters.has(col):
                    raise KeyError(f"Parameter '{col}' can not be retrieved from schema.")
                obj = self.dataset.schema.parameters.get(col)
                if not isinstance(obj, DataCategorical):
                    raise ValueError(f"Obj expected to be of type 'DataCategorical', got {obj.__class__} instead.")
                # If categorical, store mapping
                categories = sorted(obj.constraints["categories"])                
                self._categorical_mappings[col] = categories
                # Store one hot encodings as inputs
                for category in categories:
                    col_name = f"{col}_{category}"
                    self.input_columns.append(col_name)
                    self._col_norm_methods[col_name] = NormMethod.NONE
            else:
                self.input_columns.append(col)

        # Store feature methods
        for col in input_features:
            method = self._get_feature_normalize_method(col)
            self._col_norm_methods[col] = method
            self.input_columns.append(col)
                

    # === DATAMODULE OPERATIONS ===

    def prepare(self, val_size: float = 0.2, test_size: float = 0.1, recompute: bool = False) -> None:
        """
        Create initial splits and fit normalization.
        
        Args:
            val_size: Fraction of data for validation set (0.0-1.0)
            test_size: Fraction of data for test set (0.0-1.0)
            recompute: If True, overwrite existing splits. If False, raises error if splits exist.
        """
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
        """
        Update the training set with any new experiments found in the dataset.
        Does not affect validation or test sets. Refits normalization.
        
        Returns:
            Number of new experiments added to training set.
        """
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
    
    def _fit_normalize(self, split: SplitType = SplitType.TRAIN) -> None:
        """Fit normalization parameters on the specified split."""
        if not self._initialized:
            raise RuntimeError("Datamodule has not been initialized yet. Call agent_initialize_datamodule(datamodule).")

        if split not in self._split_codes:
            raise ValueError(f"Unknown split: {split}")
            
        codes = self._split_codes[split]
        if not codes:
            return
            
        X_df, y_df = self.dataset.export_to_dataframe(codes)
        if X_df.empty:
            return
            
        # Process X (One-hot)
        X_arr = self._one_hot_encode(X_df)
        y_arr = y_df.values.astype(np.float32)
        
        # Fit X
        self._parameter_stats = {}
        for i, col in enumerate(self.input_columns):
            method = self._col_norm_methods.get(col, NormMethod.NONE)
            if method != NormMethod.NONE:
                self._parameter_stats[col] = self._compute_normalization_stats(X_arr[:, i], method)
        
        # Fit y
        self._feature_stats = {}
        for i, col in enumerate(self.output_columns):
            method = self._get_feature_normalize_method(col)
            if method != NormMethod.NONE:
                self._feature_stats[col] = self._compute_normalization_stats(y_arr[:, i], method)
        
        self._is_fitted = True
    
    def get_batches(self, split: SplitType = SplitType.TRAIN) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get batched, normalized data for a split.
        Returns list of (X_batch, y_batch) tuples.
        """
        codes = self._split_codes.get(split, [])
        if not codes:
            return []
            
        X_df, y_df = self.dataset.export_to_dataframe(codes)
        if X_df.empty:
            return []
            
        X = self._one_hot_encode(X_df)
        y = y_df.values.astype(np.float32)
        
        # Normalize
        if self._is_fitted:
            self._normalize_batch(X, self.input_columns, self._parameter_stats)
            self._normalize_batch(y, self.output_columns, self._feature_stats)
        
        # Batch
        if self.batch_size is None:
            return [(X, y)]
            
        batches = []
        for i in range(0, len(X), self.batch_size):
            batches.append((X[i:i+self.batch_size], y[i:i+self.batch_size]))
            
        return batches
    
    def prepare_input(self, X_df: pd.DataFrame) -> np.ndarray:
        """Prepare input DataFrame for inference (one-hot + normalize)."""
        if not self._is_fitted:
            raise RuntimeError("DataModule not fitted.")
            
        X_arr = self._one_hot_encode(X_df)
        
        # Normalize
        self._normalize_batch(X_arr, self.input_columns, self._parameter_stats)
                
        return X_arr

    def denormalize_output(self, y_norm: np.ndarray) -> np.ndarray:
        """Reverse normalization for target features (y)."""
        return self.denormalize_values(y_norm, self.output_columns)

    def denormalize_values(self, values: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Reverse normalization for specific features."""
        if not self._is_fitted:
            return values.copy()
            
        y = values.copy()
        if y.ndim == 1:
            for i, name in enumerate(feature_names):
                if name in self._feature_stats:
                    y[i] = self._reverse_normalization(y[i], self._feature_stats[name])
        else:
            for i, name in enumerate(feature_names):
                if name in self._feature_stats:
                    y[:, i] = self._reverse_normalization(y[:, i], self._feature_stats[name])
        return y


    # === NORMALIZATION STATE ===
    
    def set_feature_normalize(self, feature_name: str, method: NormMethod) -> None:
        """Override normalization method for a specific feature."""
        self._feature_overrides[feature_name] = method
    
    def set_parameter_normalize(self, parameter_name: str, method: NormMethod) -> None:
        """Override normalization method for a specific parameter."""
        self._parameter_overrides[parameter_name] = method
    
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
    
    def get_normalization_state(self) -> Dict[str, Any]:
        """Export normalization state for inference bundle."""
        if not self._is_fitted:
            raise RuntimeError("DataModule has not been fitted yet.")
        return {
            'method': self._default_normalize,
            'is_fitted': True,
            'feature_stats': copy.deepcopy(self._feature_stats),
            'parameter_stats': copy.deepcopy(self._parameter_stats),
            'categorical_mappings': copy.deepcopy(self._categorical_mappings),
            'input_columns': copy.deepcopy(self.input_columns),
            'output_columns': copy.deepcopy(self.output_columns)
        }
    
    def set_normalization_state(self, state: Dict[str, Any]) -> None:
        """Restore normalization state from exported bundle."""
        self._default_normalize = state['method']
        self._is_fitted = state['is_fitted']
        self._feature_stats = copy.deepcopy(state.get('feature_stats', {}))
        self._parameter_stats = copy.deepcopy(state.get('parameter_stats', {}))
        self._categorical_mappings = copy.deepcopy(state.get('categorical_mappings', {}))
        self.input_columns = copy.deepcopy(state.get('input_columns', []))
        self.output_columns = copy.deepcopy(state.get('output_columns', []))
    
    # === SHARED NORMALIZATION HELPERS ===
    
    def _compute_normalization_stats(self, data: np.ndarray, method: NormMethod) -> Dict[str, Any]:
        """Compute normalization statistics for a data array."""
        if method == NormMethod.NONE:
            return {'method': NormMethod.NONE}
        elif method == NormMethod.STANDARD:
            return {
                'method': method,
                'mean': float(np.mean(data)),
                'std': float(np.std(data))
            }
        elif method == NormMethod.MIN_MAX:
            return {
                'method': method,
                'min': float(np.min(data)),
                'max': float(np.max(data))
            }
        elif method == NormMethod.ROBUST:
            return {
                'method': method,
                'median': float(np.median(data)),
                'q1': float(np.percentile(data, 25)),
                'q3': float(np.percentile(data, 75))
            }
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def _apply_normalization(self, data: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """Apply normalization to data array using pre-computed stats."""
        method = stats['method']
        
        if method == NormMethod.NONE:
            return data
        elif method == NormMethod.STANDARD:
            return (data - stats['mean']) / (stats['std'] + 1e-8)
        elif method == NormMethod.MIN_MAX:
            return (data - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
        elif method == NormMethod.ROBUST:
            iqr = stats['q3'] - stats['q1']
            return (data - stats['median']) / (iqr + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}. Expected one of {[m for m in NormMethod]}.")
    
    def _reverse_normalization(self, data_norm: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """Reverse normalization for data array."""
        method = stats['method']
        
        if method == NormMethod.NONE:
            return data_norm
        elif method == NormMethod.STANDARD:
            return data_norm * stats['std'] + stats['mean']
        elif method == NormMethod.MIN_MAX:
            return data_norm * (stats['max'] - stats['min']) + stats['min']
        elif method == NormMethod.ROBUST:
            iqr = stats['q3'] - stats['q1']
            return data_norm * iqr + stats['median']
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def _normalize_batch(self, data: np.ndarray, columns: List[str], stats: Dict[str, Dict[str, Any]]) -> None:
        """Apply normalization to a batch of data in-place."""
        if not self._is_fitted:
            return
            
        for i, col in enumerate(columns):
            if col in stats:
                data[:, i] = self._apply_normalization(data[:, i], stats[col])
    
    # === DATA EXTRACTION ===
    
    def _extract_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract all dimensional training data from metric_arrays as DataFrames."""            
        exp_codes = self.dataset.get_populated_experiment_codes()
        
        if not exp_codes:
            # Return empty DFs if no data
            return pd.DataFrame(), pd.DataFrame()
        
        X_rows = []
        y_rows = []
        
        for code in exp_codes:
            exp_data = self.dataset.get_experiment(code)
            if exp_data.features is None:
                continue
            
            # Get parameter values
            param_dict = {
                name: exp_data.parameters.get_value(name)
                for name in exp_data.parameters.keys()
                if not isinstance(exp_data.parameters.data_objects.get(name), DataDimension)
            }
            
            # Get dimensional parameters
            dim_params = {
                name: obj for name, obj in exp_data.parameters.data_objects.items()
                if isinstance(obj, DataDimension)
            }
            
            if not dim_params:
                # No dimensions - treat as single position
                y_dict = {}
                for feature_name in exp_data.features.keys():
                    value = exp_data.features.get_value(feature_name)
                    if isinstance(value, np.ndarray):
                        y_dict[feature_name] = float(value.flat[0])
                    else:
                        y_dict[feature_name] = float(value)
                
                X_rows.append(param_dict)
                y_rows.append(y_dict)
                continue
            
            # Multi-dimensional case
            first_feature = list(exp_data.features.keys())[0]
            feature_array = exp_data.features.get_value(first_feature)
            
            if not isinstance(feature_array, np.ndarray):
                continue
            
            shape = feature_array.shape
            
            for idx in np.ndindex(shape):
                row_dict = param_dict.copy()
                dim_names = list(dim_params.keys())
                for i, dim_name in enumerate(dim_names):
                    dim_obj = dim_params[dim_name]
                    row_dict[dim_obj.iterator_code] = idx[i]
                
                X_rows.append(row_dict)
                
                y_dict = {}
                for feature_name in exp_data.features.keys():
                    feature_array = exp_data.features.get_value(feature_name)
                    if isinstance(feature_array, np.ndarray):
                        value = feature_array[idx]
                        if not np.isnan(value):
                            y_dict[feature_name] = float(value)
                
                y_rows.append(y_dict)
        
        if not X_rows:
            return pd.DataFrame(), pd.DataFrame()
        
        X = pd.DataFrame(X_rows)
        y = pd.DataFrame(y_rows)
        
        return X, y

    def _one_hot_encode(self, X_df: pd.DataFrame) -> np.ndarray:
        """Apply one-hot encoding and align columns to schema."""
        X = X_df.copy()
        
        # One-hot encode
        if self._categorical_mappings:
            for col, categories in self._categorical_mappings.items():
                if col not in X.columns:
                    continue
                for category in categories:
                    col_name = f"{col}_{category}"
                    X[col_name] = (X[col] == category).astype(float)
                X = X.drop(columns=[col])
        
        # Ensure columns match and are ordered correctly
        for col in self.input_columns:
            if col not in X.columns:
                X[col] = 0.0
        
        X = X[self.input_columns]
        return X.values.astype(np.float32)

    def _decode_one_hot(self, denorm_array: np.ndarray, consumed_cols: set) -> Dict[str, Any]:
        """Decode one-hot encoded categories from array."""
        params = {}
        for original_col, categories in self._categorical_mappings.items():
            # Find indices of one-hot columns
            one_hot_cols = [f"{original_col}_{cat}" for cat in categories]
            indices = []
            
            for oh_col in one_hot_cols:
                if oh_col in self.input_columns:
                    indices.append(self.input_columns.index(oh_col))
            
            if indices:
                # Get values for this group
                group_values = denorm_array[indices]
                # Argmax to find active category
                max_idx = np.argmax(group_values)
                
                # Map back to category name
                existing_cats = [
                    cat for cat in categories 
                    if f"{original_col}_{cat}" in self.input_columns
                ]
                
                if existing_cats:
                    params[original_col] = existing_cats[max_idx]
                    
                    for idx in indices:
                        consumed_cols.add(self.input_columns[idx])
        return params

    # === CALIBRATION HELPERS ===

    def params_to_array(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Convert parameter dictionary to normalized 1D array.
        Wraps prepare_input for single sample.
        """
        if not self._is_fitted:
            raise RuntimeError("DataModule not fitted.")
            
        df = pd.DataFrame([params])
        arr = self.prepare_input(df)
        return arr[0]

    def array_to_params(self, array: np.ndarray) -> Dict[str, Any]:
        """
        Convert normalized 1D array back to parameter dictionary.
        Handles reverse normalization and reverse one-hot encoding.
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
        
        consumed_cols = set()
        
        # 1. Handle Categorical (Reverse One-Hot)
        params = self._decode_one_hot(denorm_array, consumed_cols)
        
        # 2. Handle Continuous
        for i, col in enumerate(self.input_columns):
            if self.input_columns[i] not in consumed_cols:
                params[col] = float(denorm_array[i])
                
        return params

    def copy(self) -> 'DataModule':
        """Create a deep copy of this DataModule."""
        return copy.deepcopy(self)
    
    def get_split_codes(self, split: SplitType = SplitType.TRAIN) -> List[str]:
        return self._split_codes[split]
    
    def get_split_sizes(self) -> Dict[str, int]:
        """Get the sizes of each split as dict with train/val/test keys."""
        return {
            SplitType.TRAIN: len(self._split_codes[SplitType.TRAIN]),
            SplitType.VAL: len(self._split_codes[SplitType.VAL]),
            SplitType.TEST: len(self._split_codes[SplitType.TEST])
        }
    
    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        batch_str = f"batch_size={self.batch_size}" if self.batch_size else "no batching"
        
        split_sizes = self.get_split_sizes()
        size_str = f"train={split_sizes[SplitType.TRAIN]}, val={split_sizes[SplitType.VAL]}, test={split_sizes[SplitType.TEST]}"
        
        return (
            f"DataModule(normalize='{self._default_normalize}', {batch_str}, splits=({size_str}), "
            f"{fitted_str}, overrides={len(self._feature_overrides)})"
        )
