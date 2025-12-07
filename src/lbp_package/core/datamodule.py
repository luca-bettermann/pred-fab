"""
DataModule - ML preprocessing configuration for LBP datasets.

Manages batching and normalization for machine learning workflows.
Stores normalization parameters, not normalized data (memory efficient).
"""

from typing import Optional, Dict, List, Tuple, Literal, Any, cast
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split

from .data_objects import DataDimension
from .dataset import Dataset


NormalizeMethod = Literal['standard', 'minmax', 'robust', 'none']


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
        dataset: Optional[Dataset] = None,
        batch_size: Optional[int] = None,
        normalize: NormalizeMethod = 'none',
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_seed: Optional[int] = 42
    ):
        """
        Initialize DataModule.
        
        Args:
            dataset: Dataset instance (optional, required for training)
            batch_size: Number of experiments per batch (None = single batch)
            normalize: Default normalization method for features and parameters
            test_size: Fraction of data for test set (0.0-1.0)
            val_size: Fraction of remaining data for validation set (0.0-1.0)
            random_seed: Random seed for reproducible splits (None = random)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._default_normalize = normalize
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed
        
        # Per-feature/parameter normalization overrides
        self._feature_overrides: Dict[str, NormalizeMethod] = {}
        self._parameter_overrides: Dict[str, NormalizeMethod] = {}
        
        # Fitted normalization parameters
        self._feature_stats: Dict[str, Dict[str, Any]] = {}
        self._parameter_stats: Dict[str, Dict[str, Any]] = {}
        self._categorical_mappings: Dict[str, List[str]] = {}
        self._is_fitted = False
        
        # Data storage (Numpy arrays)
        self.X_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None
        self.input_columns: List[str] = []  # Processed columns (after one-hot)
        self.output_columns: List[str] = []
        self.original_input_columns: List[str] = []  # Before one-hot
        
        # Column normalization methods map (for X)
        self._col_norm_methods: Dict[str, NormalizeMethod] = {}
        
        # Create splits
        self._split_indices: Dict[str, List[int]] = {}
        
        # Load and process data immediately if dataset provided
        if self.dataset:
            self._load_and_process_data()
            self.create_splits()
    
    def _load_and_process_data(self) -> None:
        """Load data, apply one-hot encoding, and store as numpy arrays."""
        X_df, y_df = self._extract_raw_data()
        
        self.original_input_columns = list(X_df.columns)
        self.output_columns = list(y_df.columns)
        
        # Process X (One-hot encoding)
        X_processed = X_df.copy()
        self._categorical_mappings = {}
        self._col_norm_methods = {}
        
        for col in X_df.columns:
            method = self.get_parameter_normalize_method(col)
            if method == 'categorical':
                categories = sorted(X_df[col].unique().tolist())
                self._categorical_mappings[col] = categories
                
                # One-hot encode
                for category in categories:
                    col_name = f"{col}_{category}"
                    X_processed[col_name] = (X_df[col] == category).astype(float)
                    self._col_norm_methods[col_name] = 'none'  # Don't normalize binary cols
                
                X_processed = X_processed.drop(columns=[col])
            else:
                self._col_norm_methods[col] = method
        
        self.input_columns = list(X_processed.columns)
        
        # Convert to numpy
        self.X_data = X_processed.values.astype(np.float32)
        self.y_data = y_df.values.astype(np.float32)

    def refresh(self) -> None:
        """Reload data from dataset."""
        self._load_and_process_data()
        self.create_splits()

    def create_splits(self) -> None:
        """Creates split indices based on loaded data."""
        if self.X_data is None or len(self.X_data) == 0:
            self._split_indices = {'train': [], 'val': [], 'test': []}
            return
            
        n_samples = len(self.X_data)
        indices = np.arange(n_samples)
        
        if self.test_size == 0.0 and self.val_size == 0.0:
            self._split_indices = {
                'train': indices.tolist(),
                'val': [],
                'test': []
            }
            return
        
        # Split off test set
        if self.test_size > 0.0:
            train_val_idx, test_idx = train_test_split(
                indices,
                test_size=self.test_size,
                random_state=self.random_seed
            )
        else:
            train_val_idx = indices
            test_idx = np.array([], dtype=int)
        
        # Split remaining into train/val
        if self.val_size > 0.0 and len(train_val_idx) > 0:
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=self.val_size,
                random_state=self.random_seed
            )
        else:
            train_idx = train_val_idx
            val_idx = np.array([], dtype=int)
        
        self._split_indices = {
            'train': train_idx.tolist(),
            'val': val_idx.tolist(),
            'test': test_idx.tolist()
        }
    
    def set_feature_normalize(self, feature_name: str, method: NormalizeMethod) -> None:
        """Override normalization method for a specific feature."""
        self._feature_overrides[feature_name] = method
    
    def set_parameter_normalize(self, parameter_name: str, method: NormalizeMethod) -> None:
        """Override normalization method for a specific parameter."""
        self._parameter_overrides[parameter_name] = method
    
    def get_normalize_method(self, feature_name: str) -> NormalizeMethod:
        """Get normalization method for a feature (override or default)."""
        return cast(NormalizeMethod, self._feature_overrides.get(feature_name, self._default_normalize))
    
    def get_parameter_normalize_method(self, parameter_name: str) -> NormalizeMethod:
        """Get normalization method for a parameter using schema metadata."""
        if parameter_name in self._parameter_overrides:
            return self._parameter_overrides[parameter_name]
        
        if self.dataset and self.dataset.schema:
            if parameter_name in self.dataset.schema.parameters.data_objects:
                data_obj = self.dataset.schema.parameters.data_objects[parameter_name]
                strategy = data_obj.normalize_strategy
                return cast(NormalizeMethod, self._default_normalize if strategy == 'default' else strategy)
            
            for dim_obj in self.dataset.schema.parameters.data_objects.values():
                if isinstance(dim_obj, DataDimension) and dim_obj.iterator_code == parameter_name:
                    strategy = dim_obj.normalize_strategy
                    return cast(NormalizeMethod, self._default_normalize if strategy == 'default' else strategy)
        
        return cast(NormalizeMethod, self._default_normalize)
    
    # === SHARED NORMALIZATION HELPERS ===
    
    def _compute_normalization_stats(self, data: np.ndarray, method: NormalizeMethod) -> Dict[str, Any]:
        """Compute normalization statistics for a data array."""
        if method == 'none':
            return {'method': 'none'}
        elif method == 'standard':
            return {
                'method': method,
                'mean': float(np.mean(data)),
                'std': float(np.std(data))
            }
        elif method == 'minmax':
            return {
                'method': method,
                'min': float(np.min(data)),
                'max': float(np.max(data))
            }
        elif method == 'robust':
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
        
        if method == 'none':
            return data
        elif method == 'standard':
            return (data - stats['mean']) / (stats['std'] + 1e-8)
        elif method == 'minmax':
            return (data - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
        elif method == 'robust':
            iqr = stats['q3'] - stats['q1']
            return (data - stats['median']) / (iqr + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def _reverse_normalization(self, data_norm: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """Reverse normalization for data array."""
        method = stats['method']
        
        if method == 'none':
            return data_norm
        elif method == 'standard':
            return data_norm * stats['std'] + stats['mean']
        elif method == 'minmax':
            return data_norm * (stats['max'] - stats['min']) + stats['min']
        elif method == 'robust':
            iqr = stats['q3'] - stats['q1']
            return data_norm * iqr + stats['median']
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    # === DATA EXTRACTION ===
    
    def _extract_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract all dimensional training data from metric_arrays as DataFrames."""
        if self.dataset is None:
            return pd.DataFrame(), pd.DataFrame()
            
        exp_codes = [
            code for code in self.dataset.get_experiment_codes()
            if self.dataset.get_experiment(code).features is not None
        ]
        
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

    def fit_normalize(self, split: str = 'train') -> None:
        """Fit normalization parameters on the specified split."""
        if self.X_data is None or self.y_data is None:
            return

        if split not in self._split_indices:
            raise ValueError(f"Unknown split: {split}")
            
        indices = self._split_indices[split]
        if len(indices) == 0:
            return
            
        X_subset = self.X_data[indices]
        y_subset = self.y_data[indices]
        
        # Fit X
        self._parameter_stats = {}
        for i, col in enumerate(self.input_columns):
            method = self._col_norm_methods.get(col, 'none')
            if method != 'none':
                self._parameter_stats[col] = self._compute_normalization_stats(X_subset[:, i], method)
        
        # Fit y
        self._feature_stats = {}
        for i, col in enumerate(self.output_columns):
            method = self.get_normalize_method(col)
            if method != 'none':
                self._feature_stats[col] = self._compute_normalization_stats(y_subset[:, i], method)
        
        self._is_fitted = True
    
    def get_batches(self, split: str = 'train') -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get batched, normalized data for a split.
        Returns list of (X_batch, y_batch) tuples.
        """
        if self.X_data is None or self.y_data is None:
            return []
            
        indices = self._split_indices.get(split, [])
        if len(indices) == 0:
            return []
            
        X = self.X_data[indices].copy()
        y = self.y_data[indices].copy()
        
        # Normalize
        if self._is_fitted:
            for i, col in enumerate(self.input_columns):
                if col in self._parameter_stats:
                    X[:, i] = self._apply_normalization(X[:, i], self._parameter_stats[col])
            
            for i, col in enumerate(self.output_columns):
                if col in self._feature_stats:
                    y[:, i] = self._apply_normalization(y[:, i], self._feature_stats[col])
        
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
        X_arr = X.values.astype(np.float32)
        
        # Normalize
        for i, col in enumerate(self.input_columns):
            if col in self._parameter_stats:
                X_arr[:, i] = self._apply_normalization(X_arr[:, i], self._parameter_stats[col])
                
        return X_arr

    def denormalize_output(self, y_norm: np.ndarray) -> np.ndarray:
        """Reverse normalization for target features (y)."""
        if not self._is_fitted:
            return y_norm.copy()
        
        y = y_norm.copy()
        # Handle both single sample (1D) and batch (2D)
        if y.ndim == 1:
            for i, col in enumerate(self.output_columns):
                if col in self._feature_stats:
                    y[i] = self._reverse_normalization(y[i], self._feature_stats[col])
        else:
            for i, col in enumerate(self.output_columns):
                if col in self._feature_stats:
                    y[:, i] = self._reverse_normalization(y[:, i], self._feature_stats[col])
        
        return y

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
    
    def copy(self) -> 'DataModule':
        """Create a deep copy of this DataModule."""
        return copy.deepcopy(self)
    
    def get_split_sizes(self) -> Dict[str, int]:
        """Get the sizes of each split as dict with train/val/test keys."""
        return {
            'train': len(self._split_indices['train']),
            'val': len(self._split_indices['val']),
            'test': len(self._split_indices['test'])
        }
    
    def get_normalization_state(self) -> Dict[str, Any]:
        """Export normalization state for inference bundle."""
        if not self._is_fitted:
            return {
                'method': self._default_normalize,
                'is_fitted': False,
                'feature_stats': {},
                'parameter_stats': {},
                'categorical_mappings': {},
                'input_columns': [],
                'output_columns': []
            }
        
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
    
    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        batch_str = f"batch_size={self.batch_size}" if self.batch_size else "no batching"
        split_str = f"test={self.test_size}, val={self.val_size}"
        return (
            f"DataModule(normalize='{self._default_normalize}', {batch_str}, splits=({split_str}), "
            f"{fitted_str}, overrides={len(self._feature_overrides)})"
        )
