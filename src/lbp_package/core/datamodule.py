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
    - On-the-fly transforms with stored parameters
    - Per-feature/parameter normalization overrides supported
    - Auto-detect normalization from DataObject schema metadata
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = None,
        normalize: NormalizeMethod = 'none',
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_seed: Optional[int] = 42
    ):
        """
        Initialize DataModule.
        
        Args:
            dataset: Dataset instance with experiments containing features
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
        
        # Fitted normalization parameters (populated during training)
        self._feature_stats: Dict[str, Dict[str, Any]] = {}  # feature → {mean, std, method, ...}
        self._parameter_stats: Dict[str, Dict[str, Any]] = {}  # parameter → {mean, std, method, ...}
        self._categorical_mappings: Dict[str, List[str]] = {}  # param → categories for one-hot
        self._is_fitted = False
        
        # Create splits immediately
        self._split_indices: Dict[str, List[int]] = {}
        self.create_splits()
    
    def create_splits(self) -> None:
        """Extracts data to determine sample count and creates split indices."""
        # Check if dataset has any experiments with data
        exp_codes = [
            code for code in self.dataset.get_experiment_codes()
            if self.dataset.get_experiment(code).features is not None
        ]
        
        if not exp_codes:
            # Empty dataset - create empty splits
            self._split_indices = {
                'train': [],
                'val': [],
                'test': []
            }
            return
        
        # Extract data to know how many samples we have
        X, _ = self._extract_all_data()
        n_samples = len(X)
        
        # Create split indices
        indices = np.arange(n_samples)
        
        if self.test_size == 0.0 and self.val_size == 0.0:
            # No splits - all data for training
            self._split_indices = {
                'train': indices.tolist(),
                'val': [],
                'test': []
            }
            return
        
        # Split off test set first
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
            # val_size is fraction of remaining data
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
        """
        Get normalization method for a parameter using schema metadata.
        
        Priority:
        1. User explicit override (via set_parameter_normalize)
        2. DataObject normalize_strategy ('default' → use DataModule default)
        3. DataModule default normalization
        """
        # Check user override first
        if parameter_name in self._parameter_overrides:
            return self._parameter_overrides[parameter_name]
        
        # Get DataObject from schema if available
        if self.dataset.schema:
            # Check parameters
            if parameter_name in self.dataset.schema.parameters.data_objects:
                data_obj = self.dataset.schema.parameters.data_objects[parameter_name]
                strategy = data_obj.normalize_strategy
                
                if strategy == 'default':
                    return cast(NormalizeMethod, self._default_normalize)
                else:
                    return cast(NormalizeMethod, strategy)
            
            # Check dimensions (iterator names like 'layer')
            for dim_obj in self.dataset.schema.parameters.data_objects.values():
                if isinstance(dim_obj, DataDimension) and dim_obj.dim_iterator_name == parameter_name:
                    strategy = dim_obj.normalize_strategy
                    if strategy == 'default':
                        return cast(NormalizeMethod, self._default_normalize)
                    else:
                        return cast(NormalizeMethod, strategy)
        
        # Fallback to default
        return cast(NormalizeMethod, self._default_normalize)
    
    # === SHARED NORMALIZATION HELPERS ===
    
    def _compute_normalization_stats(self, data: pd.Series, method: NormalizeMethod) -> Dict[str, Any]:
        """Compute normalization statistics for a single column."""
        if method == 'none':
            return {'method': 'none'}
        elif method == 'standard':
            return {
                'method': method,
                'mean': float(data.mean()),
                'std': float(data.std())
            }
        elif method == 'minmax':
            return {
                'method': method,
                'min': float(data.min()),
                'max': float(data.max())
            }
        elif method == 'robust':
            return {
                'method': method,
                'median': float(data.median()),
                'q1': float(data.quantile(0.25)),
                'q3': float(data.quantile(0.75))
            }
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def _apply_normalization(self, data: pd.Series, stats: Dict[str, Any]) -> pd.Series:
        """Apply normalization to a single column using pre-computed stats."""
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
    
    def _reverse_normalization(self, data_norm: pd.Series, stats: Dict[str, Any]) -> pd.Series:
        """Reverse normalization for a single column."""
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
    
    def _extract_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract all dimensional training data from metric_arrays."""        
        # Get experiments with metric arrays
        exp_codes = [
            code for code in self.dataset.get_experiment_codes()
            if self.dataset.get_experiment(code).features is not None
        ]
        
        if not exp_codes:
            raise ValueError("No experiments with metric_arrays found in dataset")
        
        # Collect all dimensional positions
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
                    # Safe conversion to float
                    if isinstance(value, np.ndarray):
                        y_dict[feature_name] = float(value.flat[0])
                    else:
                        y_dict[feature_name] = float(value)
                
                X_rows.append(param_dict)
                y_rows.append(y_dict)
                continue
            
            # Multi-dimensional case: flatten all positions
            # Get shape from first metric array
            first_feature = list(exp_data.features.keys())[0]
            feature_array = exp_data.features.get_value(first_feature)
            
            if not isinstance(feature_array, np.ndarray):
                continue
            
            shape = feature_array.shape
            
            # Iterate over all dimensional positions
            for idx in np.ndindex(shape):
                # Create row for this position
                row_dict = param_dict.copy()
                
                # Add dimension indices
                dim_names = list(dim_params.keys())
                for i, dim_name in enumerate(dim_names):
                    dim_obj = dim_params[dim_name]
                    # Use iterator name (e.g., 'layer' instead of 'n_layers')
                    row_dict[dim_obj.dim_iterator_name] = idx[i]
                
                X_rows.append(row_dict)
                
                # Extract feature values at this position
                y_dict = {}
                for feature_name in exp_data.features.keys():
                    feature_array = exp_data.features.get_value(feature_name)
                    if isinstance(feature_array, np.ndarray):
                        value = feature_array[idx]
                        if not np.isnan(value):
                            y_dict[feature_name] = float(value)
                
                y_rows.append(y_dict)
        
        if not X_rows:
            raise ValueError("No dimensional data extracted from metric_arrays")
        
        X = pd.DataFrame(X_rows)
        y = pd.DataFrame(y_rows)
        
        return X, y
    
    def get_split(self, split: Literal['train', 'val', 'test', 'all']) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get a specific data split (train/val/test/all)."""
        # Extract all data
        X, y = self._extract_all_data()
        
        # Validate training data for NaN values
        if split == 'train' and y.isnull().any().any():
            nan_counts = y.isnull().sum()
            nan_features = nan_counts[nan_counts > 0].to_dict()
            raise ValueError(
                f"Training data contains NaN values in target features: {nan_features}. "
                "Ensure all positions have measured values before training."
            )
        
        # Return requested split
        if split == 'all':
            return X, y
        elif split in ['train', 'val', 'test']:
            indices = self._split_indices[split]
            if len(indices) == 0:
                raise ValueError(f"Split '{split}' is empty")
            return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)  # type: ignore[return-value]
        else:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', 'test', or 'all'.")

    
    def fit_normalize(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Fit normalization parameters on both input parameters (X) and features (y)."""
        # Fit parameter normalization (X)
        self._parameter_stats = {}
        self._categorical_mappings = {}
        for col in X.columns:
            method = self.get_parameter_normalize_method(col)
            if method == 'categorical':
                # Store unique categories for one-hot encoding
                categories = sorted(X[col].unique().tolist())
                self._categorical_mappings[col] = categories
            elif method != 'none':
                self._parameter_stats[col] = self._compute_normalization_stats(X[col], method)
        
        # Fit feature normalization (y)
        self._feature_stats = {}
        for col in y.columns:
            method = self.get_normalize_method(col)
            if method != 'none':
                self._feature_stats[col] = self._compute_normalization_stats(y[col], method)
        
        self._is_fitted = True
    
    def normalize_parameters(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization to input parameters (X), including one-hot encoding for categoricals."""
        if not self._is_fitted:
            raise RuntimeError("DataModule not fitted. Call fit_normalize() first.")
        
        X_norm = X.copy()
        
        # Apply numeric normalization
        for col in X.columns:
            if col in self._parameter_stats:
                X_norm[col] = self._apply_normalization(X[col], self._parameter_stats[col])
        
        # Apply one-hot encoding for categorical columns
        if self._categorical_mappings:
            # Create one-hot columns
            for col, categories in self._categorical_mappings.items():
                if col not in X_norm.columns:
                    continue
                
                # Create binary columns for each category
                for category in categories:
                    col_name = f"{col}_{category}"
                    X_norm[col_name] = (X_norm[col] == category).astype(float)
                
                # Drop original categorical column
                X_norm = X_norm.drop(columns=[col])
        
        return X_norm
    
    def normalize_features(self, y: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization to target features (y)."""
        if not self._is_fitted:
            raise RuntimeError("DataModule not fitted. Call fit_normalize() first.")
        
        y_norm = y.copy()
        for col in y.columns:
            if col in self._feature_stats:
                y_norm[col] = self._apply_normalization(y[col], self._feature_stats[col])
        
        return y_norm
    
    def denormalize_parameters(self, X_norm: pd.DataFrame) -> pd.DataFrame:
        """Reverse normalization for input parameters (X), including one-hot decoding for categoricals."""
        if not self._is_fitted:
            return X_norm.copy()
        
        X = X_norm.copy()
        
        # Reverse one-hot encoding first (reconstruct categorical columns)
        if self._categorical_mappings:
            for col, categories in self._categorical_mappings.items():
                # Find one-hot encoded columns
                onehot_cols = [f"{col}_{cat}" for cat in categories]
                available_cols = [c for c in onehot_cols if c in X.columns]
                
                if available_cols:
                    # Get the category with highest value (argmax)
                    onehot_values = X[available_cols].values
                    max_indices = onehot_values.argmax(axis=1)
                    
                    # Map back to original categories
                    X[col] = [categories[i] for i in max_indices]
                    
                    # Drop one-hot columns
                    X = X.drop(columns=available_cols)
        
        # Reverse numeric normalization
        for col in X.columns:
            if col in self._parameter_stats:
                X[col] = self._reverse_normalization(X[col], self._parameter_stats[col])
        
        return X
    
    def denormalize_features(self, y_norm: pd.DataFrame) -> pd.DataFrame:
        """Reverse normalization for target features (y)."""
        if not self._is_fitted:
            return y_norm.copy()
        
        y = y_norm.copy()
        for col in y.columns:
            if col in self._feature_stats:
                y[col] = self._reverse_normalization(y_norm[col], self._feature_stats[col])
        
        return y
    
    def copy(self) -> 'DataModule':
        """Create a deep copy of this DataModule."""
        return copy.deepcopy(self)
    
    def get_feature_names(self) -> List[str]:
        """Get list of features from first experiment with metric_arrays."""
        for code in self.dataset.get_experiment_codes():
            exp_data = self.dataset.get_experiment(code)
            if exp_data.features is not None:
                return list(exp_data.features.keys())
        return []
    
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
            # Return unfitted state
            return {
                'method': self._default_normalize,
                'is_fitted': False,
                'feature_stats': {}
            }
        
        return {
            'method': self._default_normalize,
            'is_fitted': True,
            'feature_stats': copy.deepcopy(self._feature_stats)
        }
    
    def set_normalization_state(self, state: Dict[str, Any]) -> None:
        """Restore normalization state from exported bundle."""
        self._default_normalize = state['method']
        self._is_fitted = state['is_fitted']
        self._feature_stats = copy.deepcopy(state['feature_stats'])
    
    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        batch_str = f"batch_size={self.batch_size}" if self.batch_size else "no batching"
        split_str = f"test={self.test_size}, val={self.val_size}"
        return (
            f"DataModule(normalize='{self._default_normalize}', {batch_str}, splits=({split_str}), "
            f"{fitted_str}, overrides={len(self._feature_overrides)})"
        )
