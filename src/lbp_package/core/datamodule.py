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

from .dataset import Dataset


NormalizeMethod = Literal['standard', 'minmax', 'robust', 'none']


class DataModule:
    """
    Preprocessing configuration for ML workflows.
    
    - Extract and batch features from Dataset experiments
    - Fit/apply normalization to features (y not X)
    - On-the-fly transforms with stored parameters
    - Per-feature normalization overrides supported
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
            normalize: Default normalization method for all features
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
        
        # Per-feature normalization overrides
        self._feature_overrides: Dict[str, NormalizeMethod] = {}
        
        # Fitted normalization parameters (populated during training)
        self._feature_stats: Dict[str, Dict[str, Any]] = {}  # feature → {mean, std, method, ...}
        self._is_fitted = False
        
        # Split indices (populated on first extract call)
        self._split_indices: Optional[Dict[str, List[int]]] = None
    
    def set_feature_normalize(self, feature_name: str, method: NormalizeMethod) -> None:
        """Override normalization method for a specific feature."""
        self._feature_overrides[feature_name] = method
    
    def get_normalize_method(self, feature_name: str) -> NormalizeMethod:
        """Get normalization method for a feature (override or default)."""
        return cast(NormalizeMethod, self._feature_overrides.get(feature_name, self._default_normalize))
    
    def _create_splits(self, n_samples: int) -> None:
        """Create train/val/test split indices."""
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
    
    def extract_all(self, split: Optional[Literal['train', 'val', 'test']] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract parameters and features from dataset experiments as DataFrames."""
        # Get experiments with features
        exp_codes = [
            code for code in self.dataset.get_experiment_codes()
            if self.dataset.get_experiment(code).features is not None
        ]
        
        if not exp_codes:
            raise ValueError("No experiments with features found in dataset")
        
        X_data = []
        y_data = []
        for code in exp_codes:
            # Extract parameters (X)
            exp_data = self.dataset.get_experiment(code)
            param_dict = dict(exp_data.parameters.values)  # Get values dict
            X_data.append(param_dict)
        
            # Extract features (y)
            if exp_data.features is None:
                continue
            feature_dict = dict(exp_data.features.values)  # Get values dict
            y_data.append(feature_dict)

        X = pd.DataFrame(X_data)
        y = pd.DataFrame(y_data)
        
        # Create splits on first call
        if self._split_indices is None:
            self._create_splits(len(X))
        
        # Return requested split
        if split is None:
            return X, y
        elif split in ['train', 'val', 'test']:
            if self._split_indices is None:
                raise RuntimeError("Splits not initialized")
            indices = self._split_indices[split]
            if len(indices) == 0:
                raise ValueError(f"Split '{split}' is empty (configured sizes: test={self.test_size}, val={self.val_size})")
            return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)  # type: ignore[return-value]
        else:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', 'test', or None.")
    
    def get_batches(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Get data as list of (X_batch, y_batch) tuples."""
        X, y = self.extract_all()
        
        # No batching - return single batch
        if self.batch_size is None:
            return [(X, y)]
        
        # Split into batches
        n_samples = len(X)
        n_batches = int(np.ceil(n_samples / self.batch_size))
        
        batches = []
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            
            X_batch = X.iloc[start_idx:end_idx]
            y_batch = y.iloc[start_idx:end_idx]
            batches.append((X_batch, y_batch))
        
        return batches
    
    def fit_normalize(self, y: pd.DataFrame) -> None:
        """Fit normalization parameters on feature data."""
        self._feature_stats = {}
        
        for col in y.columns:
            method = self.get_normalize_method(col)
            
            if method == 'none':
                continue
            
            elif method == 'standard':
                self._feature_stats[col] = {
                    'method': 'standard',
                    'mean': float(y[col].mean()),
                    'std': float(y[col].std())
                }
            
            elif method == 'minmax':
                self._feature_stats[col] = {
                    'method': 'minmax',
                    'min': float(y[col].min()),
                    'max': float(y[col].max())
                }
            
            elif method == 'robust':
                self._feature_stats[col] = {
                    'method': 'robust',
                    'median': float(y[col].median()),
                    'q1': float(y[col].quantile(0.25)),
                    'q3': float(y[col].quantile(0.75))
                }
        
        self._is_fitted = True
    
    def normalize(self, y: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization to features using fitted parameters."""
        if not self._is_fitted:
            raise RuntimeError("DataModule not fitted. Call fit_normalize() first.")
        
        y_norm = y.copy()
        
        for col in y.columns:
            if col not in self._feature_stats:
                continue  # No normalization for this feature
            
            stats = self._feature_stats[col]
            method = stats['method']
            
            if method == 'standard':
                y_norm[col] = (y[col] - stats['mean']) / (stats['std'] + 1e-8)
            
            elif method == 'minmax':
                y_norm[col] = (y[col] - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
            
            elif method == 'robust':
                iqr = stats['q3'] - stats['q1']
                y_norm[col] = (y[col] - stats['median']) / (iqr + 1e-8)
        
        return y_norm
    
    def denormalize(self, y_norm: pd.DataFrame) -> pd.DataFrame:
        """Reverse normalization to get original scale."""
        if not self._is_fitted:
            # Not fitted, return as-is
            return y_norm.copy()
        
        y = y_norm.copy()
        
        for col in y.columns:
            if col not in self._feature_stats:
                continue
            
            stats = self._feature_stats[col]
            method = stats['method']
            
            if method == 'standard':
                y[col] = y_norm[col] * stats['std'] + stats['mean']
            
            elif method == 'minmax':
                y[col] = y_norm[col] * (stats['max'] - stats['min']) + stats['min']
            
            elif method == 'robust':
                iqr = stats['q3'] - stats['q1']
                y[col] = y_norm[col] * iqr + stats['median']
        
        return y
    
    def copy(self) -> 'DataModule':
        """Create a deep copy of this DataModule."""
        return copy.deepcopy(self)
    
    def get_feature_names(self) -> List[str]:
        """Get list of features from first experiment with features."""
        for code in self.dataset.get_experiment_codes():
            exp_data = self.dataset.get_experiment(code)
            if exp_data.features is not None:
                return list(exp_data.features.data_objects.keys())
        return []
    
    def get_split_sizes(self) -> Dict[str, int]:
        """Get the sizes of each split as dict with train/val/test keys."""
        if self._split_indices is None:
            # Force split creation
            _ = self.extract_all()
        
        if self._split_indices is None:
            raise RuntimeError("Splits not initialized")
        
        return {
            'train': len(self._split_indices['train']),
            'val': len(self._split_indices['val']),
            'test': len(self._split_indices['test'])
        }
    
    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        batch_str = f"batch_size={self.batch_size}" if self.batch_size else "no batching"
        split_str = f"test={self.test_size}, val={self.val_size}"
        return (
            f"DataModule(normalize='{self._default_normalize}', {batch_str}, splits=({split_str}), "
            f"{fitted_str}, overrides={len(self._feature_overrides)})"
        )
