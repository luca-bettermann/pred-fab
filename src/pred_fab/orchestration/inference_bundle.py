"""
Inference Bundle for production deployment.

Lightweight wrapper for trained models that enables inference without Dataset
or training dependencies. Load from exported bundle, predict features, and
optionally evaluate performance.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import pickle
import copy

from ..interfaces.prediction import IPredictionModel


class InferenceBundle:
    """
    Lightweight wrapper for production inference without training dependencies.
    
    - Load exported models from bundle file (pickle)
    - Predict features for new parameter combinations
    - Automatic input validation against schema
    - Automatic denormalization of predictions
    - No Dataset, training code, or evaluation dependencies
    """
    
    def __init__(
        self,
        prediction_models: List[IPredictionModel],
        normalization_state: Dict[str, Any],
        schema_dict: Dict[str, Any]
    ):
        """
        Initialize bundle with models and metadata.
        
        Args:
            prediction_models: List of trained prediction model instances
            normalization_state: Normalization parameters from DataModule
            schema_dict: Dataset schema for input validation
        """
        self.prediction_models = prediction_models
        self.normalization_state = normalization_state
        self.schema = schema_dict
        
        # Build feature-to-model mapping
        self.feature_to_model: Dict[str, IPredictionModel] = {}
        for model in prediction_models:
            for feat in model.outputs:
                self.feature_to_model[feat] = model
    
    @classmethod
    def load(cls, filepath: str) -> 'InferenceBundle':
        """Load trained models from exported bundle file."""
        # Load bundle dict
        with open(filepath, 'rb') as f:
            bundle_dict = pickle.load(f)
        
        # Reconstruct prediction models
        pred_models = []
        for spec in bundle_dict['prediction_models']:
            model = cls._reconstruct_model(spec)
            pred_models.append(model)
        
        return cls(
            prediction_models=pred_models,
            normalization_state=bundle_dict['normalization'],
            schema_dict=bundle_dict['schema']
        )
    
    @staticmethod
    def _reconstruct_model(spec: Dict[str, Any]) -> IPredictionModel:
        """Reconstruct model from class path and artifacts."""
        # Import model class
        class_path = spec['class_path']
        module_path, class_name = class_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        ModelClass = getattr(module, class_name)
        
        # Create instance
        model = ModelClass(logger=None)  # type: ignore[arg-type]
        
        # Restore artifacts
        model._set_model_artifacts(spec['artifacts'])
        
        return model
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict features with validation and denormalization."""
        # Validate inputs against schema
        self._validate_inputs(X)
        
        # Prepare input (one-hot + normalize)
        X_norm = self._prepare_input(X)
        
        # Collect predictions from all models
        predictions: Dict[str, Any] = {}
        for model in self.prediction_models:
            # Predict (normalized)
            y_pred_norm = model.forward_pass(X_norm)
            
            # Denormalize
            y_pred = self._denormalize_values(y_pred_norm, model.outputs)
            
            # Add to results
            for i, col in enumerate(model.outputs):
                predictions[col] = y_pred[:, i].tolist()
        
        return pd.DataFrame(predictions)
    
    def _prepare_input(self, X_df: pd.DataFrame) -> np.ndarray:
        """Prepare input DataFrame for inference (one-hot + normalize)."""
        X = X_df.copy()
        
        categorical_mappings = self.normalization_state.get('categorical_mappings', {})
        input_columns = self.normalization_state.get('input_columns', [])
        parameter_stats = self.normalization_state.get('parameter_stats', {})
        
        # One-hot encode
        if categorical_mappings:
            for col, categories in categorical_mappings.items():
                if col not in X.columns:
                    continue
                for category in categories:
                    col_name = f"{col}_{category}"
                    X[col_name] = (X[col] == category).astype(float)
                X = X.drop(columns=[col])
        
        # Ensure columns match and are ordered correctly
        for col in input_columns:
            if col not in X.columns:
                X[col] = 0.0
        
        X = X[input_columns]
        X_arr = X.values.astype(np.float32)
        
        # Normalize
        for i, col in enumerate(input_columns):
            if col in parameter_stats:
                X_arr[:, i] = self._apply_normalization(X_arr[:, i], parameter_stats[col])
                
        return X_arr

    def _denormalize_values(self, values: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Reverse normalization for specific features."""
        feature_stats = self.normalization_state.get('feature_stats', {})
        
        y = values.copy()
        if y.ndim == 1:
            for i, name in enumerate(feature_names):
                if name in feature_stats:
                    y[i] = self._reverse_normalization(y[i], feature_stats[name])
        else:
            for i, name in enumerate(feature_names):
                if name in feature_stats:
                    y[:, i] = self._reverse_normalization(y[:, i], feature_stats[name])
        return y

    def _apply_normalization(self, data: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """Apply normalization to data array using pre-computed stats."""
        method = stats['method']
        
        if method == 'none':
            return data
        elif method == 'standard':
            return (data - stats['mean']) / (stats['std'] + 1e-8)
        elif method == 'minmax':
            denom = stats['max'] - stats['min']
            if abs(denom) < 1e-12:
                return np.zeros_like(data, dtype=np.float64)
            return (data - stats['min']) / (denom + 1e-8)
        elif method == 'robust':
            iqr = stats['q3'] - stats['q1']
            return (data - stats['median']) / (iqr + 1e-8)
        else:
            return data

    def _reverse_normalization(self, data_norm: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """Reverse normalization for data array."""
        method = stats['method']
        
        if method == 'none':
            return data_norm
        elif method == 'standard':
            return data_norm * stats['std'] + stats['mean']
        elif method == 'minmax':
            denom = stats['max'] - stats['min']
            if abs(denom) < 1e-12:
                return np.full_like(data_norm, fill_value=stats['min'], dtype=np.float64)
            return data_norm * denom + stats['min']
        elif method == 'robust':
            iqr = stats['q3'] - stats['q1']
            return data_norm * iqr + stats['median']
        else:
            return data_norm
    
    def _validate_inputs(self, X: pd.DataFrame) -> None:
        """Validate parameter columns against schema."""
        # Extract parameter names from schema dict
        params_block = self.schema.get('parameters', {})
        if 'data_objects' in params_block:
            schema_params = set(params_block['data_objects'].keys())
        else:
            # Fallback: no validation if schema format unexpected
            return
        
        for param_name in X.columns:
            if param_name not in schema_params:
                raise ValueError(
                    f"Unknown parameter '{param_name}'. "
                    f"Expected parameters: {sorted(schema_params)}"
                )
            
            # TODO: Add range validation using schema constraints
    
    @property
    def feature_names(self) -> List[str]:
        """List of all predictable features."""
        return list(self.feature_to_model.keys())
    
    def __repr__(self) -> str:
        """Bundle summary with model and feature counts."""
        n_models = len(self.prediction_models)
        n_features = len(self.feature_names)
        norm_method = self.normalization_state.get('method', 'none')
        return (
            f"InferenceBundle(models={n_models}, features={n_features}, "
            f"normalization='{norm_method}')"
        )
