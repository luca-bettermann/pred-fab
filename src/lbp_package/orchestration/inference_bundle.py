"""
Inference Bundle for production deployment.

Lightweight wrapper for trained models that enables inference without Dataset
or training dependencies. Load from exported bundle, predict features, and
optionally evaluate performance.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
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
            for feat in model.predicted_features:
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
        
        # Collect predictions from all models
        predictions: Dict[str, Any] = {}
        for model in self.prediction_models:
            # Predict (normalized)
            y_pred_norm = model.forward_pass(X)
            
            # Denormalize
            y_pred = self._denormalize(y_pred_norm, model.predicted_features)
            
            # Add to results
            for col in model.predicted_features:
                if col in y_pred.columns:
                    predictions[col] = y_pred[col].tolist()
        
        return pd.DataFrame(predictions)
    
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
    
    def _denormalize(
        self, 
        y_norm: pd.DataFrame, 
        feature_names: List[str]
    ) -> pd.DataFrame:
        """Apply inverse normalization transform to predictions."""
        if not self.normalization_state['is_fitted']:
            return y_norm.copy()
        
        y_denorm = y_norm.copy()
        feature_stats = self.normalization_state['feature_stats']
        
        for feat in feature_names:
            if feat not in feature_stats:
                continue
            
            stats = feature_stats[feat]
            method = stats['method']
            
            if method == 'standard':
                y_denorm[feat] = y_norm[feat] * stats['std'] + stats['mean']
            
            elif method == 'minmax':
                y_denorm[feat] = y_norm[feat] * (stats['max'] - stats['min']) + stats['min']
            
            elif method == 'robust':
                iqr = stats['q3'] - stats['q1']
                y_denorm[feat] = y_norm[feat] * iqr + stats['median']
        
        return y_denorm
    
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
