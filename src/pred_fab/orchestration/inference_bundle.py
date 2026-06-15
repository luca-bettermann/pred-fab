"""
Inference Bundle for production deployment.

Lightweight wrapper for trained models that enables inference without Dataset
or training dependencies. Load from exported bundle, predict features, and
optionally evaluate performance.
"""

from typing import Any
import pandas as pd
import numpy as np
import torch
import pickle
import copy

from ..interfaces.prediction import IPredictionModel
from ..core.normalisers import normaliser_from_dict


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
        prediction_models: list[IPredictionModel],
        normalization_state: dict[str, Any],
        schema_dict: dict[str, Any]
    ):
        """Bundle trained models with the normalisation state and schema needed for inference."""
        self.prediction_models = prediction_models
        self.normalization_state = normalization_state
        self.schema = schema_dict
        
        # Build feature-to-model mapping
        self.feature_to_model: dict[str, IPredictionModel] = {}
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
    def _reconstruct_model(spec: dict[str, Any]) -> IPredictionModel:
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
        self._validate_inputs(X)

        X_norm_np = self._prepare_input(X)
        X_norm_t = torch.from_numpy(X_norm_np)

        input_columns = self.normalization_state.get('input_columns', [])
        col_to_idx = {c: i for i, c in enumerate(input_columns)}

        predictions: dict[str, Any] = {}
        for model in self.prediction_models:
            # Each model was trained on its own input subset (X[:, input_indices]
            # in PredictionSystem); forward_pass expects columns ordered by the
            # model's input_parameters + input_features, not the full matrix.
            model_cols = list(model.input_parameters) + list(model.input_features)
            missing = [c for c in model_cols if c not in col_to_idx]
            if missing:
                raise ValueError(
                    f"Model inputs {missing} absent from bundle input_columns; "
                    f"bundle is inconsistent with the model."
                )
            idx = [col_to_idx[c] for c in model_cols]
            y_pred_dict = model.forward_pass(X_norm_t[:, idx])
            y_pred_norm_np = np.stack(
                [y_pred_dict[f].detach().cpu().numpy() for f in model.outputs], axis=-1,
            )
            y_pred_np = self._denormalize_values(y_pred_norm_np, model.outputs)
            for i, col in enumerate(model.outputs):
                predictions[col] = y_pred_np[:, i].tolist()

        return pd.DataFrame(predictions)
    
    def _prepare_input(self, X_df: pd.DataFrame) -> np.ndarray:
        """Prepare input DataFrame for inference (cat-index encode + normalize).

        Categoricals are emitted as a single integer-index column — the index
        into the sorted category list — matching the training-time
        ``DataModule._encode_inputs`` encoding (the model expands them
        internally via embedding/one-hot). Cat columns are absent from
        ``parameter_stats``, so they pass through unnormalised, same as
        training.
        """
        X = X_df.copy()

        categorical_mappings = self.normalization_state.get('categorical_mappings', {})
        input_columns = self.normalization_state.get('input_columns', [])
        parameter_stats = self.normalization_state.get('parameter_stats', {})

        # Categorical → single cat-index column; keep the parent column name so
        # it aligns with input_columns (which holds the parent code, not one-hots).
        for col, categories in categorical_mappings.items():
            if col not in X.columns:
                continue
            cat_to_idx = {c: i for i, c in enumerate(categories)}
            X[col] = X[col].map(lambda v: cat_to_idx.get(v, 0))

        # Ensure columns match and are ordered correctly (missing → 0).
        for col in input_columns:
            if col not in X.columns:
                X[col] = 0.0

        X = X[input_columns]
        X_arr = X.values.astype(np.float32)

        # Normalize numeric columns; cat-index columns are not in
        # parameter_stats and pass through unchanged.
        for i, col in enumerate(input_columns):
            if col in parameter_stats:
                X_arr[:, i] = self._apply_normalization(X_arr[:, i], parameter_stats[col])

        return X_arr

    def _denormalize_values(self, values: np.ndarray, feature_names: list[str]) -> np.ndarray:
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

    def _apply_normalization(self, data: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
        """Apply normalization via the canonical NormaliserModule."""
        return normaliser_from_dict(stats).forward(data)

    def _reverse_normalization(self, data_norm: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
        """Reverse normalization via the canonical NormaliserModule."""
        return normaliser_from_dict(stats).reverse(data_norm)
    
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

        # Categorical values must be in the trained vocabulary — otherwise the
        # cat-index encode would silently map them to index 0 (wrong predictions
        # for a deployed model). Fail loudly instead.
        categorical_mappings = self.normalization_state.get('categorical_mappings', {})
        for col, categories in categorical_mappings.items():
            if col not in X.columns:
                continue
            vocab = set(categories)
            unknown = {v for v in X[col].tolist() if v not in vocab}
            if unknown:
                raise ValueError(
                    f"Unknown categorical value(s) {sorted(map(str, unknown))} for "
                    f"'{col}'. Trained categories: {list(categories)}"
                )

            # TODO: Add range validation using schema constraints
    
    @property
    def feature_names(self) -> list[str]:
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
