"""
Prediction System for managing prediction models.

Coordinates prediction model training and inference within the AIXD architecture.
Prediction models predict features, which can then be evaluated for performance.
Integrates with DataModule for normalization and batching.
"""

from typing import Dict, List, Optional, Type, Any
import pandas as pd
import numpy as np
import pickle

from ..core.dataset import Dataset
from ..core.datamodule import DataModule
from ..interfaces.prediction import IPredictionModel
from ..utils.logger import LBPLogger


class PredictionSystem:
    """
    Orchestrates prediction model operations with DataModule integration.
    
    - Manages prediction model registry and feature-to-model mapping
    - Coordinates training with normalization and batching via DataModule
    - Handles feature prediction with automatic denormalization
    """
    
    def __init__(self, dataset: Dataset, logger: LBPLogger):
        """Initialize prediction system."""
        self.dataset = dataset
        self.logger = logger
        self.prediction_models: List[IPredictionModel] = []  # All registered models
        self.feature_to_model: Dict[str, IPredictionModel] = {}  # feature_name -> model
        self.datamodule: Optional[DataModule] = None  # Stored after training
    
    def _get_model_name(self, model: IPredictionModel) -> str:
        """Get primary feature name from model for logging."""
        return model.feature_names[0] if model.feature_names else "unknown"
    
    def add_prediction_model(self, model: IPredictionModel) -> None:
        """Register a prediction model and create feature-to-model mappings."""
        # Add to model list
        self.prediction_models.append(model)
        
        # Create feature-to-model mappings
        for feature_name in model.feature_names:
            if feature_name in self.feature_to_model:
                self.logger.warning(
                    f"Feature '{feature_name}' already registered to another model. "
                    f"Overwriting with new model."
                )
            self.feature_to_model[feature_name] = model
        
        primary_feature = self._get_model_name(model)
        self.logger.info(f"Added prediction model for features: {model.feature_names} (primary: {primary_feature})")
    
    def train(self, datamodule: DataModule, **kwargs) -> None:
        """Train all prediction models using DataModule configuration."""
        # Store a copy to prevent mutation after training
        self.datamodule = datamodule.copy()
        
        # Check if training split is empty
        split_sizes = self.datamodule.get_split_sizes()
        if split_sizes['train'] == 0:
            raise ValueError(
                "Cannot train on empty training set. All data is in test/val splits. "
                "Reduce test_size and/or val_size in DataModule configuration."
            )
        
        self.logger.console_info("Starting prediction model training...")
        
        # Extract training split and fit normalization
        self.logger.info("Extracting training data from dataset...")
        X_train, y_train = self.datamodule.extract_all(split='train')
        
        self.logger.info(f"Fitting normalization on {len(X_train)} training experiments...")
        self.datamodule.fit_normalize(y_train)
        
        # Train each registered model
        trained_count = 0
        for model in self.prediction_models:
            # Get feature column(s) for this model
            feature_cols = model.feature_names
            missing_cols = set(feature_cols) - set(y_train.columns)
            if missing_cols:
                primary_feature = self._get_model_name(model)
                self.logger.warning(
                    f"Missing features for model '{primary_feature}': {missing_cols}, skipping"
                )
                continue
            
            # Extract and normalize features for this model
            y_feature = y_train[feature_cols]
            y_norm = self.datamodule.normalize(y_feature)  # type: ignore
            
            # Train model with user-provided kwargs
            primary_feature = feature_cols[0] if feature_cols else "unknown"
            self.logger.info(f"Training model for '{primary_feature}' on {len(X_train)} experiments...")
            model.train(X_train, y_norm, **kwargs)
            trained_count += 1
            
            self.logger.console_info(f"✓ Trained model for '{primary_feature}'")
        
        self.logger.console_success(
            f"Training complete: {trained_count}/{len(self.prediction_models)} models trained"
        )
    
    def validate(self, use_test: bool = False) -> Dict[str, Dict[str, float]]:
        """Validate prediction models on validation or test set."""
        if self.datamodule is None:
            raise RuntimeError(
                "PredictionSystem not trained yet. Call train(datamodule) first."
            )
        
        split = 'test' if use_test else 'val'
        
        # Check if split is empty before trying to extract
        split_sizes = self.datamodule.get_split_sizes()
        if split_sizes[split] == 0:
            raise ValueError(
                f"Cannot validate on {split} set: split is empty. "
                f"Configure DataModule with {'test_size' if use_test else 'val_size'} > 0.0"
            )
        
        self.logger.console_info(f"Validating models on {split} set...")
        
        # Extract validation/test data
        X_split, y_split = self.datamodule.extract_all(split=split)
        
        self.logger.info(f"Evaluating {len(self.prediction_models)} models on {len(X_split)} experiments...")
        
        # Compute metrics for each model
        results = {}
        for model in self.prediction_models:
            # Get primary feature name
            primary_feature = model.feature_names[0] if model.feature_names else "unknown"
            
            # Get feature columns
            feature_cols = model.feature_names
            missing_cols = set(feature_cols) - set(y_split.columns)
            if missing_cols:
                self.logger.warning(
                    f"Missing features for model '{primary_feature}': {missing_cols}, skipping"
                )
                continue
            
            # Get ground truth
            y_true = y_split[feature_cols]
            
            # Predict
            y_pred_norm = model.forward_pass(X_split)
            y_pred = self.datamodule.denormalize(y_pred_norm)
            
            # Compute metrics for primary feature
            y_true_vals = y_true[primary_feature].values
            y_pred_vals = y_pred[primary_feature].values
            
            mae = float(np.mean(np.abs(y_true_vals - y_pred_vals)))
            rmse = float(np.sqrt(np.mean((y_true_vals - y_pred_vals) ** 2)))
            
            # R² score
            ss_res = np.sum((y_true_vals - y_pred_vals) ** 2)
            ss_tot = np.sum((y_true_vals - np.mean(y_true_vals)) ** 2)
            r2 = float(1 - (ss_res / (ss_tot + 1e-8)))
            
            results[primary_feature] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'n_samples': len(y_true_vals)
            }
            
            self.logger.console_info(
                f"  {primary_feature}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}"
            )
        
        self.logger.console_success(
            f"Validation complete on {split} set ({len(X_split)} experiments)"
        )
        
        return results
    
    def predict(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """Predict all features for new parameter values."""
        if self.datamodule is None:
            raise RuntimeError(
                "PredictionSystem not trained yet. Call train(datamodule) first."
            )
        
        # Validate X_new columns against dataset schema
        self._validate_input_parameters(X_new)
        
        # Predict all features
        predictions = {}
        for model in self.prediction_models:
            primary_feature = self._get_model_name(model)
            
            # Predict (model expects normalized features if trained with normalization)
            # Note: X (parameters) are not normalized, only y (features)
            y_pred_norm = model.forward_pass(X_new)
            
            # Validate return type
            if not isinstance(y_pred_norm, pd.DataFrame):
                raise TypeError(
                    f"Model '{primary_feature}' forward_pass() must return pd.DataFrame, "
                    f"got {type(y_pred_norm).__name__}"
                )
            
            # Validate required columns are present
            missing_cols = set(model.feature_names) - set(y_pred_norm.columns)
            if missing_cols:
                raise ValueError(
                    f"Model '{primary_feature}' forward_pass() missing columns: {missing_cols}. "
                    f"Expected columns: {model.feature_names}"
                )
            
            # Denormalize predictions
            y_pred = self.datamodule.denormalize(y_pred_norm)
            
            # Extract columns for this feature
            for col in model.feature_names:
                if col in y_pred.columns:
                    predictions[col] = y_pred[col]
        
        return pd.DataFrame(predictions)
    
    def _validate_input_parameters(self, X: pd.DataFrame) -> None:
        """Validate that X has correct parameter columns."""
        # Get expected parameters from dataset schema
        expected_params = set(self.dataset.schema.parameters.keys())
        provided_params = set(X.columns)
        
        # Check for unexpected columns
        unexpected = provided_params - expected_params
        if unexpected:
            raise ValueError(
                f"Unexpected parameter columns in X_new: {unexpected}. "
                f"Expected parameters: {expected_params}"
            )
        
        # Check for missing required columns
        # Note: Not all parameters may be required (some might have defaults)
        # So we only warn, not error
        missing = expected_params - provided_params
        if missing:
            self.logger.warning(
                f"Missing parameter columns in X_new: {missing}. "
                f"Models may fail if these are required."
            )
    
    # === EXPORT FOR PRODUCTION INFERENCE ===
    
    def export_inference_bundle(
        self, 
        filepath: str, 
        include_evaluation: bool = False
    ) -> str:
        """Export validated inference bundle with models, normalization, and schema."""
        if self.datamodule is None:
            raise RuntimeError("Cannot export before training. Call train() first.")
        
        self.logger.console_info("Validating models for export...")
        
        # Validate prediction models
        for model in self.prediction_models:
            self._validate_model_export(model)
        
        self.logger.console_info("✓ All models validated successfully")
        
        # Build bundle dict
        bundle = self._create_bundle_dict(include_evaluation)
        
        # Save bundle
        with open(filepath, 'wb') as f:
            pickle.dump(bundle, f)
        
        self.logger.console_success(f"✓ Exported inference bundle to: {filepath}")
        return filepath
    
    def _validate_model_export(self, model: IPredictionModel) -> None:
        """Round-trip test: export artifacts and restore to verify export support."""
        try:
            # Get artifacts
            artifacts = model._get_model_artifacts()
            
            if not isinstance(artifacts, dict):
                raise TypeError(
                    f"_get_model_artifacts() must return dict, got {type(artifacts).__name__}"
                )
            
            # Verify artifacts are picklable
            try:
                pickle.dumps(artifacts)
            except Exception as e:
                raise RuntimeError(
                    f"Artifacts not picklable: {e}. Ensure all objects in artifacts dict are picklable."
                ) from e
            
            # Create fresh instance and restore
            ModelClass = model.__class__
            fresh_model = ModelClass(logger=None)  # type: ignore[arg-type]
            fresh_model._set_model_artifacts(artifacts)
            
            # Verify feature_names match
            if fresh_model.feature_names != model.feature_names:
                raise ValueError(
                    f"Round-trip validation failed: feature_names mismatch. "
                    f"Original: {model.feature_names}, Restored: {fresh_model.feature_names}"
                )
            
            self.logger.info(f"✓ Validated {ModelClass.__name__}")
            
        except NotImplementedError as e:
            raise RuntimeError(
                f"{model.__class__.__name__} does not implement "
                f"_get_model_artifacts()/_set_model_artifacts(). "
                f"These methods are required for export."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Export validation failed for {model.__class__.__name__}: {e}"
            ) from e
    
    def _create_bundle_dict(self, include_evaluation: bool) -> Dict[str, Any]:
        """Assemble bundle with models, artifacts, normalization, and schema."""
        if self.datamodule is None:
            raise RuntimeError("DataModule not initialized")
        
        bundle: Dict[str, Any] = {
            'prediction_models': [
                {
                    'class_path': f"{model.__class__.__module__}.{model.__class__.__name__}",
                    'feature_names': model.feature_names,
                    'artifacts': model._get_model_artifacts()
                }
                for model in self.prediction_models
            ],
            'normalization': self.datamodule.get_normalization_state(),
            'schema': self.dataset.schema.to_dict()
        }
        
        # NOTE: Evaluation models optional - only included if requested and available
        # InferenceBundle can work with predictions only (evaluation happens externally)
        if include_evaluation:
            self.logger.warning(
                "Evaluation model export requested but not yet implemented. "
                "Bundle will only contain prediction models."
            )
        
        return bundle




