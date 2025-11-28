"""
Prediction System for managing prediction models.

Coordinates prediction model training and inference within the AIXD architecture.
Prediction models predict features, which can then be evaluated for performance.
Integrates with DataModule for normalization and batching.
"""

from typing import Dict, List, Optional, Type, Any, Tuple
import pandas as pd
import numpy as np
import pickle

from ..core.dataset import Dataset, ExperimentData
from ..core.datamodule import DataModule
from ..core.data_blocks import MetricArrays
from ..core.data_objects import DataDimension, DataArray
from ..interfaces.prediction import IPredictionModel
from ..utils.logger import LBPLogger
from ..utils.metrics import Metrics
from .base import BaseOrchestrationSystem


class PredictionSystem(BaseOrchestrationSystem):
    """
    Orchestrates prediction model operations with DataModule integration.
    
    - Manages prediction model registry and feature-to-model mapping
    - Coordinates training with normalization and batching via DataModule
    - Handles feature prediction with automatic denormalization
    """
    
    def __init__(self, dataset: Dataset, logger: LBPLogger):
        """Initialize prediction system."""
        super().__init__(dataset, logger)
        self.prediction_models: List[IPredictionModel] = []  # All registered models
        self.feature_to_model: Dict[str, IPredictionModel] = {}  # feature_name -> model
        self.datamodule: Optional[DataModule] = None  # Stored after training
    
    def _get_model_name(self, model: IPredictionModel) -> str:
        """Get primary feature name from model for logging."""
        return model.predicted_features[0] if model.predicted_features else "unknown"
    
    def get_models(self) -> List[IPredictionModel]:
        """Return registered prediction models."""
        return self.prediction_models
    
    def get_model_specs(self) -> Dict[str, Any]:
        """Extract input/output DataObject specifications from registered prediction models."""
        # Get base specs (inputs)
        specs = super().get_model_specs()
        
        # Add predicted features
        specs["predicted_features"] = set()
        for pred_model in self.prediction_models:
            specs["predicted_features"].update(pred_model.predicted_features)
        
        return specs
    
    def add_prediction_model(self, model: IPredictionModel) -> None:
        """Register a prediction model and create feature-to-model mappings."""
        # Add to model list
        self.prediction_models.append(model)
        
        # Create feature-to-model mappings
        for feature_name in model.predicted_features:
            if feature_name in self.feature_to_model:
                self.logger.warning(
                    f"Feature '{feature_name}' already registered to another model. "
                    f"Overwriting with new model."
                )
            self.feature_to_model[feature_name] = model
        
        primary_feature = self._get_model_name(model)
        self.logger.info(f"Added prediction model for features: {model.predicted_features} (primary: {primary_feature})")
    
    def _prepare_model_data(self, model: IPredictionModel, y_norm: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Prepare normalized feature data for a specific model."""
        feature_cols = model.predicted_features
        primary_feature = self._get_model_name(model)
        
        missing_cols = set(feature_cols) - set(y_norm.columns)
        if missing_cols:
            raise ValueError(
                f"Model requires features {missing_cols} not found in data. "
                f"Available features: {list(y_norm.columns)}. "
                f"Ensure data includes all required features for '{primary_feature}'."
            )
        
        # Extract normalized features for this model
        y_model_norm = y_norm[feature_cols]
        if isinstance(y_model_norm, pd.Series):
            y_model_norm = y_model_norm.to_frame()
            
        return y_model_norm, primary_feature

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
        X_train, y_train = self.datamodule.get_split('train')
        
        self.logger.info(f"Fitting normalization on {len(X_train)} training samples...")
        self.datamodule.fit_normalize(X_train, y_train)
        
        # Normalize all data once
        X_train_norm = self.datamodule.normalize_parameters(X_train)  # type: ignore
        y_train_norm = self.datamodule.normalize_features(y_train)  # type: ignore
        
        # Train each registered model
        trained_count = 0
        for model in self.prediction_models:
            y_model_norm, primary_feature = self._prepare_model_data(model, y_train_norm)
            
            # Train model with user-provided kwargs
            self.logger.info(f"Training model for '{primary_feature}' on {len(X_train)} experiments...")
            model.train(X_train_norm, y_model_norm, **kwargs)
            trained_count += 1
            
            self.logger.console_info(f"✓ Trained model for '{primary_feature}'")
        
        self.logger.console_success(
            f"Training complete: {trained_count}/{len(self.prediction_models)} models trained"
        )
    
    def tune(self, datamodule: DataModule, **kwargs) -> None:
        """
        Fine-tune models with new data (online learning mode).
        
        Uses existing normalization fit from training - does NOT refit normalization.
        Only normalizes the new tuning data and calls model.tuning() for adaptation.
        
        Args:
            datamodule: DataModule with tuning data
            **kwargs: Additional tuning parameters passed to model.tuning()
        """
        if self.datamodule is None:
            raise RuntimeError(
                "PredictionSystem not trained yet. Call train() before tune(). "
                "Tuning requires existing normalization parameters from training."
            )
        
        self.logger.console_info("Starting online model tuning...")
        
        # Extract tuning data (use 'train' split from datamodule)
        self.logger.info("Extracting tuning data from datamodule...")
        X_tune, y_tune = datamodule.get_split('train')
        
        if len(X_tune) == 0:
            raise ValueError("Tuning datamodule has no data in 'train' split.")
        
        # Normalize using EXISTING normalization fit (no refit)
        self.logger.info(f"Normalizing {len(X_tune)} tuning samples with existing normalization...")
        X_tune_norm = self.datamodule.normalize_parameters(X_tune)  # type: ignore
        y_tune_norm = self.datamodule.normalize_features(y_tune)  # type: ignore
        
        # Tune each registered model
        tuned_count = 0
        skipped_count = 0
        for model in self.prediction_models:
            y_model_norm, primary_feature = self._prepare_model_data(model, y_tune_norm)
            
            # Try to tune model
            try:
                self.logger.info(f"Tuning model for '{primary_feature}' with {len(X_tune)} samples...")
                model.tuning(X_tune_norm, y_model_norm, **kwargs)
                tuned_count += 1
                self.logger.console_info(f"✓ Tuned model for '{primary_feature}'")
            except NotImplementedError:
                # Model doesn't implement tuning
                self.logger.info(f"Model '{primary_feature}' does not support tuning, skipping...")
                skipped_count += 1
                self.logger.console_info(f"⊘ Skipped '{primary_feature}' (tuning not implemented)")
        
        if tuned_count > 0:
            self.logger.console_success(
                f"Tuning complete: {tuned_count}/{len(self.prediction_models)} models tuned"
            )
        else:
            self.logger.console_info(
                f"Tuning called but no models implement tuning(). "
                f"Models skipped: {skipped_count}/{len(self.prediction_models)}"
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
        X_split, y_split = self.datamodule.get_split(split)
        
        self.logger.info(f"Evaluating {len(self.prediction_models)} models on {len(X_split)} samples...")
        
        # Compute metrics for each model
        results = {}
        for model in self.prediction_models:
            # Get primary feature name
            primary_feature = model.predicted_features[0] if model.predicted_features else "unknown"
            
            # Get feature columns
            feature_cols = model.predicted_features
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
            y_pred = self.datamodule.denormalize_features(y_pred_norm)
            
            # Compute metrics for primary feature
            y_true_vals = y_true[primary_feature].values
            y_pred_vals = y_pred[primary_feature].values
            
            model_metrics = Metrics.calculate_regression_metrics(y_true_vals, y_pred_vals)
            
            results[primary_feature] = model_metrics
            
            self.logger.console_info(
                f"  {primary_feature}: MAE={model_metrics['mae']:.4f}, "
                f"RMSE={model_metrics['rmse']:.4f}, R²={model_metrics['r2']:.4f}"
            )
        
        self.logger.console_success(
            f"Validation complete on {split} set ({len(X_split)} experiments)"
        )
        
        return results
    
    def _predict_from_params(
        self,
        params: Dict[str, Any],
        predict_from: int = 0,
        predict_to: Optional[int] = None,
        batch_size: int = 1000,
        overlap: int = 0
    ) -> Dict[str, np.ndarray]:
        """Core prediction logic from raw parameters with shape determined by dimensional params."""
        if self.datamodule is None:
            raise RuntimeError("PredictionSystem not trained. Call train() first.")
        
        # Extract dimensional structure from params
        dim_info = self._extract_dimensional_structure_from_params(params)
        
        # Validate prediction range
        total_positions = dim_info['total_positions']
        if predict_from < 0 or predict_from > total_positions:
            raise ValueError(f"predict_from {predict_from} out of range [0, {total_positions}]")
        if predict_to is None:
            predict_to = total_positions
        elif predict_to < 0 or predict_to > total_positions:
            raise ValueError(f"predict_to {predict_to} out of range [0, {total_positions}]")
        
        # Initialize prediction arrays
        predictions = self._initialize_prediction_dict(dim_info['shape'])
        
        # Execute batched predictions
        self._execute_batched_predictions_to_dict(
            predictions=predictions,
            dim_info=dim_info,
            predict_from=predict_from,
            predict_to=predict_to, # type: ignore
            batch_size=batch_size,
            overlap=overlap
        )
        
        self.logger.info(f"✓ Predicted {predict_to - predict_from} positions") # type: ignore
        return predictions
    
    def predict_experiment(
        self,
        exp_data: ExperimentData,
        predict_from: int = 0,
        predict_to: Optional[int] = None,
        batch_size: int = 1000,
        overlap: int = 0
    ) -> ExperimentData:
        """
        Predict dimensional features and populate exp_data.predicted_metric_arrays.
        
        Args:
            exp_data: ExperimentData with parameters set (predicted_metric_arrays populated)
            predict_from: Start position index for prediction (default: 0)
            predict_to: End position index for prediction (default: None = predict all)
            batch_size: Number of dimensional positions per batch (default: 1000)
            overlap: Number of positions to overlap between consecutive batches (default: 0).
                     Useful for context-aware models (e.g., transformers) that need continuity
                     across batch boundaries. Must be < batch_size.
        """
        # Extract parameters from exp_data
        params = self._extract_params_from_exp_data(exp_data)
        
        # Get predictions from core logic
        predictions = self._predict_from_params(
            params=params,
            predict_from=predict_from,
            predict_to=predict_to,
            batch_size=batch_size,
            overlap=overlap
        )
        
        # Store predictions in exp_data.predicted_metric_arrays
        self._store_predictions_in_exp_data(exp_data, predictions)
        
        return exp_data
    
    def _store_predictions_in_exp_data(
        self, 
        exp_data: ExperimentData, 
        predictions: Dict[str, np.ndarray]
    ) -> None:
        """Store prediction arrays in exp_data.predicted_metric_arrays."""
        for feature_name, pred_array in predictions.items():
            if feature_name not in exp_data.predicted_metric_arrays.keys():
                arr = DataArray(name=feature_name, shape=pred_array.shape)
                exp_data.predicted_metric_arrays.add(feature_name, arr)
            exp_data.predicted_metric_arrays.set_value(feature_name, pred_array)
    
    def _extract_dimensional_structure_from_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dimensional info (shape, params, positions) from schema and params dict."""
        # Get dimensional parameters from schema
        dim_params = {
            name: obj for name, obj in self.dataset.schema.parameters.data_objects.items()
            if isinstance(obj, DataDimension)
        }
        
        if not dim_params:
            raise ValueError("No dimensional parameters in schema - cannot predict dimensional features")
        
        # Calculate shape from params values
        dim_sizes = []
        dim_names = []
        for dim_name in sorted(dim_params.keys()):
            dim_obj = dim_params[dim_name]
            if dim_name not in params:
                raise ValueError(f"Missing dimensional parameter in params: {dim_name}")
            size = params[dim_name]
            dim_sizes.append(int(size))
            dim_names.append(dim_obj.dim_iterator_name)
        
        shape = tuple(dim_sizes)
        total_positions = int(np.prod(shape))
        
        # Extract non-dimensional parameters for feature matrix base
        param_base = {}
        for name in self.dataset.schema.parameters.keys():
            if name not in dim_params and name in params:
                param_base[name] = params[name]
        
        return {
            'dim_params': dim_params,
            'shape': shape,
            'dim_names': dim_names,
            'param_base': param_base,
            'total_positions': total_positions
        }
    
    def _initialize_prediction_dict(self, shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
        """Create prediction dictionary with NaN-initialized arrays for each feature."""
        predictions = {}
        
        # Collect all feature names from registered models
        for model in self.prediction_models:
            for feature_name in model.predicted_features:
                if feature_name not in predictions:
                    # Initialize with NaN (positions will be filled during prediction)
                    predictions[feature_name] = np.full(shape, np.nan)
        
        return predictions
    
    def _execute_batched_predictions_to_dict(
        self,
        predictions: Dict[str, np.ndarray],
        dim_info: Dict[str, Any],
        predict_from: int,
        predict_to: int,
        batch_size: int,
        overlap: int = 0
    ) -> None:
        """Process positions in batches: build X, predict, denormalize, store in prediction dict. Supports overlap."""
        self.logger.info(f"Predicting positions {predict_from} to {predict_to} in batches of {batch_size} (overlap={overlap})...")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if overlap >= batch_size and predict_to - predict_from > batch_size:
            raise ValueError("overlap must be less than batch_size for multiple batches")

        batch_starts = list(range(predict_from, predict_to, batch_size))
        for i, orig_batch_start in enumerate(batch_starts):
            if i == 0:
                batch_start = orig_batch_start
                batch_end = min(batch_start + batch_size, predict_to)
            else:
                batch_start = max(orig_batch_start - overlap, predict_from)
                batch_end = min(orig_batch_start + batch_size, predict_to)

            # Build feature matrix for this batch
            X_batch, batch_indices = self._build_batch_features(
                batch_start=batch_start,
                batch_end=batch_end,
                dim_info=dim_info
            )

            # Run predictions for all models
            self._predict_and_store_batch_to_dict(
                predictions=predictions,
                X_batch=X_batch,
                batch_indices=batch_indices
            )
    
    def _build_batch_features(
        self,
        batch_start: int,
        batch_end: int,
        dim_info: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, List[Tuple[int, ...]]]:
        """Build feature matrix X with params + dimensional indices for batch positions."""
        shape = dim_info['shape']
        param_base = dim_info['param_base']
        dim_params = dim_info['dim_params']
        
        X_batch_rows = []
        batch_indices = []
        
        for pos in range(batch_start, batch_end):
            # Convert linear position to multi-dimensional index
            idx = np.unravel_index(pos, shape)
            batch_indices.append(idx)
            
            # Create feature row: non-dimensional params + dimensional indices
            row = param_base.copy()
            for i, dim_name in enumerate(sorted(dim_params.keys())):
                dim_obj = dim_params[dim_name]
                row[dim_obj.dim_iterator_name] = idx[i]
            
            X_batch_rows.append(row)
        
        X_batch = pd.DataFrame(X_batch_rows)
        return X_batch, batch_indices
    
    def _predict_and_store_batch_to_dict(
        self,
        predictions: Dict[str, np.ndarray],
        X_batch: pd.DataFrame,
        batch_indices: List[Tuple[int, ...]]
    ) -> None:
        """Run all model predictions on X_batch, denormalize, and store in predictions dict."""
        for model in self.prediction_models:
            # Validate required features present
            missing_required = set(model.features_as_input) - set(X_batch.columns)
            if missing_required:
                raise ValueError(
                    f"Model requires evaluation features that are missing: {missing_required}. "
                    f"Please provide these features in the input data."
                )
            
            # Predict in normalized space
            y_pred_norm = model.forward_pass(X_batch)
            
            # Denormalize to original scale
            y_pred = self.datamodule.denormalize_features(y_pred_norm)  # type: ignore
            
            # Store predictions in arrays
            for feature_name in model.predicted_features:
                if feature_name not in y_pred.columns:
                    continue
                
                for i, idx in enumerate(batch_indices):
                    predictions[feature_name][idx] = float(y_pred[feature_name].iloc[i])
    
    def predict_uncertainty(self, params: Dict[str, Any], required_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Predict uncertainty (sigma) for parameters.
        Only works for models that implement predict_uncertainty().
        Returns dictionary of {feature_name: sigma_array_or_scalar}.
        
        Args:
            params: Parameter dictionary.
            required_features: Optional list of features to predict uncertainty for.
                               If None, predicts for all available features.
        """
        if self.datamodule is None:
            raise RuntimeError("PredictionSystem not trained. Call train() first.")
            
        dim_info = self._extract_dimensional_structure_from_params(params)
        X_batch, _ = self._build_batch_features(0, dim_info['total_positions'], dim_info)
        
        uncertainties = {}
        
        for model in self.prediction_models:
            # Optimization: Skip model if none of its features are required
            if required_features is not None:
                if not any(f in required_features for f in model.predicted_features):
                    continue

            if not hasattr(model, 'predict_uncertainty'):
                continue
                
            try:
                # Expecting model.predict_uncertainty(X) -> pd.DataFrame/dict
                sigma_norm = model.predict_uncertainty(X_batch) # type: ignore
            except Exception as e:
                self.logger.warning(f"Model {type(model).__name__} failed to predict uncertainty: {e}")
                continue
            
            # Denormalize uncertainty (scale only, no shift)
            for feature in model.predicted_features:
                # Skip if not required
                if required_features is not None and feature not in required_features:
                    continue

                if feature in sigma_norm:
                    val = sigma_norm[feature]
                    if hasattr(val, 'values'): val = val.values # Handle Series/DF
                    
                    stats = self.datamodule.normalization_state.get(feature)
                    if stats:
                        if stats['method'] == 'zscore':
                            val = val * stats['std']
                        elif stats['method'] == 'minmax':
                            val = val * (stats['max'] - stats['min'])
                    
                    uncertainties[feature] = val
                    
        return uncertainties

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
            if fresh_model.predicted_features != model.predicted_features:
                raise ValueError(
                    f"Round-trip validation failed: feature_names mismatch. "
                    f"Original: {model.predicted_features}, Restored: {fresh_model.predicted_features}"
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
                    'feature_names': model.predicted_features,
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




