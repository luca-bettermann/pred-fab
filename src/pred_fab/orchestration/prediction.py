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

from ..core import Dataset, ExperimentData, DataModule, DatasetSchema
from ..core.data_objects import DataDimension, DataArray
from ..interfaces.prediction import IPredictionModel
from ..interfaces.tuning import IResidualModel, MLPResidualModel
from ..utils import LBPLogger, Metrics, LocalData, SplitType
from .base_system import BaseOrchestrationSystem


class PredictionSystem(BaseOrchestrationSystem):
    """
    Orchestrates prediction model operations with DataModule integration.
    
    - Manages prediction model registry and feature-to-model mapping
    - Coordinates training with normalization and batching via DataModule
    - Handles feature prediction with automatic denormalization
    """
    
    def __init__(self, logger: LBPLogger, schema: DatasetSchema, local_data: LocalData, res_model: Optional[IResidualModel] = None):
        """Initialize prediction system."""
        super().__init__(logger)
        self.models: List[IPredictionModel] = []
        self.residual_model: IResidualModel = MLPResidualModel(logger) if res_model is None else res_model

        self.schema: DatasetSchema = schema
        self.local_data: LocalData = local_data
        self.datamodule: Optional[DataModule] = None  # Stored after training
    
    def _filter_batches_for_model(self, batches: List[Tuple[np.ndarray, np.ndarray]], model: IPredictionModel) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Filter batch targets to only include model outputs."""
        if self.datamodule is None:
            raise RuntimeError("DataModule not set")
        input_indices = [self.datamodule.input_columns.index(f) for f in model.input_parameters + model.input_features]
        output_indices = [self.datamodule.output_columns.index(f) for f in model.outputs]
        return [(X[:, input_indices], y[:, output_indices]) for X, y in batches]

    def train(self, datamodule: DataModule, **kwargs) -> None:
        """Train all prediction models using DataModule configuration."""
        # Store a copy to prevent mutation after training
        self.datamodule = datamodule
        
        # Check if training split is empty
        split_sizes = self.datamodule.get_split_sizes()
        if split_sizes['train'] == 0:
            raise ValueError(
                "Cannot train on empty training set. All data is in test/val splits. "
                "Reduce test_size and/or val_size in DataModule configuration."
            )
        
        self.logger.console_info("Starting prediction model training...")
        
        # Fit normalization
        self.logger.info("Fitting normalization on training data...")
        self.datamodule._fit_normalize(SplitType.TRAIN)
        
        # Get batches
        train_batches = self.datamodule.get_batches(SplitType.TRAIN)
        val_batches = self.datamodule.get_batches(SplitType.VAL)
        
        # Train each registered model
        trained_count = 0
        for model in self.models:
            # Filter batches for this model
            model_train_batches = self._filter_batches_for_model(train_batches, model)
            model_val_batches = self._filter_batches_for_model(val_batches, model)
            
            # Train model with user-provided kwargs
            self.logger.info(f"Training model for features {model.outputs}...")
            model.train(model_train_batches, model_val_batches, **kwargs)
            trained_count += 1
            
            primary_feature = model.outputs[0] if model.outputs else "unknown"
            self.logger.console_info(f"✓ Trained model for '{primary_feature}'")
        
        self.logger.console_success(
            f"Training complete: {trained_count}/{len(self.models)} models trained"
        )
    
    def tune(
            self, 
            exp_data: ExperimentData, 
            start: int, 
            end: Optional[int] = None,
            batch_size: Optional[int] = None,
            **kwargs
            ) -> DataModule:
        """
        Fine-tune models with new data (online learning mode).
        
        Uses existing normalization fit from training - does NOT refit normalization.
        Only normalizes the new tuning data and calls model.tuning() for adaptation.
        
        Args:
            exp_data: ExperimentData containing tuning data
            start: Start index of new data
            end: End index of new data
            **kwargs: Additional tuning parameters passed to model.tuning()
        
        Returns:
            Temporary DataModule used for tuning
        """
        if self.datamodule is None:
            raise RuntimeError(
                "PredictionSystem not trained yet. Call train() before tune(). "
                "Tuning requires existing normalization parameters from training."
            )
                        
        # Create a temporary Dataset with only the tuning experiment
        self.logger.info(f"Preparing tuning data from positions {start} to {end}...")
        temp_dataset = Dataset(
            schema=self.schema, 
            schema_id=self.schema.schema_id, # type: ignore
            local_data=self.local_data,
            logger=self.logger)
        temp_dataset.add_experiment(exp_data)

        # Create a temporary DataModule for tuning data
        batch_size = end - start if batch_size is None and end is not None else batch_size
        temp_datamodule = DataModule(
            dataset=temp_dataset,
            batch_size=batch_size,
            val_size=0.0,
            test_size=0.0,
        )
        
        # Copy normalization state
        temp_datamodule.set_normalization_state(self.datamodule.get_normalization_state())
        # Also copy column mappings to ensure consistency
        temp_datamodule.input_columns = self.datamodule.input_columns
        temp_datamodule.output_columns = self.datamodule.output_columns

        # Get tuning batches
        tune_batches = temp_datamodule.get_batches(SplitType.TRAIN)
        
        if not tune_batches:
            raise ValueError("Tuning datamodule has no data in 'train' split.")
        
        # Concatenate batches for residual learning
        X_list, y_list = zip(*tune_batches)
        X_tune = np.concatenate(X_list, axis=0)
        y_tune = np.concatenate(y_list, axis=0)
        
        # Initialize base predictions array
        y_pred_base = np.zeros_like(y_tune)
        
        # Get predictions from all base models
        self.logger.info("Generating base predictions for residual learning...")
        for model in self.models:
            # Get indices for this model's outputs
            output_indices = [self.datamodule.output_columns.index(f) for f in model.outputs]
            
            # Get model inputs (filter X columns)
            input_indices = [self.datamodule.input_columns.index(f) for f in model.input_parameters + model.input_features]
            X_model = X_tune[:, input_indices]
            
            # Predict (normalized)
            y_pred_model = model.forward_pass(X_model)
            
            # Place in aggregate array
            y_pred_base[:, output_indices] = y_pred_model

        # TODO: Store predicted features in exp_data.predicted_features. do we need another array?
            
        # Calculate residuals (Target - Base Prediction)
        residuals = y_tune - y_pred_base
        
        # Prepare inputs for residual model: [X, BasePredictions]
        # This allows the residual model to learn state-dependent errors (e.g. "high prediction -> high error")
        X_residual_input = np.hstack([X_tune, y_pred_base])
        
        # Train residual model
        self.logger.info(f"Training residual model on {len(X_tune)} samples...")
        self.residual_model.fit(X_residual_input, residuals)
        self.logger.console_success("✓ Residual model updated")

        return temp_datamodule
    
    def validate(self, use_test: bool = False) -> Dict[str, Dict[str, float]]:
        """Validate prediction models on validation or test set."""
        if self.datamodule is None:
            raise RuntimeError(
                "PredictionSystem not trained yet. Call train(datamodule) first."
            )
        
        split =  SplitType.TEST if use_test else SplitType.VAL
        
        # Check if split is empty before trying to extract
        split_sizes = self.datamodule.get_split_sizes()
        if split_sizes[split] == 0:
            raise ValueError(
                f"Cannot validate on {split} set: split is empty. "
                f"Configure DataModule with {'test_size' if use_test else 'val_size'} > 0.0"
            )
        
        self.logger.console_info(f"Validating models on {split} set...")
        
        # Extract validation/test data
        batches = self.datamodule.get_batches(split)
        if not batches:
            self.logger.console_warning(f"No batches returned for {split} set during validation.")
            return {}
            
        # Concatenate batches
        X_list, y_list = zip(*batches)
        X_split = np.concatenate(X_list, axis=0)
        y_split = np.concatenate(y_list, axis=0)
        
        self.logger.info(f"Evaluating {len(self.models)} models on {len(X_split)} samples...")
        
        # Compute metrics for each model
        results = {}
        for model in self.models:
            # Get indices for this model
            indices = [self.datamodule.output_columns.index(f) for f in model.outputs]
            
            # Get ground truth (denormalized)
            y_true_norm = y_split[:, indices]
            y_true = self.datamodule.denormalize_values(y_true_norm, model.outputs)
            
            # Predict
            y_pred_norm = model.forward_pass(X_split)
            y_pred = self.datamodule.denormalize_values(y_pred_norm, model.outputs)
            
            # Calculate metrics and store
            model_metrics = Metrics.calculate_regression_metrics(y_true, y_pred)
            results.update(model_metrics)
            
            self.logger.console_info(f"  Model ({', '.join(model.outputs)}):")
            
            # Calculate and log metrics per feature
            for i, feature_name in enumerate(model.outputs):
                # Extract single feature vectors
                y_true_feat = y_true[:, i]
                y_pred_feat = y_pred[:, i]
                
                feat_metrics = Metrics.calculate_regression_metrics(y_true_feat, y_pred_feat)
                
                self.logger.console_info(
                    f"    • {feature_name:<20}: "
                    f"MAE={feat_metrics['mae']:.4f}, "
                    f"RMSE={feat_metrics['rmse']:.4f}, "
                    f"R²={feat_metrics['r2']:.4f}"
                )
        
        self.logger.console_success(
            f"Validation complete on {split} set ({len(X_split)} experiments)"
        )
        
        return results

    def predict_experiment(
        self,
        exp_data: ExperimentData,
        predict_from: int = 0,
        predict_to: Optional[int] = None,
        batch_size: int = 1000,
        overlap: int = 0
    ) -> Dict[str, np.ndarray]:
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
        params = exp_data.parameters.get_values_dict()
        
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
        return predictions
    
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
    
    def _store_predictions_in_exp_data(
        self, 
        exp_data: ExperimentData, 
        predictions: Dict[str, np.ndarray]
    ) -> None:
        """Store prediction arrays in exp_data.predicted_metric_arrays."""
        for feature_name, pred_array in predictions.items():
            if feature_name not in exp_data.predicted_features.keys():
                arr = DataArray(code=feature_name)
                exp_data.predicted_features.add(feature_name, arr)
            exp_data.predicted_features.set_value(feature_name, pred_array)
    
    def _extract_dimensional_structure_from_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dimensional info (shape, params, positions) from schema and params dict."""
        # Use schema parameters to get dimensions sorted correctly by level
        sorted_dims = self.schema.parameters.get_sorted_dimensions()
        
        if not sorted_dims:
            raise ValueError("No dimensional parameters in schema - cannot predict dimensional features")
        
        dim_sizes = []
        dim_iterators = []
        dim_codes = set()
        
        for dim_obj in sorted_dims:
            name = dim_obj.code
            if name not in params:
                raise ValueError(f"Missing dimensional parameter in params: {name}")
            
            size = int(params[name])
            dim_sizes.append(size)
            dim_iterators.append(dim_obj.iterator_code)
            dim_codes.add(name)
        
        shape = tuple(dim_sizes)
        total_positions = int(np.prod(shape))
        
        # Extract non-dimensional parameters for feature matrix base
        param_base = {}
        for name in self.schema.parameters.keys():
            if name not in dim_codes and name in params:
                param_base[name] = params[name]
        
        return {
            'shape': shape,
            'dim_iterators': dim_iterators,
            'param_base': param_base,
            'total_positions': total_positions
        }
    
    def _initialize_prediction_dict(self, shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
        """Create prediction dictionary with NaN-initialized arrays for each feature."""
        predictions = {}
        
        # Collect all feature names from registered models
        for model in self.models:
            for feature_name in model.outputs:
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
        dim_iterators = dim_info['dim_iterators']
        
        X_batch_rows = []
        batch_indices = []
        
        for pos in range(batch_start, batch_end):
            # Convert linear position to multi-dimensional index
            idx = np.unravel_index(pos, shape)
            batch_indices.append(idx)
            
            # Create feature row: non-dimensional params + dimensional indices
            row = param_base.copy()
            for i, iterator_name in enumerate(dim_iterators):
                row[iterator_name] = idx[i]
            
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
        if self.datamodule is None:
             raise RuntimeError("DataModule not set")

        # Prepare input (one-hot + normalize)
        X_norm = self.datamodule.prepare_input(X_batch)
        
        for model in self.models:
            # Predict in normalized space
            y_pred_norm = model.forward_pass(X_norm)
            
            # Denormalize to original scale
            y_pred = self.datamodule.denormalize_values(y_pred_norm, model.outputs)
            
            # Store predictions in arrays
            for i, feature_name in enumerate(model.outputs):
                if feature_name not in predictions:
                    continue
                
                # y_pred is (batch, n_outputs)
                values = y_pred[:, i]
                
                for j, idx in enumerate(batch_indices):
                    predictions[feature_name][idx] = float(values[j])
    
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
        for model in self.models:
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
            if fresh_model.outputs != model.outputs:
                raise ValueError(
                    f"Round-trip validation failed: feature_names mismatch. "
                    f"Original: {model.outputs}, Restored: {fresh_model.outputs}"
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
                    'feature_names': model.outputs,
                    'artifacts': model._get_model_artifacts()
                }
                for model in self.models
            ],
            'normalization': self.datamodule.get_normalization_state(),
            'schema': self.schema.to_dict()
        }
        
        # InferenceBundle can work with predictions only (evaluation happens externally)
        if include_evaluation:
            self.logger.warning(
                "Evaluation model export requested but not yet implemented. "
                "Bundle will only contain prediction models."
            )
        
        return bundle




