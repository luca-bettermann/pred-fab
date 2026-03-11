"""
Prediction System for managing prediction models.

Coordinates prediction model training and inference within the AIXD architecture.
Prediction models predict features, which can then be evaluated for performance.
Integrates with DataModule for normalization and batching.
"""

from typing import Dict, List, Optional, Type, Any, Tuple
import copy
import pandas as pd
import numpy as np
import pickle
from scipy.stats import gaussian_kde

from ..core import Dataset, ExperimentData, DataModule, DatasetSchema
from ..core.data_objects import DataDimension, DataArray
from ..interfaces.prediction import IPredictionModel
from ..interfaces.tuning import IResidualModel, MLPResidualModel
from ..utils import PfabLogger, Metrics, LocalData, SplitType
from ..utils.enum import BlockType
from .base_system import BaseOrchestrationSystem


class PredictionSystem(BaseOrchestrationSystem):
    """
    Orchestrates prediction model operations with DataModule integration.
    
    - Manages prediction model registry and feature-to-model mapping
    - Coordinates training with normalization and batching via DataModule
    - Handles feature prediction with automatic denormalization
    """
    
    def __init__(self, logger: PfabLogger, schema: DatasetSchema, local_data: LocalData, res_model: Optional[IResidualModel] = None):
        """Initialize prediction system."""
        super().__init__(logger)
        self.models: List[IPredictionModel] = []
        self.residual_model: IResidualModel = MLPResidualModel(logger) if res_model is None else res_model

        self.schema: DatasetSchema = schema
        self.local_data: LocalData = local_data
        self.datamodule: Optional[DataModule] = None  # Stored after training

        # KDE state for NatPN-light uncertainty estimation (set after training)
        self._kde: Optional[gaussian_kde] = None
        self._q_max: Optional[float] = None
        self._n_exp: int = 0
        self._kde_bandwidth: Optional[float] = None
        self._kde_active_mask: Optional[np.ndarray] = None  # Boolean mask of non-constant dims

    def get_system_input_parameters(self) -> List[str]:
        """Get the parameter codes of all model inputs."""
        return self._get_unique_values('input_parameters')

    def get_system_input_features(self) -> List[str]:
        """Get the feature codes of all model inputs."""
        return self._get_unique_values('input_features')
    
    def get_system_outputs(self) -> List[str]:
        """Get the model outputs."""
        return self._get_unique_values('outputs')
    
    def _get_unique_values(self, attr: str) -> List[str]:
        codes = []
        for model in self.models:
            for code in getattr(model, attr):
                if code in codes:
                    continue
                codes.append(code)
        return codes
    
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
        self.datamodule.fit_normalization(SplitType.TRAIN)
        
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

        # Fit KDE on latent representations of training configs (NatPN-light)
        self._fit_kde(datamodule)
    
    # === UNCERTAINTY ESTIMATION (NatPN-light) ===

    def _fit_kde(self, datamodule: DataModule) -> None:
        """Fit weighted KDE on latent representations of all unique training configs.

        Option B: one latent point per unique effective parameter configuration.
        Non-trajectory experiments contribute 1 point (weight = sqrt(total_rows)).
        Trajectory experiments contribute K points (weight_k = sqrt(segment_rows)).
        Bandwidth: Silverman's rule.
        """
        if not self.models:
            return

        latent_points: List[np.ndarray] = []
        weights: List[float] = []
        n_exp = 0

        for code in datamodule.get_split_codes(SplitType.TRAIN):
            exp = datamodule.dataset.get_experiment(code)
            n_exp += 1
            n_rows = exp.get_num_rows()

            if not exp.parameter_updates:
                # Non-trajectory: single config
                params = exp.get_effective_parameters_for_row(0)
                z = self._encode_params(params, datamodule)
                if z is not None:
                    latent_points.append(z)
                    weights.append(float(np.sqrt(max(n_rows, 1))))
            else:
                # Trajectory: one point per segment (initial + each update event)
                events = sorted(exp.parameter_updates, key=lambda e: exp._event_start_index(e))
                seg_start = 0
                for event in events:
                    seg_end = exp._event_start_index(event)
                    seg_rows = seg_end - seg_start
                    if seg_rows > 0:
                        params = exp.get_effective_parameters_for_row(seg_start)
                        z = self._encode_params(params, datamodule)
                        if z is not None:
                            latent_points.append(z)
                            weights.append(float(np.sqrt(max(seg_rows, 1))))
                    seg_start = seg_end
                # Last segment
                seg_rows = n_rows - seg_start
                if seg_rows > 0:
                    params = exp.get_effective_parameters_for_row(seg_start)
                    z = self._encode_params(params, datamodule)
                    if z is not None:
                        latent_points.append(z)
                        weights.append(float(np.sqrt(max(seg_rows, 1))))

        if len(latent_points) < 2:
            self.logger.info("Too few training configs for KDE — uncertainty defaults to 1.0.")
            return

        latent_array = np.array(latent_points)   # (n_configs, n_latent)
        weights_array = np.array(weights)
        weights_array = weights_array / weights_array.sum()  # normalize

        # Drop constant dimensions to avoid a singular covariance matrix.
        # Dimensions where all training configs have the same latent value carry no
        # discriminative information and would cause gaussian_kde to fail.
        per_dim_std = np.std(latent_array, axis=0)
        active_mask = per_dim_std > 1e-8
        if not np.any(active_mask):
            self.logger.info("All latent dimensions are constant across training configs — uncertainty defaults to 1.0.")
            return

        projected = latent_array[:, active_mask]  # (n_samples, n_active_dims)
        n_samples, n_active_dims = projected.shape
        if n_samples <= n_active_dims:
            self.logger.info(
                f"Too few training configs ({n_samples}) for {n_active_dims}D KDE — uncertainty defaults to 1.0."
            )
            return

        try:
            # gaussian_kde expects (n_dims, n_samples)
            self._kde = gaussian_kde(projected.T, bw_method='silverman', weights=weights_array)
            self._kde_active_mask = active_mask
            # Scalar bandwidth: Silverman factor * mean std across active latent dimensions
            self._kde_bandwidth = float(self._kde.factor * np.mean(per_dim_std[active_mask] + 1e-8))
            # q_max for normalization: max KDE density over all training points
            densities = self._kde(projected.T)
            self._q_max = float(np.max(densities)) if len(densities) > 0 else 1.0
            self._n_exp = n_exp
            self.logger.info(
                f"KDE fitted on {len(latent_points)} latent configs from {n_exp} experiments "
                f"({n_active_dims}/{latent_array.shape[1]} active dims, "
                f"h={self._kde_bandwidth:.4f}, q_max={self._q_max:.6f})."
            )
        except Exception as e:
            self.logger.warning(f"KDE fitting failed: {e}. Uncertainty defaults to 1.0.")
            self._kde = None

    def _encode_params(self, params: Dict[str, Any], datamodule: DataModule) -> Optional[np.ndarray]:
        """Encode a params dict to latent representation via the first PM's encode()."""
        try:
            X_norm = datamodule.params_to_array(params)
            return self._encode_from_norm_array(X_norm)
        except Exception:
            return None

    def _encode_from_norm_array(self, X_norm: np.ndarray) -> np.ndarray:
        """Encode a 1-D normalized parameter array to latent space via first PM's encode()."""
        if not self.models or self.datamodule is None:
            return X_norm
        model = self.models[0]
        input_cols = model.input_parameters + model.input_features
        input_indices = [
            self.datamodule.input_columns.index(f)
            for f in input_cols
            if f in self.datamodule.input_columns
        ]
        if not input_indices:
            return X_norm
        X_model = X_norm[input_indices].reshape(1, -1)
        return model.encode(X_model)[0]

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode a batch of normalized parameter arrays to latent space.

        Uses the first registered PM's encode() method.  Falls back to identity
        if no models are registered or the system is not yet trained.

        Args:
            X: Normalized parameter array (batch_size, n_inputs)

        Returns:
            Latent array (batch_size, n_latent)
        """
        if not self.models or self.datamodule is None:
            return X
        model = self.models[0]
        input_cols = model.input_parameters + model.input_features
        input_indices = [
            self.datamodule.input_columns.index(f)
            for f in input_cols
            if f in self.datamodule.input_columns
        ]
        if not input_indices:
            return X
        X_model = X[:, input_indices] if X.ndim > 1 else X[input_indices].reshape(1, -1)
        return model.encode(X_model)

    def uncertainty(self, X: np.ndarray) -> float:
        """Compute epistemic uncertainty at a normalized parameter vector.

        Returns a value in [0, 1]:
            u = 1 / (1 + n_post)
        where n_post = N_exp * q_KDE(z) / q_max  (NatPN evidence posterior).

        Returns 1.0 (maximum uncertainty) before KDE is fitted.

        Args:
            X: Normalized parameter array of shape (1, n_inputs) or (n_inputs,)
        """
        if self._kde is None or self._q_max is None or self._q_max <= 0:
            return 1.0
        z = self._encode_from_norm_array(X.reshape(-1))
        if self._kde_active_mask is not None:
            z = z[self._kde_active_mask]
        q = float(self._kde(z.reshape(-1, 1))[0])
        n_post = self._n_exp * q / self._q_max
        return float(1.0 / (1.0 + n_post))

    def kernel_similarity(self, X1: np.ndarray, X2: np.ndarray) -> float:
        """Gaussian kernel similarity between two parameter vectors in latent space.

        sim(X1, X2) = exp(-||z1 - z2||^2 / h^2)

        Returns 0.0 if KDE has not been fitted yet (no bandwidth available).

        Args:
            X1, X2: Normalized parameter arrays of shape (n_inputs,) or (1, n_inputs)
        """
        if self._kde_bandwidth is None or self._kde_bandwidth < 1e-10:
            return 0.0
        z1 = self._encode_from_norm_array(X1.reshape(-1))
        z2 = self._encode_from_norm_array(X2.reshape(-1))
        if self._kde_active_mask is not None:
            z1 = z1[self._kde_active_mask]
            z2 = z2[self._kde_active_mask]
        h = self._kde_bandwidth
        return float(np.exp(-float(np.sum((z1 - z2) ** 2)) / (h ** 2)))

    def predict_for_calibration(self, params: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Any]:
        """Predict feature arrays for all dimensional positions for calibration use.

        Runs the full dimensional prediction for the given parameter configuration
        and converts each feature tensor to a tabular array suitable for evaluation
        models (rows = [dim_iter_vals..., feature_val]).

        Args:
            params: Raw (denormalized) parameter dict for the virtual experiment.

        Returns:
            Tuple of:
                - feature_arrays: Dict mapping feature code to 2-D array
                  where each row is [dim_iter_1, ..., feature_value].
                - params_block: A copy of the schema Parameters block with values
                  set from ``params``.

        Raises:
            RuntimeError: If the system has not been trained yet.
        """
        if self.datamodule is None:
            raise RuntimeError("PredictionSystem not trained. Call train() first.")

        dim_info = self._extract_dimensional_structure_from_params(params)
        shape = dim_info['shape']
        dim_iterators = dim_info['dim_iterators']

        # Full dimensional prediction
        predictions = self._initialize_prediction_dict(shape)
        self._execute_batched_predictions_to_dict(
            predictions=predictions,
            dim_info=dim_info,
            predict_from=0,
            predict_to=dim_info['total_positions'],
            batch_size=1000,
        )

        # Convert N-D tensors to tabular arrays: [dim_iter_vals..., feature_val]
        feature_arrays: Dict[str, np.ndarray] = {}
        for feat_name, tensor in predictions.items():
            flat = tensor.reshape(-1)
            rows = []
            for pos, feat_val in enumerate(flat):
                idx = np.unravel_index(pos, shape)
                rows.append(list(idx) + [float(feat_val)])
            feature_arrays[feat_name] = np.array(rows, dtype=np.float64)

        # Build Parameters block with values from params
        params_block = copy.deepcopy(self.schema.parameters)
        for code, val in params.items():
            if code in params_block.data_objects:
                try:
                    params_block.set_value(code, val)
                except Exception:
                    pass

        return feature_arrays, params_block

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
        if self.datamodule is None or not self.datamodule._is_fitted:
            raise RuntimeError(
                "PredictionSystem not trained yet. Call train() before tune(). "
                "Tuning requires existing normalization parameters from training."
            )
                        
        # Create a temporary Dataset with only the tuning experiment
        self.logger.info(f"Preparing tuning data from positions {start} to {end}...")
        temp_dataset = Dataset(schema=self.schema)
        temp_dataset.add_experiment(exp_data)

        # Create a temporary DataModule for tuning data.
        # Reuse fitted normalization from the main datamodule (no refit on tuning slice).
        batch_size = end - start if batch_size is None and end is not None else batch_size
        temp_datamodule = DataModule(dataset=temp_dataset, batch_size=batch_size)
        temp_datamodule.initialize(
            input_parameters=self.get_system_input_parameters(),
            input_features=self.get_system_input_features(),
            output_columns=self.get_system_outputs(),
        )
        temp_datamodule.set_split_codes(train_codes=[exp_data.code], val_codes=[], test_codes=[])

        # Copy fitted normalization state from offline training.
        temp_datamodule.set_normalization_state(self.datamodule.get_normalization_state())

        # Export full row-wise table and enforce requested online slice.
        X_df_all, y_df_all = temp_dataset.export_to_dataframe([exp_data.code])
        if X_df_all.empty or y_df_all.empty:
            raise ValueError("Tuning dataset has no feature rows available.")

        end_index = end if end is not None else len(X_df_all)
        if start < 0 or start >= len(X_df_all):
            raise ValueError(f"Tuning start index {start} out of bounds for {len(X_df_all)} rows.")
        if end_index <= start or end_index > len(X_df_all):
            raise ValueError(f"Tuning end index {end_index} invalid for start {start} and {len(X_df_all)} rows.")

        X_df = X_df_all.iloc[start:end_index].copy()
        y_df = y_df_all.iloc[start:end_index].copy()

        # Prepare tune arrays with training-fitted normalization.
        X_tune = temp_datamodule.prepare_input(X_df)
        y_tune = y_df[temp_datamodule.output_columns].values.astype(np.float32)
        temp_datamodule._normalize_batch(y_tune, temp_datamodule.output_columns, temp_datamodule._feature_stats)
        
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
            input_indices = [self.datamodule.input_columns.index(f) for f in model.input_parameters + model.input_features]
            indices = [self.datamodule.output_columns.index(f) for f in model.outputs]
            
            # Get ground truth (denormalized)
            y_true_norm = y_split[:, indices]
            y_true = self.datamodule.denormalize_values(y_true_norm, model.outputs)
            
            # Predict from the model-specific input slice.
            y_pred_norm = model.forward_pass(X_split[:, input_indices])
            y_pred = self.datamodule.denormalize_values(y_pred_norm, model.outputs)
            
            # Calculate metrics and store
            model_metrics = Metrics.calculate_regression_metrics(y_true, y_pred)
            results.update(model_metrics)
            
            self.logger.console_info(f"  Model ({', '.join(model.outputs)}):")
            self.logger.console_info(f"    {'Feature':<20} | {'MAE':<10} | {'RMSE':<10} | {'R²':<10}")
            self.logger.console_info(f"    {'-' * 60}")

            # Calculate and log metrics per feature
            for i, feature_name in enumerate(model.outputs):
                # Extract single feature vectors
                y_true_feat = y_true[:, i]
                y_pred_feat = y_pred[:, i]
                
                feat_metrics = Metrics.calculate_regression_metrics(y_true_feat, y_pred_feat)
                
                self.logger.console_info(
                    f"    {feature_name:<20} | "
                    f"{feat_metrics['mae']:<10.4f} | "
                    f"{feat_metrics['rmse']:<10.4f} | "
                    f"{feat_metrics['r2']:<10.4f}"
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
        # self._store_predictions_in_exp_data(exp_data, predictions)
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
    
    # def _store_predictions_in_exp_data(
    #     self, 
    #     exp_data: ExperimentData, 
    #     predictions: Dict[str, np.ndarray]
    # ) -> None:
    #     """Store prediction arrays in exp_data.predicted_metric_arrays."""
    #     for feature_name, pred_array in predictions.items():
    #         if feature_name not in exp_data.predicted_features.keys():
    #             arr = DataArray(code=feature_name, role=BlockType.FEATURE)
    #             exp_data.predicted_features.add(feature_name, arr)
    #         exp_data.predicted_features.set_value(feature_name, pred_array)
    
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
