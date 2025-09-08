from typing import Any, Dict, List, Tuple, Optional, Type, Set
import numpy as np

from lbp_package.interfaces.evaluation import IEvaluationModel

from ..interfaces import IExternalData, IPredictionModel
from ..utils import LocalDataInterface, LBPLogger


class PredictionSystem:
    """
    Orchestrates prediction models with consistent data preprocessing.
    
    Handles the complete prediction workflow:
    1. Data collection from structured (ExternalDataInterface) and unstructured (FeatureModel) sources
    2. Global preprocessing for consistency across models
    3. Model-specific preprocessing and training
    4. Prediction with denormalization
    
    Data Responsibility Boundary:
    - ExternalDataInterface: Provides structured study/experiment parameters via get_study_parameters(), get_exp_variables()
    - PredictionSystem: Coordinates parameter collection and delegates to models
    - FeatureModel: Loads domain-specific unstructured data via _load_data() and computes features
    - PredictionModel: Manages FeatureModel instances and implements ML prediction logic
    """
    
    def __init__(self, 
                 local_data: LocalDataInterface,
                 logger: LBPLogger) -> None:
        """
        Initialize prediction system.
        
        Args:
            local_data: Local file system navigation utility  
            logger: Logger instance
        """
        self.local_data = local_data
        self.logger = logger

        self.prediction_models: List[IPredictionModel] = []
        self.pred_model_by_code: Dict[str, IPredictionModel] = {}

        # Store normalization parameters for consistent denormalization
        self.normalization_params: Dict[str, Dict[str, float]] = {}
        
        # Store training metrics for summary display
        self.training_metrics: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("Initialized PredictionSystem")

    def add_prediction_model(self, performance_codes: List[str], prediction_model: Type[IPredictionModel], round_digits: int, **kwargs) -> None:
        """
        Add an prediction model to the system.
        
        Args:
            performance_codes: List of codes identifying the performance metrics
            prediction_model: Class of prediction model to instantiate
            round_digits: Number of digits to round results to
            **kwargs: Additional parameters for model initialization
        """
        # Validate if prediction_model is the correct type
        if not issubclass(prediction_model, IPredictionModel):
            raise TypeError(f"Expected a subclass of PredictionModel, got {type(prediction_model).__name__}")

        model_instance = prediction_model(
            performance_codes=performance_codes,
            logger=self.logger,
            round_digits=round_digits,
            **kwargs
        )
        self.prediction_models.append(model_instance)
        self.logger.info(f"Added model: {type(model_instance).__name__} for performance codes: {model_instance.output}")

    def get_required_performance_codes(self) -> List[str]:
        """Get all unique performance codes that need to be predicted across all models."""
        all_codes = set()
        for model in self.prediction_models:
            all_codes.update(model.output)
        return list(all_codes)

    def activate_prediction_model(self, code: str, study_params: Dict[str, Any], recompute: bool = False) -> None:
        if len(self.prediction_models) == 0:
            self.logger.warning("No prediction models available to activate.")
            return

        # Reset the performance code to prediction model mapping if necessary
        if len(self.pred_model_by_code) == 0 or recompute:
            self.pred_model_by_code = self._get_pred_model_by_code()

        # Log warning if no model is found for the given code
        if code not in self.pred_model_by_code:
            self.logger.warning(f"No prediction model for performance code '{code}' has been initialized.")

        # If a model is found, activate it
        prediction_model = self.pred_model_by_code[code]
        if not prediction_model.active:
            prediction_model.set_study_parameters(**study_params)
            prediction_model.active = True
            self.logger.info(f"Activated prediction model {type(prediction_model).__name__} for performance code '{code}' and set study parameters.")
        else:
            self.logger.info(f"Prediction model {type(prediction_model).__name__} for performance code '{code}' is already active.")

    def train(self, parameters: Dict[str, Dict], avg_features: Dict[str, Any], dim_arrays: Dict[str, Any], feature_arrays: Dict[str, Any], visualize_flag: bool, debug_flag: bool) -> None:
        """
        Train all models with unified data processing.

        Args:
            study_dataset: Dictionary of experiments {exp_code: {param_name/perf_name: value, ...}}
        """

        # High-level approach:
        # For now, we are ignorring dim_arrays and feature_arrays.
        # As of yet, it is not possible to have multi-dimensional FeatureModels,
        # hence this functionality is not relevant right now.
        # Figure out how to deal with this once FeatureModels contain dimensions.

        # 1. Filter dataset to required columns
        required_input_keys, required_output_keys, _, _ = self._get_required_keys()

        # 2. Construct input, output and array data lists
        assert parameters.keys() == avg_features.keys() == feature_arrays.keys(), "Inconsistent data keys"
        exp_codes = list(parameters.keys())

        input_data, input_keys = self._filter_for_required_columns(parameters, required_input_keys)
        output_data, output_keys = self._filter_for_required_columns(avg_features, required_output_keys)
        # input_arrays, input_keys_arrays = self._filter_for_required_columns(dim_arrays, required_input_keys_arrays)
        # output_arrays, output_keys_arrays = self._filter_for_required_columns(feature_arrays, required_output_keys_arrays)

        # Check for missing keys
        # available_input_keys = list(set(input_keys + input_keys_arrays))
        # available_output_keys = list(set(output_keys + output_keys_arrays))
        missing_input_keys = [key for key in required_input_keys if key not in input_keys]
        missing_output_keys = [key for key in required_output_keys if key not in output_keys]
        if missing_output_keys:
            raise ValueError(f"Missing required output keys from data: {missing_output_keys}")
                
        # 3. Add feature input data
        exp_codes = list(parameters.keys())
        features = self._add_feature_data(missing_input_keys, exp_codes, visualize_flag, debug_flag)
        input_data = [inp + feat for inp, feat in zip(input_data, features)]

        # 4. Prepare and normalize inputs (training mode)
        input_data = self._normalize_data(input_data, required_input_keys, is_training=True)
        output_data = self._normalize_data(output_data, required_output_keys, is_training=True)
        # input_arrays = self._normalize_data(input_arrays, required_input_keys, is_training=True)
        # output_arrays = self._normalize_data(output_arrays, required_output_keys, is_training=True)
        
        # 5. Train each model
        for pred_model in self.prediction_models:
            if not pred_model.active:
                self.logger.warning(f"Skipping inactive model: {type(pred_model).__name__}")
                continue
                
            self.logger.info(f"Training model: {type(pred_model).__name__}")
            
            # Extract relevant features for this model
            model_input_indices = [required_input_keys.index(key) for key in pred_model.input if key in required_input_keys]
            model_output_indices = [required_output_keys.index(key) for key in pred_model.output if key in required_output_keys]
            
            X = input_data[:, model_input_indices] if model_input_indices else np.array([[]])
            y = output_data[:, model_output_indices] if model_output_indices else np.array([[]])

            # Train the PredictionModel and collect metrics
            training_metrics = pred_model.train(X, y)
            
            # Store training metrics for summary
            model_name = type(pred_model).__name__
            self.training_metrics[model_name] = {
                "output_targets": pred_model.output,
                "metrics": training_metrics
            }

        self.logger.info(f"PredictionSystem training completed")

    def predict(self, exp_params: Dict[str, Any], exp_code: str, visualize_flag: bool, debug_flag: bool) -> Dict[str, float]:
        """
        Get predictions for all performance metrics for a single upcoming experiment.
        
        Args:
            exp_params: Parameter values {param_name: value} - from DataInterface
            exp_code: Experiment code for loading domain-specific data
            visualize_flag: Whether to show visualizations during feature computation
            debug_flag: Whether to run in debug mode
            
        Returns:
            Dictionary {performance_code: predicted_value} for all performance codes
        """
        self.logger.debug(f"Predicting performance for experiment {exp_code}")

        # 1. Filter dataset to required columns (same as train)
        required_input_keys, _, _, _ = self._get_required_keys()

        # 2. Create single-experiment dataset and filter
        single_exp_data = {exp_code: dict(exp_params)}
        input_data, input_keys = self._filter_for_required_columns(single_exp_data, required_input_keys)

        # 3. Add feature input data (same as train)
        missing_input_keys = [key for key in required_input_keys if key not in input_keys]
        features = self._add_feature_data(missing_input_keys, [exp_code], visualize_flag, debug_flag)
        input_data = [input_data[0] + features[0]] if features and features[0] else input_data

        # 4. Normalize inputs (inference mode)
        input_data = self._normalize_data(input_data, required_input_keys, is_training=False)
        
        # 5. Predict with each model (same pattern as train)
        all_predictions = {}
        for pred_model in self.get_active_pred_models():
            # Extract relevant features for this model (same as train)
            model_input_indices = [required_input_keys.index(key) for key in pred_model.input if key in required_input_keys]
            X = input_data[:, model_input_indices] if model_input_indices else np.array([[]])
            
            # Get predictions and denormalize
            predictions = pred_model.predict(X)  # Shape: (1, n_targets)
            denormalized = self._denormalize_data(predictions, pred_model.output)
            
            # Store results
            for i, perf_code in enumerate(pred_model.output):
                all_predictions[perf_code] = round(float(denormalized[0, i]), pred_model.round_digits)
                
        return all_predictions

    def get_active_pred_models(self) -> List[IPredictionModel]:
        """
        Get all active prediction models.
        
        Returns:
            List of active prediction models
        """
        return [model for model in self.prediction_models if model.active]
    
    def training_step_summary(self) -> str:
        """Generate summary of training results."""
        if not self.training_metrics:
            return "No training metrics available."
            
        summary = f"\033[1m{'Model Name':<20} {'Targets':<10} {'Training Score':<15} {'Samples':<10}\033[0m"
        for model_name, model_data in self.training_metrics.items():
            metrics = model_data["metrics"]
            target_count = len(model_data["output_targets"])
            
            # Format training score
            score = metrics.get("training_score", "N/A")
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            
            # Format sample count
            samples = metrics.get("training_samples", "N/A")
            summary += f"\n{model_name:<20} {target_count:<10} {score_str:<15} {samples:<10}"
            
        return summary
    
    def _get_pred_model_by_code(self) -> Dict[str, IPredictionModel]:
        # Prep prediction system output dict
        pred_model_by_code = {}
        for pred_model in self.prediction_models:
            for code in pred_model.output:
                if code in pred_model_by_code:
                    raise ValueError(f"Performance code '{code}' is predicted by multiple prediction models. Please check the configuration.")
                pred_model_by_code[code] = pred_model
        return pred_model_by_code

    def _get_required_keys(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Filter dataset to only experiments containing all required columns."""
        input_keys = set()
        output_keys = set()
        input_keys_arrays = set()
        output_keys_arrays = set()

        for model in self.prediction_models:

            if model.dataset_type == model.DatasetType.AGGR_METRICS:
                input_keys.update(model.input)
                output_keys.update(model.output)

            elif model.dataset_type == model.DatasetType.METRIC_ARRAYS:
                input_keys_arrays.update(model.input)
                output_keys_arrays.update(model.output)

            else:
                raise ValueError(f"Unknown dataset type for model {type(model).__name__}")
        return list(input_keys), list(output_keys), list(input_keys_arrays), list(output_keys_arrays)

    def _add_feature_data(self, input_keys: List[str], exp_codes: List[str], visualize_flag: bool, debug_flag: bool) -> List[List[Any]]:
        """Add feature model data to experiments in-place."""
        # Prepare mapping of performance codes to feature models
        feature_models_by_code = {}
        for pred_model in self.get_active_pred_models():
            for code, feature_model in pred_model.feature_models.items():
                if code not in feature_models_by_code:
                    feature_models_by_code[code] = feature_model

        # Run required feature models for the given exp_codes
        features = []
        for exp_code in exp_codes:
            exp_features = []
            for code in input_keys:
                if code not in feature_models_by_code:
                    raise ValueError(f"Input key '{code}' required by active prediction models has no associated feature model.")
                
                feature_model = feature_models_by_code[code]
                self.logger.info(f"Computing feature '{code}' for experiment '{exp_code}' using {type(feature_model).__name__}")
                feature_model.run(code, exp_code, self.local_data.get_experiment_folder(exp_code), visualize_flag, debug_flag)

                # Add computed feature to experiment data
                if code in feature_model.features:
                    exp_features.append(feature_model.features[code])
                else:
                    raise ValueError(f"Feature model '{type(feature_model).__name__}' did not compute expected feature '{code}' for experiment '{exp_code}'")
            features.append(exp_features)
        return features

    def _normalize_data(self, data: List[List[Any]], keys: List[str], is_training: bool = True) -> np.ndarray:
        """Normalize data for model training or inference."""
        # data array with shape (n_samples, n_features)
        data_array = np.array(data)
        if data_array.size == 0:
            return data_array
        
        for col_idx, key in enumerate(keys):
            values_array = data_array[:, col_idx]
            
            if is_training:
                # Training mode: compute and store normalization parameters
                mean_val = np.mean(values_array)
                std_val = np.std(values_array)
                
                data_array[:, col_idx] = (values_array - mean_val)
                # Avoid division by zero
                if std_val > 0:  
                    data_array[:, col_idx] /= std_val
                self.normalization_params[key] = {
                    'mean': float(mean_val), 
                    'std': float(std_val)
                }
            else:
                # Inference mode: apply stored normalization parameters
                if key in self.normalization_params:
                    params = self.normalization_params[key]
                    if params['std'] > 0:
                        data_array[:, col_idx] = (values_array - params['mean'])
                        # Avoid division by zero
                        if params['std'] > 0:
                            data_array[:, col_idx] /= params['std']
                else:
                    # Column not seen during training, use as-is
                    self.logger.warning(f"Column '{key}' not seen during training, using raw values")
        
        if is_training:
            self.logger.info(f"Normalized {data_array.shape[1]} columns using standardization")
        return data_array

    def _denormalize_data(self, data: np.ndarray, column_codes: List[str]) -> np.ndarray:
        """Denormalize data back to original scale using stored normalization parameters."""
        denormalized = np.copy(data)
        
        for col_idx, code in enumerate(column_codes):
            if code in self.normalization_params:
                params = self.normalization_params[code]
                if params['std'] > 0:
                    denormalized[:, col_idx] = (data[:, col_idx] * params['std'] + params['mean'])
            else:
                self.logger.warning(f"Column '{code}' not seen during training, using raw values")
        
        return denormalized

    def _filter_for_required_columns(self, data: Dict[str, Dict[str, Any]], required_keys: List[str]) -> Tuple[List[List[Any]], List[str]]:
        """Filter data to only include values for required keys, maintaining order of exp_codes."""
        filtered_data = []
        filtered_keys = []

        for exp_data in data.values():
            filtered_values = []
            for key, value in exp_data.items():
                if key in required_keys:
                    filtered_values.append(value)
                    if key not in filtered_keys:
                        filtered_keys.append(key)
            
            filtered_data.append(filtered_values)
        return filtered_data, filtered_keys


