from typing import Any, Dict, List, Tuple, Optional, Type, Set
import numpy as np

from ..interfaces import ExternalDataInterface, PredictionModel
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

        self.prediction_models: List[PredictionModel] = []
        self.pred_model_by_code: Dict[str, PredictionModel] = {}

        # Store normalization parameters for consistent denormalization
        self.normalization_params: Dict[str, Dict[str, float]] = {}
        
        self.logger.info("Initialized PredictionSystem")

    def add_prediction_model(self, performance_codes: List[str], prediction_model: Type[PredictionModel], round_digits: int, **kwargs) -> None:
        """
        Add an prediction model to the system.
        
        Args:
            performance_codes: List of codes identifying the performance metrics
            prediction_model: Class of prediction model to instantiate
            round_digits: Number of digits to round results to
            **kwargs: Additional parameters for model initialization
        """
        # Validate if prediction_model is the correct type
        if not issubclass(prediction_model, PredictionModel):
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

    def train(self, parameters: Dict[str, Dict], avg_features: Dict[str, Any], feature_arrays: Dict[str, Any], visualize_flag: bool, debug_flag: bool) -> None:
        """
        Train all models with unified data processing.

        Args:
            study_dataset: Dictionary of experiments {exp_code: {param_name/perf_name: value, ...}}
        """
        # 1. Filter dataset to required columns
        input_keys, output_keys, array_keys = self._get_required_keys()
       
        # 2. Construct input, output and array data lists
        assert parameters.keys() == avg_features.keys() == feature_arrays.keys(), "Inconsistent data keys"
        input_data = []
        output_data = []
        arrays = []
        for exp_code in parameters.keys():
            input_data.append([v for k, v in parameters[exp_code].items() if k in input_keys])
            output_data.append([v for k, v in avg_features[exp_code].items() if k in output_keys])
            arrays.append([v for k, v in feature_arrays[exp_code].items() if k in array_keys])

        # 3. Add feature data
        missing_keys = [key for key in input_keys if key not in list(parameters.values())[0]]
        exp_codes = list(parameters.keys())
        features = self._add_feature_data(missing_keys, exp_codes, visualize_flag, debug_flag)
        input_data = [inp + feat for inp, feat in zip(input_data, features)]

        # test normalization and denormalization
        normalized_input = self._normalize_data(input_data, input_keys, is_training=True)
        denormalized_input = self._denormalize_data(normalized_input, input_keys)

        # 4. Prepare and normalize inputs (training mode)
        input_data = self._normalize_data(input_data, input_keys, is_training=True)
        output_data = self._normalize_data(output_data, output_keys, is_training=True)
        arrays = self._normalize_data(arrays, array_keys, is_training=True)
        
        # 5. Train each model
        for model in self.prediction_models:
            if not model.active:
                self.logger.warning(f"Skipping inactive model: {type(model).__name__}")
                continue
                
            self.logger.info(f"Training model: {type(model).__name__}")
            
            # Convert to arrays in declared order
            X = self._dict_to_array(input_data, model.input)
            y = self._dict_to_array(output_data, model.output)
            
            model.train(X, y)
            self.logger.info(f"Model training completed")

    def predict(self, exp_params: Dict[str, Any], exp_code: str, visualize_flag: bool, debug_flag: bool) -> Dict[str, float]:
        """
        Get predictions for all performance metrics for a single upcoming experiment.
        
        Data Flow:
        1. Use unified input preparation (same as training)
        2. Get predictions and denormalize to original [0,1] scale
        
        Args:
            X_parameters: Parameter values {param_name: value} - from DataInterface
            exp_nr: Experiment number for loading domain-specific data
            visualize_flag: Whether to show visualizations during feature computation
            debug_flag: Whether to run in debug mode
            
        Returns:
            Dictionary {performance_code: predicted_value} for all performance codes
        """
        self.logger.debug(f"Predicting performance for experiment {exp_code}")

        all_predictions = {}
        
        for pred_model in self.prediction_models:
            if not pred_model.active:
                continue
                
            # 1. Create single-experiment dataset
            single_exp_data = {exp_code: dict(exp_params)}
            
            # 2. Add feature data
            self._add_feature_data(single_exp_data, [pred_model], visualize_flag, debug_flag)
            
            # 3. Prepare and normalize inputs only (inference mode)
            input_data = self._normalize_data(
                single_exp_data, pred_model.input, is_training=False
            )
            
            # 4. Convert to array and predict
            X = self._dict_to_array(input_data, pred_model.input)
            predictions = pred_model.predict(X)  # Shape: (1, n_targets)
            
            # 5. Denormalize and store results
            denormalized = self._denormalize_predictions(predictions.flatten(), pred_model.output)
            for i, perf_code in enumerate(pred_model.output):
                all_predictions[perf_code] = float(denormalized[i])
                
        return all_predictions   
    
    def _get_pred_model_by_code(self) -> Dict[str, PredictionModel]:
        # Prep prediction system output dict
        pred_model_by_code = {}
        for pred_model in self.prediction_models:
            for code in pred_model.output:
                if code in pred_model_by_code:
                    raise ValueError(f"Performance code '{code}' is predicted by multiple prediction models. Please check the configuration.")
                pred_model_by_code[code] = pred_model
        return pred_model_by_code

    def _get_required_keys(self) -> Tuple[List[str], List[str], List[str]]:
        """Filter dataset to only experiments containing all required columns."""
        input_keys = set()
        output_keys = set()
        array_keys = set()
        for model in self.prediction_models:
            input_keys.update(model.input)
            if model.dataset_type == model.DatasetType.AGGR_METRICS:
                output_keys.update(model.output)
            elif model.dataset_type == model.DatasetType.METRIC_ARRAYS:
                array_keys.update(model.output)
            else:
                raise ValueError(f"Unknown dataset type for model {type(model).__name__}")
        return list(input_keys), list(output_keys), list(array_keys)

    def _add_feature_data(self, input_keys: List[str], exp_codes: List[str], visualize_flag: bool, debug_flag: bool) -> List[List[Any]]:
        """Add feature model data to experiments in-place."""
        # Prepare mapping of performance codes to feature models
        feature_models_by_code = {}
        for pred_model in self.prediction_models:
            if not pred_model.active:
                continue
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
        
        for col_idx, key in enumerate(keys):
            values_array = data_array[:, col_idx]
            
            if is_training:
                # Training mode: compute and store normalization parameters
                mean_val = np.mean(values_array)
                std_val = np.std(values_array)
                
                if std_val > 0:  # Avoid division by zero
                    data_array[:, col_idx] = (values_array - mean_val) / std_val
                    self.normalization_params[key] = {
                        'mean': float(mean_val), 
                        'std': float(std_val)
                    }
                else:
                    # Keep original values when std is 0
                    self.normalization_params[key] = {
                        'mean': float(mean_val), 
                        'std': 1.0
                    }
            else:
                # Inference mode: apply stored normalization parameters
                if key in self.normalization_params:
                    params = self.normalization_params[key]
                    if params['std'] > 0:
                        data_array[:, col_idx] = (values_array - params['mean']) / params['std']
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
    
    def _dict_to_array(self, data: Dict[str, np.ndarray], column_order: List[str]) -> np.ndarray:
        """Convert dictionary of columns to 2D array in specified order."""
        if not data:
            raise ValueError("Empty data dictionary")
        
        # Get number of samples from first column
        n_samples = len(next(iter(data.values())))
        
        # Stack columns in specified order
        arrays = []
        for column in column_order:
            if column in data:
                arrays.append(data[column])
            else:
                # Fill missing columns with zeros
                arrays.append(np.zeros(n_samples))
                self.logger.warning(f"Missing column '{column}', using zeros")
        
        # Stack horizontally and transpose to get (n_samples, n_features)
        return np.column_stack(arrays)

