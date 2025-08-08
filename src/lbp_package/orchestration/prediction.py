from typing import Any, Dict, List, Tuple, Optional, Type
import numpy as np

from ..interfaces import DataInterface, PreprocessingState, PredictionModel, FeatureModel
from ..utils import FolderNavigator, LBPLogger


class PredictionSystem:
    """
    Orchestrates prediction models with consistent data preprocessing.
    
    Handles the complete prediction workflow:
    1. Data collection from structured (DataInterface) and unstructured (FeatureModel) sources
    2. Global preprocessing for consistency across models
    3. Model-specific preprocessing and training
    4. Prediction with denormalization
    
    Data Responsibility Boundary:
    - DataInterface: Provides structured study/experiment parameters via get_study_parameters(), get_exp_variables()
    - PredictionSystem: Coordinates parameter collection and delegates to models
    - FeatureModel: Loads domain-specific unstructured data via _load_data() and computes features
    - PredictionModel: Manages FeatureModel instances and implements ML prediction logic
    """
    
    def __init__(self, 
                 folder_navigator: FolderNavigator,
                 data_interface: DataInterface,
                 logger: LBPLogger) -> None:
        """
        Initialize prediction system.
        
        Args:
            folder_navigator: File system navigation utility  
            data_interface: Interface for accessing structured study configuration
            logger: Logger instance
        """
        self.nav = folder_navigator
        self.interface = data_interface
        self.logger = logger
        self.prediction_models: List[PredictionModel] = []
        
        # Store global preprocessing state for consistent normalization/denormalization
        self.preprocessing_state = PreprocessingState()
        
        self.logger.info("Initialized PredictionSystem")

    def add_prediction_model(self, prediction_model: Type[PredictionModel], study_params: Dict[str, Any], round_digits: int, **kwargs) -> None:
        """
        Add an prediction model to the system.
        
        Args:
            performance_code: Code identifying the performance metric
            evaluation_class: Class of evaluation model to instantiate
            study_params: Study parameters for model configuration
        """
        # Validate if prediction_model is the correct type
        if not issubclass(prediction_model, PredictionModel):
            raise TypeError(f"Expected a subclass of PredictionModel, got {type(prediction_model).__name__}")

        model_instance = prediction_model(
            folder_navigator=self.nav,
            logger=self.logger,
            study_params=study_params,
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
    
    def train(self, input_X: Dict[str, Any]) -> None:
        """
        Train all models with consistent data collection and preprocessing.
        
        Data Flow:
        1. Extract structured parameters from input X data (via DataInterface)
        2. For each model: collect parameters + load domain-specific data via _load_data()
        3. Apply global preprocessing for consistency across models
        4. Train individual models with model-specific preprocessing via preprocess()
        
        Args:
            historical_evaluation_data: Complete evaluation history
                Expected format: {
                    "parameters": Dict[str, List],  # {param_code: [values]} - from DataInterface
                    "performances": Dict[str, List],  # {perf_code: [values]} - from evaluation results
                    "experiment_numbers": List[int],  # Experiment identifiers
                }
        """
        self.logger.info("Training prediction models on historical data")
        
        # Step 1: Collect data from all models and apply global preprocessing
        global_X, global_y, X_keys = self._global_preprocess_and_collect(input_X)
        self.logger.info(f"Global preprocessing completed: X with {len(X_keys)} features, y{global_y.shape}")
        
        # Step 2: Train each model with model-specific preprocessing
        for i, model in enumerate(self.prediction_models):
            self.logger.info(f"Training model {i+1}/{len(self.prediction_models)}: {type(model).__name__}")
            
            # Extract relevant features and targets for this specific model
            model_X = self._extract_model_features(global_X, X_keys, model._declare_inputs())
            model_y = self._extract_model_targets(global_y, model.output, input_X["performances"])
            
            # Allow model to customize preprocessing (denormalization, custom scaling, etc.)
            processed_X, processed_y = model.preprocess(model_X, model_y, self.preprocessing_state)
            
            # Train the model with its processed data
            model.train(processed_X, processed_y)
            self.logger.info(f"Model {i+1} training completed")

    def predict(self, X_parameters: Dict[str, Any], exp_nr: int, visualize_flag: bool, debug_flag: bool) -> Dict[str, float]:
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
        self.logger.debug(f"Predicting performance for experiment {exp_nr}")

        # Collect predictions from all models
        all_predictions = {}
        
        for pred_model in self.prediction_models:
            # Use unified input preparation (same logic as training)
            input_dict = self._prepare_experiment_input(pred_model, X_parameters, exp_nr, 
                                                      visualize_flag, debug_flag)
            
            # Get model predictions (normalized scale)
            model_predictions = pred_model.predict(input_dict)  # Shape: (n_targets,)
            
            # Denormalize predictions back to original [0,1] performance scale
            denormalized_predictions = self._denormalize_predictions(model_predictions, pred_model.output)
            
            # Store predictions by performance code
            for i, perf_code in enumerate(pred_model.output):
                all_predictions[perf_code] = float(denormalized_predictions[i])
                
        return all_predictions
    
    def _prepare_experiment_input(self, model: PredictionModel, exp_params: Dict[str, Any], 
                                exp_nr: int, visualize_flag: bool = False, debug_flag: bool = False) -> Dict[str, Any]:
        """
        Unified method to prepare input data for both training and prediction.
        
        This ensures consistent data preparation between training and prediction flows.
        
        Args:
            model: PredictionModel instance
            exp_params: Experiment parameters from DataInterface  
            exp_nr: Experiment number
            visualize_flag: Whether to show visualizations during feature computation
            debug_flag: Whether to run in debug mode
            
        Returns:
            Complete input dictionary for the model
        """
        # Set parameters in model using parameter handling system
        model.set_experiment_parameters(**exp_params)
        
        # Run feature models to compute external data
        feature_data = {}
        for code, feature_model in model.feature_models.items():
            feature_model.run(code, exp_nr, visualize_flag, debug_flag)
            feature_data[code] = feature_model.features[code]
        
        # Construct complete input dictionary for the model's requirements
        input_dict = {}
        for key in model._declare_inputs():
            if key in exp_params:
                # Structured parameter from DataInterface
                input_dict[key] = exp_params[key]
            elif key in feature_data:
                # Feature data from FeatureModel computation
                input_dict[key] = feature_data[key]
            else:
                # Check if it's a model attribute (from parameter handling decorators)
                if hasattr(model, key):
                    input_dict[key] = getattr(model, key)
                    
        return input_dict
    
    def _global_preprocess_and_collect(self, historical_data: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
        """
        Collect data from all models and apply global preprocessing for consistency.
        
        Data Collection Strategy:
        1. Use parameter handling to set structured parameters (from DataInterface)
        2. Run feature models to compute domain-specific features (same as prediction)
        3. Combine into unified input representation with global normalization
        """
        self.logger.info("Collecting data from all models and applying global preprocessing")
        
        # Extract structured data from DataInterface and evaluation results
        parameters = historical_data["parameters"]  # From DataInterface
        performances = historical_data["performances"]  # From evaluation results
        exp_numbers = historical_data["experiment_numbers"]
        
        # Collect all unique input keys required by all models
        all_input_keys = set()
        for model in self.prediction_models:
            all_input_keys.update(model._declare_inputs())
        all_input_keys = list(all_input_keys)
        
        # Collect data for all experiments across all models
        collected_X = {key: [] for key in all_input_keys}
        
        for exp_nr in exp_numbers:
            # Set structured parameters for this experiment
            exp_params = {key: values[exp_numbers.index(exp_nr)] for key, values in parameters.items()}
            
            for model in self.prediction_models:
                # Use unified input preparation (same logic as prediction)
                input_dict = self._prepare_experiment_input(model, exp_params, exp_nr, 
                                                          visualize_flag=False, debug_flag=False)
                
                # Collect values for all keys required by this model
                for key in model._declare_inputs():
                    if key in input_dict:
                        collected_X[key].append(input_dict[key])
        
        # Convert to arrays and apply global normalization (default: standardization)
        X_arrays = {}
        for key, values in collected_X.items():
            if values:  # Only process non-empty lists
                X_arrays[key] = np.array(values)
                
                # Apply normalization based on preprocessing state configuration
                if self.preprocessing_state.X_normalization_method == "standardize":
                    mean_val = np.mean(X_arrays[key])
                    std_val = np.std(X_arrays[key])
                    if std_val > 0:  # Avoid division by zero
                        X_arrays[key] = (X_arrays[key] - mean_val) / std_val
                        
        # Convert performance data to array format
        y_codes = list(performances.keys())
        y = np.array([performances[code] for code in y_codes]).T  # Shape: (n_samples, n_targets)
        
        return X_arrays, y, all_input_keys
    
    def _denormalize_predictions(self, predictions: np.ndarray, y_codes: List[str]) -> np.ndarray:
        """Denormalize predictions back to original [0,1] performance scale."""
        if self.preprocessing_state.y_normalization_method == "standardize":
            # Reverse standardization if it was applied
            denormalized = np.zeros_like(predictions)
            for i, code in enumerate(y_codes):
                if (self.preprocessing_state.global_y_mean and 
                    self.preprocessing_state.global_y_std and
                    code in self.preprocessing_state.global_y_mean):
                    denormalized[i] = (predictions[i] * self.preprocessing_state.global_y_std[code] + 
                                     self.preprocessing_state.global_y_mean[code])
                else:
                    denormalized[i] = predictions[i]
            return denormalized
        else:
            return predictions  # No normalization was applied, return as-is
    
    def _extract_model_features(self, X: Dict[str, np.ndarray], all_keys: List[str], model_keys: List[str]) -> Dict[str, np.ndarray]:
        """Extract only the features relevant to a specific model from global dataset."""
        model_X = {}
        for key in model_keys:
            if key in X:
                model_X[key] = X[key]
            else:
                self.logger.warning(f"Model requires feature '{key}' not found in collected data")
        return model_X
    
    def _extract_model_targets(self, y: np.ndarray, model_codes: List[str], all_performances: Dict[str, List]) -> np.ndarray:
        """Extract only the target variables relevant to a specific model."""
        all_codes = list(all_performances.keys())
        indices = []
        for code in model_codes:
            if code in all_codes:
                indices.append(all_codes.index(code))
            else:
                raise ValueError(f"Model requires target '{code}' not found in performance data")
        return y[:, indices]
