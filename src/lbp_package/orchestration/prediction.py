from typing import Any, Dict, List, Tuple, Optional, Type, Set
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
        self.pred_model_by_code: Dict[str, PredictionModel] = {}

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

    def activate_prediction_model(self, code: str, recompute: bool = False) -> None:
        if len(self.pred_model_by_code) == 0 or recompute:
            self.pred_model_by_code = self._get_pred_model_by_code()

        assert code in self.pred_model_by_code, f"No prediction model for performance code '{code}' has been initialized."
        prediction_model = self.pred_model_by_code[code]

        if prediction_model.active:
            self.logger.warning(f"Prediction model {type(prediction_model).__name__} for performance code '{code}' is already active.")
        else:
            prediction_model.active = True
            self.logger.info(f"Activated prediction model {type(prediction_model).__name__} for performance code '{code}'")

    def train(self, study_dataset: Dict[str, Dict[str, Any]]) -> None:
        """
        Train all models with consistent data collection and preprocessing.

        Args:
            study_dataset: Dictionary of experiments {exp_code: {param_name/perf_name: value, ...}}

        The study dataset includes all parameters and performance metrics for each experiment.
        
        Data Flow:
        1. Filter study dataset to relevant parameters and performances
        2. Load additional data from FeatureModels
        3. Apply global preprocessing to inputs and performances and return X and y
        4. Train each model with its own preprocessing logic
        """

        # Step 1: Filter study dataset to relevant parameters and performances
        parameters: List[str] = []
        performances: List[str] = []
        for pred_model in self.prediction_models:
            parameters.extend(pred_model.input)
            performances.extend(pred_model.output)

        # Get all unique entry keys across all experiments
        existing_entries: Set[str] = set()
        for exp_entries in study_dataset.values():
            existing_entries.update(exp_entries.keys())

        # validate that all required entries are present in the dataset
        valid_entries = parameters + performances
        for entry in valid_entries:
            if entry not in existing_entries:
                raise ValueError(f"Study dataset is missing required entry '{entry}'")
            
        # Filter dataset to only include valid entries
        dataset = {
            exp_code: {key: value for key, value in entries.items() if key in valid_entries}
            for exp_code, entries in study_dataset.items()
        }

        # Step 2: Compute additional data from FeatureModels
        for pred_model in self.prediction_models:
            # Skip models that are not active or have no feature models defined
            if not pred_model.active or not pred_model.feature_models:
                self.logger.debug(f"Skipping prediction model {type(pred_model).__name__} for data loading - not active or no feature models defined")
                continue

            # Compute input data for each feature model
            for feature_code, feature_model in pred_model.feature_models.items():
                for exp_code, entries in dataset.items():
                    feature_model.run(feature_code, exp_code=exp_code, visualize_flag=False, debug_flag=False)
                    entries[feature_code] = feature_model.features[feature_code]
        self.logger.info(f"Feature models executed, features added to the dataset for {len(dataset)} experiments")

        # Think about adding features to airtable?
        # What do we want to store in the database, what should be stored locally?
        # For now, lets settle for: add above/below target for the airtable. (1, -1, 0)

        # Step 3: Apply global normalization to the dataset
        normalized_data, exp_codes = self._global_dataset_normalization(dataset)
        self.logger.info(f"Global preprocessing completed: {len(normalized_data)} columns normalized for {len(exp_codes)} experiments")
        
        # Step 4: Train each model with model-specific preprocessing
        for i, model in enumerate(self.prediction_models):
            if not model.active:
                continue
                
            self.logger.info(f"Training model {i+1}/{len(self.prediction_models)}: {type(model).__name__}")
            
            # Extract relevant features and targets for this specific model
            model_X = self._extract_model_data(normalized_data, model.input)
            model_y = self._extract_model_data(normalized_data, model.output)
            
            # Convert y to numpy array format expected by preprocess method
            y_array = np.column_stack([model_y[col] for col in model.output])
            
            # Allow model to customize preprocessing (denormalization, custom scaling, etc.)
            processed_X, processed_y = model.preprocess(model_X, y_array, self.preprocessing_state)
            
            # Train the model with its processed data
            model.train(processed_X, processed_y)
            self.logger.info(f"Model {i+1} training completed")

    
    def _global_dataset_normalization(self, dataset: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Normalize all columns in the dataset uniformly.
        
        Treats each column as its own entity rather than separating parameters and performances.
        This simplifies the normalization logic and makes it more flexible.
        
        Args:
            dataset: Dictionary of experiments {exp_code: {column_name: value, ...}}
            
        Returns:
            Tuple of (normalized_data, exp_codes) where:
            - normalized_data: Dict mapping column names to normalized arrays
            - exp_codes: List of experiment codes (for reference)
        """
        # Extract all experiments and their values
        exp_codes = list(dataset.keys())
        n_samples = len(exp_codes)
        
        if n_samples == 0:
            raise ValueError("Dataset is empty")
        
        # Get all unique column names across all experiments
        all_columns = set()
        for exp_data in dataset.values():
            all_columns.update(exp_data.keys())
        
        # Initialize normalized data dictionary and normalization parameters
        normalized_data = {}
        if self.preprocessing_state.normalization_params is None:
            self.preprocessing_state.normalization_params = {}
        
        # Normalize each column independently
        for column in all_columns:
            # Extract values for this column across all experiments
            values = [dataset[exp_code].get(column, 0) for exp_code in exp_codes]
            values_array = np.array(values)
            
            # Apply normalization based on preprocessing state configuration
            if self.preprocessing_state.normalization_method == "standardize":
                mean_val = np.mean(values_array)
                std_val = np.std(values_array)
                
                if std_val > 0:  # Avoid division by zero
                    normalized_values = (values_array - mean_val) / std_val
                    self.preprocessing_state.normalization_params[column] = {
                        'mean': float(mean_val), 
                        'std': float(std_val)
                    }
                else:
                    normalized_values = values_array
                    self.preprocessing_state.normalization_params[column] = {
                        'mean': float(mean_val), 
                        'std': 1.0
                    }
                    
            elif self.preprocessing_state.normalization_method == "minmax":
                min_val = np.min(values_array)
                max_val = np.max(values_array)
                
                if max_val > min_val:  # Avoid division by zero
                    normalized_values = (values_array - min_val) / (max_val - min_val)
                    self.preprocessing_state.normalization_params[column] = {
                        'min': float(min_val),
                        'max': float(max_val)
                    }
                else:
                    normalized_values = values_array
                    self.preprocessing_state.normalization_params[column] = {
                        'min': float(min_val),
                        'max': float(min_val)
                    }
            else:  # "none" or any other value
                normalized_values = values_array
                self.preprocessing_state.normalization_params[column] = None
            
            normalized_data[column] = normalized_values
        
        self.logger.info(f"Normalized {len(all_columns)} columns using '{self.preprocessing_state.normalization_method}' method")
        return normalized_data, exp_codes
    
    def predict(self, X_parameters: Dict[str, Any], exp_code: str, visualize_flag: bool) -> Dict[str, float]:
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

        # Collect predictions from all models
        all_predictions = {}
        
        for pred_model in self.prediction_models:
            # Use unified input preparation (same logic as training)
            input_dict = self._prepare_experiment_input(pred_model, X_parameters, exp_code, visualize_flag)

            # Get model predictions (normalized scale)
            model_predictions = pred_model.predict(input_dict)  # Shape: (n_targets,)
            
            # Denormalize predictions back to original [0,1] performance scale
            denormalized_predictions = self._denormalize_predictions(model_predictions, pred_model.output)
            
            # Store predictions by performance code
            for i, perf_code in enumerate(pred_model.output):
                all_predictions[perf_code] = float(denormalized_predictions[i])
                
        return all_predictions
    
    def _get_pred_model_by_code(self) -> Dict[str, PredictionModel]:
        # Prep prediction system output dict
        pred_model_by_code = {}
        for pred_model in self.prediction_models:
            for code in pred_model.output:
                assert code not in pred_model_by_code, f"Performance code '{code}' is predicted by multiple prediction models. Please check the configuration."
                pred_model_by_code[code] = pred_model
        return pred_model_by_code
    
    def _prepare_experiment_input(self, model: PredictionModel, exp_params: Dict[str, Any], exp_code: str, visualize_flag: bool) -> Dict[str, Any]:
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
            feature_model.run(code, exp_code, visualize_flag)
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

    
    def _denormalize_predictions(self, predictions: np.ndarray, column_codes: List[str]) -> np.ndarray:
        """Denormalize predictions back to original scale using unified normalization parameters."""
        if self.preprocessing_state.normalization_method == "standardize":
            # Reverse standardization if it was applied
            denormalized = np.zeros_like(predictions)
            for i, code in enumerate(column_codes):
                if (self.preprocessing_state.normalization_params and 
                    code in self.preprocessing_state.normalization_params and
                    self.preprocessing_state.normalization_params[code] is not None):
                    
                    params = self.preprocessing_state.normalization_params[code]
                    denormalized[i] = (predictions[i] * params['std'] + params['mean'])
                else:
                    denormalized[i] = predictions[i]
            return denormalized
            
        elif self.preprocessing_state.normalization_method == "minmax":
            # Reverse min-max normalization
            denormalized = np.zeros_like(predictions)
            for i, code in enumerate(column_codes):
                if (self.preprocessing_state.normalization_params and 
                    code in self.preprocessing_state.normalization_params and
                    self.preprocessing_state.normalization_params[code] is not None):
                    
                    params = self.preprocessing_state.normalization_params[code]
                    denormalized[i] = predictions[i] * (params['max'] - params['min']) + params['min']
                else:
                    denormalized[i] = predictions[i]
            return denormalized
        else:
            return predictions  # No normalization was applied, return as-is
    
    def _extract_model_data(self, normalized_data: Dict[str, np.ndarray], required_columns: List[str]) -> Dict[str, np.ndarray]:
        """Extract only the data relevant to a specific model from normalized dataset."""
        model_data = {}
        for column in required_columns:
            if column in normalized_data:
                model_data[column] = normalized_data[column]
            else:
                self.logger.warning(f"Model requires column '{column}' not found in normalized data")
        return model_data
    
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
