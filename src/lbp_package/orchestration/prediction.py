from typing import Any, Dict, List, Tuple, Optional, Type, Set
import numpy as np

from ..interfaces import DataInterface, PredictionModel
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

        # Store normalization parameters for consistent denormalization
        self.normalization_params: Dict[str, Dict[str, float]] = {}
        
        self.logger.info("Initialized PredictionSystem")

    def add_prediction_model(self, performance_codes: List[str], prediction_model: Type[PredictionModel], study_params: Dict[str, Any], round_digits: int, **kwargs) -> None:
        """
        Add an prediction model to the system.
        
        Args:
            performance_codes: List of codes identifying the performance metrics
            evaluation_class: Class of evaluation model to instantiate
            study_params: Study parameters for model configuration
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
        if len(self.pred_model_by_code) == 0 or recompute:
            self.pred_model_by_code = self._get_pred_model_by_code()

        assert code in self.pred_model_by_code, f"No prediction model for performance code '{code}' has been initialized."
        prediction_model = self.pred_model_by_code[code]

        if prediction_model.active:
            self.logger.info(f"Prediction model {type(prediction_model).__name__} for performance code '{code}' is already active.")
        else:
            prediction_model.set_model_parameters(**study_params)
            prediction_model.active = True
            self.logger.info(f"Activated prediction model {type(prediction_model).__name__} for performance code '{code}' and study parameters")

    def train(self, study_dataset: Dict[str, Dict[str, Any]]) -> None:
        """
        Train all models with unified data processing.

        Args:
            study_dataset: Dictionary of experiments {exp_code: {param_name/perf_name: value, ...}}
        """
        # 1. Filter dataset to required columns
        dataset = self._filter_dataset(study_dataset)
        
        # 2. Add feature data
        self._add_feature_data(dataset, self.prediction_models, visualize_flag=False)
        
        # 3. Get all input and output columns
        all_inputs = []
        all_outputs = []
        for model in self.prediction_models:
            all_inputs.extend(model.input)
            all_outputs.extend(model.output)
        
        # 4. Prepare and normalize inputs (training mode)
        input_data = self._prepare_and_normalize_data(
            dataset, list(set(all_inputs)), is_training=True
        )
        
        # 5. Prepare and normalize outputs (training mode) 
        output_data = self._prepare_and_normalize_data(
            dataset, list(set(all_outputs)), is_training=True
        )
        
        # 6. Train each model
        for model in self.prediction_models:
            if not model.active:
                continue
                
            self.logger.info(f"Training model: {type(model).__name__}")
            
            # Convert to arrays in declared order
            X = self._dict_to_array(input_data, model.input)
            y = self._dict_to_array(output_data, model.output)
            
            model.train(X, y)
            self.logger.info(f"Model training completed")

    def predict(self, exp_params: Dict[str, Any], exp_code: str, visualize_flag: bool) -> Dict[str, float]:
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
            self._add_feature_data(single_exp_data, [pred_model], visualize_flag)
            
            # 3. Prepare and normalize inputs only (inference mode)
            input_data = self._prepare_and_normalize_data(
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
                assert code not in pred_model_by_code, f"Performance code '{code}' is predicted by multiple prediction models. Please check the configuration."
                pred_model_by_code[code] = pred_model
        return pred_model_by_code
    
    def _extract_columns(self, data: Dict[str, np.ndarray], required_columns: List[str]) -> Dict[str, np.ndarray]:
        """Extract only the columns relevant to a model from dataset."""
        extracted_data = {}
        for column in required_columns:
            if column in data:
                extracted_data[column] = data[column]
            else:
                self.logger.warning(f"Model requires column '{column}' not found in data")
        return extracted_data

    def _filter_dataset(self, study_dataset: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Filter dataset to only experiments containing all required columns."""
        required_columns = set()
        for model in self.prediction_models:
            required_columns.update(model.input)
            required_columns.update(model.output)
        
        filtered_dataset = {}
        for exp_code, exp_data in study_dataset.items():
            # Check if experiment has all required columns
            if required_columns.issubset(exp_data.keys()):
                filtered_dataset[exp_code] = exp_data
            else:
                missing_columns = required_columns - exp_data.keys()
                self.logger.warning(f"Skipping experiment {exp_code}: missing columns {missing_columns}")
        
        self.logger.info(f"Filtered dataset: {len(filtered_dataset)}/{len(study_dataset)} experiments")
        return filtered_dataset

    def _add_feature_data(self, dataset: Dict[str, Dict[str, Any]], models: List[PredictionModel], visualize_flag: bool) -> None:
        """Add feature model data to experiments in-place."""
        for model in models:
            if hasattr(model, 'feature_models') and model.feature_models:
                for code, feature_model in model.feature_models.items():
                    # Run feature model for each experiment
                    for exp_code in dataset.keys():
                        feature_model.run(code, exp_code, self.nav.get_experiment_folder(exp_code), visualize_flag)
                        # Add computed feature to experiment data
                        if code in feature_model.features:
                            dataset[exp_code][code] = feature_model.features[code]

    def _prepare_and_normalize_data(self, dataset: Dict[str, Dict[str, Any]], target_columns: List[str], is_training: bool = True) -> Dict[str, np.ndarray]:
        """Prepare and normalize data for model training or inference."""
        exp_codes = list(dataset.keys())
        n_samples = len(exp_codes)
        
        if n_samples == 0:
            raise ValueError("Dataset is empty")
        
        # Get all unique column names across all experiments
        all_columns = set()
        for exp_data in dataset.values():
            all_columns.update(exp_data.keys())
        
        # Normalize each column independently using standardization
        normalized_data = {}
        for column in all_columns:
            # Extract values for this column across all experiments
            values = [dataset[exp_code].get(column, 0) for exp_code in exp_codes]
            values_array = np.array(values)
            
            if is_training:
                # Training mode: compute and store normalization parameters
                mean_val = np.mean(values_array)
                std_val = np.std(values_array)
                
                if std_val > 0:  # Avoid division by zero
                    normalized_values = (values_array - mean_val) / std_val
                    self.normalization_params[column] = {
                        'mean': float(mean_val), 
                        'std': float(std_val)
                    }
                else:
                    normalized_values = values_array
                    self.normalization_params[column] = {
                        'mean': float(mean_val), 
                        'std': 1.0
                    }
            else:
                # Inference mode: apply stored normalization parameters
                if column in self.normalization_params:
                    params = self.normalization_params[column]
                    if params['std'] > 0:
                        normalized_values = (values_array - params['mean']) / params['std']
                    else:
                        normalized_values = values_array
                else:
                    # Column not seen during training, use as-is
                    normalized_values = values_array
                    self.logger.warning(f"Column '{column}' not seen during training, using raw values")
            
            normalized_data[column] = normalized_values
        
        if is_training:
            self.logger.info(f"Normalized {len(all_columns)} columns using standardization")
        
        return self._extract_columns(normalized_data, target_columns)

    def _denormalize_predictions(self, predictions: np.ndarray, column_codes: List[str]) -> np.ndarray:
        """Denormalize predictions back to original scale using standardization parameters."""
        # Reverse standardization
        denormalized = np.zeros_like(predictions)
        for i, code in enumerate(column_codes):
            if code in self.normalization_params:
                params = self.normalization_params[code]
                denormalized[i] = (predictions[i] * params['std'] + params['mean'])
            else:
                denormalized[i] = predictions[i]
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

