from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from .utils.parameter_handler import ParameterHandling
from .utils.folder_navigator import FolderNavigator
from .data_interface import DataInterface
from .utils.log_manager import LBPLogger


@dataclass
class PreprocessingState:
    """
    Stores preprocessing transformation parameters for consistent denormalization.
    
    This ensures that predictions can be properly denormalized to match
    the original performance scale [0, 1].
    """
    
    # Global preprocessing state
    global_X_mean: Optional[np.ndarray] = None
    global_X_std: Optional[np.ndarray] = None
    global_X_min: Optional[np.ndarray] = None
    global_X_max: Optional[np.ndarray] = None
    
    global_y_mean: Optional[Dict[str, float]] = None
    global_y_std: Optional[Dict[str, float]] = None
    global_y_min: Optional[Dict[str, float]] = None
    global_y_max: Optional[Dict[str, float]] = None
    
    # Track which normalization methods were used
    X_normalization_method: str = "standardize"  # "standardize", "minmax", or "none"
    y_normalization_method: str = "none"  # Performance values [0,1] usually don't need normalization


@dataclass 
class PredictionModel(ParameterHandling, ABC):
    """
    Abstract interface for prediction models.
    
    Defines the structure for training and predicting performance metrics y
    based on input parameters X. Prediction models receive performance_codes
    from the system (analogous to evaluation models).
    """
    
    # === ABSTRACT METHODS (Must be implemented by subclasses) ===
    def __init__(self,
                 performance_codes: List[str],
                 logger: LBPLogger,
                 **model_params):
        """
        Initialize prediction model.
        
        Args:
            performance_codes: List of performance codes this model should predict
            logger: Logger instance
            **model_params: Model-specific parameters
        """
        self.logger = logger

        # List of input codes this model uses for prediction
        self.X_codes = self._get_input_codes()
        if not self.X_codes:
            raise ValueError("Model must specify input codes (X) for prediction")
        
        # List of performance codes this model predicts (passed by system)
        self.y_codes = performance_codes
        if not self.y_codes:
            raise ValueError("Model must specify performance codes (y) to predict")
        
        # Store preprocessing state for denormalization
        self.preprocessing_state: Optional[PreprocessingState] = None
        
        # Apply parameter handling
        self.set_model_parameters(**model_params)
        self._validate_parameters()
    
    @abstractmethod
    def _get_input_codes(self) -> List[str]:
        """
        Return which inputs (X) this model uses for prediction.

        Returns:
            List of inputs this model handles
            e.g., [parameter_1, parameter_2, "temperature"]
        """
        ...
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the prediction model on processed historical data.
        
        Args:
            X: Input features for training, shape (n_samples, n_features)
            y: Target variables for training, shape (n_samples, n_targets)
            
        Note:
            Cross-validation and hyperparameter tuning should be
            implemented within this method by the user, if needed.
        """
        ...
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict performance metrics for given input X.
        
        Args:
            X: Input features for prediction, shape (n_samples, n_features)

        Returns:
            y: Predicted performance values, shape (n_samples, n_targets)
               Order matches self.y_codes
        """
        ...

    # === OPTIONAL METHODS ===        
    def preprocess(self, 
                   global_preprocessed_X: np.ndarray, 
                   global_preprocessed_y: np.ndarray,
                   preprocessing_state: PreprocessingState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Model-specific preprocessing after global preprocessing.
        
        Args:
            global_preprocessed_X: Globally preprocessed input features
            global_preprocessed_y: Globally preprocessed target variables
            preprocessing_state: Global preprocessing transformation parameters
        
        Returns:
            Tuple of (model_specific_X, model_specific_y)
            
        Note:
            Default implementation does no additional preprocessing.
            Override this method for model-specific transformations.
        """
        self.preprocessing_state = preprocessing_state
        return global_preprocessed_X, global_preprocessed_y
    
    def _validate_parameters(self) -> None:
        """Optional parameter validation - can be overridden by subclasses."""
        pass


class PredictionSystem:
    """
    Lightweight orchestration for single or multi prediction model system.
    
    Handles global data preprocessing to ensure consistency across models,
    then delegates to individual models for training and prediction.
    """
    
    def __init__(self, 
                 folder_navigator: FolderNavigator,
                 data_interface: DataInterface,
                 logger: LBPLogger):
        """
        Initialize prediction system.
        
        Args:
            folder_navigator: File system navigation utility  
            data_interface: Interface for accessing study configuration
            logger: Logger instance
        """
        self.nav = folder_navigator
        self.interface = data_interface
        self.logger = logger
        self.models: List[PredictionModel] = []
        
        # Store global preprocessing state for denormalization
        self.preprocessing_state = PreprocessingState()
        
        self.logger.info("Initialized PredictionSystem")

    def add_model(self, model: PredictionModel) -> None:
        """
        Add a prediction model to the system.
        
        Args:
            model: Instance of PredictionModel or subclass
        """
        if not isinstance(model, PredictionModel):
            raise TypeError("model must be an instance of PredictionModel or its subclass")
        
        self.models.append(model)
        self.logger.info(f"Added model: {type(model).__name__} for performance codes: {model.y_codes}")
    
    def get_required_performance_codes(self) -> List[str]:
        """
        Get all performance codes that need to be predicted.
        """
        all_codes = set()
        for model in self.models:
            all_codes.update(model.y_codes)
        return list(all_codes)
    
    def train(self, historical_evaluation_data: Dict[str, Any]) -> None:
        """
        Train all models with global preprocessing consistency.
        
        Args:
            historical_evaluation_data: Complete evaluation history
                Expected format: {
                    "parameters": Dict[str, List],  # {param_code: [values]}
                    "performances": Dict[str, List],  # {perf_code: [values]}
                    "experiment_codes": List[str],  # Experiment identifiers
                }
        """
        self.logger.info("Training prediction models on historical data")
        
        # Step 1: Global preprocessing for data consistency
        global_X, global_y, X_codes, y_codes = self._global_preprocess(historical_evaluation_data)
        self.logger.info(f"Global preprocessing completed: X{global_X.shape}, y{global_y.shape}")
        
        # Step 2: Train each model with model-specific preprocessing
        for i, model in enumerate(self.models):
            self.logger.info(f"Training model {i+1}/{len(self.models)}: {type(model).__name__}")
            
            # Extract relevant features and targets for this model
            model_X = self._extract_model_features(global_X, X_codes, model.X_codes)
            model_y = self._extract_model_targets(global_y, y_codes, model.y_codes)
            
            # Model-specific preprocessing
            processed_X, processed_y = model.preprocess(model_X, model_y, self.preprocessing_state)
            
            # Train the model
            model.train(processed_X, processed_y)
            self.logger.info(f"Model {i+1} training completed")

    def predict(self, X: np.ndarray, parameter_codes: List[str]) -> Dict[str, np.ndarray]:
        """
        Get joint predictions for all performance metrics.
        
        Args:
            X: Array of shape (n_candidates, n_features)
        """
        self.logger.debug(f"Predicting performance for {X.shape[0]} parameter combinations")

        # Step 1: Apply global preprocessing to input parameters
        normalized_X = self._normalize_prediction_inputs(X, parameter_codes)

        # Step 2: Collect predictions from all models
        all_predictions = {}
        
        for model in self.models:
            # Extract relevant features for this model
            model_X = self._extract_model_features(normalized_X, parameter_codes, model.X_codes)
            
            # Get model predictions (normalized)
            model_predictions = model.predict(model_X)  # Shape: (n_candidates, n_targets)
            
            # Denormalize predictions to original [0,1] scale
            denormalized_predictions = self._denormalize_predictions(model_predictions, model.y_codes)
            
            # Store predictions by performance code
            for i, perf_code in enumerate(model.y_codes):
                if perf_code not in all_predictions:
                    all_predictions[perf_code] = []
                all_predictions[perf_code].append(denormalized_predictions[:, i])
        return all_predictions
    
    def _global_preprocess(self, historical_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Global preprocessing ensuring data consistency across all models.
        
        Args:
            historical_data: Raw historical evaluation data
        
        Returns:
            Tuple of (preprocessed_X, preprocessed_y, X_codes, y_codes)
        """
        self.logger.info("Applying global preprocessing")
        
        # Extract parameter data
        parameters = historical_data["parameters"]  # Dict[str, List]
        performances = historical_data["performances"]  # Dict[str, List]
        
        # Convert to arrays
        X_codes = list(parameters.keys())
        y_codes = list(performances.keys())
        
        X = np.array([parameters[code] for code in X_codes]).T  # Shape: (n_samples, n_features)
        y = np.array([performances[code] for code in y_codes]).T  # Shape: (n_samples, n_targets)
        
        # Global X normalization (parameters can have very different scales)
        if self.preprocessing_state.X_normalization_method == "standardize":
            self.preprocessing_state.global_X_mean = np.mean(X, axis=0)
            self.preprocessing_state.global_X_std = np.std(X, axis=0)
            # Avoid division by zero
            self.preprocessing_state.global_X_std = np.where(
                self.preprocessing_state.global_X_std == 0, 1.0, self.preprocessing_state.global_X_std
            )
            X_normalized = (X - self.preprocessing_state.global_X_mean) / self.preprocessing_state.global_X_std
            
        elif self.preprocessing_state.X_normalization_method == "minmax":
            self.preprocessing_state.global_X_min = np.min(X, axis=0)
            self.preprocessing_state.global_X_max = np.max(X, axis=0)
            # Avoid division by zero
            X_range = self.preprocessing_state.global_X_max - self.preprocessing_state.global_X_min
            X_range = np.where(X_range == 0, 1.0, X_range)
            X_normalized = (X - self.preprocessing_state.global_X_min) / X_range
            
        else:  # "none"
            X_normalized = X
        
        # Global y preprocessing (performances are [0,1] but might benefit from normalization)
        if self.preprocessing_state.y_normalization_method == "standardize":
            self.preprocessing_state.global_y_mean = {code: np.mean(y[:, i]) for i, code in enumerate(y_codes)}
            self.preprocessing_state.global_y_std = {code: np.std(y[:, i]) for i, code in enumerate(y_codes)}
            
            y_normalized = np.zeros_like(y)
            for i, code in enumerate(y_codes):
                std_val = self.preprocessing_state.global_y_std[code]
                if std_val == 0:
                    std_val = 1.0
                y_normalized[:, i] = (y[:, i] - self.preprocessing_state.global_y_mean[code]) / std_val
        else:  # "none" - usually for performance values [0,1]
            y_normalized = y
        
        self.logger.debug(f"Global preprocessing: X {X.shape} -> {X_normalized.shape}, y {y.shape} -> {y_normalized.shape}")
        return X_normalized, y_normalized, X_codes, y_codes
    
    def _normalize_prediction_inputs(self, X: np.ndarray, parameter_codes: List[str]) -> np.ndarray:
        """Apply the same normalization used during training to new inputs."""
        if self.preprocessing_state.X_normalization_method == "standardize":
            return (X - self.preprocessing_state.global_X_mean) / self.preprocessing_state.global_X_std
        elif self.preprocessing_state.X_normalization_method == "minmax":
            X_range = self.preprocessing_state.global_X_max - self.preprocessing_state.global_X_min
            X_range = np.where(X_range == 0, 1.0, X_range)
            return (X - self.preprocessing_state.global_X_min) / X_range
        else:
            return X
    
    def _denormalize_predictions(self, predictions: np.ndarray, y_codes: List[str]) -> np.ndarray:
        """Denormalize predictions back to original [0,1] scale."""
        if self.preprocessing_state.y_normalization_method == "standardize":
            denormalized = np.zeros_like(predictions)
            for i, code in enumerate(y_codes):
                denormalized[:, i] = (predictions[:, i] * self.preprocessing_state.global_y_std[code] + 
                                    self.preprocessing_state.global_y_mean[code])
            return denormalized
        else:
            return predictions  # No normalization was applied
    
    def _extract_model_features(self, X: np.ndarray, all_codes: List[str], model_codes: List[str]) -> np.ndarray:
        """Extract features relevant to a specific model."""
        indices = [all_codes.index(code) for code in model_codes if code in all_codes]
        if len(indices) != len(model_codes):
            missing = set(model_codes) - set(all_codes)
            raise ValueError(f"Model requires features not in data: {missing}")
        return X[:, indices]
    
    def _extract_model_targets(self, y: np.ndarray, all_codes: List[str], model_codes: List[str]) -> np.ndarray:
        """Extract targets relevant to a specific model."""
        indices = [all_codes.index(code) for code in model_codes if code in all_codes]
        if len(indices) != len(model_codes):
            missing = set(model_codes) - set(all_codes)
            raise ValueError(f"Model requires targets not in data: {missing}")
        return y[:, indices]
