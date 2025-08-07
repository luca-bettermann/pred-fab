from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Type
from dataclasses import dataclass
import numpy as np

from .features import FeatureModel
from ..utils import ParameterHandling, FolderNavigator, LBPLogger


@dataclass
class PreprocessingState:
    """
    Stores preprocessing transformation parameters for consistent denormalization.
    
    This ensures that predictions can be properly denormalized to match
    the original performance scale [0, 1].
    """
    
    # Global preprocessing state - stores normalization parameters
    global_X_mean: Optional[np.ndarray] = None
    global_X_std: Optional[np.ndarray] = None
    global_X_min: Optional[np.ndarray] = None
    global_X_max: Optional[np.ndarray] = None
    
    global_y_mean: Optional[Dict[str, float]] = None
    global_y_std: Optional[Dict[str, float]] = None
    global_y_min: Optional[Dict[str, float]] = None
    global_y_max: Optional[Dict[str, float]] = None
    
    # Track which normalization methods were used for reversal
    X_normalization_method: str = "standardize"  # "standardize", "minmax", or "none"
    y_normalization_method: str = "none"  # Performance values [0,1] usually don't need normalization


@dataclass 
class PredictionModel(ParameterHandling, ABC):
    """
    Abstract interface for prediction models.
    
    Defines the structure for training and predicting performance metrics y
    based on input parameters X. Uses the same parameter handling approach
    as FeatureModel and EvaluationModel for consistency.
    
    Data Responsibility Boundary:
    - DataInterface: Handles structured study/experiment metadata and parameters
    - PredictionModel: Loads domain-specific, unstructured data (geometry graph, sensor data, etc.)
    """

    def __init__(self,
                 performance_codes: List[str],
                 folder_navigator: FolderNavigator,
                 logger: LBPLogger,
                 round_digits: int = 3,
                 **model_params):
        """
        Initialize prediction model.
        
        Args:
            performance_codes: List of performance codes this model should predict
            folder_navigator: File system navigation utility
            logger: Logger instance
            **model_params: Model-specific parameters
        """
        self.nav = folder_navigator
        self.logger = logger

        # By default, the prediction model is deactivated from the system
        self.active: bool = False

        # Set input keys this model requires for prediction
        self.input = self.declare_inputs()
        if not self.input or not isinstance(self.input, list):
            raise ValueError("Model must declare a list of input keys (X) to predict")

        # List of performance codes this model predicts (passed by system)
        self.output = performance_codes
        if not self.output or not isinstance(self.output, list):
            raise ValueError("Model must specify a list of performance codes (y) to predict")

        # Store preprocessing state for denormalization during prediction
        self.preprocessing_state: Optional[PreprocessingState] = None

        # Initialize the model types for the extraction of additional input features
        self.feature_model_types: Dict[str, Type[FeatureModel]] = self._declare_feature_model_types()
        self.feature_models: List[FeatureModel] = []

        # Apply parameter handling - consistent with FeatureModel/EvaluationModel
        self.set_model_parameters(**model_params)
        self._validate_parameters()
    
    # === ABSTRACT METHODS (Must be implemented by subclasses) ===
    @abstractmethod
    def declare_inputs(self) -> List[str]:
        """
        Declare all input keys this model requires for prediction.
        
        This defines the complete input interface X. Can include:
        - Parameter names (from dataclass fields using parameter_handler)
        - External data keys (loaded via _load_data method)
        
        Returns:
            List of input keys this model expects in X
            e.g., ["n_layers", "temperature", "geometry_mesh", "sensor_data"]
        """
        ...
    
    @abstractmethod
    def train(self, X: Dict[str, np.ndarray], y: np.ndarray) -> None:
        """
        Train the prediction model on processed data.
        
        Args:
            X: Dictionary of input features {key: values_array}
            y: Target variables for training, shape (n_samples, n_targets)
               Order matches self.y_codes
            
        Note:
            Cross-validation and hyperparameter tuning should be
            implemented within this method by the user, if needed.
        """
        ...
    
    @abstractmethod
    def predict(self, X: Dict[str, float]) -> np.ndarray:
        """
        Predict performance metrics for given input X.
        
        Args:
            X: Dictionary of input features {key: scalar_value}
               Single experiment prediction
        Returns:
            y: Predicted performance values, shape (n_targets,)
               Order matches self.y_codes
        """
        ...

    # === PUBLIC API METHODS (Called externally) ===
    def add_feature_model(self, feature_model: FeatureModel) -> None:
        """
        Predefined logic of how feature models are added to prediction models.
        
        Args:
            feature_model: FeatureModel instance to use for feature extraction
        """
        # Append the feature model instance (one-to-many relationship)
        self.feature_models.append(feature_model)

    # === OPTIONAL METHODS ===
    def _declare_feature_model_types(self) -> Dict[str, Type[FeatureModel]]:
        """
        Declare feature model types to use for feature extraction.
        
        This method can be overridden to specify which feature models
        should be used for this prediction model.
        
        Returns:
            Dictionary of feature model types {feature_code: FeatureModelClass}
            e.g., {"energy_consumption": EnergyFeature, "path_deviation": PathDeviationFeature}
        """
        return {}
        
    def preprocess(self, 
                   global_preprocessed_X: Dict[str, Any], 
                   global_preprocessed_y: np.ndarray,
                   preprocessing_state: PreprocessingState) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Model-specific preprocessing after global preprocessing.
        
        Override this method to:
        - Denormalize specific features back to original scale
        - Apply custom normalization (robust scaling, log transforms, etc.)
        - Handle categorical encoding or feature engineering
        
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

