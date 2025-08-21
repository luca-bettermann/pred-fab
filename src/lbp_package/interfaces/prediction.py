from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type
from dataclasses import dataclass
import numpy as np

from .features import FeatureModel
from ..utils import ParameterHandling, LBPLogger


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
                 logger: LBPLogger,
                 round_digits: int = 3,
                 **kwargs):
        """
        Initialize prediction model.
        
        Args:
            performance_codes: List of performance codes this model should predict
            folder_navigator: File system navigation utility
            logger: Logger instance
            **model_params: Model-specific parameters
        """
        self.logger = logger

        # By default, the prediction model is deactivated from the system
        self.active: bool = False

        # Round digits for outputs
        self.round_digits = round_digits

        # Set input keys this model requires for prediction
        self.input = self._declare_inputs()
        if not self.input or not isinstance(self.input, list):
            raise ValueError("Model must declare a list of input keys (X) to predict")

        # Set output performance codes this model predicts
        self.output = performance_codes
        if not self.output or not isinstance(self.output, list):
            raise ValueError("Model must specify a list of performance codes (y) to predict")

        # Initialize the model types for the extraction of additional input features
        self.feature_model_types: Dict[str, Type[FeatureModel]] = self._declare_feature_model_types()
        self.feature_models: Dict[str, FeatureModel] = {}

        # Store kwargs so that they can be passed on to the feature models
        self.kwargs = kwargs

    # === ABSTRACT METHODS (Must be implemented by subclasses) ===
    @abstractmethod
    def _declare_inputs(self) -> List[str]:
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
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the prediction model on processed data.
        
        Args:
            X: Input features, shape (n_samples, n_features)
               Order matches self.input declaration
            y: Target variables for training, shape (n_samples, n_targets)
               Order matches self.output declaration
            
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
            X: Input features, shape (1, n_features) for single experiment
               Order matches self.input declaration
               
        Returns:
            y: Predicted performance values, shape (1, n_targets) 
               Order matches self.output declaration
        """
        ...

    # === PUBLIC API METHODS (Called externally) ===
    def add_feature_model(self, code: str, feature_model: FeatureModel) -> None:
        """
        Predefined logic of how feature models are added to prediction models.
        
        Args:
            feature_model: FeatureModel instance to use for feature extraction
        """
        # Append the feature model instance (one-to-many relationship)
        self.feature_models[code] = feature_model

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
                   preprocessed_X: Dict[str, Any], 
                   preprocessed_y: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Model-specific preprocessing after global preprocessing.
        
        Override this method to:
        - Denormalize specific features back to original scale
        - Apply custom normalization (robust scaling, log transforms, etc.)
        - Handle categorical encoding or feature engineering
        
        Args:
            global_preprocessed_X: Globally preprocessed input features
            global_preprocessed_y: Globally preprocessed target variables
        
        Returns:
            Tuple of (model_specific_X, model_specific_y)
            
        Note:
            Default implementation does no additional preprocessing.
            Override this method for model-specific transformations.
        """
        return preprocessed_X, preprocessed_y


