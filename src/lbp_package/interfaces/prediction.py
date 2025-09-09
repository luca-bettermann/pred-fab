from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type, final
from dataclasses import dataclass
import numpy as np
from enum import Enum

from .features import IFeatureModel
from ..utils import ParameterHandling, LBPLogger


@dataclass 
class IPredictionModel(ParameterHandling, ABC):
    """
    Abstract interface for prediction models.
    
    Defines the structure for training and predicting performance metrics y
    based on input parameters X. Uses the same parameter handling approach
    as FeatureModel and EvaluationModel for consistency.
    
    Data Responsibility Boundary:
    - DataInterface: Handles structured study/experiment metadata and parameters
    - PredictionModel: Loads domain-specific, unstructured data (geometry graph, sensor data, etc.)
    """

    # Nested enum class for dataset types
    class DatasetType(Enum):
        """Enum defining the type of dataset a prediction model uses for training."""
        AGGR_METRICS = "avg_feature"  # Uses aggregated performance metrics
        METRIC_ARRAYS = "feature_array"      # Uses granular metric arrays

    def __init__(self,
                 performance_codes: List[str],
                 logger: LBPLogger,
                 round_digits: int = 3,
                 **kwargs) -> None:
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

        # Validate input keys this model requires for prediction
        if not self.input or not isinstance(self.input, list):
            raise ValueError("Model must declare a list of input keys (X) to predict")

        # Set output performance codes this model predicts and validate
        self.output = performance_codes
        if not self.output or not isinstance(self.output, list):
            raise ValueError("Model must specify a list of performance codes (y) to predict")

        # Initialize the model types for the extraction of additional input features
        self.feature_models: Dict[str, IFeatureModel] = {}

        # Store kwargs so that they can be passed on to the feature models
        self.kwargs = kwargs

    # === ABSTRACT PROPERTIES ===
    @property
    @abstractmethod
    def input(self) -> List[str]:
        """
        Define all input keys this model requires for prediction.
        
        This defines the complete input interface X. Can include:
        - Parameter names (from dataclass fields using parameter_handler)
        - External data keys (loaded via _load_data method)
        
        Returns:
            List of input keys this model expects in X
            e.g., ["n_layers", "temperature", "geometry_mesh", "sensor_data"]
        """
        ...

    @property
    @abstractmethod
    def dataset_type(self) -> DatasetType:
        """
        Define whether this model trains on aggregated metrics or metric arrays.
        
        Returns:
            DatasetType.AGGR_METRICS: Model uses aggregated performance metrics
            DatasetType.METRIC_ARRAYS: Model uses granular metric arrays
        """
        ...

    # === OPTIONAL PROPERTIES ===
    @property
    def feature_model_types(self) -> Dict[str, Type[IFeatureModel]]:
        """
        Set feature model types to use for feature extraction.
        
        This property can be overridden to specify which feature models
        should be used for this prediction model.
        
        Returns:
            Dictionary of feature model types {feature_code: FeatureModelClass}
            e.g., {"energy_consumption": EnergyFeature, "path_deviation": PathDeviationFeature}
        """
        return {}
    
    # === ABSTRACT METHODS ===
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the prediction model on processed data.
        
        Args:
            X: Input features, shape (n_samples, n_features)
               Order matches self.input declaration
            y: Target variables for training, shape (n_samples, n_targets)
               Order matches self.output declaration
               
        Returns:
            Dictionary containing training metrics:
            {
                "training_score": float,        # Primary metric (R² for regression, accuracy for classification)
                "training_samples": int         # Number of training samples
            }
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

    # === OPTIONAL METHODS ===
    def preprocess(self, 
                   preprocessed_X: Dict[str, Any], 
                   preprocessed_y: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Model-specific preprocessing after global preprocessing.
        
        Args:
            preprocessed_X: Globally preprocessed input features
            preprocessed_y: Globally preprocessed target variables
        
        Returns:
            Tuple of (model_specific_X, model_specific_y)
        """
        return preprocessed_X, preprocessed_y
    
    # === PUBLIC API METHODS ===
    @final
    def add_feature_model(self, code: str, feature_model: IFeatureModel) -> None:
        """Add feature model to prediction model."""
        # Append the feature model instance (one-to-many relationship)
        self.feature_models[code] = feature_model
    


