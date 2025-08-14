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
    the original scale. Uses a unified approach where all columns are treated
    equally regardless of whether they are inputs or outputs.
    """
    
    # Unified normalization parameters - stores parameters for all columns
    normalization_params: Optional[Dict[str, Optional[Dict[str, float]]]] = None
    
    # Track which normalization method was used for reversal
    normalization_method: str = "standardize"  # "standardize", "minmax", or "none"
    
    def __post_init__(self):
        """Initialize normalization parameters dictionary."""
        if self.normalization_params is None:
            self.normalization_params = {}


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
                 folder_navigator: FolderNavigator,
                 logger: LBPLogger,
                 study_params: Dict[str, Any],
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
        self.nav = folder_navigator
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
        self.output = self._declare_outputs()
        if not self.output or not isinstance(self.output, list):
            raise ValueError("Model must specify a list of performance codes (y) to predict")

        # Store preprocessing state for denormalization during prediction
        self.preprocessing_state: Optional[PreprocessingState] = None

        # Initialize the model types for the extraction of additional input features
        self.feature_model_types: Dict[str, Type[FeatureModel]] = self._declare_feature_model_types()
        self.feature_models: Dict[str, FeatureModel] = {}

        # Set parameters
        self.set_model_parameters(**study_params)
    
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
    def _declare_outputs(self) -> List[str]:
        """
        Declare all output keys this model predicts.
        
        This defines the complete output interface y. Can include:
        - Performance codes (from dataclass fields using parameter_handler)
        
        Returns:
            List of output keys this model predicts
            e.g., ["energy_consumption", "path_deviation"]
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


