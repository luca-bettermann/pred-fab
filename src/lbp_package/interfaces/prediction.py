"""
Prediction Model Interface for AIXD architecture.

Defines abstract interface for prediction models that learn from experiment data
and predict features for new parameter combinations. Features can then be evaluated
to compute performance metrics.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Type, Optional, Any, final
import pandas as pd

from ..utils.logger import LBPLogger
from .features import IFeatureModel


class IPredictionModel(ABC):
    """
    Abstract base class for prediction models.
    
    - Learn parameter→feature relationships from experiment data
    - Predict feature values for new parameter combinations (virtual experiments)
    - Enable feature-based evaluation and multi-objective optimization
    - Support export/import for production inference via InferenceBundle
    - Must be dataclasses with DataObject fields for parameters (schema generation)
    """
    
    def __init__(self, logger: LBPLogger):
        """Initialize prediction model."""
        self.logger = logger
        self.feature_model_dependencies: List[IFeatureModel] = []
    
    @property
    @abstractmethod
    def feature_output_codes(self) -> List[str]:
        """
        Codes of features this model predicts.
        
        Returns:
            List of feature codes (e.g., ['filament_width', 'layer_height'])
        """
        pass
    
    @property
    def feature_input_codes(self) -> List[str]:
        """
        Features required as inputs during prediction.
        
        Lists feature names from the evaluation system that must be provided
        in X during forward_pass(). These are features already computed by
        evaluation models (e.g., ['path_deviation', 'surface_roughness']).
        Framework will NOT automatically compute these - they must be present
        in the input DataFrame.
        
        Returns:
            List of evaluation feature names to use as inputs
            Empty list if no evaluation features needed (default).
        """
        return []
    
    # @property
    # def feature_models_as_input(self) -> Dict[str, Type[IFeatureModel]]:
    #     """
    #     Additional feature models needed as inputs during prediction.
        
    #     Maps feature codes to IFeatureModel classes that compute additional
    #     input features NOT in the evaluation system (e.g., sensor data processors
    #     like temperature/humidity extractors). The system will create shared
    #     instances and attach them via add_feature_model().
        
    #     Returns:
    #         Dict mapping feature codes to IFeatureModel types
    #         (e.g., {'temp_sensor': TemperatureSensorFeature, 'humidity': HumidityFeature})
    #         Empty dict if no additional feature models needed (default).
    #     """
    #     return {}
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> None:
        """
        Train the prediction model on parameter→feature data.
        
        Args:
            X: DataFrame with parameter columns (inputs)
            y: DataFrame with feature columns (outputs to predict)
            **kwargs: Additional training parameters (e.g., learning_rate, epochs, verbose)
                     Allows user implementations to accept custom hyperparameters
        """
        pass
    
    @abstractmethod
    def forward_pass(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Forward pass of given parameter values to retrieve features.
        
        Args:
            X: DataFrame with parameter columns
        
        Returns:
            DataFrame with feature columns (all floats)
        """
        pass
    
    def tuning(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> None:
        """
        Fine-tune model with new measurements during fabrication.
        
        Override this method to implement online learning/adaptation.
        Default behavior: Raises NotImplementedError.
        
        Args:
            X: DataFrame with parameter columns (inputs)
            y: DataFrame with feature columns (new measurements)
            **kwargs: Additional tuning parameters
        
        Raises:
            NotImplementedError: If tuning not supported (default behavior)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support tuning. "
            f"Override tuning() method to enable online learning."
        )

    @final
    def add_feature_model(self, feature_model: IFeatureModel) -> None:
        """Attach feature model instance for use during training/prediction."""
        self.feature_model_dependencies.append(feature_model)
        if self.logger:
            self.logger.debug(f"Attached feature model '{feature_model}' to prediction model")
    
    # === EXPORT/IMPORT SUPPORT ===
    
    def _get_model_artifacts(self) -> Dict[str, Any]:
        """
        Serialize trained model state for production export.
        
        Override this method to enable export to InferenceBundle. Return all
        artifacts needed to restore model: weights, configuration, trained objects
        (sklearn models, neural networks, etc.). All values must be picklable.
        Raise RuntimeError if model not trained.
        
        Returns:
            Dict containing complete model state for reconstruction
            (e.g., {'model': sklearn_model, 'scaler': fitted_scaler})
        
        Raises:
            NotImplementedError: If export not supported (default behavior)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support export. "
            f"Override _get_model_artifacts() and _set_model_artifacts() to enable export."
        )
    
    def _set_model_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """
        Restore trained model state from exported artifacts.
        
        Override this method to enable import from InferenceBundle. Reconstruct
        model to its trained state from the dict returned by _get_model_artifacts().
        Must perfectly reverse _get_model_artifacts() so that round-trip
        export→import preserves model behavior.
        
        Args:
            artifacts: Dict containing model state (from _get_model_artifacts())
        
        Raises:
            NotImplementedError: If import not supported (default behavior)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support import. "
            f"Override _get_model_artifacts() and _set_model_artifacts() to enable import."
        )


