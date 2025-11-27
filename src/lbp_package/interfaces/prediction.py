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
    
    def __init__(self, logger: LBPLogger, **kwargs):
        """Initialize prediction model."""
        self.logger = logger
        self._feature_models: Dict[str, IFeatureModel] = {}
    
    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """
        Names of features this model predicts.
        
        Returns:
            List of feature names (e.g., ['filament_width', 'layer_height'])
        """
        pass
    
    @property
    def feature_model_types(self) -> Dict[str, Type[IFeatureModel]]:
        """
        Feature models this prediction model depends on.
        
        Maps feature codes to IFeatureModel classes. The system will create
        shared instances and attach them via add_feature_model().
        
        Returns:
            Dict mapping feature codes to IFeatureModel types
            (e.g., {'path_dev': PathDeviationFeature})
            Empty dict if no feature models needed (default).
        """
        return {}
    
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

    @final
    def add_feature_model(self, code: str, feature_model: IFeatureModel) -> None:
        """Attach feature model instance for use during training/prediction."""
        self._feature_models[code] = feature_model
        if self.logger:
            self.logger.debug(f"Attached feature model '{code}' to prediction model")
    
    # === EXPORT/IMPORT SUPPORT ===
    
    @abstractmethod
    def _get_model_artifacts(self) -> Dict[str, Any]:
        """
        Serialize trained model state for production export.
        
        Return all artifacts needed to restore model: weights, configuration,
        trained objects (sklearn models, neural networks, etc.). All values
        must be picklable. Raise RuntimeError if model not trained.
        
        Returns:
            Dict containing complete model state for reconstruction
            (e.g., {'model': sklearn_model, 'scaler': fitted_scaler})
        """
        pass
    
    @abstractmethod
    def _set_model_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """
        Restore trained model state from exported artifacts.
        
        Reconstruct model to its trained state from the dict returned by
        _get_model_artifacts(). Must perfectly reverse _get_model_artifacts()
        so that round-trip export→import preserves model behavior.
        
        Args:
            artifacts: Dict containing model state (from _get_model_artifacts())
        """
        pass


