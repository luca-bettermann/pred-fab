"""
Prediction Model Interface for AIXD architecture.

Defines abstract interface for prediction models that learn from experiment data
and predict features for new parameter combinations. Features can then be evaluated
to compute performance metrics.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Type, Optional, final
import pandas as pd

from ..utils.logger import LBPLogger
from .features import IFeatureModel


class IPredictionModel(ABC):
    """
    Abstract base class for prediction models.
    
    Prediction models learn relationships between input parameters and output
    features (physical measurements). They enable virtual experimentation by
    predicting feature values for untested parameter combinations.
    
    Features can then be passed to evaluation models to compute performance.
    This separation enables:
    - Interpretable predictions (actual measurements, not just performance scores)
    - Flexible optimization (same predictions, different objectives)
    - Visualization of predicted feature values
    
    IMPORTANT: Models must be dataclasses with:
    - Input parameters declared as DataObject fields (e.g., param_x: DataReal)
    - Output features declared via feature_names property (List[str])
    - Optional feature model dependencies via feature_model_types property
    
    The agent uses dataclass introspection to generate the dataset schema from
    model field declarations. This ensures type safety and automatic validation.
    
    Example:
        @dataclass
        class MyPredictionModel(IPredictionModel):
            # Inputs: declared as DataObject fields
            temperature: DataReal = DataReal("temperature", min_val=0, max_val=100)
            pressure: DataReal = DataReal("pressure", min_val=0, max_val=10)
            
            # Outputs: declared via property
            @property
            def feature_names(self) -> List[str]:
                return ["viscosity", "density"]
            
            def train(self, X, y, **kwargs):
                # Training logic...
                pass
            
            def forward_pass(self, X):
                # Prediction logic...
                return pd.DataFrame({"viscosity": [...], "density": [...]})
    
    Subclasses implement train() and forward_pass() methods to learn parameter→feature mappings.
    """
    
    def __init__(self, logger: LBPLogger, **kwargs):
        """
        Initialize prediction model.
        
        Args:
            logger: Logger instance
            **kwargs: Additional model-specific parameters
        """
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
    
    def add_feature_model(self, code: str, feature_model: IFeatureModel) -> None:
        """
        Attach a feature model instance to this prediction model.
        
        Called by the system to provide feature model dependencies declared
        in feature_model_types. Models can then use these during training/prediction.
        
        Args:
            code: Feature code (as declared in feature_model_types)
            feature_model: IFeatureModel instance to attach
        """
        self._feature_models[code] = feature_model
        if self.logger:
            self.logger.debug(f"Attached feature model '{code}' to prediction model")

    
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
    


