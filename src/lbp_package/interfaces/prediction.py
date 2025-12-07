"""
Prediction Model Interface for AIXD architecture.

Defines abstract interface for prediction models that learn from experiment data
and predict features for new parameter combinations. Features can then be evaluated
to compute performance metrics.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Type, Optional, Any, final, Tuple
import numpy as np

from .base import BaseInterface
from ..utils.logger import LBPLogger
from ..core import DataObject, Dataset


class IPredictionModel(BaseInterface):
    """
    Abstract base class for prediction models.
    
    - Learn parameter→feature relationships from experiment data
    - Predict feature values for new parameter combinations (virtual experiments)
    - Enable feature-based evaluation and multi-objective optimization
    - Support export/import for production inference via InferenceBundle
    - Must be dataclasses with DataObject fields for parameters (schema generation)
    """

    def __init__(self, dataset: Dataset, logger: LBPLogger):
        """Initialize evaluation system."""
        super().__init__(dataset, logger)
    
    # === ABSTRACT METHODS ===

    # abstract methods from BaseInterface:
    # - input_parameters
    # - input_features
    # - outputs

    @abstractmethod
    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass of given parameter values to retrieve features.
        
        Args:
            X: Numpy array with normalized parameter values (batch_size, n_params)
        
        Returns:
            Numpy array with normalized feature values (batch_size, n_features)
        """
        pass

    @abstractmethod
    def train(self, train_batches: List[Tuple[np.ndarray, np.ndarray]], val_batches: List[Tuple[np.ndarray, np.ndarray]], **kwargs) -> None:
        """
        Train the prediction model on batched data.
        
        Args:
            train_batches: List of (X, y) tuples for training
            val_batches: List of (X, y) tuples for validation
            **kwargs: Additional training parameters
        """
        pass

    # === ONLINE LEARNING ===

    def tuning(self, tune_batches: List[Tuple[np.ndarray, np.ndarray]], **kwargs) -> None:
        """
        Fine-tune model with new measurements during fabrication.
        
        Override this method to implement online learning/adaptation.
        Default behavior: Raises NotImplementedError.
        
        Args:
            tune_batches: List of (X, y) tuples for tuning
            **kwargs: Additional tuning parameters
        
        Raises:
            NotImplementedError: If tuning not supported (default behavior)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support tuning. "
            f"Override tuning() method to enable online learning."
        )
    
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

