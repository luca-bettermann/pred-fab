"""
Tuning Interface for Online Adaptation.

Defines the interface for residual models used to correct base model predictions
during online adaptation (tuning).
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Any, Literal
import numpy as np
from sklearn.neural_network import MLPRegressor

from .base_interface import BaseInterface
from ..utils.logger import PfabLogger


class IResidualModel(BaseInterface):
    """
    Abstract base class for residual models.
    
    Residual models learn the error between the base model prediction and the
    observed ground truth.
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, residuals: np.ndarray, **kwargs) -> None:
        """
        Train the residual model on the provided input features and residual errors.
        
        Args:
            X: Input features (process parameters).
            residuals: Target residuals (y_true - y_pred_base).
            **kwargs: Additional training arguments.
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the residual error for the given input features.
        
        Args:
            X: Input features.
            
        Returns:
            Predicted residuals.
        """
        pass


class MLPResidualModel(IResidualModel):
    """
    Residual model implementation using a Multi-Layer Perceptron (MLP).
    
    Uses sklearn.neural_network.MLPRegressor.
    """
    
    def __init__(
        self, 
        logger: PfabLogger,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        activation: Literal['relu', 'identity', 'logistic', 'tanh'] = 'relu',
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        random_state: Optional[int] = None
    ):
        """
        Initialize the MLP residual model.
        
        Args:
            logger: Logger instance.
            hidden_layer_sizes: Tuple of hidden layer sizes.
            activation: Activation function ('relu', 'tanh', 'logistic', 'identity').
            learning_rate_init: Initial learning rate.
            max_iter: Maximum number of iterations.
            random_state: Random seed.
        """
        super().__init__(logger)
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
            solver='adam', # Efficient for larger datasets, 'lbfgs' better for small
            warm_start=True # Allow online updates
        )
        self._is_fitted = False
        
    @property
    def input_parameters(self) -> List[str]:
        """Residual model is dynamic, no fixed schema inputs."""
        return []

    @property
    def input_features(self) -> List[str]:
        """Residual model is dynamic, no fixed schema inputs."""
        return []

    @property
    def outputs(self) -> List[str]:
        """Residual model is dynamic, no fixed schema outputs."""
        return []

    def fit(self, X: np.ndarray, residuals: np.ndarray, **kwargs) -> None:
        """
        Train the residual model.
        
        Args:
            X: Input features.
            residuals: Target residuals.
            **kwargs: Arguments passed to MLPRegressor.fit (e.g. partial_fit logic if needed)
        """
        # For online adaptation, we might want to use partial_fit if we are streaming,
        # but here we assume we get a batch (sliding window) and we might want to retrain 
        # or fine-tune. 
        # If we want to reset weights, we should re-initialize. 
        # If we want to continue learning, we use fit with warm_start=True (set in init).
        
        # However, the user requirement says: "train small model on ...". 
        # Usually for residual learning in control, we might want to overfit the local batch.
        
        self.model.fit(X, residuals)
        self._is_fitted = True
        self.logger.info(f"Residual model fitted on {len(X)} samples.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict residuals.
        
        Returns 0s if not fitted yet (neutral initialization).
        """
        if not self._is_fitted:
            # Return zeros matching the output dimension of what the model WOULD predict
            # We don't know the output dimension until fit is called usually, 
            # but we can infer from X if we assume 1D output or we wait.
            # Actually, we can't easily know the output shape without fitting or explicit config.
            # But usually residuals match the target dimension.
            # For safety, if not fitted, we return 0. 
            # But we need to know the shape. 
            # Let's assume the caller handles the "not fitted" case or we return 0 scalar 
            # which broadcasts if numpy.
            return np.zeros((X.shape[0], 1)) # Assumption: 1D residual per sample? 
            # Or we can return 0 and let numpy broadcast.
            
        return self.model.predict(X)
