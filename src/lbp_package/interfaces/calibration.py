from abc import ABC, abstractmethod
from typing import Literal, Callable
import numpy as np

from ..utils import LBPLogger


class ICalibrationStrategy(ABC):
    """
    Strategy interface for calibration optimization.
    
    - Defines how to propose next experiments based on history
    - Supports exploration (active learning) and optimization modes
    """
    
    def __init__(self, logger: LBPLogger, predict: Callable, evaluate: Callable):
        """Initialize strategy with logger."""
        self.logger = logger
        self.predict_fn = predict
        self.evaluate_fn = evaluate

    @abstractmethod
    def propose_next_points(
        self,
        X_history: np.ndarray,
        y_history: np.ndarray,
        bounds: np.ndarray,
        n_points: int = 1,
        mode: Literal['exploration', 'optimization'] = 'exploration',
        **kwargs
    ) -> np.ndarray:
        """
        Propose next parameters to evaluate.
        
        Args:
            X_history: Array of shape (n_samples, n_params) with past parameters
            y_history: Array of shape (n_samples,) with past system performance [0, 1]
            bounds: Array of shape (n_params, 2) with min/max for each parameter
            n_points: Number of points to propose
            mode: 'exploration' (use surrogate uncertainty) or 'optimization' (exploit only)
            
        Returns:
            Array of shape (n_points, n_params) with proposed parameters
        """
        pass

