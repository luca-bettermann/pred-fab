from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Callable, Optional, final
from dataclasses import dataclass
import numpy as np

from ..utils import LBPLogger


@dataclass
class ICalibrationModel(ABC):
    """
    Abstract interface for calibration models.
    
    Implements parameter optimization using prediction and evaluation models.
    Models declare their parameters as dataclass fields (DataObjects).
    """
    
    logger: LBPLogger  # Required field for logging
    performance_weights: Optional[Dict[str, float]] = None  # Set via set_performance_weights()
    
    def __post_init__(self):
        """Initialize performance_weights if not provided."""
        if self.performance_weights is None:
            self.performance_weights = {}

    # === ABSTRACT METHODS ===
    @abstractmethod
    def optimize(self, param_ranges: Dict[str, Tuple[float, float]], 
                 objective_fn: Callable[[Dict[str, float]], float]) -> Dict[str, float]:
        """
        Find optimal parameters by minimizing objective_fn.
        
        Args:
            param_ranges: {param_name: (min_val, max_val)}  
            objective_fn: Function that takes parameters dict, returns objective value (higher = better)
            
        Returns:
            Best parameters found: {param_name: optimal_value}
            
        Example:
            def optimize(self, param_ranges, objective_fn):
                from scipy.optimize import minimize
                # Use any optimization approach you want!
                result = minimize(lambda x: -objective_fn(dict(zip(param_ranges.keys(), x))), 
                                bounds=list(param_ranges.values()))
                return dict(zip(param_ranges.keys(), result.x))
        """
        ...

    
    # === PUBLIC API ===
    
    @final
    def calibrate(
        self, 
        param_ranges: Dict[str, Tuple[float, float]],
        objective_fn: Callable[[Dict[str, float]], float]
    ) -> Dict[str, float]:
        """
        Run calibration optimization.
        
        Args:
            param_ranges: {param_name: (min_val, max_val)}
            objective_fn: Function that takes parameters dict, returns objective value (higher = better)
            
        Returns:
            Optimal parameters found: {param_name: optimal_value}
        """
        self.logger.info("Starting calibration")
        self._eval_count = 0
        
        # Call user's optimization implementation
        best_params = self.optimize(param_ranges, objective_fn)
        
        # Validate return type
        if not isinstance(best_params, dict):
            raise TypeError(
                f"optimize() must return dict, "
                f"got {type(best_params).__name__}"
            )
        
        # Validate all expected parameters are present
        expected_params = set(param_ranges.keys())
        returned_params = set(best_params.keys())
        
        missing = expected_params - returned_params
        if missing:
            raise ValueError(
                f"optimize() missing parameters in return: {missing}. "
                f"Expected all parameters from param_ranges: {expected_params}"
            )
        
        # Validate dict values are numeric
        for key, value in best_params.items():
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"optimize() parameter values must be numeric, "
                    f"got {type(value).__name__} for parameter '{key}'"
                )
        
        self.logger.info(f"Calibration complete: {self._eval_count} evaluations, best: {best_params}")
        return best_params
    
    @final
    def set_performance_weights(self, weights: Dict[str, float]) -> None:
        """Set performance weights for objective function."""
        self.performance_weights = weights
        self.logger.info(f"Performance weights set: {weights}")

