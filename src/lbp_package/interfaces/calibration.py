from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Callable, Optional, final
import numpy as np

from ..utils import LBPLogger


class ICalibrationModel(ABC):
    """
    Abstract interface for calibration models.
    
    Implements parameter optimization using prediction and evaluation models.
    """
    
    def __init__(self, logger: LBPLogger):
        self.logger = logger

    # === ABSTRACT METHODS ===
    @abstractmethod
    def optimize(self, param_ranges: Dict[str, Tuple[float, float]], 
                 objective_fn: Callable[[Dict[str, float]], float],
                 fixed_params: Optional[Dict[str, Any]] = None,
                 uncertainty_fn: Optional[Callable[[Dict[str, float]], float]] = None) -> Dict[str, float]:
        """
        Find optimal parameters by minimizing objective_fn.
        
        Args:
            param_ranges: {param_name: (min_val, max_val)} for free parameters
            objective_fn: Function that takes parameters dict, returns objective value (higher = better)
            fixed_params: Optional dictionary of fixed parameter values
            uncertainty_fn: Optional function that returns uncertainty (sigma) for a given parameter set.
                            If provided, the optimizer may use this directly instead of a surrogate model.
            
        Returns:
            Best parameters found: {param_name: optimal_value} (including fixed params)
            
        Example:
            def optimize(self, param_ranges, objective_fn, fixed_params=None, uncertainty_fn=None):
                fixed = fixed_params or {}
                # ... optimization logic ...
                return {**best_free_params, **fixed}
        """
        ...

    
    # === PUBLIC API ===
    @final
    def calibrate(
        self, 
        param_ranges: Dict[str, Tuple[float, float]],
        objective_fn: Callable[[Dict[str, float]], float],
        fixed_params: Optional[Dict[str, Any]] = None,
        uncertainty_fn: Optional[Callable[[Dict[str, float]], float]] = None
    ) -> Dict[str, float]:
        """Run calibration optimization and validate results."""
        self.logger.info("Starting calibration")
        self._eval_count = 0
        
        # Call user's optimization implementation
        best_params = self.optimize(param_ranges, objective_fn, fixed_params, uncertainty_fn)
        
        # Validate return type
        if not isinstance(best_params, dict):
            raise TypeError(
                f"optimize() must return dict, "
                f"got {type(best_params).__name__}"
            )
        
        # Validate all expected parameters are present
        expected_params = set(param_ranges.keys())
        if fixed_params:
            expected_params.update(fixed_params.keys())
            
        returned_params = set(best_params.keys())
        
        missing = expected_params - returned_params
        if missing:
            raise ValueError(
                f"optimize() missing parameters in return: {missing}. "
                f"Expected all parameters from param_ranges and fixed_params: {expected_params}"
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

