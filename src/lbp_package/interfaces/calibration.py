from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass

from ..utils import ParameterHandling, LBPLogger


class ICalibrationModel(ParameterHandling, ABC):
    """
    Simple calibration interface: just implement optimize() method.
    Interface handles evaluation orchestration, you handle optimization logic.
    """

    def __init__(self, logger: LBPLogger, **kwargs):
        self.logger = logger
        self.performance_weights: Dict[str, float] = {}

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

    def calibrate(self, exp_code: str, predict_fn: Callable, evaluate_fn: Callable, 
                  param_keys: List[str], param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Template method: creates objective function and calls user's optimize() method.
        Interface handles evaluation orchestration, user has complete optimization freedom.
        """
        self.logger.info(f"Starting calibration for {exp_code}")
        
        # Store context for objective function
        self._exp_code = exp_code
        self._predict_fn = predict_fn  
        self._evaluate_fn = evaluate_fn
        self._eval_count = 0
        
        # Let user optimize however they want!
        best_params = self.optimize(param_ranges, self._objective_function)
        
        self.logger.info(f"Calibration complete: {self._eval_count} evaluations, best: {best_params}")
        return best_params
    
    def _objective_function(self, params: Dict[str, float]) -> float:
        """
        Objective function that handles predict→evaluate→objective pipeline.
        Called by user's optimization algorithm.
        """
        try:
            self._eval_count += 1
            obj_val = self._evaluate_objective(params, self._exp_code, self._predict_fn, self._evaluate_fn)
            self.logger.debug(f"Eval {self._eval_count}: {params} -> {obj_val:.6f}")
            return obj_val
        except Exception as e:
            self.logger.warning(f"Evaluation failed for {params}: {e}")
            return float('-inf')  # Return very bad objective for failed evaluations

    def set_performance_weights(self, weights: Dict[str, float]) -> None:
        """Set performance weights for objective function."""
        self.performance_weights = weights

    def _evaluate_objective(self, params: Dict[str, float], exp_code: str, 
                           predict_fn: Callable, evaluate_fn: Callable) -> float:
        """Evaluate objective: predict → evaluate → weighted sum."""
        features = predict_fn(params)
        performances = evaluate_fn(exp_code, features)
        return sum(self.performance_weights.get(k, 1.0) * v for k, v in performances.items())


    def calibration_step_summary(self, exp_code: str, param_ranges: Dict[str, Tuple[float, float]], 
                                  optimal_params: Dict[str, float], evaluation_count: int,
                                  predicted_features: Optional[Dict[str, float]],
                                  performance_values: Optional[Dict[str, float]]) -> str:
        """
        Generate summary of calibration results.
        
        Args:
            exp_code: Experiment code being calibrated
            param_ranges: Parameter ranges used in optimization  
            optimal_params: Optimal parameters found
            evaluation_count: Number of objective function evaluations performed
            predicted_features: Feature values predicted for optimal parameters (optional)
            performance_values: Performance values for optimal parameters (optional)
            
        Returns:
            Formatted summary string for console display
        """
        summary = f"\033[1mCalibration Model: {type(self).__name__}\033[0m\n"
        summary += f"Total Evaluations: {evaluation_count}\n\n"
        
        summary += f"\033[1m{'Parameter':<20} {'Min Value':<15} {'Max Value':<15} {'Optimal Value':<15}\033[0m"
        for param_name, optimal_value in optimal_params.items():
            param_range = param_ranges[param_name]
            min_str = f"{param_range[0]:.3f}"
            max_str = f"{param_range[1]:.3f}"
            summary += f"\n{param_name:<20} {min_str:<15} {max_str:<15} {optimal_value:<15.6f}"
            
        summary += f"\n\n\033[1m{'Performance Code':<20} {'Weight':<15} {'Pred. Feature':<15} {'Pred. Performance':<15}\033[0m"
        for code, weight in self.performance_weights.items():
            feature_val = predicted_features.get(code, 'N/A') if predicted_features else 'N/A'
            perf_val = performance_values.get(code, 'N/A') if performance_values else 'N/A'
            
            # Format values for display
            feature_str = f"{feature_val:.4f}" if isinstance(feature_val, (int, float)) else str(feature_val)
            perf_str = f"{perf_val:.4f}" if isinstance(perf_val, (int, float)) else str(perf_val)
            
            summary += f"\n{code:<20} {weight:<15.3f} {feature_str:<15} {perf_str:<15}"
            
        return summary