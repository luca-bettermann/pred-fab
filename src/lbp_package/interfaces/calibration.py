from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Callable
from dataclasses import dataclass

from ..utils import ParameterHandling, LBPLogger


@dataclass 
class ICalibrationModel(ParameterHandling, ABC):
    """
    Abstract interface for optimization-based calibration models.
    
    Defines the structure for finding optimal input parameters X that maximize
    the weighted objective function of multiple performance evaluations.
    
    The calibration workflow:
    1. User defines parameter ranges and performance weights
    2. CalibrationModel explores parameter space using optimizer
    3. For each parameter combination: predict -> evaluate -> objective
    4. Returns optimal parameters for upcoming experiment
    """

    def __init__(self,
                 logger: LBPLogger,
                 **kwargs):
        """
        Initialize calibration model.
        
        Args:
            logger: Logger instance
            **optimizer_params: Optimizer-specific parameters
        """
        self.logger = logger
        
        # Performance weights will be set during calibration from EvaluationModels
        self.performance_weights: Dict[str, float] = {}
        
        # Store kwargs for optimizer initialization
        self.kwargs = kwargs

    # === NO ABSTRACT PROPERTIES ===
    # Users handle optimizer instantiation themselves to avoid property issues

    # === ABSTRACT METHODS ===
    @abstractmethod
    def calibrate(self, 
                  exp_code: str,
                  predict_fn: Callable[[Dict[str, float]], Dict[str, float]],
                  evaluate_fn: Callable[[str, Dict[str, float]], Dict[str, float]], 
                  param_keys: List[str],
                  param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Find optimal parameters for upcoming experiment.
        
        Args:
            exp_code: Experiment code for the upcoming experiment
            predict_fn: Function that takes parameters and returns predicted features
                       Signature: predict_fn(params) -> {feature_code: predicted_value}
            evaluate_fn: Function that takes exp_code and features, returns performances
                        Signature: evaluate_fn(exp_code, features) -> {perf_code: perf_value}
            param_keys: List of parameter names to optimize
            param_ranges: Parameter bounds {param_name: (min_val, max_val)}
            
        Returns:
            Dictionary of optimal parameters {param_name: optimal_value}
        """
        ...

    # === PREDEFINED METHODS ===
    def set_performance_weights(self, weights: Dict[str, float]) -> None:
        """
        Set performance weights from EvaluationModels.
        
        Args:
            weights: Dictionary mapping performance codes to weights
        """
        self.performance_weights = weights
        self.logger.info(f"Set performance weights: {weights}")

    def objective_function(self, evaluations: Dict[str, float]) -> float:
        """
        Combine multiple performance evaluations into single objective value.
        
        Uses weighted sum approach where higher values are better.
        
        Args:
            evaluations: Dictionary {performance_code: performance_value}
            
        Returns:
            Weighted objective value (higher is better)
        """
        if not self.performance_weights:
            raise ValueError("Performance weights not set. Call set_performance_weights() first.")
            
        weighted_sum = 0.0
        for perf_code, value in evaluations.items():
            if perf_code not in self.performance_weights:
                self.logger.warning(f"No weight specified for performance code '{perf_code}', using weight=1.0")
                weight = 1.0
            else:
                weight = self.performance_weights[perf_code]
            weighted_sum += weight * value
            
        return weighted_sum

    # === HELPER METHODS ===
    def _predict_and_evaluate(self, 
                             params: Dict[str, float], 
                             exp_code: str,
                             predict_fn: Callable, 
                             evaluate_fn: Callable) -> Dict[str, float]:
        """
        Helper method: predict features then evaluate performances.
        
        Args:
            params: Parameter values to evaluate
            exp_code: Experiment code
            predict_fn: Prediction function
            evaluate_fn: Evaluation function
            
        Returns:
            Dictionary of performance evaluations {perf_code: perf_value}
        """
        self.logger.debug(f"Calibration: Evaluating parameters {params}")
        
        # 1. Predict features using current parameters
        predicted_features = predict_fn(params)
        self.logger.debug(f"Calibration: Predicted features {predicted_features}")
        
        # 2. Evaluate predicted features to get performances
        performances = evaluate_fn(exp_code, predicted_features)
        self.logger.debug(f"Calibration: Performance evaluations {performances}")
        
        return performances
    
    def _evaluate_objective(self, 
                           params: Dict[str, float], 
                           exp_code: str,
                           predict_fn: Callable, 
                           evaluate_fn: Callable) -> float:
        """
        Helper method: get objective value for parameter combination.
        
        Args:
            params: Parameter values to evaluate
            exp_code: Experiment code
            predict_fn: Prediction function
            evaluate_fn: Evaluation function
            
        Returns:
            Objective function value (higher is better)
        """
        performances = self._predict_and_evaluate(params, exp_code, predict_fn, evaluate_fn)
        objective_value = self.objective_function(performances)
        self.logger.debug(f"Calibration: Objective value = {objective_value}")
        return objective_value
