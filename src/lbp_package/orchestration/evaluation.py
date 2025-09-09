from typing import Any, Dict, Type, Optional, List, final
import numpy as np

from ..utils import LBPLogger
from ..interfaces.evaluation import IEvaluationModel


class EvaluationSystem:
    """
    Orchestrates multiple evaluation models for a complete performance assessment.
    
    Manages the execution of evaluation models, handles database interactions,
    and coordinates the overall evaluation workflow.
    """
    
    def __init__(
        self,
        logger: LBPLogger):
        """
        Initialize evaluation system.
        
        Args:
            logger: Logger instance for debugging and monitoring

        Structure of memory storage dicts:
            aggr_metrics: {exp_code: {performance_code: {metric_name: value}}}
            metrics_arrays: {exp_code: {performance_code: np.ndarray}}
        """
        self.logger: LBPLogger = logger
        self.evaluation_models: Dict[str, IEvaluationModel] = {}

        # Storage of metrics in memory 
        self.aggr_metrics: Dict[str, Dict[str, Dict[str, Optional[np.floating]]]] = {}  # dict_keys: exp_code, perf_code
        self.metric_arrays: Dict[str, Dict[str, np.ndarray]] = {}                       # dict_keys: perf_code, exp_code
        self.dim_sizes: Dict[str, Dict[str, List[int]]] = {}                            # dict_keys: perf_code, exp_code

    # === PUBLIC API METHODS ===
    def add_evaluation_model(self, performance_code: str, evaluation_class: Type[IEvaluationModel], round_digits: int, weight: Optional[float] = None, **kwargs) -> None:
        """
        Add an evaluation model to the system.
        
        Args:
            performance_code: Code identifying the performance metric
            evaluation_class: Class of evaluation model to instantiate
            round_digits: Number of digits to round results to
            calibration_weight: Optional weight for calibration objective function
            **kwargs: Additional parameters for model initialization
        """
        # Validate if evaluation_model is the correct type
        if not issubclass(evaluation_class, IEvaluationModel):
            raise TypeError(f"Expected a subclass of EvaluationModel, got {type(evaluation_class).__name__}")

        self.logger.info(f"Adding '{evaluation_class.__name__}' model to evaluate performance '{performance_code}'")
        eval_model = evaluation_class(
            performance_code=performance_code,
            logger=self.logger,
            round_digits=round_digits,
            weight=weight,
            **kwargs
        )
        self.evaluation_models[performance_code] = eval_model

    def activate_evaluation_model(self, code: str, study_params: Dict[str, Any]) -> None:
        # Activate evaluation model and apply dataclass-based parameter handling
        if code not in self.evaluation_models:
            raise ValueError(f"No evaluation model for performance code '{code}' has been initialized.")
        self.evaluation_models[code].active = True
        self.evaluation_models[code].set_study_parameters(**study_params)

        # Initialize metrics array, dim array and dim size storage for this performance code
        if code not in self.metric_arrays:
            self.metric_arrays[code] = {}
        if code not in self.dim_sizes:
            self.dim_sizes[code] = {}

        self.logger.info(f"Activated evaluation model for performance code '{code}' and set study parameters.")

    def initialize_for_exp(self, exp_code, **exp_params) -> None:
        """
        Set experiment parameters for all evaluation models and their feature models.
        
        Args:
            exp_code: Experiment code
            exp_params: Experiment parameters
        """
        # Initialize arrays dictionaries for exp_code, if they have not been loaded yet
        if exp_code not in self.aggr_metrics:
            self.aggr_metrics[exp_code] = {}

        for eval_model in self.evaluation_models.values():
            assert eval_model.feature_model is not None, f"Feature model for evaluation '{eval_model.performance_code}' is not set."
            self.logger.info(f"Set exp parameters for '{eval_model.performance_code}' and '{type(eval_model.feature_model).__name__}'")

            # Configure models with experiment parameters
            eval_model.set_exp_parameters(**exp_params)
            eval_model.feature_model.set_exp_parameters(**exp_params)

            # Initialize feature model arrays with correct dimensions
            dim_sizes = eval_model._get_dim_sizes()
            eval_model.feature_model.reset_for_new_experiment(eval_model.performance_code, dim_sizes)

            # Store dimension sizes for performance code and experiment
            self.dim_sizes[eval_model.performance_code][exp_code] = dim_sizes

    def run(self, exp_code: str, exp_folder: str, visualize_flag: bool = False, debug_flag: bool = True) -> None:
        """
        Execute evaluation for all models.
        
        Args:
            exp_code: Experiment code
            exp_folder: Experiment folder path
            visualize_flag: Whether to show visualizations
            debug_flag: Whether to run in debug mode (no writing/saving)
            **exp_params: Experiment parameters
        """
        # Make sure at least one evaluation model has been activated
        active_models = self.get_active_eval_models()
        if len(active_models) == 0:
            raise ValueError("No evaluation models have been activated.")
        self.logger.info(f"Running evaluation system for experiment {exp_code}")

        # Execute each evaluation model
        for performance_code, eval_model in active_models.items():
            self.logger.info(f"Running evaluation for '{performance_code}' performance with '{type(eval_model).__name__}'...")

            # Run evaluation and return dict with "Value", "Performance", "Robustness" and "Resilience" as keys
            aggr_metrics, metrics_array, dim_array = eval_model.run(exp_code, exp_folder, visualize_flag, debug_flag)
            
            # Store the computed features and performances in memory
            self.aggr_metrics[exp_code][performance_code] = aggr_metrics
            self.metric_arrays[performance_code][exp_code] = metrics_array

        self.logger.console_info("All evaluations completed successfully.")

    def evaluation_step_summary(self, exp_code: str) -> str:
        """Generate summary of evaluation results."""
        active_eval_models = {code: model for code, model in self.evaluation_models.items() if model.active}
        
        summary = f"\n\033[1m{'Performance Code':<20} {'Evaluation Model':<20} {'Dimensions':<30} {'Performance Value':<20}\033[0m"

        for code, eval_model in active_eval_models.items():
            dimensions = ', '.join([f"{name}: {size}" for name, size in zip(eval_model.dim_names, self.dim_sizes[code][exp_code])])
            dimensions = "(" + dimensions + ")"
            summary += f"\n{code:<20} {type(eval_model).__name__:<20} {dimensions:<30} {self.aggr_metrics[exp_code][code].get('Performance_Avg', None):<20}"
        return summary
    
    def invert_dict_structure(self, dict_to_invert: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Invert a nested dictionary structure from {outer_key: {inner_key: value}} to {inner_key: {outer_key: value}}.
        
        Args:
            dict_to_invert: Dictionary to invert

        Returns:
            Inverted dictionary
        """
        inverted_dict: Dict[str, Dict[str, Any]] = {}
        for outer_key, inner_dict in dict_to_invert.items():
            for inner_key, value in inner_dict.items():
                if inner_key not in inverted_dict:
                    inverted_dict[inner_key] = {}
                inverted_dict[inner_key][outer_key] = value
        return inverted_dict
    
    def get_calibration_weights(self) -> Dict[str, float]:
        """
        Get calibration weights from all evaluation models.
        
        Returns:
            Dictionary mapping performance codes to their calibration weights
            
        Raises:
            ValueError: If any active evaluation model lacks calibration weight
        """
        weights = {}
        for code, eval_model in self.get_active_eval_models().items():
            if eval_model.weight is None:
                raise ValueError(f"Evaluation model '{code}' is active but has no calibration weight defined. "
                               f"Set calibration_weight when adding the model to enable calibration.")
            weights[code] = eval_model.weight
        
        if not weights:
            raise ValueError("No active evaluation models with calibration weights found.")
            
        return weights
    
    def get_active_eval_models(self) -> Dict[str, IEvaluationModel]:
        """
        Get all active evaluation models.
        
        Returns:
            Dictionary of active evaluation models {performance_code: model}
        """
        return {code: model for code, model in self.evaluation_models.items() if model.active}