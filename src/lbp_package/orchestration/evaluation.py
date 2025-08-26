from typing import Any, Dict, Type, Optional, List
import numpy as np

from ..utils import LBPLogger
from ..interfaces.evaluation import EvaluationModel


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

        self.evaluation_models: Dict[str, EvaluationModel] = {}

        # Storage of metrics in memory  
        self.aggr_metrics: Dict[str, Dict[str, Dict[str, Optional[np.floating]]]] = {}
        self.metrics_arrays: Dict[str, Dict[str, np.ndarray]] = {}

    # === PUBLIC API METHODS (Called externally) ===
    def add_evaluation_model(self, performance_code: str, evaluation_class: Type[EvaluationModel], round_digits: int, **kwargs) -> None:
        """
        Add an evaluation model to the system.
        
        Args:
            performance_code: Code identifying the performance metric
            evaluation_class: Class of evaluation model to instantiate
            study_params: Study parameters for model configuration
        """
        # Validate if evaluation_model is the correct type
        if not issubclass(evaluation_class, EvaluationModel):
            raise TypeError(f"Expected a subclass of EvaluationModel, got {type(evaluation_class).__name__}")

        self.logger.info(f"Adding '{evaluation_class.__name__}' model to evaluate performance '{performance_code}'")
        eval_model = evaluation_class(
            performance_code=performance_code,
            logger=self.logger,
            round_digits=round_digits,
            **kwargs
        )
        self.evaluation_models[performance_code] = eval_model

    def activate_evaluation_model(self, code: str, study_params: Dict[str, Any]) -> None:
        # Activate evaluation model and apply dataclass-based parameter handling
        if code not in self.evaluation_models:
            raise ValueError(f"No evaluation model for performance code '{code}' has been initialized.")
        self.evaluation_models[code].active = True
        self.evaluation_models[code].set_study_parameters(**study_params)
        self.logger.info(f"Activated evaluation model for performance code '{code}' and set study parameters.")

    def run(self, exp_code: str, exp_folder: str, visualize_flag: bool = False, debug_flag: bool = True, exp_params: dict = {}) -> None:
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
        active_models = {code: model for code, model in self.evaluation_models.items() if model.active}
        if len(active_models) == 0:
            raise ValueError("No evaluation models have been activated.")
        self.logger.info(f"Running evaluation system for experiment {exp_code}")

        # Initialize all models before execution
        self._model_exp_initialization(**exp_params)

        # Initialize arrays dictionaries for exp_code
        self.aggr_metrics[exp_code] = {}
        self.metrics_arrays[exp_code] = {}

        # Execute each evaluation model
        for performance_code, eval_model in active_models.items():
            self.logger.info(f"Running evaluation for '{performance_code}' performance with '{type(eval_model).__name__}'...")

            # Run evaluation and return dict with "Value", "Performance", "Robustness" and "Resilience" as keys
            eval_aggr_metrics, eval_metrics_array = eval_model.run(exp_code, exp_folder, visualize_flag, debug_flag, **exp_params)
            
            # Store the computed features and performances in memory
            self.aggr_metrics[exp_code][performance_code] = eval_aggr_metrics
            self.metrics_arrays[exp_code][performance_code] = eval_metrics_array

        self.logger.console_info("All evaluations completed successfully.")

    def evaluation_step_summary(self, exp_code: str) -> str:
        """Generate summary of evaluation results."""
        active_eval_models = {code: model for code, model in self.evaluation_models.items() if model.active}
        
        summary = f"\n\033[1m{'Performance Code':<20} {'Evaluation Model':<20} {'Dimensions':<30} {'Performance Value':<20}\033[0m"

        for code, eval_model in active_eval_models.items():
            dimensions = ', '.join([f"{name}: {size}" for name, size in zip(eval_model.dim_names, eval_model._compute_dim_sizes())])
            dimensions = "(" + dimensions + ")"
            summary += f"\n{code:<20} {type(eval_model).__name__:<20} {dimensions:<30} {self.aggr_metrics[exp_code][code].get('Performance_Avg', None):<20}"
        return summary + "\n"

    # === PRIVATE/INTERNAL METHODS (Internal use only) ===
    def _model_exp_initialization(self, **exp_params) -> None:
        """
        Initialize all evaluation models for the current experiment.
        
        Args:
            exp_params: Experiment parameters
        """
        for eval_model in self.evaluation_models.values():
            assert eval_model.feature_model is not None, f"Feature model for evaluation '{eval_model.performance_code}' is not set."
            self.logger.info(f"Initializing arrays of evaluation model '{eval_model.performance_code}' and its feature model '{type(eval_model.feature_model).__name__}'")
            
            # Configure models with experiment parameters
            eval_model.set_exp_parameters(**exp_params)
            eval_model.feature_model.set_exp_parameters(**exp_params)

            # Initialize arrays with correct dimensions
            dim_sizes = eval_model._compute_dim_sizes()
            eval_model.feature_model.reset_for_new_experiment(eval_model.performance_code, dim_sizes)

