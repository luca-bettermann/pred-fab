from typing import Any, Dict, Type, Optional
import numpy as np

from ..utils import FolderNavigator, LBPLogger
from ..interfaces.evaluation import EvaluationModel
from ..interfaces.data import DataInterface


class EvaluationSystem:
    """
    Orchestrates multiple evaluation models for a complete performance assessment.
    
    Manages the execution of evaluation models, handles database interactions,
    and coordinates the overall evaluation workflow.
    """
    
    def __init__(
        self,
        folder_navigator: FolderNavigator,
        data_interface: DataInterface,
        logger: LBPLogger):
        """
        Initialize evaluation system.
        
        Args:
            folder_navigator: File system navigation utility
            data_interface: Database interface for data access
            logger: Logger instance for debugging and monitoring
        """
        self.nav: FolderNavigator = folder_navigator
        self.interface: DataInterface = data_interface
        self.logger: LBPLogger = logger

        self.evaluation_models: Dict[str, EvaluationModel] = {}
        self.performance_metrics: Dict[str, Dict[str, Optional[np.floating]]] = {}

        self.study_params: Dict[str, Any] = {}

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
        assert code in self.evaluation_models, f"No evaluation model for performance code '{code}' has been initialized."
        self.evaluation_models[code].active = True
        self.evaluation_models[code].set_model_parameters(**study_params)
        self.logger.info(f"Activated evaluation model for performance code '{code}' and set study parameters.")

    def run(self, study_record: Dict[str, Any], exp_code: str, exp_record: Dict[str, Any], visualize_flag: bool = False, debug_flag: bool = True, **exp_params) -> None:
        """
        Execute evaluation for all models.
        
        Args:
            exp_nr: Experiment number
            exp_record: Experiment record from database
            visualize_flag: Whether to show visualizations
            debug_flag: Whether to run in debug mode (no writing/saving)
            **exp_params: Experiment parameters
        """
        # Make sure at least one evaluation model has been activated
        active_models = {code: model for code, model in self.evaluation_models.items() if model.active}
        assert len(active_models) > 0, "No evaluation models have been activated."
        self.logger.info(f"Running evaluation system for experiment {exp_code}")

        # Initialize all models before execution
        self._model_exp_initialization(**exp_params)

        # Execute each evaluation model
        for performance_code, eval_model in active_models.items():
            self.logger.console_info(f"Running evaluation for '{performance_code}' performance with '{type(eval_model).__name__}'...")
            exp_folder = self.nav.get_experiment_folder(exp_code)

            # Run evaluation and return dict with "Value", "Performance", "Robustness" and "Resilience" as keys
            aggr_performance_metrics = eval_model.run(exp_code, exp_folder, visualize_flag, debug_flag, **exp_params)
            self.performance_metrics[performance_code] = aggr_performance_metrics

        self.logger.console_info("All evaluations completed successfully.")

        # Push results to database if not in debug mode
        if not debug_flag:
            self.logger.info(f"Pushing results to database for {self.performance_metrics.keys()}...")
            self.interface.push_to_database(exp_record, self.performance_metrics)
        else:
            self.logger.info(f"Debug mode: Skipping database push for {self.performance_metrics.keys()}")

        # Update system-wide performance metrics
        if not debug_flag:
            self.logger.info("Updating system performance...")
            self.interface.update_system_performance(study_record)
        else:
            self.logger.info("Debug mode: Skipping system performance update")

    # === PRIVATE/INTERNAL METHODS (Internal use only) ===
    def _model_exp_initialization(self, **exp_params) -> None:
        """
        Initialize all evaluation models for the current experiment.
        
        Args:
            **exp_params: Experiment parameters
        """
        for eval_model in self.evaluation_models.values():
            assert eval_model.feature_model is not None, f"Feature model for evaluation '{eval_model.performance_code}' is not set."
            self.logger.info(f"Initializing arrays of evaluation model '{eval_model.performance_code}' and its feature model '{type(eval_model.feature_model).__name__}'")
            
            # Configure models with experiment parameters
            eval_model.set_experiment_parameters(**exp_params)
            eval_model.feature_model.set_experiment_parameters(**exp_params)

            # Initialize arrays with correct dimensions
            dim_sizes = eval_model._compute_dim_sizes()
            eval_model.reset_for_new_experiment(dim_sizes)
            eval_model.feature_model.reset_for_new_experiment(eval_model.performance_code, dim_sizes)

