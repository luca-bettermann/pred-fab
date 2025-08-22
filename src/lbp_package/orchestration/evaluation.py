from typing import Any, Dict, Type, Optional, List
import numpy as np

from ..utils import LocalDataInterface, LBPLogger
from ..interfaces.evaluation import EvaluationModel
from ..interfaces.external_data import ExternalDataInterface


class EvaluationSystem:
    """
    Orchestrates multiple evaluation models for a complete performance assessment.
    
    Manages the execution of evaluation models, handles database interactions,
    and coordinates the overall evaluation workflow.
    """
    
    def __init__(
        self,
        local_data: LocalDataInterface,
        external_data: ExternalDataInterface,
        logger: LBPLogger):
        """
        Initialize evaluation system.
        
        Args:
            local_data: Local file system navigation and operations utility
            external_data: External database interface for data access
            logger: Logger instance for debugging and monitoring
        """
        self.local_data: LocalDataInterface = local_data
        self.external_data: ExternalDataInterface = external_data
        self.logger: LBPLogger = logger

        self.evaluation_models: Dict[str, EvaluationModel] = {}

        # Storage of metrics in memory
        self.aggr_metrics: Dict[str, Dict[str, Optional[np.floating]]] = {}
        self.metrics_arrays: Dict[str, np.ndarray] = {}

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
        self.evaluation_models[code].set_study_parameters(**study_params)
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
            self.logger.info(f"Running evaluation for '{performance_code}' performance with '{type(eval_model).__name__}'...")
            exp_folder = self.local_data.get_experiment_folder(exp_code)

            # Run evaluation and return dict with "Value", "Performance", "Robustness" and "Resilience" as keys
            eval_aggr_metrics, eval_metrics_array = eval_model.run(exp_code, exp_folder, visualize_flag, debug_flag, **exp_params)
            
            # Store the computed features and performances in memory
            self.aggr_metrics[performance_code] = eval_aggr_metrics
            self.metrics_arrays[performance_code] = eval_metrics_array

        self.logger.console_info("All evaluations completed successfully.")

        # Push results to database if not in debug mode
        if not debug_flag:
            self.logger.info(f"Pushing results to database for {self.aggr_metrics.keys()}...")
            self.external_data.push_to_database(exp_record, self.aggr_metrics, self.metrics_arrays)
        else:
            self.logger.info(f"Debug mode: Skipping database push for {self.aggr_metrics.keys()}")

        # Update system-wide performance metrics
        if not debug_flag:
            self.logger.info("Updating system performance...")
            self.external_data.update_system_performance(study_record)
        else:
            self.logger.info("Debug mode: Skipping system performance update")

    def has_experiments_data(self, exp_codes: List[str]) -> List[str]:
        """
        Check which experiments have data in memory.
        
        Args:
            exp_codes: List of experiment codes to check
            
        Returns:
            List of experiment codes that are missing from memory
        """
        missing_exp_codes = []
        active_codes = [code for code, model in self.evaluation_models.items() if model.active]
        
        for exp_code in exp_codes:
            # Check if we have both aggregated metrics and arrays for active models for this experiment
            has_aggr = all(f"{exp_code}_{code}" in self.aggr_metrics for code in active_codes)
            has_arrays = all(f"{exp_code}_{code}" in self.metrics_arrays for code in active_codes)
            
            if not (has_aggr and has_arrays and len(active_codes) > 0):
                missing_exp_codes.append(exp_code)
                
        return missing_exp_codes

    def load_experiments_to_memory(self, exp_codes: List[str], 
                                  aggr_metrics_dict: Dict[str, Dict[str, Any]], 
                                  metrics_arrays_dict: Dict[str, Dict[str, np.ndarray]]) -> None:
        """
        Load experiment data from external sources to memory.
        
        Args:
            exp_codes: List of experiment codes to load
            aggr_metrics_dict: Aggregated metrics by experiment
            metrics_arrays_dict: Metrics arrays by experiment
            exp_params_dict: Experiment parameters by experiment
        """
        self.logger.info(f"Loading experiment data to memory for {len(exp_codes)} experiments")
        
        for exp_code in exp_codes:
            # Load aggregated metrics with exp_code prefix
            if exp_code in aggr_metrics_dict:
                for perf_code, metrics in aggr_metrics_dict[exp_code].items():
                    self.aggr_metrics[f"{exp_code}_{perf_code}"] = metrics
                    
            # Load metrics arrays with exp_code prefix
            if exp_code in metrics_arrays_dict:
                for perf_code, array in metrics_arrays_dict[exp_code].items():
                    self.metrics_arrays[f"{exp_code}_{perf_code}"] = array

    # === PRIVATE/INTERNAL METHODS (Internal use only) ===    # === PRIVATE/INTERNAL METHODS (Internal use only) ===
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
            eval_model.set_exp_parameters(**exp_params)
            eval_model.feature_model.set_exp_parameters(**exp_params)

            # Initialize arrays with correct dimensions
            dim_sizes = eval_model._compute_dim_sizes()
            eval_model.feature_model.reset_for_new_experiment(eval_model.performance_code, dim_sizes)

