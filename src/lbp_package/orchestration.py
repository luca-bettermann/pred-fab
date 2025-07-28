import yaml
from importlib import import_module
from typing import Any, Dict, List, Type, Tuple, Optional

from .utils.folder_navigator import FolderNavigator
from .utils.log_manager import LBPLogger
from .evaluation import EvaluationModel
from .data_interface import DataInterface

# TODO
# Move all interfaces into interfaces.py file
# Then we have a clean separation of orchestration and interfaces

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
        logger: LBPLogger
    ) -> None:
        """
        Initialize evaluation system.
        
        Args:
            folder_navigator: File system navigation utility
            data_interface: Database interface for data access
            logger: Logger instance for debugging and monitoring
        """
        self.nav = folder_navigator
        self.interface = data_interface
        self.logger = logger
        self.evaluation_models = {}

    # === PUBLIC API METHODS (Called externally) ===
    def add_evaluation_model(self, evaluation_class: Type[EvaluationModel], performance_code: str, study_params: Dict[str, Any]) -> None:
        """
        Add an evaluation model to the system.
        
        Args:
            evaluation_class: Class of evaluation model to instantiate
            performance_code: Code identifying the performance metric
            study_params: Study parameters for model configuration
        """
        self.logger.info(f"Adding '{evaluation_class.__name__}' model to evaluate performance '{performance_code}'")
        eval_model = evaluation_class(
            performance_code,
            folder_navigator=self.nav,
            logger=self.logger,
            **study_params
        )
        self.evaluation_models[performance_code] = eval_model

    def add_feature_model_instances(self, study_params: Dict[str, Any]) -> None:
        """
        Create feature model instances for evaluation models.
        
        Optimizes by sharing feature model instances where possible.
        
        Args:
            study_params: Study parameters for feature model configuration
        """
        feature_model_dict = {}

        for eval_model in self.evaluation_models.values():
            feature_model_type = eval_model.feature_model_type
            
            # Share feature model instances of the same type
            if feature_model_type not in feature_model_dict:
                eval_model.feature_model = feature_model_type(
                    performance_code=eval_model.performance_code, 
                    folder_navigator=eval_model.nav, 
                    logger=self.logger, 
                    round_digits=eval_model.round_digits,
                    **study_params
                )
                feature_model_dict[feature_model_type] = eval_model.feature_model
                self.logger.info(f"Adding feature model instance '{type(eval_model.feature_model).__name__}' to evaluation model '{type(eval_model).__name__}'")
            else:
                # Reuse existing feature model instance
                eval_model.feature_model = feature_model_dict[feature_model_type]
                eval_model.feature_model.initialize_for_performance_code(eval_model.performance_code)
                self.logger.info(f"Reusing existing feature model instance '{type(eval_model.feature_model).__name__}' for evaluation model '{type(eval_model).__name__}'")
            
    def run(self, study_record: Dict[str, Any], exp_nr: int, exp_record: Dict[str, Any], visualize_flag: bool = False, debug_flag: bool = True, **exp_params) -> None:
        """
        Execute evaluation for all models.
        
        Args:
            exp_nr: Experiment number
            exp_record: Experiment record from database
            visualize_flag: Whether to show visualizations
            debug_flag: Whether to run in debug mode (no writing/saving)
            **exp_params: Experiment parameters
        """
        self.logger.info(f"Running evaluation system for experiment {exp_nr}")
        
        # Initialize all models before execution
        self._model_exp_initialization(**exp_params)

        # Execute each evaluation model
        for performance_code, eval_model in self.evaluation_models.items():
            self.logger.console_info(f"Running evaluation for '{performance_code}' performance with '{type(eval_model).__name__}' evaluation model...")
            eval_model.run(exp_nr, visualize_flag, debug_flag, **exp_params)
            self.logger.info(f"Finished evaluation for '{performance_code}' with '{type(eval_model).__name__}' model.")

            # Push results to database if not in debug mode
            if not debug_flag:
                self.logger.info(f"Pushing results to database for '{performance_code}'...")
                self.interface.push_to_database(exp_record, performance_code, eval_model.performance_metrics)
            else:
                self.logger.info(f"Debug mode: Skipping database push for '{performance_code}'")

        self.logger.console_info("All evaluations completed successfully.")

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
            self.logger.info(f"Initializing arrays of evaluation model '{eval_model.performance_code}' and its feature model '{type(eval_model.feature_model).__name__}'")

            # Configure models with experiment parameters
            eval_model.set_experiment_parameters(**exp_params)
            eval_model.feature_model.set_experiment_parameters(**exp_params)

            # Initialize arrays with correct dimensions
            dim_sizes = eval_model._compute_dim_sizes()
            eval_model.reset_for_new_experiment(dim_sizes)
            eval_model.feature_model.reset_for_new_experiment(eval_model.performance_code, dim_sizes)



class LBPManager:
    """
    Main orchestration class for the Learning by Printing system.
    
    Manages the complete workflow including study initialization,
    evaluation execution, and coordination between different subsystems.
    """
    
    def __init__(
            self, 
            root_folder: str,
            local_folder: str, 
            log_folder: str, 
            data_interface: DataInterface,
            server_folder: Optional[str] = None
            ):
        """
        Initialize LBP Manager.
        
        Args:
            local_folder: Path to local data storage
            server_folder: Path to server data storage
            log_folder: Path to log file storage
            data_interface: Interface for database operations
        """
        self.logger_name = "LBPManager"
        self.logger = LBPLogger(self.logger_name, log_folder)
        self.logger.info("Initializing LBP Manager")
        self.interface = data_interface
        
        # Initialize file system navigator
        self.nav = FolderNavigator(root_folder, local_folder, server_folder)
        
        # Study and experiment state
        self.study_code = None
        self.study_record = None
        self.exp_record = None
        self.study_params: Dict = {}
        
        # Configuration storage
        self.config: Dict[str, Any] = {}
        self.evaluation_config: Dict[str, str] = {}
        self.prediction_config: Dict[str, str] = {}
        self.calibration_config: Dict[str, str] = {}
        self.system_config: Dict[str, Any] = {}
        
        # System defaults (loaded from config)
        self.default_debug_mode: bool = False
        self.default_visualize_flag: bool = False
        self.default_round_digits: int = 3
        
        # Performance mapping
        self.performance_mapping: Dict[str, Type[EvaluationModel]] = {}

        # Subsystem components
        self.eval_system = None
        self.pred_system = None

    # === PUBLIC API METHODS (Called externally) ===
    def initialize_study(self, study_code: str, debug_flag: bool = False) -> None:
        """
        Initialize the system for a specific study.
        
        Args:
            study_code: Unique identifier for the study
            debug_flag: Whether to run in debug mode
        """
        self.logger.console_info(f"\n------- Study Initialization: '{study_code}' -------")
        
        # Configure study context
        self.study_code = study_code
        self.nav.set_study_code(study_code)
        self.logger.info(f"Folder navigator initialized for study '{study_code}'")
        
        # Load configuration file
        self._load_config()
        
        # Load study data from database
        self.study_record = self.interface.get_study_record(study_code)
        self.logger.info(f"Study record retrieved from database")
        
        # Validate study parameters
        self.study_params = self.interface.get_study_parameters(self.study_record)
        if not self.study_params or not isinstance(self.study_params, dict):
            raise ValueError(f"Study parameters for study code '{study_code}' empty or not a dict. Please check the database configuration.")
        
        # Load performance configuration
        performance_records = self.interface.get_performance_records(self.study_record)
        if not performance_records or not isinstance(performance_records, list):
            raise ValueError(f"Performance records for study code '{study_code}' empty or not a list. Please check the database configuration.")

        # Initialize evaluation system
        self.eval_system = EvaluationSystem(self.nav, self.interface, self.logger)
        
        # Load and configure evaluation models
        self.performance_mapping = self._load_performance_mapping()
        for performance_record in performance_records:
            code = performance_record.get("Code")
            assert code is not None, f"Performance record missing 'Code' field: {performance_record}"
            assert code in self.performance_mapping, f"Performance code '{code}' not found in performance mapping."

            evaluation_class = self.performance_mapping[code]
            if evaluation_class is not None:
                self.eval_system.add_evaluation_model(
                        evaluation_class,
                        code,
                        self.study_params
                    )
            else:
                self.logger.warning(f"Performance mapping '{code}' is None, skipping evaluation model creation.")

        # Initialize feature model instances
        self.eval_system.add_feature_model_instances(self.study_params)

        # Display initialization summary
        summary = self._initialization_step_summary()
        self.logger.console_summary(summary)
        self.logger.console_success(f"Successfully initialized evaluation system of study '{study_code}'.")

    def run_evaluation(self, exp_nr: int, visualize_flag: Optional[bool] = None, debug_flag: Optional[bool] = None) -> None:
        """
        Execute one iteration of the learning loop.
        
        Args:
            exp_nr: Experiment number to process
            visualize_flag: Whether to show visualizations (uses system default if None)
            debug_flag: Whether to run in debug mode (uses system default if None)
        """
        if not self.eval_system or not self.study_record:
            raise RuntimeError("Evaluation system is not initialized. Please call initialize_study() first.")

        # Use system defaults if not explicitly provided
        actual_debug_flag = debug_flag if debug_flag is not None else self.default_debug_mode
        actual_visualize_flag = visualize_flag if visualize_flag is not None else self.default_visualize_flag
        
        self.logger.info(f"Execution flags: debug_mode={actual_debug_flag}, visualize_flag={actual_visualize_flag}")

        # Configure debug mode if needed
        if actual_debug_flag and not self.logger.debug_mode:
            self.logger.switch_to_debug_mode()

        exp_code = self.nav.get_experiment_code(exp_nr)
        self.logger.console_info(f"\n------- Run Experiment: '{exp_code}' -------")

        # Load experiment data
        self.exp_record = self.interface.get_exp_record(exp_code)
        self.logger.info(f"Experiment record retrieved from database")
        
        # Merge study and experiment parameters
        exp_vars = self.interface.get_exp_variables(self.exp_record)
        exp_vars.update(self.study_params)

        # Execute evaluation pipeline
        self.eval_system.run(
            study_record=self.study_record,
            exp_nr=exp_nr, 
            exp_record=self.exp_record, 
            visualize_flag=actual_visualize_flag, 
            debug_flag=actual_debug_flag,
            **exp_vars
        )

        # Display evaluation summary
        summary = self._evaluation_step_summary()
        self.logger.console_summary(summary)
        self.logger.console_success(f"Successfully evaluated the performance attributes of study '{self.study_code}'.")

    # === PRIVATE/INTERNAL METHODS (Internal use only) ===
    def _load_config(self) -> None:
        """
        Load configuration from config.yaml file and organize into sections.
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file has invalid YAML syntax
            ValueError: If config structure is invalid
        """
        self.logger.info("Loading configuration from config.yaml file")
        config_path = self.nav.root_folder + "/config.yaml"

        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_path}. Please ensure the file exists.")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")

        if not isinstance(self.config, dict):
            raise ValueError("Configuration file must contain a valid dictionary structure.")

        # Extract and store configuration sections
        self.evaluation_config = self.config.get('evaluation', {})
        self.prediction_config = self.config.get('prediction', {})
        self.calibration_config = self.config.get('calibration', {})
        self.system_config = self.config.get('system', {})
        
        # Load system defaults
        self.default_debug_mode = self.system_config.get('debug_mode', False)
        self.default_visualize_flag = self.system_config.get('visualize_flag', False)
        self.default_round_digits = self.system_config.get('round_digits', 3)
        
        self.logger.info(f"Loaded configuration with {len(self.evaluation_config)} evaluation models, "
                        f"{len(self.prediction_config)} prediction models, "
                        f"{len(self.calibration_config)} calibration models, "
                        f"and {len(self.system_config)} system settings")
        self.logger.info(f"System defaults: debug_mode={self.default_debug_mode}, "
                        f"visualize_flag={self.default_visualize_flag}, "
                        f"round_digits={self.default_round_digits}")

    def _load_performance_mapping(self) -> Dict[str, Type[EvaluationModel]]:
        """
        Load performance-to-evaluation-model mapping from stored configuration.
        
        Returns:
            Dictionary mapping performance codes to evaluation model classes
        """
        self.logger.info("Loading performance mapping from stored configuration")

        if not self.evaluation_config:
            raise ValueError("No performance to class mappings found in the evaluation config.")

        mapping = {}

        # Dynamically import evaluation model classes
        for code, class_path in self.evaluation_config.items():
            try:
                module_path, class_name = class_path.rsplit('.', 1)
                module = import_module(module_path)
                evaluation_class = getattr(module, class_name)
                mapping[code] = evaluation_class
                self.logger.debug(f"Loaded evaluation class '{class_name}' for performance code '{code}'")
            except (ValueError, ImportError, AttributeError) as e:
                raise ImportError(f"Error loading class '{class_path}' for performance code '{code}': {e}")
            
        return mapping
    
    def _initialization_step_summary(self) -> str:
        """Generate summary of initialization results."""
        assert self.eval_system is not None
        
        summary = f"Loaded {len(self.performance_mapping)} performance attributes, {len(self.eval_system.evaluation_models)} evaluation models and {len(set([type(e.feature_model).__name__ for e in self.eval_system.evaluation_models.values()]))} feature models.\n\n"
        summary += f"\033[1m{'Performance Code':<20} {'Evaluation Model':<20} {'Feature Model':<20}\033[0m"
        
        for code, eval_model in self.eval_system.evaluation_models.items():
            summary += f"\n{code:<20} {type(eval_model).__name__:<20} {type(eval_model.feature_model).__name__:<20}"
        return summary + "\n"

    def _evaluation_step_summary(self) -> str:
        """Generate summary of evaluation results."""
        assert self.eval_system is not None

        summary = f"Evaluated {len(self.performance_mapping)} performance attributes, using {len(self.eval_system.evaluation_models)} evaluation models and {len(set([type(e.feature_model).__name__ for e in self.eval_system.evaluation_models.values()]))} feature models.\n\n"
        summary += f"\033[1m{'Performance Code':<20} {'Evaluation Model':<20} {'Dimensions':<30} {'Performance Value':<20}\033[0m"

        for code, eval_model in self.eval_system.evaluation_models.items():
            dimensions = ', '.join([f"{name}: {size}" for name, size in zip(eval_model.dim_names, eval_model._compute_dim_sizes())])
            dimensions = "(" + dimensions + ")"
            summary += f"\n{code:<20} {type(eval_model).__name__:<20} {dimensions:<30} {eval_model.performance_metrics.get('Value', 0):<20}"
        return summary + "\n"


