import yaml
from importlib import import_module
from typing import Any, Dict, List, Type, Tuple, Optional

from src.lbp_package.utils.folder_navigator import FolderNavigator
from src.lbp_package.utils.log_manager import LBPLogger
from src.lbp_package.evaluation import EvaluationModel, EvaluationSystem
from src.lbp_package.data_interface import DataInterface

class LBPManager:
    """
    Main orchestration class for the Learning by Printing system.
    
    Manages the complete workflow including study initialization,
    evaluation execution, and coordination between different subsystems.
    """
    
    def __init__(
            self, 
            local_folder: str, 
            server_folder: str, 
            log_folder: str, 
            data_interface: DataInterface
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
        self.nav = FolderNavigator(local_folder, server_folder)
        
        # Study and experiment state
        self.study_code = None
        self.study_record = None
        self.exp_record = None
        self.study_params: Dict = {}
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

    def run_evaluation(self, exp_nr: int, visualize_flag: bool = False, debug_flag: bool = True) -> None:
        """
        Execute one iteration of the learning loop.
        
        Args:
            exp_nr: Experiment number to process
            visualize_flag: Whether to show visualizations
            debug_flag: Whether to run in debug mode (no writing/saving)
        """
        if not self.eval_system:
            raise RuntimeError("Evaluation system is not initialized. Please call initialize_study() first.")

        # Configure debug mode if needed
        if debug_flag and not self.logger.debug_mode:
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
            exp_nr=exp_nr, 
            exp_record=self.exp_record, 
            visualize_flag=visualize_flag, 
            debug_flag=debug_flag,
            **exp_vars
        )

        # Display evaluation summary
        summary = self._evaluation_step_summary()
        self.logger.console_summary(summary)
        self.logger.console_success(f"Successfully evaluated the performance attributes of study '{self.study_code}'.")

    # === PRIVATE/INTERNAL METHODS (Internal use only) ===
    def _load_performance_mapping(self) -> Dict[str, Type[EvaluationModel]]:
        """
        Load performance-to-evaluation-model mapping from configuration.
        
        Returns:
            Dictionary mapping performance codes to evaluation model classes
        """
        self.logger.info("Loading performance mapping from config.yaml file")
        config_path = "config.yaml"

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_path}. Please ensure the file exists.")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")

        mapping = {}
        performance_mapping = config.get('evaluation', {})

        if not performance_mapping:
            raise ValueError("No performance to class mappings found in the evaluation config.")

        # Dynamically import evaluation model classes
        for code, class_path in performance_mapping.items():
            try:
                module_path, class_name = class_path.rsplit('.', 1)
                module = import_module(module_path)
                evaluation_class = getattr(module, class_name)
                mapping[code] = evaluation_class
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


