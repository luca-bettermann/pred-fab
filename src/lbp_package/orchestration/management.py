import yaml
from importlib import import_module
from typing import Any, Dict, List, Type, Tuple, Optional

from .evaluation import EvaluationSystem
from .prediction import PredictionSystem
from ..interfaces import DataInterface, EvaluationModel, PredictionModel
from ..utils import FolderNavigator, LBPLogger

# TODO:

# - Add example prediction model, using feature models
# - Continue with the prediction model and system implementation
# - Adapt the testing now that we moved _add_feature_model_instances from evaluation to LBPmanager
# - Write tests for the predictionmodel and system

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
        
        # Initialize system defaults (loaded from config)
        self.defaults = {}
        
        # Performance mapping
        self.performance_records: List[Dict[str, Any]] = []

        # Initialize system components
        self.eval_system = EvaluationSystem(self.nav, self.interface, self.logger)
        self.pred_system = PredictionSystem(self.nav, self.interface, self.logger)

        # Load configuration file
        self._load_config()

    # === PUBLIC API METHODS (Called externally) ===
    def add_evaluation_model(self, performance_code: str, evaluation_class: Type[EvaluationModel], **kwargs) -> None:
        """
        Add an evaluation model to the system.
        Args:
            performance_code: Code identifying the performance metric
            evaluation_class: Class of evaluation model to instantiate
            **kwargs: Additional parameters for model initialization
        """
        round_digits = self._get_default_attribute('round_digits', kwargs.get('round_digits', None))
        self.eval_system.add_evaluation_model(performance_code, evaluation_class, round_digits, **kwargs)

    def add_prediction_model(self, performance_codes: List[str], prediction_class: Type[PredictionModel], **kwargs) -> None:
        """
        Add a prediction model to the system.
        
        Args:
            performance_codes: List of codes identifying the performance metrics
            prediction_class: Class of prediction model to instantiate
            **kwargs: Additional parameters for model initialization
        """
        round_digits = self._get_default_attribute('round_digits', kwargs.get('round_digits', None))
        self.pred_system.add_prediction_model(performance_codes, prediction_class, self.study_params, round_digits, **kwargs)

    def initialize_for_study(self, study_code: str, debug_flag: Optional[bool] = None, eval_flag: Optional[bool] = None, pred_flag: Optional[bool] = None) -> None:
        """
        Initialize the system for a specific study.
        
        Args:
            study_code: Unique identifier for the study
            debug_flag: Whether to run in debug mode
        """
        self.logger.console_info(f"\n------- Study Initialization: '{study_code}' -------")

        # Use system defaults if not explicitly provided
        debug_flag = self._get_default_attribute('debug_flag', debug_flag)
        eval_flag = self._get_default_attribute('eval_flag', eval_flag)
        pred_flag = self._get_default_attribute('pred_flag', pred_flag)

        # Configure study context
        self.study_code = study_code
        self.nav.set_study_code(study_code)
        self.logger.info(f"Folder navigator initialized for study '{study_code}'")
        
        # Load study data from database
        self.study_record = self.interface.get_study_record(study_code)
        self.logger.info(f"Study record retrieved from database")
        
        # Validate study parameters
        study_params = self.interface.get_study_parameters(self.study_record)
        if not study_params or not isinstance(study_params, dict):
            raise ValueError(f"Study parameters for study code '{study_code}' empty or not a dict. Please check the database configuration.")
        
        # Load performance configuration
        performance_records = self.interface.get_performance_records(self.study_record)
        assert performance_records, f"Performance records for study code '{study_code}' empty. Please check the database configuration."
        assert isinstance(performance_records, list), f"Performance records for study code '{study_code}' not a list. Please check the database configuration."

        # Activate the evaluation and prediction models based on performance records
        for performance_record in performance_records:
            # Validate whether the performance records are dicts and contain 'Code' key, then retrieve the code
            assert isinstance(performance_record, dict), f"Performance record for study code '{study_code}' is not a dict: {performance_record}"
            assert "Code" in performance_record, f"Performance record missing 'Code' field: {performance_record}"
            code = str(performance_record.get("Code"))

            # Activate evaluation model
            if eval_flag:
                self.eval_system.activate_evaluation_model(code, study_params)

            # Activate prediction model if it matches the output codes
            if pred_flag:
                self.pred_system.activate_prediction_model(code, study_params)

            # Add performance record to the list
            self.performance_records.append(performance_record)

        # Initialize feature model instances
        self._add_feature_model_instances(study_params)

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
        if not self.study_record:
            raise RuntimeError("No study has been initialized. Please call initialize_study() first.")

        # Use system defaults if not explicitly provided
        debug_flag = self._get_default_attribute('debug_flag', debug_flag)
        visualize_flag = self._get_default_attribute('visualize_flag', visualize_flag)
        assert debug_flag is not None and visualize_flag is not None, "flags must be set to True or False."

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
            study_record=self.study_record,
            exp_code=exp_code, 
            exp_record=self.exp_record, 
            visualize_flag=visualize_flag,
            debug_flag=debug_flag,
            **exp_vars
        )

        # Display evaluation summary
        summary = self._evaluation_step_summary()
        self.logger.console_summary(summary)
        self.logger.console_success(f"Successfully evaluated the performance attributes of study '{self.study_code}'.")

    def run_training(self, restrict_to_exp_codes: List[str] = []) -> None:
        """
        Run training for all prediction models.
        
        This method orchestrates the training process for all active prediction models
        using the evaluation history and feature models.
        """
        if not self.study_record:
            raise RuntimeError("No study has been initialized. Please call initialize_study() first.")

        self.logger.console_info(f"\n------- Run Training for Study: '{self.study_code}' -------")

        # Load study dataset
        study_dataset = self.interface.get_study_dataset(self.study_record, restrict_to_exp_codes)

        # Run training for all prediction models
        self.pred_system.train(study_dataset)

        # Display training summary
        self.logger.console_success(f"Successfully trained prediction models for study '{self.study_code}'.")

    def run_calibration(self, exp_nr: int, visualize_flag: bool = False) -> None:

        # Load experiment parameters
        exp_code = self.nav.get_experiment_code(exp_nr)
        exp_record = self.interface.get_exp_record(exp_code)
        exp_params = self.interface.get_exp_variables(exp_record)

        # Predict -> this will not be used like this
        predictions = self.pred_system.predict(exp_params, exp_code, visualize_flag)

        # Add code that calibrates upcoming experiment
        ...

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
        # self.evaluation_config = self.config.get('evaluation', {})
        # self.prediction_config = self.config.get('prediction', {})
        # self.calibration_config = self.config.get('calibration', {})
        self.system_config = self.config.get('system', {})
        
        # Set system defaults
        for key, value in self.system_config.items():
            self.defaults[key] = value

        self.logger.info(f"Loaded configuration with {len(self.system_config)} system settings")
        defaults_str = ', '.join([f"{key}={value}" for key, value in self.defaults.items()])
        self.logger.info(f"System defaults: {defaults_str}")

        # self.logger.info(f"Loaded configuration with {len(self.evaluation_config)} evaluation models, "
        #                 f"{len(self.prediction_config)} prediction models, "
        #                 f"{len(self.calibration_config)} calibration models, "
        #                 f"and {len(self.system_config)} system settings")


    def _get_default_attribute(self, key: str, value: Any) -> Any:
        """
        Retrieve a default attribute value from the system configuration.

        Args:
            key: The key of the default attribute to retrieve
        
        Returns:
            The value of the default attribute
        """
        if value is None:
            assert key in self.defaults, f"Default attribute '{key}' not found in system configuration."
            value = self.defaults[key]
            assert value is not None, f"Default attribute '{key}' is None. Please check the configuration."
            self.logger.debug(f"Retrieved default attribute '{key}': {value}")
            return value
        else:
            self.logger.debug(f"Using provided value for '{key}': {value}")
            return value

    def _add_feature_model_instances(self, study_params: Dict[str, Any]) -> None:
        """
        Create feature model instances for evaluation and prediction models.

        Optimizes by sharing feature model instances where possible.
        """
        assert self.eval_system is not None, "Evaluation system is not initialized."
        assert self.pred_system is not None, "Prediction system is not initialized."

        # Create a list of tuples consisting of (model, code, feature_model_type)
        feature_model_collection = []

        # Iterate over evaluation models
        for code, eval_model in self.eval_system.evaluation_models.items():
            # Create a collection instance for each evaluation model
            collection_instance = (eval_model, code, eval_model.feature_model_type)
            feature_model_collection.append(collection_instance)

        # Iterate over prediction models and their feature model types
        for pred_model in self.pred_system.prediction_models:
            for code, feature_model_type in pred_model.feature_model_types.items():
                # Create a collection instance for feature model mapped to the prediction model
                collation_instance = (pred_model, code, feature_model_type)
                feature_model_collection.append(collation_instance)

        # Create a dictionary to store unique feature model instances
        feature_model_dict = {}

        # Iterate over the collection and initiate or reuse feature model instances
        for model, code, feature_model_type in feature_model_collection:

            # Check whether an instance of the feature model type already exists
            if feature_model_type not in feature_model_dict:

                # Create a new feature model instance
                feature_model_instance = feature_model_type(
                    performance_code=code,
                    logger=self.logger,
                    study_params=study_params,
                    round_digits=model.round_digits,
                    **model.kwargs
                )
                
                # Store the feature model instance in the dictionary
                self.logger.info(f"Added feature model instance '{type(feature_model_instance).__name__}' to '{type(model).__name__}' model")

            else:
                # Reuse existing feature model instance
                feature_model_instance = feature_model_dict[feature_model_type]
                feature_model_instance.initialize_for_code(model.performance_code)
                self.logger.info(f"Reusing existing feature model instance '{type(feature_model_instance).__name__}' for model '{type(model).__name__}'")

            # Add the feature model instance to the model
            model.add_feature_model(code=code, feature_model=feature_model_instance)
            feature_model_dict[feature_model_type] = feature_model_instance

    def _initialization_step_summary(self) -> str:
        """Generate summary of initialization results."""
        assert self.eval_system is not None and self.pred_system is not None
        val_eval_models = {code: eval_model for code, eval_model in self.eval_system.evaluation_models.items() if eval_model.active}
        val_pred_models = {code: pred_model for code, pred_model in self.pred_system.pred_model_by_code.items() if pred_model.active}

        summary = f"Loaded {len(self.performance_records)} performance attributes, {len(val_eval_models)} evaluation models and {len(set([type(e.feature_model).__name__ for e in val_eval_models.values()]))} feature models.\n\n"
        summary += f"\033[1m{'Performance Code':<20} {'Feature Model':<20} {'Evaluation Model':<20} {'Prediction Model':<20}\033[0m"
        
        for performance_record in self.performance_records:
            code = performance_record.get("Code")
            assert isinstance(code, str), f"Performance record code must be a string: {performance_record}"
                
            eval_model = val_eval_models.get(code, None)
            feature_model = eval_model.feature_model if eval_model else None
            pred_model = val_pred_models.get(code, None)
            summary += f"\n{code:<20} {type(feature_model).__name__:<20} {type(eval_model).__name__:<20} {type(pred_model).__name__:<20}"
        return summary + "\n"

    def _evaluation_step_summary(self) -> str:
        """Generate summary of evaluation results."""
        assert self.eval_system is not None

        active_eval_models = {code: model for code, model in self.eval_system.evaluation_models.items() if model.active}
        
        summary = f"Evaluated {len(active_eval_models)} performance attributes, using {len(active_eval_models)} evaluation models and {len(set([type(e.feature_model).__name__ for e in active_eval_models.values()]))} feature models.\n\n"
        summary += f"\033[1m{'Performance Code':<20} {'Evaluation Model':<20} {'Dimensions':<30} {'Performance Value':<20}\033[0m"

        for code, eval_model in active_eval_models.items():
            dimensions = ', '.join([f"{name}: {size}" for name, size in zip(eval_model.dim_names, eval_model._compute_dim_sizes())])
            dimensions = "(" + dimensions + ")"
            summary += f"\n{code:<20} {type(eval_model).__name__:<20} {dimensions:<30} {self.eval_system.performance_metrics[code].get('Value', 0):<20}"
        return summary + "\n"


