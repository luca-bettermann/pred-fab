from typing import Any, Dict, List, Type, Tuple, Optional, Callable

from .evaluation import EvaluationSystem
from .prediction import PredictionSystem
from ..interfaces import IExternalData, IEvaluationModel, IPredictionModel, ICalibrationModel
from ..utils import LocalDataInterface, LBPLogger


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
            external_data_interface: IExternalData,
            debug_flag: bool = False,
            recompute_flag: bool = False,
            visualize_flag: bool = True,
            round_digits: int = 3,
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
        self.external_data = external_data_interface

        # System settings
        self.debug_flag = debug_flag
        self.recompute_flag = recompute_flag
        self.visualize_flag = visualize_flag
        self.round_digits = round_digits

        # Initialize local data handler
        self.local_data = LocalDataInterface(root_folder, local_folder, server_folder)
        
        # Study and experiment state
        self.study_records: Dict[str, Dict[str, Any]] = {}
        self.exp_records: Dict[str, Dict[str, Any]] = {}

        # required fields of records
        self._required_study_record_fields = ['id', 'Code', 'Parameters', 'Performance']
        self._required_exp_record_fields = ['id', 'Code', 'Parameters']

        # Configuration storage
        self.config: Dict[str, Any] = {}
        self.evaluation_config: Dict[str, str] = {}
        self.prediction_config: Dict[str, str] = {}
        self.calibration_config: Dict[str, str] = {}
        self.system_config: Dict[str, Any] = {}
        
        # Initialize system defaults (loaded from config)
        self.defaults = {}

        # Initialize system components
        self.eval_system = EvaluationSystem(self.logger)
        self.pred_system = PredictionSystem(self.local_data, self.logger)
        self.calibration_model: Optional[ICalibrationModel] = None

        self.logger.console_info("\n------- Welcome to Learning by Printing -------\n")

    # === PUBLIC API METHODS (Called externally) ===
    def add_evaluation_model(self, performance_code: str, evaluation_class: Type[IEvaluationModel], round_digits: Optional[int] = None, weight: Optional[float] = None, **kwargs) -> None:
        """
        Add an evaluation model to the system.
        Args:
            performance_code: Code identifying the performance metric
            evaluation_class: Class of evaluation model to instantiate
            round_digits: Number of decimal places to round evaluations (optional)
            calibration_weight: Optional weight for calibration objective function
            **kwargs: Additional parameters for model initialization
        """
        round_digits = self._get_default_attribute('round_digits', round_digits)
        if not isinstance(round_digits, int):
            raise ValueError("Round digits must be an integer.")
        self.eval_system.add_evaluation_model(performance_code, evaluation_class, round_digits, weight, **kwargs)
        
        weight_msg = f" with calibration weight {weight}" if weight is not None else ""
        self.logger.console_info(f"Added evaluation model '{evaluation_class.__name__}' for performance '{performance_code}'{weight_msg}.")

    def add_prediction_model(self, performance_codes: List[str], prediction_class: Type[IPredictionModel], round_digits: Optional[int] = None, **kwargs) -> None:
        """
        Add a prediction model to the system.
        
        Args:
            performance_codes: List[str]: List of codes identifying the performance metrics
            prediction_class: Class of prediction model to instantiate
            round_digits: Number of decimal places to round predictions (optional)
            **kwargs: Additional parameters for model initialization
        """
        round_digits = self._get_default_attribute('round_digits', round_digits)
        if not isinstance(round_digits, int):
            raise ValueError("Round digits must be an integer.")
        self.pred_system.add_prediction_model(performance_codes, prediction_class, round_digits, **kwargs)
        self.logger.console_info(f"Added prediction model '{prediction_class.__name__}' for performance codes: {performance_codes}.")

    def set_calibration_model(self, calibration_class: Type[ICalibrationModel], **kwargs) -> None:
        """
        Add a calibration model to the system.
        
        Args:
            calibration_class: Class of calibration model to instantiate
            **kwargs: Additional parameters for model initialization
        """
        self.calibration_model = calibration_class(logger=self.logger, **kwargs)
        self.logger.console_info(f"\nSet calibration model to '{calibration_class.__name__}'.")

    def initialize_for_study(self, study_code: str, debug_flag: Optional[bool] = None, recompute_flag: Optional[bool] = None) -> None:
        """
        Initialize the system for a specific study.
        
        Args:
            study_code: Unique identifier for the study
            debug_flag: Whether to run in debug mode
        """
        self.logger.console_info(f"\n------- Study Initialization: '{study_code}' -------\n")

        # Use system defaults if not explicitly provided
        _, debug_flag, recompute_flag = self._validate_system_flags(debug_flag=debug_flag, recompute_flag=recompute_flag)

        # Configure study context
        self.local_data.set_study_code(study_code)
        self.logger.info(f"Local data handler initialized for study '{study_code}'")
        
        # Hierarchical load and save of study record
        study_record = self._load_and_save_study_record(study_code, debug_flag, recompute_flag)

        # Validate keys in study record
        if not all(field in study_record for field in self._required_study_record_fields):
            missing_fields = [field for field in self._required_study_record_fields if field not in study_record]
            raise ValueError(f"Study record for '{study_code}' is missing required fields: {missing_fields}")
        
        # Validate study parameters
        study_params = study_record['Parameters']
        if not study_params or not isinstance(study_params, dict):
            raise ValueError(f"Study parameters for study code '{study_code}' empty or not a dict. Please check the database configuration.")

        # Activate the evaluation and prediction models based on performance records
        for performance_code in study_record['Performance']:
            self.eval_system.activate_evaluation_model(performance_code, study_params)
            self.pred_system.activate_prediction_model(performance_code, study_params)

        # Initialize feature model instances
        self._add_feature_model_instances(study_params)

        # Display initialization summary
        summary = self._initialization_step_summary()
        self.logger.console_summary(summary)
        self.logger.console_success(f"Successfully initialized evaluation system of study '{study_code}'.")

    def run_evaluation(
            self, 
            study_code: str, 
            exp_nrs: Optional[List[int]] = None, 
            exp_nr: Optional[int] = None, 
            visualize_flag: Optional[bool] = None, 
            debug_flag: Optional[bool] = None, 
            recompute_flag: Optional[bool] = None) -> None:
        """
        Execute evaluation for one or multiple experiments.
        
        Args:
            exp_nrs: List of experiment numbers to process (use this OR exp_nr)
            exp_nr: Single experiment number to process (use this OR exp_nrs)
            visualize_flag: Whether to show visualizations (uses system default if None)
            debug_flag: Whether to run in debug mode (uses system default if None)
            recompute_flag: Force a recompute of the evaluation (uses system default if None)
        """
        # Check if study has been initialized
        self.local_data.check_availability(study_code, self.study_records)

        # Handle input compatibility
        exp_codes = self._convert_exp_nrs(exp_nr, exp_nrs)

        # Set flags with system defaults if not explicitly provided
        visualize_flag, debug_flag, recompute_flag = self._validate_system_flags(visualize_flag, debug_flag, recompute_flag)

        # Configure debug mode if needed
        if debug_flag and not self.logger.debug_mode:
            self.logger.switch_to_debug_mode()

        # Start the evaluation
        self.logger.console_info(f"\n------- Evaluation of {len(exp_codes)} Experiment{'s' if len(exp_codes) > 1 else ''} -------\n")

        # Hierarchical load and save of experiment records in batch
        exp_records = self._load_and_save_exp_records(exp_codes, debug_flag, recompute_flag)

        # Hierarchical load of evaluation data in batch
        exp_codes_with_missing_data = self._load_evaluation_data(exp_codes, debug_flag, recompute_flag)

        # Iterate over experiment codes
        for exp_code in exp_codes:
            self.logger.console_info(f"\nStart evaluation of '{exp_code}'...")

            # Set experiment parameters for all evaluation and feature models
            exp_params = exp_records[exp_code]['Parameters']
            self.eval_system.initialize_for_exp(exp_code, **exp_params)

            # Check whether evaluation is needed
            if exp_code in exp_codes_with_missing_data or recompute_flag:

                # Execute evaluation pipeline
                self.eval_system.run(
                    exp_code=exp_code,
                    exp_folder=self.local_data.get_experiment_folder(exp_code),
                    visualize_flag=visualize_flag,
                    debug_flag=debug_flag
                )

            # Display evaluation summary
            summary = self.eval_system.evaluation_step_summary(exp_code)
            self.logger.console_summary(summary)
        self.logger.console_success(f"Concluded evaluation of '{exp_codes[0] if len(exp_codes) == 1 else exp_codes}'.")

        # Save experiment data in batch
        self._save_evaluation_data(exp_codes, debug_flag, recompute_flag)

    def run_training(
            self, 
            study_code: str, 
            exp_nrs: List[int], 
            visualize_flag: Optional[bool] = None,
            debug_flag: Optional[bool] = None,
            recompute_flag: Optional[bool] = None) -> None:
        """
        Run training for all prediction models.
        
        This method orchestrates the training process for all active prediction models
        using the evaluation history and feature models.
        """
        # Check if study has been initialized
        self.local_data.check_availability(study_code, self.study_records)

        # Set flags with system defaults if not explicitly provided
        visualize_flag, debug_flag, recompute_flag = self._validate_system_flags(visualize_flag, debug_flag, recompute_flag)

        self.logger.console_info(f"\n------- Run Training for {len(exp_nrs)} Experiments of Study '{study_code}' -------\n")

        # Get experiment codes and load the corresponding records
        exp_codes = [self.local_data.get_experiment_code(exp_nr) for exp_nr in exp_nrs]
        exp_records = self._load_and_save_exp_records(exp_codes, debug_flag, recompute_flag=False)

        # Load complete experiment data using hierarchical pattern
        missing_codes = self._load_evaluation_data(exp_codes, debug_flag, recompute_flag)
        if missing_codes:
            raise ValueError(f"Cannot run training, missing evaluation data for experiments: {missing_codes}")

        # Prepare training inputs and outputs
        parameters, avg_features, dim_arrays, feature_arrays = self._get_train_inputs(exp_records)

        # Run training for all prediction models
        self.pred_system.train(parameters, avg_features, dim_arrays, feature_arrays, visualize_flag, debug_flag)

        # Display training summary
        summary = self.pred_system.training_step_summary()
        self.logger.console_summary(summary)
        self.logger.console_success(f"Successfully trained prediction models for study '{study_code}'.")

    def run_calibration(self, exp_nr: int, param_ranges: Dict[str, Tuple[float, float]], 
                       visualize_flag: Optional[bool] = None, debug_flag: Optional[bool] = None) -> Dict[str, float]:
        """
        Run calibration to find optimal parameters for upcoming experiment.
        
        Args:
            exp_nr: Experiment number for the upcoming experiment
            param_ranges: Parameter bounds {param_name: (min_val, max_val)}
            visualize_flag: Whether to show visualizations during optimization
            debug_flag: Whether to run in debug mode
            
        Returns:
            Dictionary of optimal parameters {param_name: optimal_value}
        """
        # Validate calibration model exists
        if self.calibration_model is None:
            raise ValueError("No calibration model added. Use add_calibration_model() first.")
            
        # Set flags with system defaults if not explicitly provided
        visualize_flag, debug_flag, _ = self._validate_system_flags(visualize_flag=visualize_flag, debug_flag=debug_flag)

        # Get experiment code
        exp_code = self.local_data.get_experiment_code(exp_nr)
        
        self.logger.console_info(f"\n------- Run Calibration for Experiment '{exp_code}' -------\n")
        
        # Get calibration weights from evaluation models
        try:
            calibration_weights = self.eval_system.get_calibration_weights()
            self.calibration_model.set_performance_weights(calibration_weights)
        except ValueError as e:
            raise ValueError(f"Cannot run calibration: {e}")
        
        # Create prediction function
        def predict_fn(params: Dict[str, float]) -> Dict[str, float]:
            return self.pred_system.predict(params, exp_code, visualize_flag=False, debug_flag=True)
        
        # Create evaluation function  
        def evaluate_fn(exp_code: str, predicted_features: Dict[str, float]) -> Dict[str, float]:
            """Evaluate predicted features using existing evaluation logic."""
            import numpy as np
            performances = {}
            
            for code, eval_model in self.eval_system.get_active_eval_models().items():
                if eval_model.weight is None:
                    continue
                    
                if code in predicted_features:
                    feature_value = np.array(predicted_features[code])
                    performance_value = eval_model._compute_performance(
                        feature_value, eval_model.target_value, eval_model.scaling_factor
                    )
                    performances[code] = float(performance_value) if performance_value is not None else 0.0
                else:
                    raise ValueError(f"Predicted features do not contain required performance code '{code}' for evaluation.")
            
            return performances
        
        # Extract parameter keys from ranges
        param_keys = list(param_ranges.keys())
        
        # Run calibration
        optimal_params = self.calibration_model.calibrate(
            exp_code=exp_code,
            predict_fn=predict_fn,
            evaluate_fn=evaluate_fn,
            param_keys=param_keys,
            param_ranges=param_ranges
        )
        
        # Get evaluation count from calibration model
        evaluation_count = getattr(self.calibration_model, '_eval_count', 0)
        
        # Get predicted features and performance values for optimal parameters for summary
        # Validate optimal parameters before evaluation
        predicted_features = None
        performance_values = None
        
        if optimal_params and all(isinstance(v, (int, float)) for v in optimal_params.values()):
            predicted_features = predict_fn(optimal_params)
            calibration_weights = self.eval_system.get_calibration_weights()
            if predicted_features and all(code in predicted_features for code in calibration_weights.keys()):
                performance_values = evaluate_fn(exp_code, predicted_features)
            else:
                self.logger.warning(f"Prediction function returned incomplete features for summary")
        else:
            self.logger.warning(f"Invalid optimal parameters for summary: {optimal_params}")
        
        # Display calibration summary
        summary = self.calibration_model.calibration_step_summary(
            exp_code, param_ranges, optimal_params, evaluation_count, 
            predicted_features, performance_values
        )
        self.logger.console_summary(summary)
        
        # Format optimal params for console display (remove numpy types)
        formatted_params = {k: round(float(v), self.round_digits) for k, v in optimal_params.items()}
        self.logger.console_success(f"Calibration completed. Optimal parameters: {formatted_params}")
        return optimal_params

    # === PRIVATE/INTERNAL METHODS (Internal use only) ===
    def _convert_exp_nrs(self, exp_nr: Optional[int], exp_nrs: Optional[List[int]]) -> List[str]:
        if exp_nr is not None and exp_nrs is not None:
            raise ValueError("Provide either exp_nr OR exp_nrs, not both")
        elif exp_nr is not None:
            exp_nrs = [exp_nr]
        elif exp_nrs is None:
            raise ValueError("Must provide either exp_nr or exp_nrs")
        exp_codes = [self.local_data.get_experiment_code(exp_nr) for exp_nr in exp_nrs]
        return exp_codes

    def _load_and_save_study_record(self, study_code: str, debug_flag: bool, recompute_flag: bool) -> Dict[str, Any]:
        """
        Load and save study record using hierarchical process.
        """
        # Hierarchical load of study record
        missing_study = self._hierarchical_load(
            "study record",
            [study_code],
            self.study_records,
            self.local_data.load_study_records,
            self.external_data.pull_study_records,
            debug_flag,
            recompute_flag
        )
        if missing_study:
            raise ValueError(f"Could not load study record for {study_code}.")

        # Hierarchical save of study record
        self._hierarchical_save(
            "study record",
            [study_code],
            self.study_records,
            self.local_data.save_study_records,
            self.external_data.push_study_records,
            debug_flag,
            recompute_flag
        )

        # retrieve study record from memory and return
        return self.study_records[study_code]

    def _load_and_save_exp_records(self, exp_codes: List[str], debug_flag: bool, recompute_flag: bool) -> Dict[str, Dict[str, Any]]:
        """
        Load and save experiment records using hierarchical process.
        """
        # Hierarchical load experiment records
        missing_exp = self._hierarchical_load(
            "experiment records",
            exp_codes,
            self.exp_records,
            self.local_data.load_exp_records,
            self.external_data.pull_exp_records,
            debug_flag,
            recompute_flag
        )
        if missing_exp:
            raise ValueError(f"Could not load experiment records for: {missing_exp}")

        # Hierarchical save experiment records
        self._hierarchical_save(
            "experiment records",
            exp_codes,
            self.exp_records,
            self.local_data.save_exp_records,
            self.external_data.push_exp_records,
            debug_flag,
            recompute_flag
        )

        # retrieve exp records from memory and return as list
        return {code: self.exp_records[code] for code in exp_codes}

    def _load_evaluation_data(self, exp_codes: List[str], debug_flag: bool, recompute_flag: bool) -> List[str]:
        """
        Load experiment data using hierarchical process (memory → local → external).
        
        Args:
            exp_codes: List of experiment codes to load data for
            debug_flag: If True, skip pulling and pushing from external source

        Returns:
            List of experiment codes that have missing or incomplete data
        """
        self.logger.info(f"Loading experiment data for {len(exp_codes)} experiments: {exp_codes}")

        # Load aggregated metrics
        missing_aggr_codes = self._hierarchical_load(
            "aggregated metrics",
            exp_codes,
            self.eval_system.aggr_metrics,
            self.local_data.load_aggr_metrics,
            self.external_data.pull_aggr_metrics,
            debug_flag,
            recompute_flag
        )

        for perf_code, metric_array in self.eval_system.metric_arrays.items():
            # Load metrics arrays
            missing_array_codes = self._hierarchical_load(
                f"'{perf_code}' metric array",
                exp_codes,
                metric_array,
                self.local_data.load_metrics_arrays,
                self.external_data.pull_metrics_arrays,
                debug_flag,
                recompute_flag,
                perf_code=perf_code
            )

        missing_codes = set(missing_aggr_codes + missing_array_codes)
        if missing_codes:
            self.logger.info(f"No metrics found for {len(missing_codes)} experiments: {missing_codes} - will generate during evaluation.")
        return list(missing_codes)

    def _save_evaluation_data(self, exp_codes: List[str], debug_flag: bool, recompute_flag: bool) -> None:
        """
        Save experiment data using hierarchical process (memory → local → external).
        
        Args:
            exp_codes: List of experiment codes to save data for
        """
        if not exp_codes:
            return
            
        self.logger.info(f"Saving experiment data for {len(exp_codes)} experiments: {exp_codes}")
                
        # Save aggregated metrics
        self._hierarchical_save(
            "aggregated metrics",
            exp_codes,
            self.eval_system.aggr_metrics,
            self.local_data.save_aggr_metrics,
            self.external_data.push_aggr_metrics,
            debug_flag,
            recompute_flag
        )

        # Iterate over performances
        for perf_code, metric_array in self.eval_system.metric_arrays.items():
            # Get column names for saving
            dim_iterator_names = self.eval_system.evaluation_models[perf_code].dim_iterator_names
            metric_names = self.eval_system.evaluation_models[perf_code].metric_names

            # Save metrics arrays
            self._hierarchical_save(
                f"'{perf_code}' metric array",
                exp_codes,
                metric_array,
                self.local_data.save_metrics_arrays,
                self.external_data.push_metrics_arrays,
                debug_flag,
                recompute_flag,
                perf_code=perf_code,
                column_names=dim_iterator_names + metric_names
            )

        self.logger.info(f"Completed saving experiment data for {len(exp_codes)} experiments")

    def _hierarchical_load(self, 
                           dtype: str,
                           target_codes: List[str],
                           memory_storage: Dict[str, Any],
                           local_loader: Callable[[List[str]], Tuple[List[str], Dict[str, Any]]],
                           external_loader: Callable[[List[str]], Tuple[List[str], Any]],
                           debug: Optional[bool],
                           recompute: Optional[bool],
                           **kwargs) -> List[str]:
        """
        Universal hierarchical data loading with missing-only processing.
        
        Args:
            target_codes: List of codes to load (study_codes, exp_codes, etc.)
            memory_storage: Dictionary where loaded data will be stored/retrieved
            local_loader: Function to load from local files
            external_loader: Function to load from external source
            
        Returns:
            List of codes that couldn't be loaded from any source
        """
        # 1. Check memory - filter out already loaded codes
        missing_memory = [code for code in target_codes if code not in memory_storage]
        self._check_for_retrieved_codes(target_codes, missing_memory, dtype, "memory")

        # check break condition
        if not missing_memory:
            return []
        
        # 2. Load from local files
        if not recompute:
            missing_local, local_data = local_loader(missing_memory, **kwargs)
            memory_storage.update(local_data)
            self._check_for_retrieved_codes(missing_memory, missing_local, dtype, "local files", console_output=True)
        else:
            missing_local = missing_memory
            self.logger.info(f"Recompute mode: Skipping loading {dtype} from local files")

        # check break condition
        if not missing_local:
            return []

        # 3. Load from external sources (skip if in debug mode)
        if not debug:
            missing_external, external_data = external_loader(missing_local)
            self._check_for_retrieved_codes(missing_local, missing_external, dtype, "external source", console_output=True)
            memory_storage.update(external_data)
        else:
            missing_external = missing_local 
            self.logger.info(f"Debug mode: Skipping loading {dtype} from external source")
  
        return missing_external
    
    def _check_for_retrieved_codes(self, target_pre: List[str], target_post: List[str], dtype: str, source: str, console_output: bool = False) -> List[str]:
        """Check which codes were successfully retrieved and log to console."""
        retrieved_codes = [code for code in target_pre if code not in target_post]
        if retrieved_codes:
            message = f"Retrieved {dtype} {retrieved_codes} from {source}."
            self.logger.console_info(message) if console_output else self.logger.info(message)
        return retrieved_codes

    def _hierarchical_save(self, 
                           dtype: str,
                           target_codes: List[str],
                           memory_storage: Dict[str, Any],
                           local_saver: Callable[[List[str], Dict[str, Any], bool], bool],
                           external_saver: Callable[[List[str], Dict[str, Any], bool], bool],
                           debug_flag: bool,
                           recompute_flag: bool,
                           **kwargs) -> None:
        """
        Universal hierarchical data saving: Memory → Local Files → External Source
        
        Args:
            target_codes: List of codes to save (exp_codes, study_codes, etc.)
            memory_storage: Dictionary containing data to save
            local_saver: Function to save to local files
            external_saver: Function to save to external source
        """
        # 1. Filter to codes that exist in memory
        codes_to_save = [code for code in target_codes if code in memory_storage]
        if not codes_to_save:
            self.logger.debug(f"{dtype} {target_codes} found in memory.")
            return
        
        # 2. Extract data for the codes to save
        data_to_save = {code: memory_storage[code] for code in codes_to_save}
        
        # 3. Save to local files
        saved = local_saver(codes_to_save, data_to_save, recompute_flag, **kwargs)
        if saved:
            self.logger.console_info(f"Saved {dtype} {codes_to_save} as local files.")
        else:
            self.logger.info(f"{dtype.capitalize()} {codes_to_save} already exist as local files.")

        # 4. Save to external source (skip if in debug mode)
        if not debug_flag:
            pushed = external_saver(codes_to_save, data_to_save, recompute_flag, **kwargs)
            if pushed:
                self.logger.console_info(f"Pushed {dtype} {codes_to_save} to external source.")
            else:
                self.logger.info(f"{dtype} {codes_to_save} already exists in external source.")
        else:
            self.logger.info(f"Debug mode: Skipped pushing {dtype} {codes_to_save} to external source.")

    def _get_default_attribute(self, key: str, value: Any) -> Any:
        """
        Retrieve a default attribute value from the LBP manager configuration.

        Args:
            key: The key of the default attribute to retrieve
        
        Returns:
            The value of the default attribute
        """
        if value is None:
            if not hasattr(self, key):
                raise ValueError(f"Default attribute '{key}' not found in LBP manager configuration.")
            value = getattr(self, key)
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

    def _get_train_inputs(self, exp_records: Dict[str, Dict[str, Any]]
                          ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        # Initialize input dicts
        parameters = {}
        avg_features = {}
        dim_arrays = {}
        feature_arrays = {}

        for exp_code, exp_record in exp_records.items():
            # Verify that the experiment record contains all required fields
            if all(field not in exp_record for field in self._required_exp_record_fields):
                missing_fields = [field for field in self._required_exp_record_fields if field not in exp_record]
                raise ValueError(f"Missing required fields in experiment record for {exp_code}: {missing_fields}")

            # Add the relevant data to the input dictionaries
            parameters[exp_code] = exp_record['Parameters']
            
            # Initialize avg_features and feature_arrays dicts
            if exp_code not in avg_features:
                avg_features[exp_code] = {}
            if exp_code not in feature_arrays:
                feature_arrays[exp_code] = {}
            if exp_code not in dim_arrays:
                dim_arrays[exp_code] = {}

            # Populate avg_features and feature_arrays
            for perf_code, eval_model in self.eval_system.get_active_eval_models().items():
                # Build feature dicts with keys exp_code, perf_code
                avg_features[exp_code][perf_code] = self.eval_system.aggr_metrics[exp_code][perf_code].get("Feature_Avg", {})
                array = self.eval_system.metric_arrays[perf_code][exp_code]

                # Extract feature array regardless of dimensionality
                boundary = len(eval_model.dim_iterator_names)
                dim_arrays[exp_code][perf_code] = array[..., :boundary]
                feature_arrays[exp_code][perf_code] = array[..., boundary]

        return parameters, avg_features, dim_arrays, feature_arrays

    def _initialization_step_summary(self) -> str:
        """Generate summary of initialization results."""
        assert self.eval_system is not None and self.pred_system is not None
        val_eval_models = self.eval_system.get_active_eval_models()
        val_pred_models = {code: pred_model for code, pred_model in self.pred_system.pred_model_by_code.items() if pred_model.active}

        summary = f"Activated {len(val_eval_models)} evaluation models and {len(set([type(e.feature_model).__name__ for e in val_eval_models.values()]))} feature models.\n\n"
        summary += f"\033[1m{'Performance Code':<20} {'Feature Model':<20} {'Evaluation Model':<20} {'Prediction Model':<20}\033[0m"

        for code in self.eval_system.evaluation_models:
            eval_model = val_eval_models.get(code, None)
            feature_model = eval_model.feature_model if eval_model else None
            pred_model = val_pred_models.get(code, None)
            summary += f"\n{code:<20} {type(feature_model).__name__:<20} {type(eval_model).__name__:<20} {type(pred_model).__name__:<20}"
        return summary

    def _validate_system_flags(self, visualize_flag: Optional[bool] = None, debug_flag: Optional[bool] = None, recompute_flag: Optional[bool] = None) -> Tuple[bool, bool, bool]:
        visualize_flag = self._get_default_attribute('visualize_flag', visualize_flag)
        debug_flag = self._get_default_attribute('debug_flag', debug_flag)
        recompute_flag = self._get_default_attribute('recompute_flag', recompute_flag)
        assert isinstance(visualize_flag, bool), "Visualize flag must be boolean."
        assert isinstance(debug_flag, bool), "Debug flag must be boolean."
        assert isinstance(recompute_flag, bool), "Recompute flag must be boolean."
        return visualize_flag, debug_flag, recompute_flag
