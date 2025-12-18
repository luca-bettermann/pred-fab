"""
LBPAgent - Main orchestration class for AIXD architecture.

Replaces study-based logic with dataset-centric approach while maintaining
existing workflow patterns. Wraps EvaluationSystem, PredictionSystem, and
manages schema generation and dataset initialization.
"""

import numpy as np
from typing import Any, Dict, List, Set, Type, Optional, Tuple, Literal

from pred_fab.utils.enum import Mode
from ..core.schema import DatasetSchema, SchemaRegistry
from ..core.dataset import Dataset, ExperimentData
from ..core.datamodule import DataModule
from ..orchestration import (
    FeatureSystem,
    EvaluationSystem,
    PredictionSystem,
    CalibrationSystem
)

from ..interfaces import IFeatureModel, IEvaluationModel, IPredictionModel, IExternalData
from ..utils import LocalData, LBPLogger, StepType, Phase

SystemNames = Literal['feature', 'evaluation', 'prediction', 'calibration']

class PfabAgent:
    """
    Main orchestration class for AIXD-based LBP framework.
    
    - Model registration and activation
    - Schema generation from active models
    - Dataset initialization and loading
    - Coordination between EvaluationSystem, PredictionSystem, CalibrationSystem
    """
    
    def __init__(
        self,
        root_folder: str,
        external_data: Optional[IExternalData] = None,
        debug_flag: bool = False,
        recompute_flag: bool = False,
        visualize_flag: bool = True,
    ):
        """
        Initialize LBP Agent.
        
        Args:
            root_folder: Project root folder
            local_folder: Path to local data storage
            external_data_interface: Optional interface for database operations
            debug_flag: Skip external data operations if True
            recompute_flag: Force recomputation/overwrite if True
            visualize_flag: Enable visualizations if True
        """
        
        # Initialize local data handler
        self.local_data = LocalData(root_folder)

        # Initialize logger
        self.logger_name = "LBPAgent"
        self.logger = LBPLogger(self.logger_name, self.local_data.get_log_folder('logs'))
        self.logger.info("Initializing LBP Agent")
        
        # Initialize external data interface
        self.external_data = external_data
        if external_data is None:
            self.logger.console_warning(
                "No external data interface provided: "
                "Experiments need to be available as local JSON files."
            )
        
        # Flag settings
        self.debug_flag = debug_flag
        self.recompute_flag = recompute_flag
        self.visualize_flag = visualize_flag
        
        # Initialize system components (will be set later)
        self.feature_system: FeatureSystem
        self.eval_system: EvaluationSystem
        self.pred_system: PredictionSystem
        self.calibration_system: CalibrationSystem
        
        # Model registry (store classes and params until dataset is initialized)
        self._feature_model_specs: List[Tuple[Type[IFeatureModel], dict]] = []  # List of (class, kwargs)
        self._evaluation_model_specs: List[Tuple[Type[IEvaluationModel], dict]] = []  # List of (class, kwargs)
        self._prediction_model_specs: List[Tuple[Type[IPredictionModel], dict]] = []  # List of (class, kwargs)
        
        # Initialization state guard
        self._initialized = False
        
        # Progress tracking
        self.active_exp: Optional[ExperimentData] = None
        
        self.logger.console_info("\n===== Welcome to PFAB - Predictive Fabrication =====\n")

    # === MODEL REGISTRATION ===

    def _register_model(
            self, 
            model_class: Type[Any], 
            model_specs: List[Tuple[Type[Any], dict]], 
            kwargs: dict
        ) -> None:
        """General registration logic."""
        if self._initialized:
            raise RuntimeError(
                "Cannot register models after initialize() has been called. "
                "Create a new LBPAgent instance for other configurations."
            )
        
        # Store class and params until dataset is created
        model_specs.append((model_class, kwargs))
        self.logger.info(f"Registered model: {model_class.__name__}")

    def register_feature_model(self, feature_class: Type[Any], **kwargs) -> None:
        """Register a feature model."""
        self._register_model(feature_class, self._feature_model_specs, kwargs)

    def register_evaluation_model(self, evaluation_class: Type[IEvaluationModel], **kwargs) -> None:
        """Register an evaluation model."""
        self._register_model(evaluation_class, self._evaluation_model_specs, kwargs)
    
    def register_prediction_model(self, prediction_class: Type[IPredictionModel], **kwargs) -> None:
        """Register a prediction model."""
        self._register_model(prediction_class, self._prediction_model_specs, kwargs)
    
    # === MODEL INITIALIZATION & VALIDATION ===

    def initialize(self, schema: DatasetSchema) -> Dataset:
        """Initialize dataset and orchestration systems from registered models and validate with schema."""

        self.logger.console_info("\n===== Initializing Dataset =====")
                
        # Step 1: Initialize systems (dataset will be set later)
        self.logger.info("Instantiating models from registered classes...")
        if not self._feature_model_specs:
            raise ValueError("No feature models registered. Call register_feature_model() first.")

        
        # Instantiate feature models
        self.feature_system = FeatureSystem(logger=self.logger)
        self._instantiate_model_group(
            self._feature_model_specs,
            self.feature_system.models,
            "feature"
        )

        # Instantiate evaluation models
        self.eval_system = EvaluationSystem(logger=self.logger)
        self._instantiate_model_group(
            self._evaluation_model_specs,
            self.eval_system.models,
            "evaluation"
        )

        # Initialize schema: it requires feature models to set dimension codes
        registry = SchemaRegistry(self.local_data.local_folder)
        new_schema_id = schema.initialize(registry, self.feature_system.models)
        self.local_data.set_schema(new_schema_id)
        self.logger.info(f"\nUsing schema ID: {new_schema_id}")

        # Prediction system requires schema to be set
        self.pred_system = PredictionSystem(logger=self.logger, schema=schema, local_data=self.local_data)
        self._instantiate_model_group(
            self._prediction_model_specs,
            self.pred_system.models,
            "prediction"
        )

        # Calibration system requires prediction, evaluation and schema to be set
        self.calibration_system = CalibrationSystem(
            schema=schema,
            logger=self.logger, 
            predict_fn=self.pred_system._predict_from_params,
            evaluate_fn=self.eval_system._evaluate_feature_dict,
            residual_predict_fn=self.pred_system.residual_model.predict
            )

        # validate against schema
        self._validate_systems_against_schema(schema)
        self._initialized = True

        # Step 3: Create dataset with storage interfaces
        dataset = Dataset(
            schema=schema,
            schema_id=new_schema_id,
            local_data=self.local_data,
            external_data=self.external_data,
            logger=self.logger,
            debug_mode=self.debug_flag,
        )

        # Display summary
        summary_lines = [
            "\nDataset:",
            f"  - Schema ID: {new_schema_id}",
            f"  - Parameters: {len(schema.parameters.data_objects)} ({len(schema.parameters.get_dim_names())} dims)",
            f"  - Features: {len(schema.features.data_objects)}",
            f"  - Performance Attributes: {len(schema.performance.data_objects)}",
        ]
        self.logger.console_summary("\n".join(summary_lines))
        self.logger.console_success(f"Successfully initialized dataset '{new_schema_id}'.")
        return dataset        
        
    def _validate_systems_against_schema(self, schema: DatasetSchema) -> None:
        """Validate all orchestration systems against the provided schema."""
        
        # Extract specs from systems
        feat_specs = self.feature_system.get_model_specs()
        input_params = feat_specs["input_parameters"]
        input_features = feat_specs["input_features"]
        output_features = feat_specs["outputs"]
        output_predicted_features = []
        output_performance_attrs = []

        if self.eval_system:
            eval_specs = self.eval_system.get_model_specs()
            input_params.extend(eval_specs["input_parameters"])
            input_features.extend(eval_specs["input_features"])
            output_performance_attrs.extend(eval_specs["outputs"])

        if self.pred_system:
            pred_specs = self.pred_system.get_model_specs()
            input_params.extend(pred_specs["input_parameters"])
            input_features.extend(pred_specs["input_features"])
            output_predicted_features.extend(pred_specs["outputs"])

        # Validate that all lists are represented in schema
        self._check_sets_against_keys(set(input_params), schema.parameters.keys())
        self._check_sets_against_keys(set(input_features), schema.features.keys())
        self._check_sets_against_keys(set(output_features), schema.features.keys())
        self._check_sets_against_keys(set(output_performance_attrs), schema.performance.keys())

        # Validate that all input features are represented as outputs features
        uncomputed_inputs = set(input_features) - set(output_features)
        if uncomputed_inputs:
            raise ValueError(
                f"The following input features are not computed by any model: "
                f"{uncomputed_inputs}"
            )
        
        # Check if there are any predicted features that are not computed by feature models
        unpredicted_features = set(output_predicted_features) - set(output_features)
        if unpredicted_features:
            raise ValueError(
                f"The following predicted features are not computed by any feature model: "
                f"{unpredicted_features}"
            )
        
    def activate_system(self, system_name: SystemNames) -> None:
        """Activate a specific orchestration system."""
        if not self._initialized:
            raise RuntimeError("Cannot activate systems before initialize() is called.")
        self._toggle_system(system_name, activate=True)

    def deactivate_system(self, system_name: SystemNames) -> None:
        """Deactivate a specific orchestration system."""
        if not self._initialized:
            raise RuntimeError("Cannot deactivate systems before initialize() is called.")
        self._toggle_system(system_name, activate=False)

    def set_active_experiment(self, exp_data: ExperimentData) -> None:
        """Set the active experiment for online operations."""
        if not self._initialized:
             raise RuntimeError("Agent not initialized.")
        
        self.active_exp = exp_data
        self.logger.info(f"Active experiment set to: {exp_data.exp_code}")

    # === FULL STEP OPERATIONS ==

    def step_offline(
        self,
        exp_data: ExperimentData,
        datamodule: Optional[DataModule] = None,
        step_type: StepType = StepType.FULL,
        phase: Phase = Phase.INFERENCE,
        w_explore: float = 0.5,
        n_optimization_rounds: int = 10,
        recompute: bool = False,
        visualize: bool = False,
    ) -> Optional[ExperimentData]:
        """Perform a full offline step of all active systems."""
        self._check_systems(step_type)

        # Set start and end values
        start, end = 0, None
        
        # 1. Extract Features
        self.feature_system.compute_exp_features(exp_data, start, end, recompute=recompute, visualize=visualize)
        self._log_step_completion(exp_data.exp_code, start, end, action="had features extracted")
        
        # 2. Evaluate Performance
        self.eval_system.compute_exp_evaluation(exp_data, start, end, recompute=recompute)
        self._log_step_completion(exp_data.exp_code, start, end, action="evaluated")

        # End step here if only evaluation is requested
        if step_type == StepType.EVAL:
            return
        
        # TODO: figure out when datamodule is needed, and when not
        if datamodule is None:
            raise ValueError("DataModule must be provided for training in offline step.")
        
        # 3. Train Prediction Model, only in LEARNING phase
        # TODO: Here, we actually should be training surrogate model, first and foremost
        # TODO: Do we even want to train prediction model in learning mode?
        if phase == Phase.LEARNING:
            datamodule.prepare()
            self.pred_system.train(datamodule)
            self._log_step_completion(exp_data.exp_code, start, end, action="included in training")

        # 4. Train Exploration Model and calibrate new experiment
        current_params = exp_data.parameters.get_values_dict()
        new_params = self.calibration_system.propose_params(
            datamodule, Mode.OFFLINE, phase, current_params, w_explore, n_optimization_rounds)
        new_exp_data = datamodule.dataset.create_experiment("new_exp", new_params)
        self._log_step_completion(exp_data.exp_code, start, end, action="calibrated")
        
        # 5. Predict features of new experiment
        self.pred_system.predict_experiment(new_exp_data)
        self._log_step_completion(new_exp_data.exp_code, start, end, action="predicted features")
        return new_exp_data
        
    def step_online(
        self,
        exp_data: Optional[ExperimentData] = None,
        step_type: StepType = StepType.FULL,
        phase: Phase = Phase.INFERENCE,
        dimension: Optional[str] = None,
        step_index: Optional[int] = None,
        batch_size: Optional[int] = None,
        n_optimization_rounds: int = 10,
        recompute: bool = False,
        visualize: bool = False,
        **kwargs
        ) -> Optional[Dict[str, Any]]:
        """Perform a full online step of all active systems."""
        
        # Run validations whether systems are initialized and active
        self._check_systems(step_type)

        # Retrieve experiment data
        exp_data, start, end = self._step_config(exp_data, dimension, step_index)
        
        # 1. Extract Features
        self.feature_system.compute_exp_features(exp_data, start, end, recompute=recompute, visualize=visualize)
        self._log_step_completion(exp_data.exp_code, start, end, action="had features extracted")

        # 2. Evaluate Performance
        self.eval_system.compute_exp_evaluation(exp_data, start, end, recompute=recompute)
        self._log_step_completion(exp_data.exp_code, start, end, action="evaluated")

        if step_type == StepType.EVAL:
            return
        
        # 3. Tune Prediction Model
        temp_datamodule = self.pred_system.tune(exp_data, start, end, batch_size, **kwargs)
        self._log_step_completion(exp_data.exp_code, start, end, action="used for tuning")

        # TODO: How are we handling if a feature does not contain the dimension we are stepping over?

        # 4. Train Exploration Model and calibrate process parameters
        current_params = exp_data.parameters.get_values_dict()
        new_params = self.calibration_system.propose_params(
            temp_datamodule, Mode.OFFLINE, phase, current_params, n_optimization_rounds)
        self._log_step_completion(exp_data.exp_code, start, end, action="calibrated")
        return new_params


    # === PARTIAL STEP OPERATIONS ===

    def feature_step(
        self,
        exp_data: Optional[ExperimentData],
        dimension: Optional[str] = None,
        step_index: Optional[int] = None,
        recompute: bool = False,
        visualize: bool = False
    ) -> Dict[str, np.ndarray]:
        """Evaluate experiment and mutate exp_data with results."""
        if self.eval_system is None:
            raise RuntimeError("EvaluationSystem not initialized. Call initialize() first.")
        
        # Retrieve experiment data
        exp_data, start, end = self._step_config(exp_data, dimension, step_index)

        # Delegate to evaluation system
        feature_dict = self.feature_system.compute_exp_features(
            exp_data=exp_data,
            evaluate_from=start,
            evaluate_to=end,
            recompute=recompute,
            visualize=visualize
        )
        
        # Logging
        self._log_step_completion(exp_data.exp_code, start, end, action="computed features")
        return feature_dict

    def evaluation_step(
        self,
        exp_data: Optional[ExperimentData],
        dimension: Optional[str] = None,
        step_index: Optional[int] = None,
        recompute: bool = False
    ) -> Dict[str, Optional[float]]:
        """Evaluate experiment and mutate exp_data with results."""
        if self.eval_system is None:
            raise RuntimeError("EvaluationSystem not initialized. Call initialize() first.")
        
        # Retrieve experiment data
        exp_data, start, end = self._step_config(exp_data, dimension, step_index)

        # Delegate to evaluation system
        performance_dict = self.eval_system.compute_exp_evaluation(
            exp_data=exp_data,
            evaluate_from=start,
            evaluate_to=end,
            recompute=recompute
        )
        
        # Logging
        self._log_step_completion(exp_data.exp_code, start, end, action="evaluated")
        return performance_dict

    # === PREDICTION STEPS ===

    def training_step(
        self,
        datamodule: DataModule,
        validate: bool = True,
        test: bool = False,
        **kwargs
    ) -> None:
        """Train all prediction models (offline learning)."""
        if self.pred_system is None:
            raise RuntimeError("PredictionSystem not initialized. Call initialize() first.")
        
        # Train prediction models using provided DataModule
        self.pred_system.train(datamodule, **kwargs)

        # Run validation on trained models if requested
        if validate or test:
            self.pred_system.validate(use_test=test)

    def tuning_step(
        self,
        dimension: str,
        step_index: int,
        exp_data: Optional[ExperimentData] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> None:
        """Train prediction models with experiment data, offline and online."""
        if self.pred_system is None:
            raise RuntimeError("PredictionSystem not initialized. Call initialize() first.")
        
        # Retrieve experiment data
        exp_data, start, end = self._step_config(exp_data, dimension, step_index)

        # Delegate to prediction system
        self.pred_system.tune(
            exp_data=exp_data, 
            start=start, 
            end=end, 
            batch_size=batch_size,
            **kwargs
            )
        
    def prediction_step(
        self,
        exp_data: Optional[ExperimentData] = None,
        dimension: Optional[str] = None,
        step_index: Optional[int] = None,
        batch_size: int = 1000
    ) -> Dict[str, np.ndarray]:
        """Predict dimensional features and mutate exp_data with results."""
        if self.pred_system is None:
            raise RuntimeError("PredictionSystem not initialized. Call initialize() first.")
        
        # Retrieve experiment data
        exp_data, start, end = self._step_config(exp_data, dimension, step_index)

        # Delegate to prediction system
        predictions = self.pred_system.predict_experiment(
            exp_data=exp_data,
            predict_from=start,
            predict_to=end,
            batch_size=batch_size
        )
        
        self._log_step_completion(exp_data.exp_code, start, end, action="predicted")
        return predictions
    
    # === CALIBRATION ===
    
    def configure_calibration(
        self,
        offline_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        online_deltas: Optional[Dict[str, float]] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
        performance_weights: Optional[Dict[str, float]] = None
    ) -> None:
        """Configure calibration system parameters."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized.")
            
        if performance_weights:
            self.calibration_system.set_performance_weights(performance_weights)
            
        if offline_bounds:
            self.calibration_system.configure_param_bounds(offline_bounds, fixed_params)
            
        if online_deltas:
            self.calibration_system.configure_trust_regions(online_deltas, fixed_params)
            
        self.logger.info("Configured calibration system.")

    def propose_new_parameters(
        self,
        online: bool = False,
        exploration_weight: float = 0.5,
        current_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Propose next parameter set using calibration system.
        
        Args:
            online: If True, use Trust Region (deltas) and force exploitation.
            exploration_weight: 0.0 (Exploitation) to 1.0 (Exploration).
            current_params: Current parameters (required for online mode).
        """
        if not self._initialized or self.calibration_system is None:
             raise RuntimeError("Agent not initialized.")
        
        # Get or create DataModule
        if self.pred_system and self.pred_system.datamodule:
            dm = self.pred_system.datamodule
        else:
            # Create temp datamodule
            self.logger.info("Creating temporary DataModule for calibration...")
            dm = DataModule(self.calibration_system.dataset)
            dm._fit_normalize()
            
        return self.calibration_system.propose_new_parameters(
            datamodule=dm,
            current_params=current_params,
            online=online,
            w_explore=exploration_weight
        )

    # === Helper Functions ===

    def _instantiate_model_group(
            self,
            model_specs: List[Tuple[Type[Any], dict]],
            system_model_list: List[Any],
            model_type: str
        ) -> None:
        """Instantiate a group of models and add to system."""
        # Instanticate feature model instances from registered classes
        for _class, _kwargs in model_specs:
            self.logger.info(f"Instantiating {model_type} model: {_class.__name__}")
            model = _class(logger=self.logger, **_kwargs)
            if model in system_model_list:
                raise ValueError(f"{model_type} model {_class.__name__} registered multiple times.")
            system_model_list.append(model)

    def _check_sets_against_keys(self, model_codes: Set[str], schema_keys: List[str]) -> None:
        not_represented = model_codes - set(schema_keys)
        if not_represented:
            raise ValueError(
                f"The following codes are not represented in the schema: "
                f"{not_represented}"
            )
        
        unused = set(schema_keys) - model_codes
        if unused:
            self.logger.warning(
                f"The following codes are defined in the schema but not used by any model: "
                f"{unused}"
            )        

    def _toggle_system(self, system_name: SystemNames, activate: bool) -> None:
        """Toggle a system on or off."""
        system = self._get_system(system_name)
        action = "Activated" if activate else "Deactivated"
        
        if activate:
            system.activate()
        else:
            system.deactivate()
        
        self.logger.info(f"{action} {system_name.capitalize()} System.")

    def _get_system(self, system_name: SystemNames) -> Any:
        """Get a system by name, raising an error if not initialized."""
        systems = {
            'evaluation': self.eval_system,
            'prediction': self.pred_system,
            'calibration': self.calibration_system,
        }
        
        if system_name not in systems:
            raise ValueError(f"Unknown system name: {system_name}")
        
        system = systems[system_name]
        if system is None:
            raise RuntimeError(f"{system_name.capitalize()} System not initialized. Call initialize() first.")
        
        return system
    
    def _check_systems(self, step: StepType) -> None:
        """Validate that all systems are initialized and active for a full step."""
        if not self._initialized:
            raise RuntimeError("Cannot perform step before initialize() is called.")
        
        if step == StepType.EVAL:
            rel_systems = [self.feature_system, self.eval_system]
        elif step == StepType.FULL:
            rel_systems = [
                self.feature_system,
                self.eval_system,
                self.pred_system,
                self.calibration_system
            ]
        else:
            raise ValueError(f"Unknown step type: {step}")
        
        # Validate that all required systems are active
        if not all([rel_systems]):
            raise RuntimeError(f"One or more required systems not initialized for {step.value} step.")
        
    def _step_config(
            self, 
            exp_data: Optional[ExperimentData], 
            dimension: Optional[str] = None, 
            step_index: Optional[int] = None
            ) -> Tuple[ExperimentData, int, Optional[int]]:
        """Helper to configure step parameters and retrieve exp_data."""
        # Retrieve experiment data
        if exp_data is None:
            if self.active_exp is None:
                raise RuntimeError("No active experiment set. Call set_active_experiment() first.")
            exp_data = self.active_exp

        # Determine evaluation range
        if any([dimension, step_index]) and not all([dimension, step_index]):
            raise ValueError("Both dimension and step_index must be provided for partial evaluation.")
        # Partial evaluation
        elif dimension and step_index:
            start, end = exp_data.parameters.get_start_and_end_indices(dimension, step_index)
        # Full evaluation
        else:
            start, end = 0, None
        return exp_data, start, end
    
    def _log_step_completion(
            self,
            exp_code: str,
            start: int,
            end: Optional[int],
            action: str
        ) -> None:
        """Helper to log step completion messages."""
        log_text = f"Experiment '{exp_code}' {action} successfully"
        if start and end:
            log_text += f" for dimension range [{start}:{end}]"
        self.logger.console_info(log_text)
