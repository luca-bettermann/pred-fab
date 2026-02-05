"""
LBPAgent - Main orchestration class for AIXD architecture.

Replaces study-based logic with dataset-centric approach while maintaining
existing workflow patterns. Wraps EvaluationSystem, PredictionSystem, and
manages schema generation and dataset initialization.
"""

import numpy as np
import textwrap
from typing import Any, Dict, List, Set, Type, Optional, Tuple, Literal

from pred_fab.core.data_objects import DataArray
from pred_fab.utils.enum import Domain, SystemName
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
from ..utils import LocalData, PfabLogger, StepType, Mode

print("\n===== Welcome to PFAB - Predictive Fabrication =====\n")


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
        self.logger = PfabLogger.get_logger(self.local_data.get_log_folder('logs'))
        self.logger.info("Initializing PfabAgent")
        
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
        specs = (model_class, kwargs)
        if specs in model_specs:
            raise ValueError(f"Model {model_class.__name__} already registered.")
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

    def initialize_systems(self, schema: DatasetSchema, verbose_flag: bool = False) -> None:
        """Initialize dataset and orchestration systems from registered models and validate with schema."""
                
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
        self.feature_system._set_feature_column_names(schema)
        self.feature_system.set_ref_objects(schema)

        # Instantiate evaluation models
        self.eval_system = EvaluationSystem(logger=self.logger)
        self._instantiate_model_group(
            self._evaluation_model_specs,
            self.eval_system.models,
            "evaluation"
        )
        self.eval_system.set_ref_objects(schema)

        # Instantiate prediction models
        self.pred_system = PredictionSystem(logger=self.logger, schema=schema, local_data=self.local_data)
        self._instantiate_model_group(
            self._prediction_model_specs,
            self.pred_system.models,
            "prediction"
        )
        self.pred_system.set_ref_objects(schema)

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

        self.logger.console_success(f"Successfully initialized agentic systems.")
        if verbose_flag:
            self.state_report()
            self.logger.console_new_line()
        
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
        self._check_sets_against_keys(set(output_performance_attrs), schema.performance_attrs.keys())

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
        
    def activate_system(self, system_name: SystemName) -> None:
        """Activate a specific orchestration system."""
        if not self._initialized:
            raise RuntimeError("Cannot activate systems before initialize() is called.")
        self._toggle_system(system_name, activate=True)

    def deactivate_system(self, system_name: SystemName) -> None:
        """Deactivate a specific orchestration system."""
        if not self._initialized:
            raise RuntimeError("Cannot deactivate systems before initialize() is called.")
        self._toggle_system(system_name, activate=False)

    def set_active_experiment(self, exp_data: ExperimentData) -> None:
        """Set the active experiment for online operations."""
        if not self._initialized:
             raise RuntimeError("Agent not initialized.")
        
        self.active_exp = exp_data
        self.logger.info(f"Active experiment set to: {exp_data.code}")

    def state_report(self) -> None:
        """Log an overview of the registered models and their I/O to the console."""
        if not self._initialized:
            self.logger.console_warning("Agent not initialized. No models to report.")
            return

        summary = ["\n===== Agent State Report ====="]
        

        def _add_section(name: str, models: List[Any], width=25) -> None:
            if not models:
                return
            
            # Header with newline before it
            header = f"\n{name:<{width}} | {'Inputs':<{width}} | {'Outputs':<{width}}"
            summary.append(header)
            divider = "-" * (3 * width + 6)
            summary.append(divider)
            
            for model in models:
                inp_str = ", ".join(model.input_parameters + model.input_features)
                out_str = ", ".join(model.outputs)
                
                # Wrap text to multiple lines if needed
                inp_lines = textwrap.wrap(inp_str, width=width) if inp_str else [""]
                out_lines = textwrap.wrap(out_str, width=width) if out_str else [""]
                
                # If wrap returns empty list for empty string (python behavior varies), ensure list has one element
                if not inp_lines: inp_lines = [""]
                if not out_lines: out_lines = [""]

                max_lines = max(len(inp_lines), len(out_lines))
                model_name = model.__class__.__name__

                for i in range(max_lines):
                    col1 = model_name if i == 0 else ""
                    col2 = inp_lines[i] if i < len(inp_lines) else ""
                    col3 = out_lines[i] if i < len(out_lines) else ""
                    summary.append(f"{col1:<{width}} | {col2:<{width}} | {col3:<{width}}")

        # Report on systems
        if self.feature_system:
            _add_section("Feature System", self.feature_system.models)
            
        if self.eval_system:
            _add_section("Evaluation System", self.eval_system.models)
            
        if self.pred_system:
            _add_section("Prediction System", self.pred_system.models)
            
        print("\n".join(summary))

    def calibration_state_report(self) -> None:
        self.calibration_system.state_report()

    # === OFFLINE STEP OPERATIONS ==

    def evaluation_step(
        self,
        exp_data: ExperimentData,
        recompute_flag: bool = False,
        visualize: bool = False
    ) -> None:
        """Perform a exploration step of all active systems."""
        self._check_systems(StepType.EVAL)

        # Set start and end values
        start, end = 0, None
        
        # Extract Features and Evaluate Performance
        self.feature_system.run_feature_extraction(exp_data, start, end, recompute=recompute_flag, visualize=visualize)
        self.eval_system.run_evaluation(exp_data, start, end, recompute=recompute_flag)
        self.logger.console_success(f"Successfully evaluated experiment '{exp_data.code}'.")
    
    def exploration_step(
        self,
        datamodule: DataModule,
        w_explore: float = 0.5,
        n_optimization_rounds: int = 10
    ) -> Dict[str, Any]:
        """Perform a exploration step of all active systems."""
        self._check_systems(StepType.FULL)

        # Train Exploration Model and calibrate new experiment
        new_params = self.calibration_system.run_calibration(
            datamodule=datamodule,
            mode=Mode.EXPLORATION,
            fixed_context=None,
            w_explore=w_explore,
            n_optimization_rounds=n_optimization_rounds
        )

        self.logger.console_success("Successfully completed exploration step. Proposed new parameters:")
        for key, value in new_params.items():
            self.logger.console_success(f"  {key}: {value}")

        return new_params

    def inference_step(
        self,
        exp_data: ExperimentData,
        datamodule: DataModule,
        w_explore: float = 0.5,
        n_optimization_rounds: int = 10,
        recompute: bool = False,
        visualize: bool = False,
    ) -> Dict[str, Any]:
        """Perform a exploration step of all active systems."""
        self._check_systems(StepType.FULL)

        # Set start and end values
        start, end = 0, None
        
        # 1. Extract Features
        self.feature_system.run_feature_extraction(exp_data, start, end, recompute=recompute, visualize=visualize)

        # 2. Evaluate Performance
        self.eval_system.run_evaluation(exp_data, start, end, recompute=recompute)

        # 3. Train Exploration Model and calibrate new experiment
        new_params = self.calibration_system.run_calibration(
            datamodule=datamodule,
            mode=Mode.EXPLORATION,
            fixed_context=None,
            w_explore=w_explore,
            n_optimization_rounds=n_optimization_rounds
        )

        self.logger.console_success("Successfully completed exploration step. Proposed new parameters:")
        for key, value in new_params.items():
            self.logger.console_success(f"  {key}: {value}")

        return new_params

    def step_offline(
        self,
        exp_data: ExperimentData,
        datamodule: Optional[DataModule] = None,
        step_type: StepType = StepType.FULL,
        phase: Mode = Mode.INFERENCE,
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
        self.feature_system.run_feature_extraction(exp_data, start, end, recompute=recompute, visualize=visualize)
        self._log_step_completion(exp_data.code, start, end, action="had features extracted")
        
        # 2. Evaluate Performance
        self.eval_system.run_evaluation(exp_data, start, end, recompute=recompute)
        self._log_step_completion(exp_data.code, start, end, action="evaluated")

        # End step here if only evaluation is requested
        if step_type == StepType.EVAL:
            return
        
        # TODO: figure out when datamodule is needed, and when not
        if datamodule is None:
            raise ValueError("DataModule must be provided for training in offline step.")

        # 4. Train Exploration Model and calibrate new experiment
        current_params = exp_data.parameters.get_values_dict()
        new_params = self.calibration_system.run_calibration(
            datamodule, Domain.OFFLINE, phase, current_params, w_explore, n_optimization_rounds)
        new_exp_data = datamodule.dataset.create_experiment("new_exp", new_params)
        self._log_step_completion(exp_data.code, start, end, action="calibrated")
        
        # 5. Predict features of new experiment -> we only do this in inference mode
        self.pred_system.predict_experiment(new_exp_data)
        self._log_step_completion(new_exp_data.code, start, end, action="predicted features")
        return new_exp_data
        
    def step_online(
        self,
        exp_data: Optional[ExperimentData] = None,
        step_type: StepType = StepType.FULL,
        phase: Mode = Mode.INFERENCE,
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
        self.feature_system.run_feature_extraction(exp_data, start, end, recompute=recompute, visualize=visualize)
        self._log_step_completion(exp_data.code, start, end, action="had features extracted")

        # 2. Evaluate Performance
        self.eval_system.run_evaluation(exp_data, start, end, recompute=recompute)
        self._log_step_completion(exp_data.code, start, end, action="evaluated")

        if step_type == StepType.EVAL:
            return
        
        # 3. Tune Prediction Model
        temp_datamodule = self.pred_system.tune(exp_data, start, end, batch_size, **kwargs)
        self._log_step_completion(exp_data.code, start, end, action="used for tuning")

        # TODO: How are we handling if a feature does not contain the dimension we are stepping over?

        # 4. Train Exploration Model and calibrate process parameters
        current_params = exp_data.parameters.get_values_dict()
        new_params = self.calibration_system.run_adaptation(
            datamodule=temp_datamodule,
            mode=phase,
            current_params=current_params,
            w_explore=0.0 # Default/Unused for Adaptation/Inference
        )
        self._log_step_completion(exp_data.code, start, end, action="calibrated")
        return new_params

    # === PARTIAL STEP OPERATIONS ===

    # def feature_step(
    #     self,
    #     exp_data: Optional[ExperimentData],
    #     dimension: Optional[str] = None,
    #     step_index: Optional[int] = None,
    #     recompute: bool = False,
    #     visualize: bool = False
    # ) -> Dict[str, np.ndarray]:
    #     """Evaluate experiment and mutate exp_data with results."""
    #     if self.eval_system is None:
    #         raise RuntimeError("EvaluationSystem not initialized. Call initialize() first.")
        
    #     # Retrieve experiment data
    #     exp_data, start, end = self._step_config(exp_data, dimension, step_index)

    #     # Delegate to evaluation system
    #     feature_dict = self.feature_system.run_feature_extraction(
    #         exp_data=exp_data,
    #         evaluate_from=start,
    #         evaluate_to=end,
    #         recompute=recompute,
    #         visualize=visualize
    #     )
        
    #     # Logging
    #     self._log_step_completion(exp_data.exp_code, start, end, action="computed features")
    #     return feature_dict

    # def evaluation_step(
    #     self,
    #     exp_data: Optional[ExperimentData],
    #     dimension: Optional[str] = None,
    #     step_index: Optional[int] = None,
    #     recompute: bool = False
    # ) -> Dict[str, Optional[float]]:
    #     """Evaluate experiment and mutate exp_data with results."""
    #     if self.eval_system is None:
    #         raise RuntimeError("EvaluationSystem not initialized. Call initialize() first.")
        
    #     # Retrieve experiment data
    #     exp_data, start, end = self._step_config(exp_data, dimension, step_index)

    #     # Delegate to evaluation system
    #     performance_dict = self.eval_system.run_evaluation(
    #         exp_data=exp_data,
    #         evaluate_from=start,
    #         evaluate_to=end,
    #         recompute=recompute
    #     )
        
    #     # Logging
    #     self._log_step_completion(exp_data.exp_code, start, end, action="evaluated")
    #     return performance_dict

    # === PREDICTION STEPS ===

    def training_step(
        self,
        datamodule: DataModule,
        validate: bool = True,
        test: bool = False,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Train all prediction models (offline learning)."""
        if self.pred_system is None:
            raise RuntimeError("PredictionSystem not initialized. Call initialize() first.")
        
        # Train prediction models using provided DataModule
        self.pred_system.train(datamodule, **kwargs)

        # Run validation on trained models if requested
        if validate or test:
            return self.pred_system.validate(use_test=test)

    def adaptation_step(
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
        
        self._log_step_completion(exp_data.code, start, end, action="predicted")
        return predictions
    
    # === CALIBRATION ===
    
    def configure_calibration(
        self,
        performance_weights: Optional[Dict[str, float]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
        adaptation_delta: Optional[Dict[str, float]] = None,
        force: bool = False
    ) -> None:
        """Configure calibration system parameters."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized.")
            
        if performance_weights:
            self.calibration_system.set_performance_weights(performance_weights)
            self.logger.info("Configured performance weights for calibration system.")
        if bounds:
            self.calibration_system.configure_param_bounds(bounds, force=force)
            self.logger.info("Configured parameter bounds for calibration system.")
        if fixed_params:
            self.calibration_system.configure_fixed_params(fixed_params, force=force)
            self.logger.info("Configured fixed parameters for calibration system.")
        if adaptation_delta:
            self.calibration_system.configure_adaptation_delta(adaptation_delta, force=force)
            self.logger.info("Configured adaptation delta for calibration system.")

    # def propose_new_parameters(
    #     self,
    #     online: bool = False,
    #     exploration_weight: float = 0.5,
    #     current_params: Optional[Dict[str, Any]] = None
    # ) -> Dict[str, Any]:
    #     """
    #     Propose next parameter set using calibration system.
        
    #     Args:
    #         online: If True, use adaptation delta (trust region) and force exploitation.
    #         exploration_weight: 0.0 (Exploitation) to 1.0 (Exploration).
    #         current_params: Current parameters (required for online mode).
    #     """
    #     if not self._initialized or self.calibration_system is None:
    #          raise RuntimeError("Agent not initialized.")
        
    #     # Get or create DataModule
    #     if self.pred_system and self.pred_system.datamodule:
    #         dm = self.pred_system.datamodule
    #     else:
    #         # Create temp datamodule
    #         self.logger.info("Creating temporary DataModule for calibration...")
    #         dm = DataModule(self.calibration_system.dataset)
    #         dm._fit_normalize()
            
    #     params = current_params or {}
        
    #     if online:
    #         return self.calibration_system.run_adaptation(
    #             datamodule=dm,
    #             mode=Mode.EXPLORATION,
    #             current_params=params,
    #             w_explore=exploration_weight
    #         )
    #     else:
    #          return self.calibration_system.run_calibration(
    #             datamodule=dm,
    #             mode=Mode.EXPLORATION,
    #             fixed_context=params,
    #             w_explore=exploration_weight
    #         )

    # === Helper Functions ===

    def _instantiate_model_group(
            self,
            model_specs: List[Tuple[Type[Any], dict]],
            system_model_list: List[Any],
            model_type: str
        ) -> None:
        """Instantiate a group of models and add to system."""
        # Instanticate model instances from registered classes
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

    def _toggle_system(self, system_name: SystemName, activate: bool) -> None:
        """Toggle a system on or off."""
        system = self._get_system(system_name)
        action = "Activated" if activate else "Deactivated"
        
        if activate:
            system.activate()
        else:
            system.deactivate()
        
        self.logger.info(f"{action} {system_name.value.capitalize()} System.")

    def _get_system(self, system_name: SystemName) -> Any:
        """Get a system by name, raising an error if not initialized."""
        systems = {
            SystemName.EVALUATION: self.eval_system,
            SystemName.PREDICTION: self.pred_system,
            SystemName.CALIBRATION: self.calibration_system,
        }
        
        if system_name not in systems:
            raise ValueError(f"Unknown system name: {system_name}")
        
        system = systems[system_name]
        if system is None:
            raise RuntimeError(f"{system_name.value.capitalize()} System not initialized. Call initialize() first.")
        
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
