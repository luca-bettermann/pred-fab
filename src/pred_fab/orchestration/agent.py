"""PfabAgent — main orchestration class for the PFAB framework."""

import textwrap
from typing import Any, Dict, List, Set, Type, Optional, Tuple
import numpy as np

from pred_fab.utils.enum import SystemName
from ..core.schema import DatasetSchema
from ..core.dataset import Dataset, ExperimentData
from ..core.datamodule import DataModule
from ..core import ExperimentSpec
from ..orchestration import (
    FeatureSystem,
    EvaluationSystem,
    PredictionSystem,
    CalibrationSystem
)

from ..interfaces import IFeatureModel, IEvaluationModel, IPredictionModel
from ..interfaces.calibration import GaussianProcessSurrogate
from ..utils import LocalData, PfabLogger, StepType, Mode, SplitType


class PfabAgent:
    """Main orchestration class: coordinates EvaluationSystem, PredictionSystem, and CalibrationSystem."""

    def __init__(
        self,
        root_folder: str,
        debug_flag: bool = False,
        recompute_flag: bool = False,
        visualize_flag: bool = True,
    ):
        
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

        # GP surrogate for uncertainty estimation (fitted after each train() call)
        self._gp_surrogate: Optional[GaussianProcessSurrogate] = None

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

        # Calibration system requires prediction, evaluation and schema to be set.
        # perf_fn encapsulates predict_for_calibration + _evaluate_feature_dict so
        # CalibrationSystem never calls internal PM/EM methods directly.
        _pred = self.pred_system
        _eval = self.eval_system

        def _perf_fn(params_dict):
            try:
                feature_arrays, params_block = _pred.predict_for_calibration(params_dict)
                return _eval._evaluate_feature_dict(feature_arrays, params_block)
            except Exception:
                return {}

        # GP surrogate for uncertainty estimation.
        # is_fitted=False before train() — uncertainty_fn returns 1.0 (max) until fitted.
        self._gp_surrogate = GaussianProcessSurrogate(self.logger)
        _gp = self._gp_surrogate

        def _gp_uncertainty_fn(X: np.ndarray) -> float:
            """Return GP-based std as uncertainty ∈ [0, 1]; 1.0 before GP is fitted."""
            if not _gp.is_fitted:
                return 1.0
            _, std = _gp.predict(X.reshape(1, -1))
            return float(np.clip(np.mean(std), 0.0, 1.0))

        def _gp_similarity_fn(X1: np.ndarray, X2: np.ndarray) -> float:
            """Gaussian kernel similarity in normalized parameter space."""
            diff = X1.reshape(-1) - X2.reshape(-1)
            return float(np.exp(-float(np.dot(diff, diff))))

        self.calibration_system = CalibrationSystem(
            schema=schema,
            logger=self.logger,
            perf_fn=_perf_fn,
            uncertainty_fn=_gp_uncertainty_fn,
            similarity_fn=_gp_similarity_fn,
        )

        # validate against schema
        self._validate_systems_against_schema(schema)
        self._initialized = True

        self.logger.console_success(f"Successfully initialized agentic systems.")
        if verbose_flag:
            self.state_report()
            self.logger.console_new_line()

    def create_datamodule(self, dataset: Dataset) -> DataModule:
        # Get the input and output columns
        system_input_param = self.pred_system.get_system_input_parameters()
        system_input_feats = self.pred_system.get_system_input_features()
        system_outputs = self.pred_system.get_system_outputs()
        
        # Initialize datamodule
        datamodule = DataModule(dataset)
        datamodule.initialize(system_input_param, system_input_feats, system_outputs)
        return datamodule
        
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
            
        self.logger.console_new_line()
        self.logger.console_info("\n".join(summary))
        self.logger.console_new_line()

    def calibration_state_report(self) -> None:
        self.calibration_system.state_report()

    # === STEP METHODS ==
    
    def exploration_step(
        self,
        datamodule: DataModule,
        w_explore: float = 0.5,
        n_optimization_rounds: int = 10,
        current_params: Optional[Dict[str, Any]] = None,
    ) -> ExperimentSpec:
        """UCB-based exploration proposal. Iterates over trajectory dimensions when configured."""
        self._check_systems(StepType.FULL)

        result = self.calibration_system.run_calibration(
            datamodule=datamodule,
            mode=Mode.EXPLORATION,
            current_params=current_params,
            w_explore=w_explore,
            n_optimization_rounds=n_optimization_rounds,
        )

        self.logger.console_success("Successfully completed exploration step.")
        return result

    def inference_step(
        self,
        exp_data: ExperimentData,
        datamodule: DataModule,
        w_explore: float = 0.5,
        n_optimization_rounds: int = 10,
        recompute: bool = False,
        visualize: bool = False,
        current_params: Optional[Dict[str, Any]] = None,
    ) -> ExperimentSpec:
        """Extract features, evaluate, then return an inference-guided proposal."""
        self._check_systems(StepType.FULL)

        start, end = 0, None

        # 1. Extract Features
        self.feature_system.run_feature_extraction(exp_data, start, end, recompute=recompute, visualize=visualize)

        # 2. Evaluate Performance
        self.eval_system.run_evaluation(exp_data, start, end, recompute=recompute)

        # 3. Calibrate
        result = self.calibration_system.run_calibration(
            datamodule=datamodule,
            mode=Mode.INFERENCE,
            current_params=current_params,
            w_explore=w_explore,
            n_optimization_rounds=n_optimization_rounds,
        )

        self.logger.console_success("Successfully completed inference step.")
        return result

    # === ADDITIONAL API CALLS ===

    def evaluate(
        self,
        exp_data: ExperimentData,
        recompute_flag: bool = False,
        visualize: bool = False
    ) -> None:
        """Evaluate an experiment and mutate features/performance in place."""
        self._check_systems(StepType.EVAL)

        # Set start and end values
        start, end = 0, None
        
        # Extract Features and Evaluate Performance
        self.feature_system.run_feature_extraction(exp_data, start, end, recompute=recompute_flag, visualize=visualize)
        self.eval_system.run_evaluation(exp_data, start, end, recompute=recompute_flag)
        self.logger.console_success(f"Successfully evaluated experiment '{exp_data.code}'.")

    def train(
        self,
        datamodule: DataModule,
        validate: bool = True,
        test: bool = False,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Train prediction models, fit GP surrogate, and optionally validate/test."""
        if self.pred_system is None:
            raise RuntimeError("PredictionSystem not initialized. Call initialize() first.")

        # Train prediction models using provided DataModule
        self.pred_system.train(datamodule, **kwargs)

        # Fit GP surrogate on experiment-level (params → performance) data
        self._fit_gp_surrogate(datamodule)

        # Run validation on trained models if requested
        if validate or test:
            return self.pred_system.validate(use_test=test)
        
    def predict(
        self,
        exp_data: Optional[ExperimentData] = None,
        dimension: Optional[str] = None,
        step_index: Optional[int] = None,
        batch_size: int = 1000
    ) -> Dict[str, np.ndarray]:
        """Predict features for an experiment slice and mutate feature tensors."""
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

    def baseline_step(
        self,
        n: int,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        n_optimization_rounds: int = 10,
    ) -> List[ExperimentSpec]:
        """Generate n space-filling baseline proposals (greedy maximin). No trained model required."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized.")
        result = self.calibration_system.run_baseline(
            n=n,
            param_bounds=param_bounds,
            n_optimization_rounds=n_optimization_rounds,
        )
        self.logger.console_success(f"Successfully completed baseline step ({n} proposals).")
        return result

    # === Helper Functions ===

    def _fit_gp_surrogate(self, datamodule: DataModule) -> None:
        """Fit GP surrogate on experiment-level (params → performance) data from the training split."""
        if self._gp_surrogate is None or self.calibration_system is None:
            return
        perf_names = self.calibration_system.perf_names_order
        X_list, y_list = [], []
        for code in datamodule.get_split_codes(SplitType.TRAIN):
            exp = datamodule.dataset.get_experiment(code)
            if not exp.performance:
                continue
            try:
                params = exp.parameters.get_values_dict()
                x = datamodule.params_to_array(params)
                y = [float(exp.performance.get_value(name)) for name in perf_names]
                X_list.append(x)
                y_list.append(y)
            except Exception as e:
                self.logger.debug(f"Skipping experiment '{code}' for GP training: {e}")
        if len(X_list) >= 2:
            self._gp_surrogate.fit(np.array(X_list), np.array(y_list))
        else:
            self.logger.info("Not enough experiments with performance data to fit GP surrogate.")

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
        if not all(rel_systems):
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

        # Determine evaluation range (use explicit None checks to handle step_index=0)
        if (dimension is None) != (step_index is None):
            raise ValueError("Both dimension and step_index must be provided for partial evaluation.")
        # Partial evaluation
        elif dimension is not None and step_index is not None:
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
