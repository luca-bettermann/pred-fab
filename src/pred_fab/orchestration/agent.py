"""PfabAgent — main orchestration class for the PFAB framework."""

import textwrap
from typing import Any
import numpy as np

from pred_fab.utils.enum import SystemName
from ..core.schema import DatasetSchema
from ..core.dataset import Dataset, ExperimentData
from ..core.datamodule import DataModule
from ..core.data_objects import DataArray
from ..core import ParameterProposal, ExperimentSpec
from ..orchestration import (
    FeatureSystem,
    EvaluationSystem,
    PredictionSystem,
    CalibrationSystem,
    Optimizer,
)

from ..interfaces import IFeatureModel, IEvaluationModel, IPredictionModel
from ..utils import LocalData, PfabLogger, ConsoleReporter, StepType, Mode, SourceStep


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
        self._feature_model_specs: list[tuple[type[IFeatureModel], dict]] = []  # List of (class, kwargs)
        self._evaluation_model_specs: list[tuple[type[IEvaluationModel], dict]] = []  # List of (class, kwargs)
        self._prediction_model_specs: list[tuple[type[IPredictionModel], dict]] = []  # List of (class, kwargs)
        
        # Initialization state guard
        self._initialized = False

        # Context snapshot: current measured values of context features, injected into perf_fn.
        self._context_snapshot: dict[str, float] = {}

        # Console reporter (created in initialize_systems)
        self._console: ConsoleReporter | None = None

        # Progress tracking
        self.active_exp: ExperimentData | None = None
        
    def _assert_initialized(self) -> None:
        """Raise if the agent has not been initialized yet."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize_systems() first.")

    @property
    def console(self) -> ConsoleReporter:
        """Schema-aware console reporter; available after initialize_systems()."""
        if self._console is None:
            raise RuntimeError("Console not available. Call initialize_systems() first.")
        return self._console

    # === MODEL REGISTRATION ===

    def _register_model(
            self, 
            model_class: type[Any], 
            model_specs: list[tuple[type[Any], dict]], 
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

    def register_feature_model(self, feature_class: type[Any], **kwargs) -> None:
        """Register a feature model."""
        self._register_model(feature_class, self._feature_model_specs, kwargs)

    def register_evaluation_model(self, evaluation_class: type[IEvaluationModel], **kwargs) -> None:
        """Register an evaluation model."""
        self._register_model(evaluation_class, self._evaluation_model_specs, kwargs)
    
    def register_prediction_model(self, prediction_class: type[IPredictionModel], **kwargs) -> None:
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
        _ctx = self._context_snapshot  # mutable dict; closure captures reference

        def _perf_fn(params_dict):
            try:
                # Merge current context snapshot so prediction model receives observed covariates.
                merged = dict(params_dict)
                if _ctx:
                    merged.update(_ctx)
                feature_arrays, params_block = _pred.predict_for_calibration(merged)
                return _eval._evaluate_feature_dict(feature_arrays, params_block)
            except Exception:
                return {}

        self.calibration_system = CalibrationSystem(
            schema=schema,
            logger=self.logger,
            perf_fn=_perf_fn,
            uncertainty_fn=_pred.uncertainty,
            similarity_fn=_pred.kernel_similarity,
        )

        # Wire up virtual KDE point callbacks for within-trajectory spacing.
        # The DataModule reference is captured at call time via the closure.
        def _add_vp(params: dict[str, Any]) -> None:
            dm = self.calibration_system._active_datamodule
            if dm is not None:
                _pred.add_virtual_point(params, dm)

        self.calibration_system._add_virtual_point_fn = _add_vp
        self.calibration_system._clear_virtual_points_fn = _pred.clear_virtual_points

        # validate against schema
        self._validate_systems_against_schema(schema)

        # Build console reporter from schema metadata
        from ..core.data_objects import DataCategorical
        param_codes = list(schema.parameters.keys())
        perf_codes = list(schema.performance_attrs.keys())
        param_categories = {
            code: obj.constraints.get("categories", [])
            for code, obj in schema.parameters.items()
            if isinstance(obj, DataCategorical)
        }
        self._console = ConsoleReporter(
            logger=self.logger,
            param_codes=param_codes,
            perf_codes=perf_codes,
            param_categories=param_categories,
        )

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

        # Recursive features are derived by the FeatureSystem (tensor shifting),
        # not by any registered feature model — treat them as computed.
        recursive_codes = {
            code for code, obj in schema.features.items()
            if isinstance(obj, DataArray) and obj.is_recursive
        }
        output_features_set = set(output_features) | recursive_codes

        # Validate that all input features are represented as outputs features
        uncomputed_inputs = set(input_features) - output_features_set
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
        self._assert_initialized()
        
        self.active_exp = exp_data
        self.logger.info(f"Active experiment set to: {exp_data.code}")

    def state_report(self) -> None:
        """Log an overview of the registered models and their I/O to the console."""
        if not self._initialized:
            self.logger.console_warning("Agent not initialized. No models to report.")
            return

        summary = ["\n===== Agent State Report ====="]
        
        def _add_section(name: str, models: list[Any], width=25) -> None:
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
        n_optimization_rounds: int = 5,
        current_params: dict[str, Any] | None = None,
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

        self.logger.info("Successfully completed exploration step.")
        return result

    def inference_step(
        self,
        exp_data: ExperimentData,
        datamodule: DataModule,
        w_explore: float = 0.5,
        n_optimization_rounds: int = 5,
        recompute: bool = False,
        visualize: bool = False,
        current_params: dict[str, Any] | None = None,
    ) -> ExperimentSpec:
        """Extract features, evaluate, then return an inference-guided proposal."""
        self._check_systems(StepType.FULL)

        # 1. Extract Features
        self.feature_system.run_feature_extraction(exp_data, 0, None, recompute=recompute, visualize=visualize)

        # 2. Evaluate Performance
        self.eval_system.run_evaluation(exp_data, recompute=recompute)

        # 3. Calibrate
        result = self.calibration_system.run_calibration(
            datamodule=datamodule,
            mode=Mode.INFERENCE,
            current_params=current_params,
            w_explore=w_explore,
            n_optimization_rounds=n_optimization_rounds,
        )

        self.logger.info("Successfully completed inference step.")
        return result

    def adaptation_step(
        self,
        dimension: str | None = None,
        step_index: int | None = None,
        exp_data: ExperimentData | None = None,
        mode: Mode = Mode.INFERENCE,
        w_explore: float = 0.0,
        record: bool = False,
        **kwargs
    ) -> ExperimentSpec:
        """Tune on a step slice then return an online calibration proposal.

        batch_size is derived automatically (one batch per dimension step);
        pass ``batch_size=N`` via ``**kwargs`` to override.
        """
        if self.pred_system is None:
            raise RuntimeError("PredictionSystem not initialized. Call initialize() first.")
        if self.calibration_system is None:
            raise RuntimeError("CalibrationSystem not initialized. Call initialize() first.")

        # Retrieve experiment data
        exp_data, start, end = self._step_config(exp_data, dimension, step_index)

        # Tune prediction system on the requested online slice.
        temp_datamodule = self.pred_system.tune(
            exp_data=exp_data,
            start=start,
            end=end,
            **kwargs
        )
        self._log_step_completion(exp_data.code, start, end, action="used for tuning")

        # Calibrate around effective current parameters (online = single step).
        current_params = exp_data.get_effective_parameters_at_step(dimension=dimension, step_index=step_index)
        target_indices = {dimension: step_index} if dimension is not None and step_index is not None else {}
        result = self.calibration_system.run_calibration(
            datamodule=temp_datamodule,
            mode=mode,
            current_params=current_params,
            target_indices=target_indices,
            w_explore=w_explore,
        )
        # Tag as adaptation (run_calibration uses mode-derived source_step).
        proposal = ParameterProposal.from_dict(
            result.initial_params.to_dict(), source_step=SourceStep.ADAPTATION,
        )
        result = ExperimentSpec(initial_params=proposal, schedules=result.schedules)

        # Record only if user confirms that proposed changes were applied physically.
        if record:
            exp_data.record_parameter_update(proposal, dimension=dimension, step_index=step_index)
            self._log_step_completion(exp_data.code, start, end, action="recorded parameter update")

        self.logger.info("Successfully completed adaptation step.")
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

        # Extract Features and Evaluate Performance
        self.feature_system.run_feature_extraction(exp_data, 0, None, recompute=recompute_flag, visualize=visualize)
        self.eval_system.run_evaluation(exp_data, recompute=recompute_flag)
        self.logger.info(f"Successfully evaluated experiment '{exp_data.code}'.")

    def train(
        self,
        datamodule: DataModule,
        validate: bool = True,
        test: bool = False,
        **kwargs
    ) -> dict[str, Any] | None:
        """Train prediction models and optionally validate/test."""
        if self.pred_system is None:
            raise RuntimeError("PredictionSystem not initialized. Call initialize() first.")
        
        # Train prediction models using provided DataModule
        self.pred_system.train(datamodule, **kwargs)

        # Update running performance range for acquisition normalization
        if self.calibration_system is not None:
            self.calibration_system.update_perf_range(datamodule)

        # Run validation on trained models if requested
        if validate or test:
            perf_weights = None
            if self.calibration_system is not None:
                perf_weights = self.calibration_system.performance_weights
            return self.pred_system.validate(
                use_test=test, performance_weights=perf_weights,
            )
        
    def predict(
        self,
        exp_data: ExperimentData | None = None,
        dimension: str | None = None,
        step_index: int | None = None,
        batch_size: int = 1000
    ) -> dict[str, np.ndarray]:
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
    
    def configure(
        self,
        *,
        bounds: dict[str, tuple[float, float]] | None = None,
        performance_weights: dict[str, float] | None = None,
        fixed_params: dict[str, Any] | None = None,
        adaptation_delta: dict[str, float] | None = None,
        step_parameters: dict[str, str] | None = None,
        ofat_strategy: list[str] | None = None,
        exploration_radius: float | None = None,
        optimizer: Optimizer | None = None,
        online_optimizer: Optimizer | None = None,
        mpc_lookahead: int | None = None,
        mpc_discount: float | None = None,
        boundary_buffer: tuple[float, float, float] | None = None,
        de_maxiter: int | None = None,
        de_popsize: int | None = None,
        lbfgsb_maxfun: int | None = None,
        lbfgsb_eps: float | None = None,
        trajectory_smoothing: float | None = None,
        force: bool = False,
    ) -> None:
        """Configure the agent.  All parameters are keyword-only and optional.

        Call configure() one or more times at any phase — only the supplied
        arguments are applied; the rest remain unchanged.

        bounds              — search space for offline calibration, keyed by parameter code.
        performance_weights — relative importance of each performance attribute (default 1.0).
        fixed_params        — parameters held constant during calibration (e.g. design intent).
        adaptation_delta    — trust-region half-widths for online/layer-by-layer adaptation.
        step_parameters     — {param_code: dimension_code} pairs declaring which runtime
                              parameters are re-optimised at each step of the given dimension.
        ofat_strategy       — cycle through these parameter codes one-at-a-time per adaptation
                              step (OFAT).  Requires adaptation_delta to be set first.
        exploration_radius  — NatPN evidence model bubble size c:
                              h = c·√d/√N (radius), γ = max(1, c·√N) (sharpness).
                              Larger c → slower transition from exploration to exploitation.
        optimizer           — offline optimizer: Optimizer.DE (default) or Optimizer.LBFGSB.
        online_optimizer    — online/adaptation optimizer: Optimizer.LBFGSB (default) or Optimizer.DE.
        mpc_lookahead       — N-step lookahead for MPC (default 0 = greedy single-step).
        mpc_discount        — discount factor γ for MPC: step j counts as γʲ (default 0.9).
        boundary_buffer     — (extent, strength, exponent) for boundary penalty.
                              extent: fraction of range from each boundary (e.g. 0.1 = 10%).
                              strength: penalty at boundary — score *= (1 - strength).
                              exponent: curve shape (1=linear, 2=quadratic, >2=steeper).
                              Pass (0, 0, 0) or None to disable.
        de_maxiter          — DE: maximum generations (default 100).
        de_popsize          — DE: population size per dimension (default 10).
        lbfgsb_maxfun       — L-BFGS-B: max function evaluations per start (default: auto).
        lbfgsb_eps          — L-BFGS-B: finite-difference step size (default 1e-3).
        trajectory_smoothing — penalize speed changes between adjacent trajectory layers.
                              0 = disabled, 0.1 = mild, 0.3 = strong. Encourages monotonic
                              trajectories by reducing the acquisition score proportional to
                              |speed_change| / trust_region_width.
        force               — overwrite already-configured settings without warning.
        """
        self._assert_initialized()

        if bounds is not None:
            self.calibration_system.configure_param_bounds(bounds, force=force)
        if performance_weights is not None:
            self.calibration_system.set_performance_weights(performance_weights)
            if self._console is not None:
                self._console._perf_weights = performance_weights
            # Map performance weights to feature names for per-model KDE aggregation.
            feature_weights: dict[str, float] = {}
            for eval_model in self.eval_system.models:
                perf_name = eval_model.output_performance
                feat_name = eval_model.input_feature
                if perf_name in performance_weights:
                    feature_weights[feat_name] = performance_weights[perf_name]
            if feature_weights:
                self.pred_system.set_uncertainty_weights(feature_weights)
        if fixed_params is not None:
            self.calibration_system.configure_fixed_params(fixed_params, force=force)
        if adaptation_delta is not None:
            self.calibration_system.configure_adaptation_delta(adaptation_delta, force=force)
        if step_parameters is not None:
            for param_code, dim_code in step_parameters.items():
                self.calibration_system.configure_step_parameter(param_code, dim_code, force=force)
        if ofat_strategy is not None:
            self.calibration_system.configure_ofat_strategy(ofat_strategy)
        if exploration_radius is not None:
            self.pred_system.configure_exploration(exploration_radius)
        if optimizer is not None:
            self.calibration_system.optimizer = optimizer
        if mpc_lookahead is not None:
            self.calibration_system.default_mpc_lookahead = mpc_lookahead
        if mpc_discount is not None:
            self.calibration_system.default_mpc_discount = mpc_discount
        if online_optimizer is not None:
            self.calibration_system.online_optimizer = online_optimizer
        if boundary_buffer is not None:
            extent, strength, exponent = boundary_buffer
            self.calibration_system.boundary_buffer_extent = extent
            self.calibration_system.boundary_buffer_strength = strength
            self.calibration_system.boundary_buffer_exponent = exponent
        if de_maxiter is not None:
            self.calibration_system.de_maxiter = de_maxiter
        if de_popsize is not None:
            self.calibration_system.de_popsize = de_popsize
        if lbfgsb_maxfun is not None:
            self.calibration_system.lbfgsb_maxfun = lbfgsb_maxfun
        if lbfgsb_eps is not None:
            self.calibration_system.lbfgsb_eps = lbfgsb_eps
        if trajectory_smoothing is not None:
            self.calibration_system.trajectory_smoothing = trajectory_smoothing

    # ── Optimizer telemetry (read-only, set after each calibration step) ────────

    @property
    def last_opt_score(self) -> float:
        """Best acquisition score found in the most recent calibration step."""
        return self.calibration_system.last_opt_score

    @property
    def last_opt_nfev(self) -> int:
        """Number of objective evaluations in the most recent calibration step."""
        return self.calibration_system.last_opt_nfev

    @property
    def last_opt_n_starts(self) -> int:
        """Number of optimizer restarts used in the most recent calibration step."""
        return self.calibration_system.last_opt_n_starts

    # ── Acquisition introspection ───────────────────────────────────────────────

    def predict_performance(self, params: dict[str, Any]) -> dict[str, float | None]:
        """Run the prediction + evaluation pipeline at params and return performance scores.

        Uses the same perf_fn the optimizer uses internally — useful for logging what the
        model predicts at a proposed point before committing to an experiment.
        Requires the agent to be trained (pred_system must have a fitted model).
        """
        self._assert_initialized()
        return self.calibration_system.perf_fn(params)

    def predict_uncertainty(self, params: dict[str, Any], datamodule: DataModule) -> float:
        """Return the predicted uncertainty (0–1) at params.

        Uses the same evidence model the acquisition function queries.
        Pass the current datamodule so params are normalized consistently.
        Returns 1.0 (maximum uncertainty) if the evidence model is not yet fitted.
        """
        self._assert_initialized()
        return float(self.pred_system.uncertainty(datamodule.params_to_array(params)))

    def update_context_snapshot(self, values: dict[str, float]) -> None:
        """Update the context feature snapshot injected into the calibration perf_fn.

        Call this with the latest measured context values (e.g. temperature, humidity)
        before running exploration or inference steps so the prediction model receives
        correct observed covariate inputs during calibration.
        """
        self._context_snapshot.clear()
        self._context_snapshot.update(values)

    def baseline_step(
        self,
        n: int,
        param_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> list[ExperimentSpec]:
        """Generate n space-filling baseline proposals using LHS. No trained model required."""
        self._assert_initialized()
        result = self.calibration_system.run_baseline(
            n=n,
            param_bounds=param_bounds,
        )
        self.logger.console_success(f"Successfully completed baseline step ({n} proposals).")
        return result

    # === Helper Functions ===

    def _instantiate_model_group(
            self,
            model_specs: list[tuple[type[Any], dict]],
            system_model_list: list[Any],
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

    def _check_sets_against_keys(self, model_codes: set[str], schema_keys: list[str]) -> None:
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
        self._assert_initialized()
        
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
            exp_data: ExperimentData | None, 
            dimension: str | None = None, 
            step_index: int | None = None
            ) -> tuple[ExperimentData, int, int | None]:
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
            end: int | None,
            action: str
        ) -> None:
        """Helper to log step completion messages."""
        log_text = f"Experiment '{exp_code}' {action} successfully"
        if start and end:
            log_text += f" for dimension range [{start}:{end}]"
        self.logger.info(log_text)
