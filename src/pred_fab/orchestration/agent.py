"""PfabAgent — main orchestration class for the PFAB framework."""

import copy
import functools
from typing import Any

import numpy as np
import torch


def requires(*systems: "SystemName"):
    """Decorator: assert that the named systems are initialized before the method runs."""
    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            self._require(*systems)
            return method(self, *args, **kwargs)
        wrapper._required_systems = systems  # type: ignore[attr-defined]
        return wrapper
    return decorator

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
    EvidenceBackend,
)

from ..interfaces import IFeatureModel, IEvaluationModel, IPredictionModel
from ..utils import LocalData, PfabLogger, ConsoleReporter, StepType, Mode, SourceStep
from .evidence import EstimatorConfig


class PfabAgent:
    """Main orchestration class: coordinates EvaluationSystem, PredictionSystem, and CalibrationSystem."""

    # Dependency graph: system → set of systems that must be initialized first.
    # Evaluation has no system deps — it defines target/scaling independently.
    # Feature is only needed when actually running extraction (evaluate method).
    SYSTEM_DEPS: dict[SystemName, set[SystemName]] = {
        SystemName.FEATURE:     set(),
        SystemName.EVALUATION:  set(),
        SystemName.PREDICTION:  set(),
        SystemName.CALIBRATION: {SystemName.PREDICTION, SystemName.EVALUATION},
    }

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

    def _get_system(self, name: SystemName) -> "BaseOrchestrationSystem":
        """Return the orchestration system for the given name."""
        return {
            SystemName.FEATURE: self.feature_system,
            SystemName.EVALUATION: self.eval_system,
            SystemName.PREDICTION: self.pred_system,
            SystemName.CALIBRATION: self.calibration_system,
        }[name]

    def _require(self, *systems: SystemName) -> None:
        """Assert that the named systems (and their transitive deps) are initialized."""
        self._assert_initialized()
        needed: set[SystemName] = set()
        stack = list(systems)
        while stack:
            s = stack.pop()
            if s not in needed:
                needed.add(s)
                stack.extend(self.SYSTEM_DEPS.get(s, set()))
        missing = [
            s.value for s in needed
            if (sys := self._get_system(s)) is None or not sys.is_initialized
        ]
        if missing:
            missing.sort()
            asked = ", ".join(s.value for s in systems)
            raise RuntimeError(
                f"Systems not initialized: {', '.join(missing)}. "
                f"Register the required models before calling this method "
                f"(needs: {asked})."
            )

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
        self.schema = schema

        # Step 1: Initialize systems that have registered models
        self.logger.info("Instantiating models from registered classes...")

        self.feature_system = FeatureSystem(logger=self.logger)
        if self._feature_model_specs:
            self._instantiate_model_group(self._feature_model_specs, self.feature_system.models, "feature")
            self.feature_system._set_feature_column_names(schema)
            self.feature_system.set_ref_objects(schema)
            self.feature_system._initialized = True

        self.eval_system = EvaluationSystem(logger=self.logger)
        if self._evaluation_model_specs:
            self._instantiate_model_group(self._evaluation_model_specs, self.eval_system.models, "evaluation")
            self.eval_system.set_ref_objects(schema)
            self.eval_system._initialized = True

        self.pred_system = PredictionSystem(logger=self.logger, schema=schema, local_data=self.local_data)
        if self._prediction_model_specs:
            self._instantiate_model_group(self._prediction_model_specs, self.pred_system.models, "prediction")
            self.pred_system.set_ref_objects(schema)
            self.pred_system._initialized = True

        # Calibration system — requires Prediction + Evaluation (per SYSTEM_DEPS)
        if self.pred_system.is_initialized and self.eval_system.is_initialized:
            _pred = self.pred_system
            _eval = self.eval_system
            _ctx = self._context_snapshot

            def _perf_fn_tensor(
                params_dicts: list[dict[str, Any]],
            ) -> dict[str, torch.Tensor]:
                if not params_dicts:
                    return {}
                merged_list: list[dict[str, Any]] = []
                for pd_ in params_dicts:
                    m = dict(pd_)
                    if _ctx:
                        m.update(_ctx)
                    merged_list.append(m)
                feat_dicts_S = _pred.predict_for_calibration_tensor(merged_list)
                params_blocks: list[Any] = []
                for pd_ in merged_list:
                    block = copy.deepcopy(schema.parameters)
                    for code, val in pd_.items():
                        if code not in block.data_objects:
                            continue
                        v = val.item() if hasattr(val, "item") and torch.is_tensor(val) else val
                        try:
                            block.set_value(code, v)
                        except Exception:
                            pass
                    params_blocks.append(block)
                return _eval._evaluate_feature_dict_tensor(feat_dicts_S, params_blocks)

            self.calibration_system = CalibrationSystem(
                schema=schema,
                logger=self.logger,
                perf_fn_tensor=_perf_fn_tensor,
                uncertainty_fn=_pred.uncertainty,
                evidence=EvidenceBackend(
                    batched_tensor=_pred.delta_integrated_evidence_batched_tensor,
                    joint_batched_tensor=_pred.delta_integrated_evidence_joint_batched_tensor,
                ),
                n_exp_fn=lambda: _pred._n_exp,
                fit_empty_kde_fn=_pred.fit_empty_kde,
            )
            self.calibration_system._initialized = True
        else:
            self.calibration_system = None  # type: ignore[assignment]

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
        """Validate initialized orchestration systems against the schema."""

        input_params: list[str] = []
        input_features: list[str] = []
        output_features: list[str] = []
        output_predicted_features: list[str] = []
        output_performance_attrs: list[str] = []

        if self.feature_system.is_initialized:
            specs = self.feature_system.get_model_specs()
            input_params.extend(specs["input_parameters"])
            input_features.extend(specs["input_features"])
            output_features.extend(specs["outputs"])

        if self.eval_system.is_initialized:
            specs = self.eval_system.get_model_specs()
            input_params.extend(specs["input_parameters"])
            input_features.extend(specs["input_features"])
            output_performance_attrs.extend(specs["outputs"])

        if self.pred_system.is_initialized:
            specs = self.pred_system.get_model_specs()
            input_params.extend(specs["input_parameters"])
            input_features.extend(specs["input_features"])
            output_predicted_features.extend(specs["outputs"])

        domain_iterator_codes: set[str] = set()
        for domain in schema.domains.values():
            domain_iterator_codes.update(domain.iterator_input_codes)
        valid_input_feature_codes = set(schema.features.keys()) | domain_iterator_codes

        self._check_sets_against_keys(set(input_params), schema.parameters.keys())
        unknown_input_features = set(input_features) - valid_input_feature_codes
        if unknown_input_features:
            raise ValueError(
                f"The following input features are not in the schema: "
                f"{unknown_input_features}"
            )
        if output_features:
            self._check_sets_against_keys(set(output_features), schema.features.keys())
        if output_performance_attrs:
            self._check_sets_against_keys(set(output_performance_attrs), schema.performance_attrs.keys())

        output_features_set = set(output_features) | domain_iterator_codes

        if self.feature_system.is_initialized:
            uncomputed_inputs = set(input_features) - output_features_set
            if uncomputed_inputs:
                raise ValueError(
                    f"The following input features are not computed by any model: "
                    f"{uncomputed_inputs}"
                )
            if self.pred_system.is_initialized:
                unpredicted = set(output_predicted_features) - set(output_features)
                if unpredicted:
                    raise ValueError(
                        f"The following predicted features are not computed by any feature model: "
                        f"{unpredicted}"
                    )

    def set_active_experiment(self, exp_data: ExperimentData) -> None:
        """Set the active experiment for online operations."""
        self._assert_initialized()
        
        self.active_exp = exp_data
        self.logger.info(f"Active experiment set to: {exp_data.code}")

    def state_report(self) -> None:
        """Log an overview of all agent systems to the console."""
        _B = "\033[1m"
        _D = "\033[2m"
        _R = "\033[0m"

        if not self._initialized:
            self.logger.console_warning("Agent not initialized. No models to report.")
            return

        lines = [f"\n  {_B}Agent{_R}"]

        def _add_model_section(name: str, models: list[Any]) -> None:
            if not models:
                return
            lines.append(f"\n  {_D}{name}{_R}")
            for model in models:
                inp_str = ", ".join(model.input_parameters + model.input_features)
                out_str = ", ".join(model.outputs)
                lines.append(f"    {model.__class__.__name__:<20s} {inp_str:<30s} → {out_str}")

        if self.feature_system:
            _add_model_section("Feature System", self.feature_system.models)
        if self.eval_system:
            _add_model_section("Evaluation System", self.eval_system.models)
        if self.pred_system:
            _add_model_section("Prediction System", self.pred_system.models)

        # # Calibration System (config — not a trained system)
        # cal = self.calibration_system
        # lines.append(f"\n  {_D}Calibration System{_R}")
        #
        # pw_parts = [f"{k}={v:g}" for k, v in cal.performance_weights.items()]
        # lines.append(f"    {_D}Weights: {', '.join(pw_parts)}{_R}")
        #
        # explore_parts = [f"radius={cal._exploration_radius:g}",
        #                  f"buffer={cal._buffer:g}",
        #                  f"decay_exp={cal._decay_exp:g}"]
        # lines.append(f"    {_D}Exploration: {', '.join(explore_parts)}{_R}")
        #
        # for code in cal.data_objects.keys():
        #     low, high = cal._get_hierarchical_bounds_for_code(code)
        #     bounds_str = f"[{low}, {high}]"
        #     delta = cal.trust_regions.get(code, "─")
        #     lines.append(f"    {code:<20s} {bounds_str:<15s} {_D}{delta}{_R}")

        self.logger.console_summary("\n".join(lines))
        self.logger.console_new_line()

    def calibration_state_report(self) -> None:
        self.calibration_system.state_report()

    # === STEP METHODS ==
    
    @requires(SystemName.CALIBRATION)
    def acquisition_step(
        self,
        datamodule: DataModule,
        kappa: float | None = None,
        n_optimization_rounds: int = 5,
        current_params: dict[str, Any] | None = None,
    ) -> ExperimentSpec:
        """Unified proposal step: κ>0 = exploration, κ=0 = inference.

        ``kappa=None`` resolves to the persistent default set by
        ``configure_exploration(kappa=...)`` (initial value 0.5).
        """
        if kappa is None:
            kappa = self.calibration_system.kappa_default

        mode = Mode.INFERENCE if kappa == 0.0 else Mode.EXPLORATION
        result = self.calibration_system.run_calibration(
            datamodule=datamodule,
            mode=mode,
            current_params=current_params,
            kappa=kappa,
            n_optimization_rounds=n_optimization_rounds,
        )

        self.logger.info(f"Completed acquisition step (kappa={kappa}).")
        return result

    # Backward-compatible aliases
    def exploration_step(self, datamodule: DataModule, kappa: float | None = None,
                         n_optimization_rounds: int = 5,
                         current_params: dict[str, Any] | None = None) -> ExperimentSpec:
        """Alias for acquisition_step with kappa > 0; ``None`` uses the configured default."""
        return self.acquisition_step(datamodule, kappa=kappa,
                                     n_optimization_rounds=n_optimization_rounds,
                                     current_params=current_params)

    def inference_step(
        self,
        exp_data: ExperimentData,
        datamodule: DataModule,
        n_optimization_rounds: int = 5,
        recompute: bool = False,
        visualize: bool = False,
        current_params: dict[str, Any] | None = None,
    ) -> ExperimentSpec:
        """Feature extraction + evaluation, then acquisition_step with kappa=0."""
        self._check_systems(StepType.FULL)

        self.feature_system.run_feature_extraction(exp_data, 0, None, recompute=recompute, visualize=visualize)
        self.eval_system.run_evaluation(exp_data, recompute=recompute)

        return self.acquisition_step(datamodule, kappa=0.0,
                                     n_optimization_rounds=n_optimization_rounds,
                                     current_params=current_params)

    def adaptation_step(
        self,
        dimension: str | None = None,
        step_index: int | None = None,
        exp_data: ExperimentData | None = None,
        mode: Mode = Mode.INFERENCE,
        kappa: float = 0.0,
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
            kappa=kappa,
        )
        # Tag as adaptation (run_calibration uses mode-derived source_step).
        proposal = ParameterProposal.from_dict(
            result.initial_params.to_dict(), source_step=SourceStep.ADAPTATION,
        )
        result = ExperimentSpec(initial_params=proposal, trajectories=result.trajectories)

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

    @requires(SystemName.PREDICTION)
    def train(
        self,
        datamodule: DataModule,
        validate: bool = True,
        test: bool = False,
        **kwargs
    ) -> dict[str, Any] | None:
        """Train prediction models and optionally validate/test."""
        
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
    
    def to(self, device: str | torch.device) -> "PfabAgent":
        """Move all torch state to ``device``.

        Calls ``.to(device)`` on every ``nn.Module`` instance held by the
        framework: prediction model networks, categorical embeddings, and
        ``nn.Module``-based normalisers. Records the device on
        ``pred_system._device`` so KDE-side torch conversions inside the
        acquisition graph target it. After this call, ``forward_pass`` /
        ``encode`` and the gradient acquisition path operate on the new
        device.

        Returns self for chaining (``agent.to('cuda').baseline_step(3)``).

        Note: KDE storage (``_model_kdes`` latent_points / point_weights)
        remains numpy on CPU; the torch estimators convert + move to
        ``_device`` at call boundaries. RNG state stays on CPU.
        """
        self._assert_initialized()
        device_t = torch.device(device) if isinstance(device, str) else device

        for model in self.pred_system.models:
            if hasattr(model, "_model") and model._model is not None:
                model._model = model._model.to(device_t)
            if hasattr(model, "_cat_embeddings"):
                model._cat_embeddings = model._cat_embeddings.to(device_t)
            if hasattr(model, "_compiled_forward") and model._compiled_forward is not None:
                # torch.compile-d module; rebuild the compiled wrapper around
                # the moved net at next first-call.
                model._compiled_forward = None

        if self.pred_system.datamodule is not None:
            for stats in self.pred_system.datamodule._parameter_stats.values():
                stats.to(device_t)
            for stats in self.pred_system.datamodule._feature_stats.values():
                stats.to(device_t)

        # Record on PredictionSystem so KDE torch conversions target it.
        self.pred_system._device = device_t

        return self

    def configure_performance(
        self,
        *,
        weights: dict[str, float],
    ) -> None:
        """Set performance weights for calibration and uncertainty aggregation."""
        self._assert_initialized()
        self.calibration_system.set_performance_weights(weights)
        if self._console is not None:
            self._console._perf_weights = weights
        # Map performance weights to feature names for per-model KDE aggregation.
        feature_weights: dict[str, float] = {}
        for eval_model in self.eval_system.models:
            perf_name = eval_model.output_performance
            feat_name = eval_model.input_feature
            if perf_name in weights:
                feature_weights[feat_name] = weights[perf_name]
        if feature_weights:
            self.pred_system.set_uncertainty_weights(feature_weights)

    def configure_exploration(
        self,
        *,
        sigma: float | None = None,
        kappa: float | None = None,
    ) -> None:
        """Configure the exploration objective.

        ``sigma`` — kernel bandwidth for Δ∫E.
        ``kappa`` — persistent default κ ∈ [0, 1] for ``acquisition_step`` /
        ``exploration_step``. ``inference_step`` is unaffected (always κ=0).
        """
        self._assert_initialized()
        if sigma is not None:
            self.pred_system.configure_exploration(sigma=sigma)
        if kappa is not None:
            if not 0.0 <= kappa <= 1.0:
                raise ValueError(f"kappa must be in [0, 1], got {kappa!r}")
            self.calibration_system.kappa_default = float(kappa)

    def configure_evidence(
        self,
        *,
        estimator: str | None = None,
        radii: tuple[float, ...] | None = None,
        angular_gap_deg: float | None = None,
        box: float | None = None,
        n_samples: int | None = None,
        seed: int | None = None,
        cutoff_sigmas: float | None = None,
        truncation_threshold: int | None = None,
    ) -> None:
        """Configure the evidence estimator and its tuning knobs.

        ``estimator``: ``"kernel_field"`` (deterministic shell quadrature;
        probe count grows with D) or ``"sobol_local"`` (QMC cube with
        fixed ``n_samples`` per kernel — the high-D escape hatch).

        Per-estimator knobs:
          KernelField — ``radii``, ``angular_gap_deg``
          SobolLocal  — ``box``, ``n_samples``, ``seed``
        Shared — ``cutoff_sigmas``, ``truncation_threshold``.

        Per-estimator-irrelevant knobs are accepted but ignored.
        """
        self._assert_initialized()
        self.pred_system.configure_evidence(
            estimator=estimator,
            radii=radii,
            angular_gap_deg=angular_gap_deg,
            box=box,
            n_samples=n_samples,
            seed=seed,
            cutoff_sigmas=cutoff_sigmas,
            truncation_threshold=truncation_threshold,
        )

    def configure_optimizer(
        self,
        *,
        de_maxiter: int | None = None,
        de_popsize: int | None = None,
        n_starts: int | None = None,
        n_iters: int | None = None,
        lr: float | None = None,
        convergence_window: float | None = None,
        convergence_eps: float | None = None,
    ) -> None:
        """Set optimiser tuning parameters.

        ``de_maxiter`` / ``de_popsize`` control the global DE phase;
        ``n_starts`` / ``n_iters`` / ``lr`` control the local LBFGS phase.
        ``convergence_window`` (fraction of maxiter) and ``convergence_eps``
        (fraction of initial objective) apply to both.
        """
        self._assert_initialized()
        cal = self.calibration_system
        if de_maxiter is not None:
            cal.de_maxiter = de_maxiter
        if de_popsize is not None:
            cal.de_popsize = de_popsize
        if n_starts is not None:
            cal.engine.gradient_n_starts = n_starts
        if n_iters is not None:
            cal.engine.gradient_n_iters = n_iters
        if lr is not None:
            cal.engine.gradient_lr = lr
        if convergence_window is not None:
            cal.engine.convergence_window_frac = convergence_window
        if convergence_eps is not None:
            cal.engine.convergence_eps_frac = convergence_eps

    def configure_trajectory(
        self,
        parameter: str,
        dimension: str,
        *,
        delta: float | None = None,
        force: bool = False,
    ) -> None:
        """Configure a parameter to vary per step of a dimension.

        ``smoothing`` knob dropped — under the
        gradient schedule path, smoothness emerges naturally from the
        differentiable autoreg coupling between adjacent steps and the
        delta-constraint penalty.
        """
        self._assert_initialized()
        cal = self.calibration_system
        cal.configure_trajectory_parameter(parameter, dimension, force=force)
        if delta is not None:
            cal.configure_adaptation_delta({parameter: delta}, force=force)

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

    @property
    def last_baseline_nfev(self) -> int:
        """Number of DE evaluations in the most recent baseline step."""
        return getattr(self.calibration_system, 'last_baseline_nfev', 0)

    # ── Acquisition introspection ───────────────────────────────────────────────

    def predict_performance(self, params: dict[str, Any]) -> dict[str, float | None]:
        """Run the prediction + evaluation pipeline at params and return performance scores.

        Uses the same perf_fn the optimizer uses internally — useful for logging what the
        model predicts at a proposed point before committing to an experiment.
        Requires the agent to be trained (pred_system must have a fitted model).
        """
        self._assert_initialized()
        # The scalar perf_fn was collapsed into perf_fn_tensor during the
        # tensor-only migration — wrap a single dict in/out around the
        # batched tensor call site to preserve this method's contract.
        perf_fn_tensor = self.calibration_system.perf_fn_tensor
        if perf_fn_tensor is None:
            raise RuntimeError("Agent has no perf_fn_tensor configured.")
        out = perf_fn_tensor([params])
        return {code: float(t[0].item()) for code, t in out.items()}

    def predict_uncertainty(self, params: dict[str, Any], datamodule: DataModule) -> float:
        """Return the predicted uncertainty (0–1) at params.

        Uses the same evidence model the acquisition function queries.
        Boundary evidence is included in the uncertainty computation.
        Pass the current datamodule so params are normalized consistently.
        Returns 1.0 (maximum uncertainty) if the evidence model is not yet fitted.
        """
        self._assert_initialized()
        X = datamodule.params_to_array(params)
        return float(self.pred_system.uncertainty(X))

    def update_context_snapshot(self, values: dict[str, float]) -> None:
        """Update the context feature snapshot injected into the calibration perf_fn.

        Call this with the latest measured context values (e.g. temperature, humidity)
        before running exploration or inference steps so the prediction model receives
        correct observed covariate inputs during calibration.
        """
        self._context_snapshot.clear()
        self._context_snapshot.update(values)

    @requires(SystemName.CALIBRATION)
    def baseline_step(
        self,
        n: int,
    ) -> list[ExperimentSpec]:
        """Generate n space-filling baseline proposals via batch-aware evidence maximization.

        Uses the acquisition objective with κ=1 (pure exploration: maximize ΔI,
        the integrated evidence gain). Each new proposal is added to the
        evidence field so subsequent proposals naturally space-fill.
        """
        # Bypass the encoder during baseline so the (random-init) prediction
        # model can't taint placement: evidence/KDE operates on raw normalised
        # input space here. Auto-managed; not user-facing.
        self.pred_system._bypass_encoder = True
        try:
            result = self.calibration_system.run_baseline(n=n)
        finally:
            self.pred_system._bypass_encoder = False
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

    
    def _check_systems(self, step: StepType) -> None:
        """Legacy bridge — maps StepType to _require()."""
        if step == StepType.EVAL:
            self._require(SystemName.FEATURE, SystemName.EVALUATION)
        elif step == StepType.FULL:
            self._require(SystemName.FEATURE, SystemName.EVALUATION,
                          SystemName.PREDICTION, SystemName.CALIBRATION)
        else:
            self._assert_initialized()
        
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
