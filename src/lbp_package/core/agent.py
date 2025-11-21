"""
LBPAgent - Main orchestration class for AIXD architecture.

Replaces study-based logic with dataset-centric approach while maintaining
existing workflow patterns. Wraps EvaluationSystem, PredictionSystem, and
manages schema generation and dataset initialization.
"""

from typing import Any, Dict, List, Type, Optional, Callable
from dataclasses import fields
import inspect
import json
import os

from .schema import DatasetSchema
from .schema_registry import SchemaRegistry
from .dataset import Dataset
from .data_objects import DataReal, DataInt, DataBool, DataCategorical, DataString, DataDimension
from ..orchestration.evaluation import EvaluationSystem
from ..orchestration.prediction import PredictionSystem
from ..interfaces import IEvaluationModel, IPredictionModel, ICalibrationModel, IExternalData
from ..utils import LocalData, LBPLogger


class LBPAgent:
    """
    Main orchestration class for AIXD-based LBP framework.
    
    Manages:
    - Model registration and activation
    - Schema generation from active models
    - Dataset initialization and loading
    - Coordination between EvaluationSystem, PredictionSystem, CalibrationModel
    """
    
    def __init__(
        self,
        root_folder: str,
        local_folder: str,
        log_folder: str,
        external_data_interface: Optional[IExternalData] = None,
        server_folder: Optional[str] = None,
        debug_flag: bool = False,
        recompute_flag: bool = False,
        visualize_flag: bool = True,
        round_digits: int = 3,
    ):
        """
        Initialize LBP Agent.
        
        Args:
            root_folder: Project root folder
            local_folder: Path to local data storage
            log_folder: Path to log file storage
            external_data_interface: Optional interface for database operations
            server_folder: Optional path to server data storage
            debug_flag: Skip external data operations if True
            recompute_flag: Force recomputation/overwrite if True
            visualize_flag: Enable visualizations if True
            round_digits: Number of decimal places for rounding
        """
        self.logger_name = "LBPAgent"
        self.logger = LBPLogger(self.logger_name, log_folder)
        self.logger.info("Initializing LBP Agent")
        
        self.external_data = external_data_interface
        if external_data_interface is None:
            self.logger.console_warning(
                "No external data interface provided: "
                "Dataset records need to be available as local JSON files."
            )
        
        # System settings
        self.debug_flag = debug_flag
        self.recompute_flag = recompute_flag
        self.visualize_flag = visualize_flag
        self.round_digits = round_digits
        
        # Initialize local data handler
        self.local_data = LocalData(root_folder, local_folder, server_folder)
        
        # Initialize system components
        self.eval_system = EvaluationSystem(self.logger)
        self.pred_system = PredictionSystem(self.local_data, self.logger)
        self.calibration_model: Optional[ICalibrationModel] = None
        
        # Current dataset (set during initialize_for_dataset)
        self.current_dataset: Optional[Dataset] = None
        
        self.logger.console_info("\n------- Welcome to Learning by Printing (AIXD) -------\n")
    
    # === MODEL REGISTRATION (same API as LBPManager) ===
    
    def add_evaluation_model(
        self,
        performance_code: str,
        evaluation_class: Type[IEvaluationModel],
        round_digits: Optional[int] = None,
        weight: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Register an evaluation model.
        
        Args:
            performance_code: Code identifying the performance metric
            evaluation_class: IEvaluationModel subclass
            round_digits: Decimal places for rounding (uses system default if None)
            weight: Calibration weight for multi-objective optimization
            **kwargs: Additional model configuration
        """
        round_digits = round_digits if round_digits is not None else self.round_digits
        self.eval_system.add_evaluation_model(
            performance_code, evaluation_class, round_digits, weight, **kwargs
        )
        self.logger.console_info(f"Registered evaluation model: {evaluation_class.__name__}")
    
    def add_prediction_model(
        self,
        performance_codes: List[str],
        prediction_class: Type[IPredictionModel],
        round_digits: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Register a prediction model.
        
        Args:
            performance_codes: List of performance codes this model predicts
            prediction_class: IPredictionModel subclass
            round_digits: Decimal places for rounding
            **kwargs: Additional model configuration
        """
        round_digits = round_digits if round_digits is not None else self.round_digits
        self.pred_system.add_prediction_model(performance_codes, prediction_class, round_digits, **kwargs)
        self.logger.console_info(
            f"Registered prediction model: {prediction_class.__name__} "
            f"for performance codes: {performance_codes}"
        )
    
    def add_calibration_model(
        self,
        calibration_class: Type[ICalibrationModel],
        **kwargs
    ) -> None:
        """
        Register a calibration model.
        
        Args:
            calibration_class: ICalibrationModel subclass
            **kwargs: Additional model configuration
        """
        self.calibration_model = calibration_class(logger=self.logger, **kwargs)
        self.logger.console_info(f"Set calibration model to '{calibration_class.__name__}'.")
    
    # === SCHEMA GENERATION ===
    
    def generate_schema_from_active_models(self) -> DatasetSchema:
        """
        Generate DatasetSchema from currently active evaluation models.
        
        Introspects active models to extract:
        - Static parameters (@parameter_static)
        - Dynamic parameters (@parameter_dynamic, excluding dimensional)
        - Dimensional parameters (from dim_param_names)
        - Performance attributes (from performance codes)
        
        Returns:
            DatasetSchema instance
            
        Raises:
            ValueError: If no models are active or if validation fails
        """
        schema = DatasetSchema()
        
        active_models = {
            code: model for code, model in self.eval_system.evaluation_models.items()
            if model.active
        }
        
        if not active_models:
            raise ValueError("No evaluation models are active. Call activate models first.")
        
        # Track all parameters to avoid duplicates
        seen_static = set()
        seen_dynamic = set()
        seen_dimensional = set()
        
        for code, model in active_models.items():
            # Extract static parameters
            static_param_names = model.get_param_names_by_type('model')
            for param_name in static_param_names:
                if param_name not in seen_static:
                    field = model.get_param_field(param_name)
                    data_obj = self._infer_data_object(param_name, field)
                    schema.static_params.add(param_name, data_obj)
                    seen_static.add(param_name)
            
            # Extract dynamic parameters (exclude dimensional)
            dynamic_param_names = model.get_param_names_by_type('experiment')
            for param_name in dynamic_param_names:
                if param_name not in model.dim_param_names and param_name not in seen_dynamic:
                    field = model.get_param_field(param_name)
                    data_obj = self._infer_data_object(param_name, field)
                    schema.dynamic_params.add(param_name, data_obj)
                    seen_dynamic.add(param_name)
            
            # Extract dimensional parameters
            for i, dim_param_name in enumerate(model.dim_param_names):
                if dim_param_name not in seen_dimensional:
                    dim_obj = DataDimension(
                        dim_name=model.dim_names[i],
                        dim_param_name=dim_param_name,
                        dim_iterator_name=model.dim_iterator_names[i]
                    )
                    schema.dimensional_params.add(dim_param_name, dim_obj)
                    seen_dimensional.add(dim_param_name)
            
            # Add performance attribute
            # Infer type from target_value or default to DataReal
            perf_obj = DataReal(code)  # Default to real-valued performance
            schema.performance_attrs.add(code, perf_obj)
            
            # Set calibration weight if defined
            if model.weight is not None:
                schema.performance_attrs.set_weight(code, model.weight)
        
        # Validate: dimensional params must be in dynamic params
        for dim_param in schema.dimensional_params.keys():
            if dim_param not in schema.dynamic_params.data_objects:
                raise ValueError(
                    f"Dimensional parameter '{dim_param}' must also be marked with "
                    f"@parameter_dynamic (not just used in dim_param_names)"
                )
        
        # Also process feature models for their parameters
        self._add_feature_model_params_to_schema(schema, active_models)
        
        self.logger.info(
            f"Generated schema with {len(schema.static_params.data_objects)} static, "
            f"{len(schema.dynamic_params.data_objects)} dynamic, "
            f"{len(schema.dimensional_params.data_objects)} dimensional params, "
            f"{len(schema.performance_attrs.data_objects)} performance attributes"
        )
        
        return schema
    
    def _infer_data_object(self, param_name: str, field) -> Any:
        """
        Infer DataObject type from field annotation or default.
        
        Args:
            param_name: Parameter name
            field: Dataclass field
            
        Returns:
            DataObject instance
        """
        # Try to infer from type annotation
        if hasattr(field, 'type'):
            field_type = field.type
            
            # Handle Optional types
            if hasattr(field_type, '__origin__'):
                if field_type.__origin__ is type(None) or str(field_type.__origin__) == 'typing.Union':
                    # Extract actual type from Optional[T]
                    args = getattr(field_type, '__args__', ())
                    field_type = next((arg for arg in args if arg is not type(None)), field_type)
            
            if field_type == int or field_type == 'int':
                return DataInt(param_name)
            elif field_type == float or field_type == 'float':
                return DataReal(param_name)
            elif field_type == bool or field_type == 'bool':
                return DataBool(param_name)
            elif field_type == str or field_type == 'str':
                return DataString(param_name)
        
        # Default to DataReal
        return DataReal(param_name)
    
    def _add_feature_model_params_to_schema(
        self,
        schema: DatasetSchema,
        active_models: Dict[str, IEvaluationModel]
    ) -> None:
        """
        Add feature model parameters to schema.
        
        Feature models are already deduplicated by _add_feature_model_instances,
        so we can extract their parameters without duplication.
        """
        for model in active_models.values():
            if model.feature_model is not None:
                # Extract static params from feature model
                static_names = model.feature_model.get_param_names_by_type('model')
                for param_name in static_names:
                    if param_name not in schema.static_params.data_objects:
                        field = model.feature_model.get_param_field(param_name)
                        data_obj = self._infer_data_object(param_name, field)
                        schema.static_params.add(param_name, data_obj)
                
                # Extract dynamic params from feature model
                dynamic_names = model.feature_model.get_param_names_by_type('experiment')
                for param_name in dynamic_names:
                    if param_name not in schema.dynamic_params.data_objects:
                        field = model.feature_model.get_param_field(param_name)
                        data_obj = self._infer_data_object(param_name, field)
                        schema.dynamic_params.add(param_name, data_obj)
    
    # === TWO-PHASE INITIALIZATION ===
    
    def initialize_for_dataset(
        self,
        schema_id: Optional[str] = None,
        performance_codes: Optional[List[str]] = None,
        static_params: Optional[Dict[str, Any]] = None,
    ) -> Dataset:
        """
        Two-phase initialization pattern integrating with existing LBP workflow.
        
        Supports two modes:
        1. New dataset: Provide performance_codes + static_params
        2. Load existing: Provide schema_id only (loads from registry)
        
        Phase 1: Model Setup
        - Load study record (hierarchical load/save)
        - Extract static params and performance codes from record
        - Activate models for performance codes
        - Initialize feature models with static params
        - Generate schema from active models
        
        Phase 2: Dataset Setup
        - Get or create schema ID via registry
        - Create dataset instance
        - Attach storage paths
        - Initialize evaluation system storage references
        
        Args:
            schema_id: Optional schema ID to load existing dataset
            performance_codes: List of performance codes to activate (for new dataset)
            static_params: Static parameter values (for new dataset)
        
        Returns:
            Initialized Dataset instance
            
        Raises:
            ValueError: If parameters are invalid or study record missing
        """
        if schema_id is not None:
            # Mode 2: Load existing dataset
            return self.load_existing_dataset(schema_id)
        
        # Mode 1: New dataset - require both performance_codes and static_params
        if performance_codes is None or static_params is None:
            raise ValueError(
                "For new dataset, both 'performance_codes' and 'static_params' required. "
                "For loading existing dataset, provide 'schema_id'."
            )
        
        self.logger.console_info("\n===== Phase 1: Model Setup =====")
        
        # Step 1: Activate models for each performance code
        self.logger.info(f"Activating models for performance codes: {performance_codes}")
        for perf_code in performance_codes:
            self.eval_system.activate_evaluation_model(perf_code, static_params)
            self.pred_system.activate_prediction_model(perf_code, static_params)
        
        # Step 2: Initialize feature models with static params
        self.logger.info("Initializing feature models")
        self._add_feature_model_instances(static_params)
        
        # Step 3: Generate schema from active models
        self.logger.info("Generating schema from active models")
        schema = self.generate_schema_from_active_models()
        
        self.logger.console_info("\n===== Phase 2: Dataset Setup =====")
        
        # Step 4: Get or create schema ID
        registry = SchemaRegistry(self.local_data.local_folder)
        schema_hash = schema._compute_schema_hash()
        schema_struct = schema.to_dict()
        new_schema_id = registry.get_or_create_schema_id(schema_hash, schema_struct)
        self.logger.console_info(f"Using schema ID: {new_schema_id}")
        
        # Step 5: Create dataset
        dataset = Dataset(name=new_schema_id, schema=schema, schema_id=new_schema_id)
        dataset.set_static_values(static_params)
        
        # Step 6: Attach storage paths
        self.local_data.set_schema_id(new_schema_id)
        
        # Step 7: Attach dataset to evaluation system
        self.logger.info("Attaching dataset to evaluation system")
        self._attach_dataset_to_systems(dataset)
        
        # Store current dataset
        self.current_dataset = dataset
        
        # Display summary
        summary = self._initialization_summary(dataset, performance_codes)
        self.logger.console_summary(summary)
        self.logger.console_success(f"Successfully initialized dataset '{new_schema_id}'.")
        
        return dataset
    
    def _add_feature_model_instances(self, static_params: Dict[str, Any]) -> None:
        """
        Create and assign feature model instances to evaluation and prediction models.
        
        Deduplicates feature models by type to avoid redundant instantiation.
        
        Args:
            static_params: Static parameter values to pass to feature models
        """
        # Create collection of (model, code, feature_model_type) tuples
        feature_model_collection = []
        
        # Collect from evaluation models
        for code, eval_model in self.eval_system.evaluation_models.items():
            if eval_model.active and eval_model.feature_model_type is not None:
                feature_model_collection.append((eval_model, code, eval_model.feature_model_type))
        
        # Collect from prediction models
        for pred_model in self.pred_system.prediction_models:
            if pred_model.active:
                for code, feature_model_type in pred_model.feature_model_types.items():
                    feature_model_collection.append((pred_model, code, feature_model_type))
        
        # Create unique feature model instances (deduplicated by type)
        feature_model_dict = {}
        
        for model, code, feature_model_type in feature_model_collection:
            # Check if instance already exists for this type
            if feature_model_type not in feature_model_dict:
                # Create new instance
                feature_model_instance = feature_model_type(
                    performance_code=code,
                    logger=self.logger,
                    study_params=static_params,
                    round_digits=model.round_digits,
                    **model.kwargs
                )
                feature_model_dict[feature_model_type] = feature_model_instance
                self.logger.info(f"Created feature model: {feature_model_type.__name__} for {code}")
            
            # Assign feature model to the model (reuse if already exists)
            model.feature_model = feature_model_dict[feature_model_type]
    
    def _attach_dataset_to_systems(self, dataset: Dataset) -> None:
        """
        Attach dataset storage to evaluation and prediction systems.
        
        Systems get references to dataset's master storage dictionaries.
        
        Args:
            dataset: Dataset instance with master storage
        """
        # Evaluation system gets references to dataset storage
        self.eval_system.aggr_metrics = dataset._aggr_metrics
        self.eval_system.metric_arrays = dataset._metric_arrays
        
        # Prediction system will access via dataset in future steps
        # For now, no change needed (pred_system uses local_data directly)
    
    def _initialization_summary(self, dataset: Dataset, performance_codes: List[str]) -> str:
        """
        Generate initialization summary for console output.
        
        Args:
            dataset: Initialized dataset
            performance_codes: Active performance codes
            
        Returns:
            Formatted summary string
        """
        schema = dataset.schema
        
        summary_lines = [
            f"Dataset Schema: {dataset.schema_id}",
            f"Performance Codes: {', '.join(performance_codes)}",
            "",
            "Schema Structure:",
            f"  - Static Parameters: {len(schema.static_params.data_objects)}",
            f"  - Dynamic Parameters: {len(schema.dynamic_params.data_objects)}",
            f"  - Dimensional Parameters: {len(schema.dimensional_params.data_objects)}",
            f"  - Performance Attributes: {len(schema.performance_attrs.data_objects)}",
            "",
            "Active Models:",
        ]
        
        # List active evaluation models
        for code, model in self.eval_system.evaluation_models.items():
            if model.active:
                feature_info = ""
                if model.feature_model is not None:
                    feature_info = f" (Feature: {type(model.feature_model).__name__})"
                summary_lines.append(f"  - Evaluation: {type(model).__name__}{feature_info}")
        
        # List active prediction models
        for model in self.pred_system.prediction_models:
            if model.active:
                summary_lines.append(f"  - Prediction: {type(model).__name__}")
        
        return "\n".join(summary_lines)
    
    def _initialize_feature_models(self, static_params: Dict[str, Any]) -> None:
        """
        Initialize feature models with static parameters.
        
        Args:
            static_params: Static parameter values
        """
        for model in self.eval_system.evaluation_models.values():
            if model.feature_model is not None:
                # Set static params on feature model
                static_param_names = model.feature_model.get_param_names_by_type('model')
                feature_static = {k: v for k, v in static_params.items() if k in static_param_names}
                if feature_static:
                    model.feature_model.set_parameters_static(**feature_static)
    
    def _initialize_evaluation_for_dataset(
        self,
        dataset: Dataset,
        static_params: Dict[str, Any]
    ) -> None:
        """
        Initialize evaluation system for dataset (replaces initialize_for_study).
        
        Args:
            dataset: Dataset instance
            static_params: Static parameter values
        """
        # Set static params on eval models
        for model in self.eval_system.evaluation_models.values():
            static_param_names = model.get_param_names_by_type('model')
            model_static = {k: v for k, v in static_params.items() if k in static_param_names}
            if model_static:
                model.set_parameters_static(**model_static)
        
        # Aggregate metrics storage will use dataset._aggr_metrics
        # (to be migrated in later steps)
    
    # === DATASET LOADING ===
    
    def load_existing_dataset(self, schema_id: str) -> Dataset:
        """
        Load an existing dataset from local storage with compatibility checking.
        
        Args:
            schema_id: Schema identifier (e.g., "schema_001")
        
        Returns:
            Loaded Dataset instance
            
        Raises:
            ValueError: If schema is incompatible with active models
            FileNotFoundError: If schema file not found
        """
        self.logger.console_info(f"\n===== Loading Dataset: {schema_id} =====")
        
        # Step 1: Load schema from registry
        registry = SchemaRegistry(self.local_data.local_folder)
        schema_dict = registry.get_schema_by_id(schema_id)
        if schema_dict is None:
            raise FileNotFoundError(f"Schema '{schema_id}' not found in registry")
        
        stored_schema = DatasetSchema.from_dict(schema_dict['structure'])
        
        # Step 2: Extract performance codes from schema
        performance_codes = list(stored_schema.performance_attrs.keys())
        
        # Step 3: Extract static params from stored schema
        # TODO: Load actual values from schema.json file when we implement persistence
        # For now, create empty dict (caller should call set_static_values)
        static_params = {}
        
        self.logger.console_info("\n===== Phase 1: Model Setup =====")
        
        # Step 4: Activate models for performance codes
        self.logger.info(f"Activating models for performance codes: {performance_codes}")
        for perf_code in performance_codes:
            # Use empty dict for now until we load actual static params
            self.eval_system.activate_evaluation_model(perf_code, static_params)
            self.pred_system.activate_prediction_model(perf_code, static_params)
        
        # Step 5: Initialize feature models
        self.logger.info("Initializing feature models")
        self._add_feature_model_instances(static_params)
        
        # Step 6: Generate current schema from active models
        current_schema = self.generate_schema_from_active_models()
        
        # Step 7: Check compatibility (raises ValueError if incompatible)
        try:
            stored_schema.is_compatible_with(current_schema)
            self.logger.console_info(f"Schema compatibility check passed")
        except ValueError as e:
            raise ValueError(
                f"Stored schema '{schema_id}' is incompatible with active models:\n{str(e)}"
            )
        
        self.logger.console_info("\n===== Phase 2: Dataset Setup =====")
        
        # Step 8: Create dataset from stored schema
        dataset = Dataset(name=schema_id, schema=stored_schema, schema_id=schema_id)
        
        # Step 9: Attach storage paths
        self.local_data.set_schema_id(schema_id)
        
        # Step 10: Attach dataset to systems
        self.logger.info("Attaching dataset to systems")
        self._attach_dataset_to_systems(dataset)
        
        # Store current dataset
        self.current_dataset = dataset
        
        self.logger.console_info(f"Dataset '{schema_id}' loaded successfully\n")
        
        return dataset
    
    # === EXPERIMENT OPERATIONS (delegate to current_dataset) ===
    
    def add_experiment(
        self,
        exp_code: str,
        exp_params: Dict[str, Any],
        perf_metrics: Optional[Dict[str, Dict[str, float]]] = None
    ) -> None:
        """
        Add experiment to current dataset.
        
        Args:
            exp_code: Experiment code
            exp_params: Experiment parameters
            perf_metrics: Optional performance metrics (perf_code -> metric_name -> value)
        """
        if self.current_dataset is None:
            raise RuntimeError("No dataset initialized. Call initialize_for_dataset first.")
        
        self.current_dataset.add_experiment(exp_code, exp_params, perf_metrics)
        self.logger.info(f"Added experiment '{exp_code}' to dataset")
    
    def get_experiment_codes(self) -> List[str]:
        """Get all experiment codes in current dataset."""
        if self.current_dataset is None:
            raise RuntimeError("No dataset initialized")
        return self.current_dataset.get_experiment_codes()
    
    def get_experiment_params(self, exp_code: str) -> Dict[str, Any]:
        """Get parameters for specific experiment."""
        if self.current_dataset is None:
            raise RuntimeError("No dataset initialized")
        return self.current_dataset.get_experiment_params(exp_code)
    
    # === EXPERIMENT EXECUTION ===
    
    def initialize_for_exp(self, exp_code: str, exp_params: Dict[str, Any]) -> None:
        """
        Initialize evaluation system for a specific experiment.
        
        Sets experiment parameters on all active models and their feature models.
        Creates storage entries in dataset for this experiment.
        
        Args:
            exp_code: Experiment code
            exp_params: Experiment parameters
        """
        if self.current_dataset is None:
            raise RuntimeError("No dataset initialized. Call initialize_for_dataset first.")
        
        # Delegate to evaluation system
        self.eval_system.initialize_for_exp(exp_code, **exp_params)
        
        self.logger.info(f"Initialized experiment '{exp_code}' with parameters")
    
    def evaluate_experiment(
        self,
        exp_code: str,
        exp_params: Dict[str, Any],
        visualize: bool = False,
        recompute: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a single experiment.
        
        Args:
            exp_code: Experiment code
            exp_params: Experiment parameters
            visualize: Show visualizations if True
            recompute: Force recomputation if True
            
        Returns:
            Performance metrics dict: perf_code -> metric_name -> value
        """
        if self.current_dataset is None:
            raise RuntimeError("No dataset initialized")
        
        # Initialize for experiment
        self.initialize_for_exp(exp_code, exp_params)
        
        # Check if already computed (unless recompute=True)
        if not recompute and self.current_dataset.has_experiment(exp_code):
            self.logger.console_info(f"Experiment '{exp_code}' already evaluated (use recompute=True to override)")
            return self.current_dataset._aggr_metrics.get(exp_code, {})
        
        # Get experiment folder
        exp_folder = self.local_data.get_experiment_folder(exp_code)
        
        # Run evaluation system
        self.eval_system.run(
            exp_code=exp_code,
            exp_folder=exp_folder,
            visualize_flag=visualize,
            debug_flag=self.debug_flag
        )
        
        # Results are now in dataset._aggr_metrics (attached to eval_system.aggr_metrics)
        results = self.current_dataset._aggr_metrics.get(exp_code, {})
        
        # Add experiment to dataset if not already present
        if not self.current_dataset.has_experiment(exp_code):
            self.current_dataset.add_experiment(exp_code, exp_params, results)
        
        return results
    
    # === HIERARCHICAL LOAD/SAVE ===
    
    def load_experiments_hierarchical(
        self,
        exp_codes: List[str],
        recompute: bool = False
    ) -> List[str]:
        """
        Load experiment data using hierarchical pattern: Memory → Local → External.
        
        Args:
            exp_codes: List of experiment codes to load
            recompute: Skip loading from local/external if True
            
        Returns:
            List of experiment codes that could not be loaded
        """
        if self.current_dataset is None:
            raise RuntimeError("No dataset initialized")
        
        # 1. Check memory - filter already loaded experiments
        missing_memory = [
            code for code in exp_codes
            if not self.current_dataset.has_experiment(code)
        ]
        
        if not missing_memory:
            self.logger.info(f"All experiments {exp_codes} already in memory")
            return []
        
        # 2. Load from local files (unless recompute)
        missing_local = missing_memory
        if not recompute:
            missing_local = self._load_from_local(missing_memory)
            loaded_count = len(missing_memory) - len(missing_local)
            if loaded_count > 0:
                self.logger.console_info(
                    f"Loaded {loaded_count} experiments from local files"
                )
        else:
            self.logger.info("Recompute mode: Skipping local file loading")
        
        if not missing_local:
            return []
        
        # 3. Load from external source (unless debug)
        missing_external = missing_local
        if not self.debug_flag and self.external_data is not None:
            missing_external = self._load_from_external(missing_local)
            loaded_count = len(missing_local) - len(missing_external)
            if loaded_count > 0:
                self.logger.console_info(
                    f"Loaded {loaded_count} experiments from external source"
                )
        elif self.debug_flag:
            self.logger.info("Debug mode: Skipping external data loading")
        else:
            self.logger.warning("No external interface: Skipping external loading")
        
        return missing_external
    
    def _load_from_local(self, exp_codes: List[str]) -> List[str]:
        """
        Load experiment records and metrics from local JSON files.
        
        Args:
            exp_codes: Experiment codes to load
            
        Returns:
            List of codes that were not found locally
        """
        assert self.current_dataset is not None
        missing = []
        
        for exp_code in exp_codes:
            try:
                # Load experiment record
                exp_folder = self.local_data.get_experiment_folder(exp_code)
                record_file = os.path.join(exp_folder, f"{exp_code}_record.json")
                
                if not os.path.exists(record_file):
                    missing.append(exp_code)
                    continue
                
                with open(record_file, 'r') as f:
                    record = json.load(f)
                
                exp_params = record.get('Parameters', {})
                
                # Load performance metrics if available
                aggr_metrics = {}
                for perf_code in self.current_dataset.schema.performance_attrs.keys():
                    metrics_file = os.path.join(
                        exp_folder,
                        f"{exp_code}_{perf_code}_metrics.json"
                    )
                    if os.path.exists(metrics_file):
                        with open(metrics_file, 'r') as f:
                            aggr_metrics[perf_code] = json.load(f)
                
                # Add to dataset
                self.current_dataset.add_experiment(
                    exp_code,
                    exp_params,
                    aggr_metrics if aggr_metrics else None
                )
                
            except Exception as e:
                self.logger.warning(
                    f"Failed to load experiment {exp_code} from local: {str(e)}"
                )
                missing.append(exp_code)
        
        return missing
    
    def _load_from_external(self, exp_codes: List[str]) -> List[str]:
        """
        Load experiment data from external database.
        
        Args:
            exp_codes: Experiment codes to load
            
        Returns:
            List of codes that were not found externally
        """
        assert self.current_dataset is not None
        
        if self.external_data is None:
            return exp_codes
        
        # Use external interface to pull experiment records
        # This delegates to the IExternalData implementation
        try:
            missing, external_records = self.external_data.pull_exp_records(exp_codes)
            
            # Add loaded experiments to dataset
            for exp_code, record in external_records.items():
                exp_params = record.get('Parameters', {})
                # Note: Performance metrics would be loaded separately
                # via pull_performance_records if needed
                self.current_dataset.add_experiment(exp_code, exp_params)
            
            return missing
            
        except Exception as e:
            self.logger.error(f"External data loading failed: {str(e)}")
            return exp_codes
    
    def save_experiments_hierarchical(
        self,
        exp_codes: List[str],
        recompute: bool = False
    ) -> None:
        """
        Save experiment data using hierarchical pattern: Memory → Local → External.
        
        Args:
            exp_codes: List of experiment codes to save
            recompute: Overwrite existing files if True
        """
        if self.current_dataset is None:
            raise RuntimeError("No dataset initialized")
        
        # Filter to experiments that exist in dataset
        codes_to_save = [
            code for code in exp_codes
            if self.current_dataset.has_experiment(code)
        ]
        
        if not codes_to_save:
            self.logger.warning(f"No experiments to save from {exp_codes}")
            return
        
        # 1. Save to local files
        saved_local = self._save_to_local(codes_to_save, recompute)
        if saved_local:
            self.logger.console_info(f"Saved {len(saved_local)} experiments to local files")
        
        # 2. Save to external source (unless debug)
        if not self.debug_flag and self.external_data is not None:
            saved_external = self._save_to_external(codes_to_save, recompute)
            if saved_external:
                self.logger.console_info(
                    f"Saved {len(saved_external)} experiments to external source"
                )
        elif self.debug_flag:
            self.logger.info("Debug mode: Skipping external save")
    
    def _save_to_local(self, exp_codes: List[str], recompute: bool) -> List[str]:
        """
        Save experiment records and metrics to local JSON files.
        
        Args:
            exp_codes: Experiment codes to save
            recompute: Overwrite existing if True
            
        Returns:
            List of codes that were successfully saved
        """
        assert self.current_dataset is not None
        saved = []
        
        for exp_code in exp_codes:
            try:
                exp_folder = self.local_data.get_experiment_folder(exp_code)
                os.makedirs(exp_folder, exist_ok=True)
                
                record_file = os.path.join(exp_folder, f"{exp_code}_record.json")
                
                # Skip if exists and not recomputing
                if os.path.exists(record_file) and not recompute:
                    continue
                
                # Get experiment data from dataset
                exp_params = self.current_dataset.get_experiment_params(exp_code)
                
                # Save experiment record
                record = {
                    "Code": exp_code,
                    "Parameters": exp_params
                }
                with open(record_file, 'w') as f:
                    json.dump(record, f, indent=2)
                
                # Save performance metrics
                if exp_code in self.current_dataset._aggr_metrics:
                    for perf_code, metrics in self.current_dataset._aggr_metrics[exp_code].items():
                        metrics_file = os.path.join(
                            exp_folder,
                            f"{exp_code}_{perf_code}_metrics.json"
                        )
                        with open(metrics_file, 'w') as f:
                            json.dump(metrics, f, indent=2)
                
                saved.append(exp_code)
                
            except Exception as e:
                self.logger.error(f"Failed to save experiment {exp_code} locally: {str(e)}")
        
        return saved
    
    def _save_to_external(self, exp_codes: List[str], recompute: bool) -> List[str]:
        """
        Save experiment data to external database.
        
        Args:
            exp_codes: Experiment codes to save
            recompute: Overwrite existing if True
            
        Returns:
            List of codes successfully saved
        """
        assert self.current_dataset is not None
        
        if self.external_data is None:
            return []
        
        try:
            # Prepare data for external save
            records_to_save = {}
            for exp_code in exp_codes:
                exp_params = self.current_dataset.get_experiment_params(exp_code)
                records_to_save[exp_code] = {
                    "Code": exp_code,
                    "Parameters": exp_params
                }
            
            # Push to external source
            pushed = self.external_data.push_exp_records(
                exp_codes, records_to_save, recompute
            )
            
            return exp_codes if pushed else []
            
        except Exception as e:
            self.logger.error(f"External save failed: {str(e)}")
            return []

