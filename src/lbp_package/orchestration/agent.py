"""
LBPAgent - Main orchestration class for AIXD architecture.

Replaces study-based logic with dataset-centric approach while maintaining
existing workflow patterns. Wraps EvaluationSystem, PredictionSystem, and
manages schema generation and dataset initialization.
"""

from typing import Any, Dict, List, Type, Optional
from dataclasses import fields

from typing import TYPE_CHECKING
import pandas as pd

from ..core.schema import DatasetSchema
from ..core.schema_registry import SchemaRegistry
from ..core.dataset import Dataset
from ..core.datamodule import DataModule
from ..core.data_objects import DataReal, DataInt, DataBool, DataCategorical, DataString, DataDimension
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
        external_data: Optional[IExternalData] = None,
        server_folder: Optional[str] = None,
        debug_flag: bool = False,
        recompute_flag: bool = False,
        visualize_flag: bool = True,
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
        """
        self.logger_name = "LBPAgent"
        self.logger = LBPLogger(self.logger_name, log_folder)
        self.logger.info("Initializing LBP Agent")
        
        self.external_data = external_data
        if external_data is None:
            self.logger.console_warning(
                "No external data interface provided: "
                "Dataset records need to be available as local JSON files."
            )
        
        # System settings
        self.debug_flag = debug_flag
        self.recompute_flag = recompute_flag
        self.visualize_flag = visualize_flag
        
        # Initialize local data handler
        self.local_data = LocalData(root_folder, local_folder, server_folder)
        
        # Initialize system components (will be set with dataset reference)
        self.eval_system: Optional[EvaluationSystem] = None
        self.pred_system: Optional[PredictionSystem] = None
        self.calibration_model: Optional[ICalibrationModel] = None
        
        # Model registry (store classes and params until dataset is initialized)
        self._evaluation_model_specs: Dict[str, tuple] = {}  # perf_code -> (class, kwargs)
        self._prediction_model_specs: Dict[str, tuple] = {}  # perf_code -> (class, kwargs)
        
        # Initialization state guard
        self._initialized = False
        
        self.logger.console_info("\n------- Welcome to Learning by Printing (AIXD) -------\n")
    
    # === MODEL REGISTRATION ===
    
    def register_evaluation_model(
        self,
        performance_code: str,
        evaluation_class: Type[IEvaluationModel],
        **kwargs
    ) -> None:
        """
        Register an evaluation model.
        
        Args:
            performance_code: Code identifying the performance metric
            evaluation_class: IEvaluationModel subclass (dataclass)
            **kwargs: Additional parameters for model instantiation
            
        Raises:
            RuntimeError: If called after initialize()
        """
        if self._initialized:
            raise RuntimeError(
                "Cannot register models after initialize() has been called. "
                "Create a new LBPAgent instance for additional models."
            )
        
        # Store class and params until dataset is created
        self._evaluation_model_specs[performance_code] = (evaluation_class, kwargs)
        
        self.logger.console_info(f"Registered evaluation model: {evaluation_class.__name__}")
    
    def register_prediction_model(
        self,
        performance_codes: List[str],
        prediction_class: Type[IPredictionModel],
        **kwargs
    ) -> None:
        """
        Register a prediction model.
        
        Args:
            performance_codes: List of performance codes this model predicts
            prediction_class: IPredictionModel subclass (dataclass)
            **kwargs: Additional model parameters
            
        Raises:
            RuntimeError: If called after initialize()
        """
        if self._initialized:
            raise RuntimeError(
                "Cannot register models after initialize() has been called. "
                "Create a new LBPAgent instance for additional models."
            )
        
        # Store class and params for each performance code
        for perf_code in performance_codes:
            self._prediction_model_specs[perf_code] = (prediction_class, kwargs)
        
        self.logger.console_info(
            f"Registered prediction model: {prediction_class.__name__} "
            f"for performance codes: {performance_codes}"
        )
    
    def register_calibration_model(
        self,
        calibration_class: Type[ICalibrationModel],
        **kwargs
    ) -> None:
        """
        Register a calibration model.
        
        Args:
            calibration_class: ICalibrationModel subclass
            **kwargs: Additional model configuration
            
        Raises:
            RuntimeError: If called after initialize()
        """
        if self._initialized:
            raise RuntimeError(
                "Cannot register models after initialize() has been called. "
                "Create a new LBPAgent instance for additional models."
            )
        
        self.calibration_model = calibration_class(logger=self.logger, **kwargs)
        self.logger.console_info(f"Set calibration model to '{calibration_class.__name__}'.")
    
    # === SCHEMA GENERATION ===
    
    def generate_schema_from_active_models(self) -> DatasetSchema:
        """
        Generate DatasetSchema from registered evaluation/prediction model CLASSES.
        
        Uses dataclass introspection (fields()) to extract:
        - Parameters from model class fields
        - Dimensions (if models have dimensional properties)
        - Performance attributes (from evaluation model codes)
        
        Returns:
            DatasetSchema instance
            
        Raises:
            ValueError: If no models are registered
        """
        if not self._evaluation_model_specs:
            raise ValueError("No evaluation models registered. Call register_evaluation_model() first.")
        
        schema = DatasetSchema()
        seen_params = set()
        
        self.logger.info("Generating schema from registered model classes...")
        
        # 1. Extract parameters from evaluation model classes and their feature model classes
        for perf_code, (eval_class, eval_kwargs) in self._evaluation_model_specs.items():
            self.logger.info(f"Processing evaluation model for '{perf_code}': {eval_class.__name__}")
            
            # Extract from evaluation model class fields
            self._extract_parameters_from_class(eval_class, schema, seen_params)
            
            # Get feature model class from evaluation model (as a property)
            # We need to inspect the class to find the feature_model_type property
            if hasattr(eval_class, 'feature_model_type'):
                # This is a property - we'd need an instance to call it
                # For now, skip feature model extraction during schema generation
                # Feature model params will be extracted when models are instantiated
                pass
            
            # Add performance attribute
            perf_obj = DataReal(perf_code)
            schema.performance_attrs.add(perf_code, perf_obj)
            self.logger.info(f"Added performance attribute: {perf_code}")
        
        # 2. Extract parameters from prediction model classes
        for perf_code, (pred_class, pred_kwargs) in self._prediction_model_specs.items():
            self.logger.info(f"Processing prediction model for '{perf_code}': {pred_class.__name__}")
            self._extract_parameters_from_class(pred_class, schema, seen_params)
        
        self.logger.info(
            f"Generated schema with {len(schema.parameters.data_objects)} parameters, "
            f"{len(schema.dimensions.data_objects)} dimensions, "
            f"{len(schema.performance_attrs.data_objects)} performance attributes"
        )
        
        return schema
    
    def _extract_parameters_from_class(
        self,
        model_class: type,
        schema: DatasetSchema,
        seen_params: set
    ) -> None:
        """
        Extract DataObject parameters from a model CLASS's dataclass fields.
        
        Args:
            model_class: Model class (evaluation, feature, or prediction)
            schema: Schema to populate
            seen_params: Set of already-seen parameter names (for deduplication)
        """
        from ..core.data_objects import DataReal, DataInt, DataBool, DataCategorical, DataString, DataDimension
        
        # Get dataclass fields from class
        model_fields = fields(model_class)
        
        for field in model_fields:
            # Skip special fields
            if field.name in ('logger', 'feature_model', 'dataset'):
                continue
            
            # Check if field has a default value that's a DataObject
            if field.default is not field.default_factory:  # type: ignore
                # Field has a default value
                default_val = field.default
                data_object_types = (DataReal, DataInt, DataBool, DataCategorical, DataString, DataDimension)
                
                if isinstance(default_val, data_object_types):
                    param_name = field.name
                    
                    if param_name not in seen_params:
                        # Add to appropriate schema block
                        if isinstance(default_val, DataDimension):
                            schema.dimensions.add(param_name, default_val)
                            self.logger.info(f"  Added dimension: {param_name}")
                        else:
                            schema.parameters.add(param_name, default_val)
                            self.logger.info(f"  Added parameter: {param_name}")
                        
                        seen_params.add(param_name)
    
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
    
    # === INITIALIZATION ===
    
    def initialize(
        self,
        schema_id: Optional[str] = None,
        static_params: Optional[Dict[str, Any]] = None,
    ) -> Dataset:
        """
        Initialize dataset and orchestration systems from registered models.
        
        Supports two modes:
        1. New dataset: Provide performance_codes + static_params
           - Generates schema from registered models
           - Creates new dataset with schema registry
        2. Load existing: Provide schema_id only
           - Loads existing dataset from registry
        
        Args:
            schema_id: Optional schema ID to load existing dataset
            static_params: Static parameter values (for new dataset)
        
        Returns:
            Initialized Dataset instance
            
        Raises:
            ValueError: If parameters are invalid
        """
        if schema_id is not None:
            raise NotImplementedError(
                "Loading existing datasets is now handled by Dataset.populate(). "
                "Create dataset, then call dataset.populate() to load experiments."
            )
        
        # Mode 1: New dataset
        if static_params is None:
            raise ValueError("For new dataset, 'static_params' required.")
        
        self.logger.console_info("\n===== Initializing Dataset =====")
        
        # Step 1: Generate schema from registered models
        self.logger.info("Generating schema from registered models")
        schema = self.generate_schema_from_active_models()
        
        # Step 2: Get or create schema ID via registry
        registry = SchemaRegistry(self.local_data.local_folder)
        schema_hash = schema._compute_schema_hash()
        schema_struct = schema.to_dict()
        new_schema_id = registry.get_or_create_schema_id(schema_hash, schema_struct)
        self.logger.console_info(f"Using schema ID: {new_schema_id}")
        
        # Step 3: Create dataset with storage interfaces
        dataset = Dataset(
            name=new_schema_id,
            schema=schema,
            schema_id=new_schema_id,
            local_data=self.local_data,
            external_data=self.external_data,
            logger=self.logger,
            debug_mode=self.debug_flag,
        )
        dataset.set_static_values(static_params)
        
        # Step 4: Initialize orchestration systems with dataset
        self.logger.info("Initializing orchestration systems with dataset")
        self.eval_system = EvaluationSystem(dataset, self.logger)
        self.pred_system = PredictionSystem(dataset, self.logger)
        
        # Step 5: Register models with systems (instantiate from specs)
        self.logger.info("Instantiating and registering models with systems")
        for perf_code, (eval_class, eval_kwargs) in self._evaluation_model_specs.items():
            # Instantiate evaluation model (no dataset parameter)
            eval_model = eval_class(logger=self.logger, **eval_kwargs)
            
            # Get feature model class from evaluation model property
            feature_model_class = eval_model.feature_model_type  # type: ignore
            
            # Add to evaluation system (which will create and attach feature model with dataset)
            self.eval_system.add_evaluation_model(perf_code, eval_model, feature_model_class)
        
        for perf_code, (pred_class, pred_kwargs) in self._prediction_model_specs.items():
            # Instantiate prediction model (no dataset parameter)
            pred_model = pred_class(logger=self.logger, **pred_kwargs)
            # Register with system (model declares what it predicts via feature_names)
            self.pred_system.add_prediction_model(pred_model)
        
        # Step 6: Create and attach shared feature model instances to prediction models
        self._add_feature_model_instances(dataset)
        
        # Attach storage paths
        self.local_data.set_schema_id(new_schema_id)
        
        # Mark as initialized
        self._initialized = True
        
        # Display summary
        summary_lines = [
            "Dataset Initialization Summary:",
            f"  - Schema ID: {new_schema_id}",
            f"  - Parameters: {len(schema.parameters.data_objects)}",
            f"  - Dimensions: {len(schema.dimensions.data_objects)}",
            f"  - Performance Attributes: {len(schema.performance_attrs.data_objects)}",
        ]
        self.logger.console_summary("\n".join(summary_lines))
        self.logger.console_success(f"Successfully initialized dataset '{new_schema_id}'.")
        
        return dataset
    
    def _add_feature_model_instances(self, dataset: Dataset) -> None:
        """
        Create shared feature model instances and attach to prediction models.
        
        Feature model instances are shared across all models that need the same type.
        This prevents duplication and ensures consistency.
        
        Args:
            dataset: Dataset instance to pass to feature model constructors
        """
        from ..interfaces.features import IFeatureModel
        from typing import Type, Dict
        
        if not self.pred_system:
            return  # No prediction models to process
        
        # Collect all feature model types needed across all prediction models
        feature_model_collection: List[tuple] = []  # [(model, code, feature_model_type), ...]
        
        for pred_model in self.pred_system.prediction_models:
            feature_model_types = pred_model.feature_model_types
            for code, feature_model_type in feature_model_types.items():
                feature_model_collection.append((pred_model, code, feature_model_type))
        
        if not feature_model_collection:
            self.logger.debug("No feature model dependencies declared by prediction models")
            return
        
        # Create shared instances by type (same type = same instance)
        feature_model_dict: Dict[Type[IFeatureModel], IFeatureModel] = {}
        
        for pred_model, code, feature_model_type in feature_model_collection:
            # Check if we already created an instance of this type
            if feature_model_type not in feature_model_dict:
                # Create new instance
                self.logger.debug(f"Creating feature model instance: {feature_model_type.__name__}")
                feature_model_instance = feature_model_type(
                    dataset=dataset,
                    logger=self.logger
                )
                feature_model_dict[feature_model_type] = feature_model_instance
            else:
                # Reuse existing instance
                feature_model_instance = feature_model_dict[feature_model_type]
                self.logger.debug(
                    f"Reusing feature model instance: {feature_model_type.__name__}"
                )
            
            # Attach to prediction model
            pred_model.add_feature_model(code, feature_model_instance)
        
        self.logger.info(
            f"Attached {len(feature_model_collection)} feature model dependencies "
            f"({len(feature_model_dict)} unique types)"
        )
    
    # === EVALUATION OPERATIONS ===
    
    def evaluate_experiment(
        self,
        exp_data: 'ExperimentData',  # type: ignore
        visualize: bool = False,
        recompute: bool = False
    ) -> None:
        """
        Evaluate experiment and mutate exp_data with results.
        
        Delegates to EvaluationSystem for execution.
        
        Args:
            exp_data: ExperimentData with parameters populated
            visualize: Show visualizations if True
            recompute: Force recomputation if True
            
        Side effects:
            Mutates exp_data.performance and exp_data.metric_arrays
        """
        # Delegate to evaluation system
        self.eval_system.evaluate_experiment(  # type: ignore[attr-defined]
            exp_data=exp_data,
            visualize=visualize,
            recompute=recompute
        )
        
        self.logger.console_info(f"Experiment '{exp_data.exp_code}' evaluated successfully")
    
    # === PREDICTION OPERATIONS ===
    
    def train(self, datamodule: DataModule, **kwargs) -> None:
        """
        Train all prediction models using DataModule configuration.
        
        Delegates to PredictionSystem for execution.
        
        Args:
            datamodule: Configured DataModule with normalization/batching settings
            **kwargs: Additional training parameters passed to all prediction models
                     Examples:
                     - learning_rate: float (e.g., 0.001)
                     - epochs: int (e.g., 100)
                     - verbose: bool (enable training logs)
                     - early_stopping: bool
            
        Raises:
            RuntimeError: If prediction system not initialized
        
        Example:
            agent.train(datamodule, learning_rate=0.001, epochs=50, verbose=True)
        """
        if self.pred_system is None:
            raise RuntimeError(
                "PredictionSystem not initialized. Call initialize() first."
            )
        
        self.pred_system.train(datamodule, **kwargs)
    
    def predict(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """
        Predict all features for new parameter values.
        
        Delegates to PredictionSystem for execution with automatic
        normalization/denormalization.
        
        Args:
            X_new: DataFrame with parameter columns
            
        Returns:
            DataFrame with predicted feature values (denormalized)
            
        Raises:
            RuntimeError: If prediction system not initialized or trained
        """
        if self.pred_system is None:
            raise RuntimeError(
                "PredictionSystem not initialized. Call initialize() first."
            )
        
        return self.pred_system.predict(X_new)
