"""
LBPAgent - Main orchestration class for AIXD architecture.

Replaces study-based logic with dataset-centric approach while maintaining
existing workflow patterns. Wraps EvaluationSystem, PredictionSystem, and
manages schema generation and dataset initialization.
"""

from typing import Any, Dict, List, Type, Optional, Tuple
from ..core.schema import DatasetSchema, SchemaRegistry
from ..core.dataset import Dataset
from ..core.datamodule import DataModule
from ..orchestration.evaluation import EvaluationSystem
from ..orchestration.prediction import PredictionSystem
from ..orchestration.calibration import CalibrationSystem
from ..interfaces import IFeatureModel, IEvaluationModel, IPredictionModel, IExternalData
from ..utils import LocalData, LBPLogger


class LBPAgent:
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
        local_folder: str,
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
        self.local_data = LocalData(root_folder, local_folder)

        # Initialize logger
        self.logger_name = "LBPAgent"
        self.logger = LBPLogger(self.logger_name, self.local_data.get_log_folder('logs'))
        self.logger.info("Initializing LBP Agent")
        
        # Initialize external data interface
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
        
        # Initialize system components (will be set with dataset reference)
        self.eval_system: Optional[EvaluationSystem] = None
        self.pred_system: Optional[PredictionSystem] = None
        self.calibration_system: Optional[CalibrationSystem] = None
        
        # Model registry (store classes and params until dataset is initialized)
        self._feature_model_specs: List[Tuple[Type[IFeatureModel], dict]] = []  # List of (class, kwargs)
        self._evaluation_model_specs: List[Tuple[Type[IEvaluationModel], dict]] = []  # List of (class, kwargs)
        self._prediction_model_specs: List[Tuple[Type[IPredictionModel], dict]] = []  # List of (class, kwargs)
        
        # Initialization state guard
        self._initialized = False
        
        self.logger.console_info("\n------- Welcome to PFAB - Predictive Fabrication -------\n")
    
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
        self.logger.console_info(f"Registered model: {model_class.__name__}")

    def register_feature_model(self, feature_class: Type[Any], **kwargs) -> None:
        """Register a feature model."""
        self._register_model(feature_class, self._feature_model_specs, kwargs)

    def register_evaluation_model(self, evaluation_class: Type[IEvaluationModel], **kwargs) -> None:
        """Register an evaluation model."""
        self._register_model(evaluation_class, self._evaluation_model_specs, kwargs)
    
    def register_prediction_model(self, prediction_class: Type[IPredictionModel], **kwargs) -> None:
        """Register a prediction model."""
        self._register_model(prediction_class, self._prediction_model_specs, kwargs)

    
    # === SCHEMA GENERATION & INITIALIZATION ===
    
    def _instantiate_models(self) -> None:
        """Instantiate model instances from registered classes and populate systems."""
        if not self._evaluation_model_specs:
            raise ValueError("No evaluation models registered. Call register_evaluation_model() first.")
        
        self.logger.info("Instantiating models from registered classes...")
        
        # Initialize systems (dataset will be set later)
        self.eval_system = EvaluationSystem(dataset=None, logger=self.logger)  # type: ignore
        self.pred_system = PredictionSystem(dataset=None, logger=self.logger)  # type: ignore
        self.calibration_system = CalibrationSystem(
            dataset=None, # type: ignore
            logger=self.logger, 
            predict_fn=self.pred_system._predict_from_params,
            evaluate_fn=self.eval_system.evaluate)
        
        # Instanticate feature model instances from registered classes
        feature_model_mappings = {}
        for feature_class, feature_kwargs in self._feature_model_specs:
            self.logger.info(f"Instantiating feature model: {feature_class.__name__}")
            feature_model = feature_class(logger=self.logger, **feature_kwargs)
            if feature_model in feature_model_mappings.values():
                raise ValueError(f"Feature model {feature_class.__name__} registered multiple times.")
            # Store instance for later attachment
            for feature_code in feature_model.feature_codes:
                if feature_code in feature_model_mappings:
                    raise ValueError(
                        f"Feature model for code '{feature_code}' already registered. "
                        f"Cannot register multiple models for the same feature code."
                    )
                feature_model_mappings[feature_code] = feature_model
        
        # Instantiate evaluation model instances from registered classes
        for eval_class, eval_kwargs in self._evaluation_model_specs:
            self.logger.info(f"Instantiating evaluation model {eval_class.__name__}")
            eval_model = eval_class(logger=self.logger, **eval_kwargs)

            # Ensure feature model dependency is satisfied
            if eval_model.feature_input_code not in feature_model_mappings:
                raise ValueError(
                    f"Feature model for code '{eval_model.feature_input_code}' "
                    f"not registered but required by evaluation model '{eval_class.__name__}'."
                )
            eval_model.set_feature_model(feature_model_mappings[eval_model.feature_input_code])

            # Store instance in system
            if eval_model in self.eval_system.evaluation_models:
                raise ValueError(f"Evaluation model {eval_class.__name__} registered multiple times.")
            self.eval_system.evaluation_models.append(eval_model)
        
        # Instantiate prediction model instances from registered classes (unique classes only)
        for pred_class, pred_kwargs in self._prediction_model_specs:
            self.logger.info(f"Instantiating prediction model: {pred_class.__name__}")
            pred_model = pred_class(logger=self.logger, **pred_kwargs)

            for feature_code in pred_model.feature_input_codes:
                # Ensure feature model dependencies are satisfied
                if feature_code not in feature_model_mappings:
                    raise ValueError(
                        f"Feature model for code '{feature_code}' not registered "
                        f"but required by prediction model '{pred_class.__name__}'."
                    )
                pred_model.add_feature_model(feature_model_mappings[feature_code])

            # Store instance in system
            if pred_model in self.pred_system.prediction_models:
                raise ValueError(f"Prediction model {pred_class.__name__} registered multiple times.")
            self.pred_system.prediction_models.append(pred_model)

    def _generate_schema_from_registered_models(self) -> DatasetSchema:
        """Generate DatasetSchema from registered models."""
        # Instantiate models if not already done
        if self.eval_system is None or self.pred_system is None:
            self._instantiate_models()
        
        self.logger.info("Generating schema from registered model classes...")
        
        # Extract specs from systems
        eval_specs = self.eval_system.get_model_specs()  # type: ignore
        pred_specs = self.pred_system.get_model_specs()  # type: ignore
        
        # Delegate schema construction to DatasetSchema
        schema = DatasetSchema.from_model_specs(
            eval_specs=eval_specs,
            pred_specs=pred_specs,
            logger=self.logger
        )
        
        return schema
        
    def initialize(self) -> Dataset:
        """Initialize schema, dataset and orchestration systems from registered models."""

        self.logger.console_info("\n===== Initializing Dataset =====")
        
        # Step 1: Generate schema from registered models
        self.logger.info("Generating schema from registered models")
        schema = self._generate_schema_from_registered_models()
        
        # Step 2: Get or create schema ID via registry
        registry = SchemaRegistry(self.local_data.local_folder)
        schema_hash = schema._compute_schema_hash()
        schema_struct = schema.to_dict()
        new_schema_id = registry.get_or_create_schema_id(schema_hash, schema_struct)
        self.local_data.set_schema(new_schema_id)
        self.logger.console_info(f"Using schema ID: {new_schema_id}")
        
        # Step 3: Create dataset with storage interfaces
        dataset = Dataset(
            schema=schema,
            schema_id=new_schema_id,
            local_data=self.local_data,
            external_data=self.external_data,
            logger=self.logger,
            debug_mode=self.debug_flag,
        )
        
        # Step 4: Attach dataset to already-initialized systems
        self.logger.info("Attaching dataset to orchestration systems")
        self.eval_system.dataset = dataset  # type: ignore
        self.pred_system.dataset = dataset  # type: ignore
        self.calibration_system.dataset = dataset # type: ignore
        
        # # Step 5: Attach feature models to evaluation models (now that dataset exists)
        # self.logger.info("Attaching feature models to evaluation models")
        # for eval_model in self.eval_system.evaluation_models:
        #     feature_model_class = eval_model.feature_model_class  # type: ignore
        #     feature_model = feature_model_class(dataset=dataset, logger=self.logger)
        #     eval_model.add_feature_model(feature_model)
        
        # # Step 6: Setup prediction model mappings (feature_name -> model)
        # self.logger.info("Setting up prediction model mappings")
        # for pred_model in self.pred_system.prediction_models:  # type: ignore
        #     # Create feature-to-model mappings
        #     for feature_name in pred_model.feature_output_codes:
        #         if feature_name in self.pred_system.feature_to_model:  # type: ignore
        #             self.logger.warning(  # type: ignore
        #                 f"Feature '{feature_name}' already registered to another model. "
        #                 f"Overwriting with new model."
        #             )
        #         self.pred_system.feature_to_model[feature_name] = pred_model  # type: ignore
        
        # # Step 7: Create and attach shared feature model instances to prediction models
        # self._add_feature_model_instances(dataset)
                
        # Mark as initialized
        self._initialized = True
        
        # Display summary
        summary_lines = [
            "Dataset Initialization Summary:",
            f"  - Schema ID: {new_schema_id}",
            f"  - Parameters: {len(schema.parameters.data_objects)}",
            f"  - Dimensions: {len(schema.dimensions.data_objects)}",
            f"  - Performance Attributes: {len(schema.performance.data_objects)}",
        ]
        self.logger.console_summary("\n".join(summary_lines))
        self.logger.console_success(f"Successfully initialized dataset '{new_schema_id}'.")
        
        return dataset
    
    # def _add_feature_model_instances(self, dataset: Dataset) -> None:
    #     """Create shared feature model instances and attach to prediction models."""
    #     from ..interfaces.features import IFeatureModel
    #     from typing import Type, Dict
        
    #     if not self.pred_system:
    #         return  # No prediction models to process
        
    #     # Collect all feature model types needed across all prediction models
    #     feature_model_collection: List[tuple] = []  # [(model, code, feature_model_type), ...]
        
    #     for pred_model in self.pred_system.prediction_models:
    #         feature_model_types = pred_model.feature_models_as_input
    #         for code, feature_model_type in feature_model_types.items():
    #             feature_model_collection.append((pred_model, code, feature_model_type))
        
    #     if not feature_model_collection:
    #         self.logger.debug("No feature model dependencies declared by prediction models")
    #         return
        
    #     # Create shared instances by type (same type = same instance)
    #     feature_model_dict: Dict[Type[IFeatureModel], IFeatureModel] = {}
        
    #     for pred_model, code, feature_model_type in feature_model_collection:
    #         # Check if we already created an instance of this type
    #         if feature_model_type not in feature_model_dict:
    #             # Create new instance
    #             self.logger.debug(f"Creating feature model instance: {feature_model_type.__name__}")
    #             feature_model_instance = feature_model_type(
    #                 dataset=dataset,
    #                 logger=self.logger
    #             )
    #             feature_model_dict[feature_model_type] = feature_model_instance
    #         else:
    #             # Reuse existing instance
    #             feature_model_instance = feature_model_dict[feature_model_type]
    #             self.logger.debug(
    #                 f"Reusing feature model instance: {feature_model_type.__name__}"
    #             )
            
    #         # Attach to prediction model
    #         pred_model.add_feature_model(code, feature_model_instance)
        
    #     self.logger.info(
    #         f"Attached {len(feature_model_collection)} feature model dependencies "
    #         f"({len(feature_model_dict)} unique types)"
    #     )
    
    # === EVALUATION OPERATIONS ===

    def evaluate(
        self,
        exp_data: 'ExperimentData',  # type: ignore
        evaluate_from: int = 0,
        evaluate_to: Optional[int] = None,
        visualize: bool = False,
        recompute: bool = False
    ) -> None:
        """Evaluate experiment and mutate exp_data with results."""
        if self.eval_system is None:
            raise RuntimeError("EvaluationSystem not initialized. Call initialize() first.")
        
        # Delegate to evaluation system
        self.eval_system.evaluate_experiment(
            exp_data=exp_data,
            evaluate_from=evaluate_from,
            evaluate_to=evaluate_to,
            visualize=visualize,
            recompute=recompute
        )
        
        self.logger.console_info(f"Experiment '{exp_data.exp_code}' evaluated successfully")

    # === PREDICTION OPERATIONS ===
    
    def train(self, datamodule: DataModule, **kwargs) -> None:
        """Train all prediction models (offline learning)."""
        if self.pred_system is None:
            raise RuntimeError(
                "PredictionSystem not initialized. Call initialize() first."
            )
        
        self.pred_system.train(datamodule, **kwargs)
    
    def tune(self, datamodule: DataModule, **kwargs) -> None:
        """Fine-tune prediction models with new data (online learning)."""
        if self.pred_system is None:
            raise RuntimeError(
                "PredictionSystem not initialized. Call initialize() first."
            )
        
        self.pred_system.tune(datamodule, **kwargs)
        
    def predict(
        self,
        exp_data: 'ExperimentData',  # type: ignore
        predict_from: int = 0,
        predict_to: Optional[int] = None,
        batch_size: int = 1000
    ) -> None:
        """Predict dimensional features and mutate exp_data with results."""
        if self.pred_system is None:
            raise RuntimeError("PredictionSystem not initialized. Call initialize() first.")
        
        # Delegate to prediction system
        self.pred_system.predict_experiment(
            exp_data=exp_data,
            predict_from=predict_from,
            predict_to=predict_to,
            batch_size=batch_size
        )
        
        self.logger.console_info(
            f"Predicted dimensions [{predict_from}:{predict_to}] for '{exp_data.exp_code}'"
        )
    
    # === CALIBRATION ===
    
    def set_performance_weights(
        self,
        performance_weights: Dict[str, float]
    ) -> None:
        """
        Configure the calibration system weights.
        
        Args:
            performance_weights: Weights for system performance calculation.
        """
        if self.calibration_system:
            self.calibration_system.set_performance_weights(performance_weights)
            self.logger.info(f"Configured calibration weights: {performance_weights}")
        else:
            self.logger.warning("Calibration system not initialized. Call initialize() first.")

    def propose_next_experiments(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        n_points: int = 1,
        mode: Any = 'exploration', # using Any to avoid import issues if Literal not imported, but it is.
        fixed_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Propose next experiments using the calibration system.
        
        Args:
            param_ranges: Ranges for parameters to optimize.
            n_points: Number of experiments to propose.
            mode: 'exploration' (Active Learning) or 'optimization' (Exploitation).
            fixed_params: Parameters to keep fixed.
        """
        if not self._initialized or self.calibration_system is None:
             raise RuntimeError("Agent not initialized.")
        
        return self.calibration_system.propose_new_experiments(
            param_ranges=param_ranges,
            n_points=n_points,
            mode=mode,
            fixed_params=fixed_params,
            **kwargs
        )

