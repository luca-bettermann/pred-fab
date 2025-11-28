"""
LBPAgent - Main orchestration class for AIXD architecture.

Replaces study-based logic with dataset-centric approach while maintaining
existing workflow patterns. Wraps EvaluationSystem, PredictionSystem, and
manages schema generation and dataset initialization.
"""

from typing import Any, Dict, List, Type, Optional, Tuple
import numpy as np
from ..core.schema import DatasetSchema, SchemaRegistry
from ..core.dataset import Dataset
from ..core.datamodule import DataModule
from ..orchestration.evaluation import EvaluationSystem
from ..orchestration.prediction import PredictionSystem
from ..orchestration.calibration import BayesianCalibration
from ..interfaces import IEvaluationModel, IPredictionModel, ICalibrationModel, IExternalData
from ..utils import LocalData, LBPLogger


class LBPAgent:
    """
    Main orchestration class for AIXD-based LBP framework.
    
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
        """Register an evaluation model."""
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
        """Register a prediction model."""
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
        """Register a calibration model."""
        if self._initialized:
            raise RuntimeError(
                "Cannot register models after initialize() has been called. "
                "Create a new LBPAgent instance for additional models."
            )
        
        self.calibration_model = calibration_class(logger=self.logger, **kwargs)
        self.logger.console_info(f"Set calibration model to '{calibration_class.__name__}'.")
    
    # === SCHEMA GENERATION & INITIALIZATION ===
    
    def _instantiate_models(self) -> None:
        """Instantiate model instances from registered classes and populate systems."""
        if not self._evaluation_model_specs:
            raise ValueError("No evaluation models registered. Call register_evaluation_model() first.")
        
        self.logger.info("Instantiating models from registered classes...")
        
        # Initialize systems (dataset will be set later)
        self.eval_system = EvaluationSystem(dataset=None, logger=self.logger)  # type: ignore
        self.pred_system = PredictionSystem(dataset=None, logger=self.logger)  # type: ignore
        
        # Instantiate evaluation model instances from registered classes
        for perf_code, (eval_class, eval_kwargs) in self._evaluation_model_specs.items():
            self.logger.info(f"Instantiating evaluation model for '{perf_code}': {eval_class.__name__}")
            eval_model = eval_class(logger=self.logger, **eval_kwargs)
            # Store instance in system (feature model attachment happens later with dataset)
            self.eval_system.evaluation_models[perf_code] = eval_model
        
        # Instantiate prediction model instances from registered classes (unique classes only)
        seen_pred_classes = set()
        for perf_code, (pred_class, pred_kwargs) in self._prediction_model_specs.items():
            if pred_class not in seen_pred_classes:
                self.logger.info(f"Instantiating prediction model: {pred_class.__name__}")
                pred_model = pred_class(logger=self.logger, **pred_kwargs)
                # Store instance in system
                self.pred_system.prediction_models.append(pred_model)
                seen_pred_classes.add(pred_class)
    
    def generate_schema_from_registered_models(self) -> DatasetSchema:
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
    
    # === INITIALIZATION ===
    
    def initialize(
        self,
        static_params: Dict[str, Any],
    ) -> Dataset:
        """
        Initialize dataset and orchestration systems from registered models.
        
        Args:
            static_params: Static parameter values for dataset
        
        Returns:
            Initialized Dataset instance
        """
        self.logger.console_info("\n===== Initializing Dataset =====")
        
        # Step 1: Generate schema from registered models
        self.logger.info("Generating schema from registered models")
        schema = self.generate_schema_from_registered_models()
        
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
        
        # Step 4: Attach dataset to already-initialized systems
        self.logger.info("Attaching dataset to orchestration systems")
        self.eval_system.dataset = dataset  # type: ignore
        self.pred_system.dataset = dataset  # type: ignore
        
        # Step 5: Attach feature models to evaluation models (now that dataset exists)
        self.logger.info("Attaching feature models to evaluation models")
        for perf_code, eval_model in self.eval_system.evaluation_models.items():  # type: ignore
            feature_model_class = eval_model.feature_model_type  # type: ignore
            feature_model = feature_model_class(dataset=dataset, logger=self.logger)
            eval_model.add_feature_model(feature_model)
        
        # Step 6: Setup prediction model mappings (feature_name -> model)
        self.logger.info("Setting up prediction model mappings")
        for pred_model in self.pred_system.prediction_models:  # type: ignore
            # Create feature-to-model mappings
            for feature_name in pred_model.predicted_features:
                if feature_name in self.pred_system.feature_to_model:  # type: ignore
                    self.logger.warning(  # type: ignore
                        f"Feature '{feature_name}' already registered to another model. "
                        f"Overwriting with new model."
                    )
                self.pred_system.feature_to_model[feature_name] = pred_model  # type: ignore
        
        # Step 7: Create and attach shared feature model instances to prediction models
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
        """Create shared feature model instances and attach to prediction models."""
        from ..interfaces.features import IFeatureModel
        from typing import Type, Dict
        
        if not self.pred_system:
            return  # No prediction models to process
        
        # Collect all feature model types needed across all prediction models
        feature_model_collection: List[tuple] = []  # [(model, code, feature_model_type), ...]
        
        for pred_model in self.pred_system.prediction_models:
            feature_model_types = pred_model.feature_models_as_input
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
    
    def evaluate(
        self,
        exp_data: 'ExperimentData',  # type: ignore
        evaluate_from: int = 0,
        evaluate_to: Optional[int] = None,
        visualize: bool = False,
        recompute: bool = False
    ) -> None:
        """Evaluate experiment and mutate exp_data with results."""
        # Delegate to evaluation system
        self.eval_system.evaluate_experiment(  # type: ignore[attr-defined]
            exp_data=exp_data,
            evaluate_from=evaluate_from,
            evaluate_to=evaluate_to,
            visualize=visualize,
            recompute=recompute
        )
        
        self.logger.console_info(f"Experiment '{exp_data.exp_code}' evaluated successfully")

    # === CALIBRATION ===
    
    def configure_calibration(
        self,
        calibration_model: Optional[ICalibrationModel] = None,
        **kwargs
    ) -> None:
        """
        Configure the calibration model.
        
        Args:
            calibration_model: Custom ICalibrationModel instance. If None, uses BayesianCalibration.
            **kwargs: Arguments passed to BayesianCalibration if calibration_model is None.
                      (n_iterations, n_initial_points, exploration_weight, random_seed)
        """
        if calibration_model is not None:
            self.calibration_model = calibration_model
            self.logger.info(f"Configured custom calibration model: {type(calibration_model).__name__}")
        else:
            # Default to BayesianCalibration
            self.calibration_model = BayesianCalibration(
                logger=self.logger,
                **kwargs
            )
            self.logger.info("Configured default BayesianCalibration model")

    def calibrate(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        objectives: Dict[str, Dict[str, Any]],
        fixed_params: Optional[Dict[str, Any]] = None,
        direct_mode: bool = False
    ) -> Dict[str, float]:
        """
        Run calibration to find optimal parameters using the configured calibration model.
        
        Args:
            param_ranges: Ranges for parameters to optimize.
            objectives: Dict of {perf_code: {'weight': float, 'feature': str}}.
            fixed_params: Parameters to keep fixed.
            direct_mode: If True, uses prediction model's uncertainty directly (skips GP surrogate).
        """
        if not self._initialized:
             raise RuntimeError("Agent not initialized.")
        
        # Auto-configure if not set
        if self.calibration_model is None:
            self.logger.info("No calibration model configured. Using default BayesianCalibration.")
            self.configure_calibration()
             
        # 2. Define Objective Function
        def objective_fn(params: Dict[str, float]) -> float:
            # Predict all features
            predictions = self.pred_system._predict_from_params(params) # type: ignore
            
            total_score = 0.0
            
            for perf_code, config in objectives.items():
                weight = config.get('weight', 1.0)
                feature_name = config['feature']
                
                if feature_name not in predictions:
                    raise ValueError(f"Feature '{feature_name}' not predicted.")
                
                feature_val = predictions[feature_name]
                
                if perf_code not in self.eval_system.evaluation_models: # type: ignore
                    raise ValueError(f"Unknown performance code: {perf_code}")
                    
                eval_model = self.eval_system.evaluation_models[perf_code] # type: ignore
                
                # Compute target & scaling
                target = eval_model._compute_target_value(**params)
                scaling = eval_model._compute_scaling_factor(**params)
                
                # Compute performance
                # feature_val might be a 0-d array or scalar, ensure float
                if hasattr(feature_val, 'item'):
                    feature_val = feature_val.item()
                perf = eval_model._compute_performance(float(feature_val), target, scaling)
                
                total_score += perf * weight
                
            return total_score

        # 3. Define Uncertainty Function (if direct_mode)
        uncertainty_fn = None
        if direct_mode:
            # Identify required features for uncertainty
            required_features = [config['feature'] for config in objectives.values()]
            
            def u_fn(params: Dict[str, float]) -> float:
                # Only request uncertainty for relevant features
                uncertainties = self.pred_system.predict_uncertainty(params, required_features=required_features) # type: ignore
                
                total_sigma = 0.0
                for perf_code, config in objectives.items():
                    weight = config.get('weight', 1.0)
                    feature_name = config['feature']
                    
                    if feature_name not in uncertainties:
                        raise ValueError(f"Direct mode requested but feature '{feature_name}' has no uncertainty prediction.")
                    
                    sigma = uncertainties[feature_name]
                    if hasattr(sigma, 'mean'):
                        sigma = float(np.mean(sigma))
                    else:
                        sigma = float(sigma)
                        
                    total_sigma += sigma * weight
                return total_sigma
            uncertainty_fn = u_fn

        # 4. Run Calibration
        # We know self.calibration_model is not None here
        best_params = self.calibration_model.calibrate( # type: ignore
            param_ranges=param_ranges,
            objective_fn=objective_fn,
            fixed_params=fixed_params,
            uncertainty_fn=uncertainty_fn
        )
        
        return best_params

