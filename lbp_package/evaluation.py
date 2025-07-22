import os
import numpy as np
import pandas as pd
import itertools
from typing import Any, Dict, List, Type, Tuple, Optional
from dataclasses import dataclass

from abc import ABC, abstractmethod
from numpy.typing import NDArray
from lbp_package.utils.folder_navigator import FolderNavigator
from lbp_package.utils.parameter_handler import ParameterHandling
from lbp_package.utils.log_manager import LBPLogger
from lbp_package.data_interface import DataInterface


@dataclass
class FeatureModel(ParameterHandling, ABC):
    """
    Abstract base class for feature extraction models.
    
    Provides a standardized interface for extracting features from experimental data.
    Uses dataclass-based parameter handling for clean configuration management.
    """
    
    def __init__(self, 
                 performance_code: str,
                 folder_navigator: FolderNavigator, 
                 logger: LBPLogger,
                 round_digits: int,
                 **study_params) -> None:
        """
        Initialize feature extraction model.

        Args:
            performance_code: Code identifying the performance metric
            folder_navigator: File system navigation utility
            logger: Logger instance for debugging and monitoring
            **study_params: Study parameters for configuration
        """
        self.nav = folder_navigator
        self.logger = logger
        self.round_digits = round_digits

        # Feature storage - supports multiple performance codes per model
        self.features: Dict[str, NDArray[np.float64]] = {}
        self.performance_codes: List[str] = []
        self.initialize_for_performance_code(performance_code)

        # Track processed dimensions to avoid duplicate computation
        self.processed_dims: List = []
        self.is_processed_state: bool = False

        # Temporary storage for current feature computation
        self.current_feature: Dict[str, float] = {}

        # Apply dataclass-based parameter handling
        self.set_model_parameters(**study_params)
        self._validate_parameters()

    # === ABSTRACT METHODS (Must be implemented by subclasses) ===
    @abstractmethod
    def _load_data(self, exp_nr: int) -> Any:
        """
        Load data for feature extraction.
        
        Args:
            exp_nr: Experiment number
            
        Returns:
            Loaded data object
        """
        ...

    @abstractmethod
    def _compute_features(self, data: Any, visualize_flag: bool) -> Dict[str, float]:
        """
        Extract features from loaded data.
        
        Args:
            data: Data object from _load_data
            visualize_flag: Whether to show visualizations
            
        Returns:
            Dictionary mapping performance codes to feature values
        """
        ...

    # === PUBLIC API METHODS (Called externally) ===
    def initialize_for_performance_code(self, performance_code: str) -> None:
        """
        Initialize feature storage for a new performance code.
        
        Args:
            performance_code: Code identifying the performance metric
        """
        self.performance_codes.append(performance_code)
        self.features[performance_code] = np.empty(0)

    def run(self, performance_code: str, exp_nr: int, visualize_flag: bool, **dims_dict) -> None:
        """
        Execute the feature extraction pipeline.

        Args:
            performance_code: Code identifying the performance metric
            exp_nr: Experiment number
            visualize_flag: Whether to show visualizations
            **dims_dict: Runtime parameters for dimensional indexing
        """
        # Set runtime parameters for current extraction
        self.set_runtime_parameters(**dims_dict)

        # Optional initialization step
        self._initialization_step(performance_code)

        # Check if dimensions already processed
        self._set_processed_state(**dims_dict)

        if not self.is_processed_state:
            # Fetch and load data
            self._fetch_data(exp_nr)
            current_data = self._load_data(exp_nr)
            
            # Compute features
            feature_dict = self._compute_features(current_data, visualize_flag)
            
            # Store results in feature arrays
            for perf_code, value in feature_dict.items():
                indices = tuple(dims_dict.values())

                assert perf_code in self.performance_codes, f"Performance code '{perf_code}' not initialized in feature model."
                assert self.features[perf_code] is not None, f"Feature storage for '{perf_code}' is not initialized."
                self.features[perf_code][indices] = value
                self.logger.debug(f"Extracted feature '{perf_code}': {round(value, self.round_digits) if value is not None else value}")

        else:
            self.logger.info("Data already processed for these dimensions, skipping")

        # Optional cleanup step
        self._cleanup_step(performance_code)

    def reset_for_new_experiment(self, performance_code: str, dim_sizes: List[int]) -> None:
        """
        Reset feature storage for a new experiment.
        
        Args:
            performance_code: Code identifying the performance metric
            dim_sizes: Dimensions of the feature array to create
        """
        self.logger.info(f"Resetting '{type(self).__name__}' feature model for new experiment")
        self._validate_parameters()
        self.features[performance_code] = np.empty(dim_sizes)
        
    # === OPTIONAL METHODS ===
    def _fetch_data(self, exp_nr: int) -> None:
        """
        Fetch data from external sources if needed.
        
        Args:
            exp_nr: Experiment number
        """
        pass

    def _validate_parameters(self) -> None:
        """Optional parameter validation logic to check values after initialization."""
        pass

    def _initialization_step(self, performance_code: str) -> None:
        """
        Optional initialization logic before feature extraction.
        
        Args:
            performance_code: Code identifying the performance metric
        """
        pass

    def _cleanup_step(self, performance_code: str) -> None:
        """
        Optional cleanup logic after feature extraction.
        
        Args:
            performance_code: Code identifying the performance metric
        """
        pass
    
    # === PRIVATE API METHODS (Called internally) ===
    def _set_processed_state(self, **dims_indices) -> None:
        """Check if current dimensions have been processed."""
        if dims_indices in self.processed_dims:
            self.is_processed_state = True
        else:
            self.processed_dims.append(dims_indices)
            self.is_processed_state = False


@dataclass
class EvaluationModel(ParameterHandling, ABC):
    """
    Abstract base class for evaluation models.
    
    Evaluates feature values against target values to compute performance metrics.
    Supports multi-dimensional evaluation with configurable aggregation strategies.
    """
    
    def __init__(
            self, 
            performance_code: str,
            folder_navigator: FolderNavigator,
            dimension_names: List[Tuple[str, str, str]],
            feature_model_type: Type[FeatureModel],
            logger: LBPLogger,
            round_digits: int = 3,
            **study_params
    ) -> None:
        """
        Initialize evaluation model.

        Args:
            performance_code: Code identifying the performance metric
            folder_navigator: File system navigation utility
            dimension_names: List of (dimension_name, iterator_name, parameter_name) tuples
            feature_model_type: Class of feature model to use
            logger: Logger instance for debugging and monitoring
            round_digits: Number of decimal places for rounding results
            **study_params: Study parameters for configuration
        """
        self.nav = folder_navigator
        self.logger = logger

        # Feature model configuration
        self.feature_model_type: Type[FeatureModel] = feature_model_type
        self.feature_model: Optional[FeatureModel] = None

        # Dimensional configuration
        self.dim_names: List[str] = []
        self.dim_iterator_names: List[str] = []
        self.dim_param_names: List[str] = []

        # Initialize the dimensional layers, in the form of [('dim_name', 'dim_iterator_name', 'dim_parameter_name'), (...), ...]
        # Note that the names must represent the namig convention in the code
        # e.g. [('layers', 'layer_id', 'limitLayers), [...], ...)
        # 'layers' is the name of the dimension, 'layer_id' is used to define the current id, limitLayers is the name of the variable that defines the number of layers.
        self._initialize_dimensions(dimension_names)

        # Performance storage and configuration
        self.round_digits: int = round_digits
        self.performance_code = performance_code
        self.performance_array: NDArray[np.float64] = np.empty((0,))
        self.performance_metrics: Dict[str, Optional[np.floating]] = {}

        # Apply dataclass-based parameter handling
        self.set_model_parameters(**study_params)

    # === ABSTRACT METHODS (Must be implemented by subclasses) ===
    @abstractmethod
    def _compute_target_value(self) -> float:
        """
        Compute the target value for performance evaluation.
        
        Returns:
            Target value for the current evaluation context
        """
        ...

    # === PUBLIC API METHODS (Called externally) ===
    def run(self, exp_nr: int, visualize_flag: bool, debug_flag: bool, **exp_params) -> None:
        """
        Execute the evaluation pipeline.
        
        Args:
            exp_nr: Experiment number
            visualize_flag: Whether to show visualizations
            debug_flag: Whether to run in debug mode (no saving)
            **exp_params: Experiment parameters
        """
        self.logger.info(f"Starting evaluation for experiment {exp_nr}")

        # Configure models with experiment parameters
        self.set_experiment_parameters(**exp_params)
        assert self.feature_model is not None, "Feature model must be initialized before running evaluation."
        self.feature_model.set_experiment_parameters(**exp_params)

        # Process all dimensional combinations
        total_dims = len(list(itertools.product(*self._compute_dim_ranges())))
        self.logger.info(f"Processing {total_dims} dimensional combinations")
        
        for i, dims in enumerate(itertools.product(*self._compute_dim_ranges())):

            # Create runtime parameters for current dimensions
            dims_dict = dict(zip(self.dim_iterator_names, dims))
            self.logger.debug(f"Processing dimension {i+1}/{total_dims}: {dims_dict}...")
            self.set_runtime_parameters(**dims_dict)

            # Extract features
            self.feature_model.run(self.performance_code, exp_nr, visualize_flag, **dims_dict)

            # Compute performance for current dimensions
            self._initialization_step()
            self._compute_performance(dims)
            self._cleanup_step()

        # Save results and aggregate performance
        if not debug_flag:
            self._save_results_locally(exp_nr)
        else:
            self.logger.info("Debug mode: Skipping result saving")
            
        self._aggregate_performance(self.performance_array[..., 2])
        self.logger.info(f"Evaluation completed: {self.performance_metrics}")

    def reset_for_new_experiment(self, dim_sizes: List[int]) -> None:
        """
        Reset evaluation model for a new experiment.
        
        Args:
            dim_sizes: Dimensions of the performance array to create
        """
        self.logger.info(f"Resetting evaluation model '{type(self).__name__}' for new experiment")
        self._validate_parameters()

        # Initialize performance array: [target_value, diff, performance_value]
        self.performance_array = np.empty(dim_sizes + [3,])

        # Reset performance metrics
        for key in self.performance_metrics.keys():
            self.performance_metrics[key] = None
            
    # === OPTIONAL METHODS ===
    def _compute_scaling_factor(self) -> Optional[float]:
        """
        Optionally compute scaling factor for performance normalization.
        
        Returns:
            Scaling factor or None for default scaling
        """
        return None

    def _validate_parameters(self) -> None:
        """Validate parameter values after initialization."""
        pass

    def _aggregate_performance(self, performance_value_array: NDArray[np.float64]) -> None:
        """
        Aggregate performance values across dimensions.
        
        Default implementation computes mean performance value.
        Subclasses can override for custom aggregation strategies.
        
        Args:
            performance_value_array: Array of performance values to aggregate
        """
        # Default: compute mean performance value
        self.performance_metrics['Value'] = np.round(np.average(performance_value_array), self.round_digits)

        # Initialize other metrics as None (can be overridden)
        self.performance_metrics['Performance'] = None
        self.performance_metrics['Robustness'] = None
        self.performance_metrics['Resilience'] = None

    def _initialization_step(self) -> None:
        """Optional initialization before performance computation."""
        pass

    def _cleanup_step(self) -> None:
        """Optional cleanup after performance computation."""
        pass

    # === PRIVATE API METHODS (Called internally) ===
    def _initialize_dimensions(self, dim_list: List[Tuple[str, str, str]]) -> None:
        """Parse dimension configuration into separate lists."""
        self.dim_names = [dim[0] for dim in dim_list]
        self.dim_iterator_names = [dim[1] for dim in dim_list]
        self.dim_param_names = [dim[2] for dim in dim_list]

    def _compute_performance(self, dims: Tuple) -> None:
        """
        Compute performance for current dimensions.
        
        Args:
            dims: Tuple of current dimension indices
        """
        # Extract feature value
        assert self.feature_model is not None, "Feature model must be initialized before computing performance."
        if not dims:
            feature_value = self.feature_model.features[self.performance_code].item()
        else:
            feature_value = self.feature_model.features[self.performance_code][dims]

        # Compute evaluation components
        target_value = self._compute_target_value()
        scaling_factor = self._compute_scaling_factor()

        # Evaluate performance based on feature and target values
        if (feature_value is None or np.isnan(feature_value)) or target_value is None:
            performance_value = None
            diff = None
            self.logger.warning(f"Feature or target value is None for dims {dims}, skipping performance computation.")
        elif feature_value == 0 or feature_value is None:
            diff = -target_value
            performance_value = 0.0
        else:
            # Compute difference from target
            diff = feature_value - target_value

            # Apply scaling for performance normalization
            if scaling_factor is not None and scaling_factor > 0:
                performance_value = 1.0 - np.abs(diff) / scaling_factor
            elif target_value > 0:
                performance_value = 1.0 - np.abs(diff) / target_value
            else:
                performance_value = np.abs(diff)
                self.logger.warning(f"Performance value has not been scaled.")
        
        # Store results in performance array
        self.performance_array[dims][0] = target_value
        self.performance_array[dims][1] = diff
        self.performance_array[dims][2] = performance_value

        self.logger.debug(
            f"Performance computed for dims {dims}: "
            f"feature={np.round(feature_value, self.round_digits) if feature_value is not None else None}, "
            f"target={np.round(target_value, self.round_digits) if target_value is not None else None}, "
            f"performance={np.round(performance_value, self.round_digits) if performance_value is not None else None}"
        )

    def _save_results_locally(self, exp_nr: int) -> None:
        """Save evaluation results to CSV file."""
        folder_path = os.path.join(self.nav.get_experiment_folder(exp_nr), 'results')
        exp_code = self.nav.get_experiment_code(exp_nr) 

        # Create results directory if needed
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        results = []

        # Export results for all dimensional combinations
        for indices in itertools.product(*self._compute_dim_ranges()):
            interval_dict = {}
            interval_dict['exp_code'] = exp_code

            # Add dimensional indices
            indices_dict = dict(zip(self.dim_iterator_names, indices))
            interval_dict.update(indices_dict)

            # Add evaluation results
            assert self.feature_model is not None, "Feature model must be initialized before adding evaluation results."
            feature_value = self.feature_model.features[self.performance_code][indices]
            target_value = self.performance_array[indices][0]
            diff_value = self.performance_array[indices][1]
            performance_value = self.performance_array[indices][2]

            interval_dict['feature_value'] = round(feature_value, self.round_digits)
            interval_dict['target_value'] = round(target_value, self.round_digits)
            interval_dict['diff_value'] = round(diff_value, self.round_digits)
            interval_dict['performance_value'] = round(performance_value, self.round_digits)
            results.append(interval_dict)

        # Save to CSV
        results_csv = os.path.join(folder_path, f"{exp_code}_{self.performance_code}.csv")
        df = pd.DataFrame(results)
        df.to_csv(results_csv, index=False)
        self.logger.info(f"Results saved locally to: {results_csv}")

    def _compute_dim_sizes(self) -> List[int]:
        """Get sizes of all dimensions from parameter values."""
        return [getattr(self, attr_name) for attr_name in self.dim_param_names]

    def _compute_dim_ranges(self) -> List[range]:
        """Create range objects for all dimensions."""
        dim_sizes = self._compute_dim_sizes()
        return [range(size) for size in dim_sizes]


class EvaluationSystem:
    """
    Orchestrates multiple evaluation models for a complete performance assessment.
    
    Manages the execution of evaluation models, handles database interactions,
    and coordinates the overall evaluation workflow.
    """
    
    def __init__(
        self,
        folder_navigator: FolderNavigator,
        data_interface: DataInterface,
        logger: LBPLogger
    ) -> None:
        """
        Initialize evaluation system.
        
        Args:
            folder_navigator: File system navigation utility
            data_interface: Database interface for data access
            logger: Logger instance for debugging and monitoring
        """
        self.nav = folder_navigator
        self.interface = data_interface
        self.logger = logger
        self.evaluation_models = {}

    # === PUBLIC API METHODS (Called externally) ===
    def add_evaluation_model(self, evaluation_class: Type[EvaluationModel], performance_code: str, study_params: Dict[str, Any]) -> None:
        """
        Add an evaluation model to the system.
        
        Args:
            evaluation_class: Class of evaluation model to instantiate
            performance_code: Code identifying the performance metric
            study_params: Study parameters for model configuration
        """
        self.logger.info(f"Adding '{evaluation_class.__name__}' model to evaluate performance '{performance_code}'")
        eval_model = evaluation_class(
            performance_code,
            folder_navigator=self.nav,
            logger=self.logger,
            **study_params
        )
        self.evaluation_models[performance_code] = eval_model

    def add_feature_model_instances(self, study_params: Dict[str, Any]) -> None:
        """
        Create feature model instances for evaluation models.
        
        Optimizes by sharing feature model instances where possible.
        
        Args:
            study_params: Study parameters for feature model configuration
        """
        feature_model_dict = {}

        for eval_model in self.evaluation_models.values():
            feature_model_type = eval_model.feature_model_type
            
            # Share feature model instances of the same type
            if feature_model_type not in feature_model_dict:
                eval_model.feature_model = feature_model_type(
                    performance_code=eval_model.performance_code, 
                    folder_navigator=eval_model.nav, 
                    logger=self.logger, 
                    round_digits=eval_model.round_digits,
                    **study_params
                )
                feature_model_dict[feature_model_type] = eval_model.feature_model
                self.logger.info(f"Adding feature model instance '{type(eval_model.feature_model).__name__}' to evaluation model '{type(eval_model).__name__}'")
            else:
                # Reuse existing feature model instance
                eval_model.feature_model = feature_model_dict[feature_model_type]
                eval_model.feature_model.initialize_for_performance_code(eval_model.performance_code)
                self.logger.info(f"Reusing existing feature model instance '{type(eval_model.feature_model).__name__}' for evaluation model '{type(eval_model).__name__}'")
            
    def run(self, exp_nr: int, exp_record: Dict[str, Any], visualize_flag: bool = False, debug_flag: bool = True, **exp_params) -> None:
        """
        Execute evaluation for all models.
        
        Args:
            exp_nr: Experiment number
            exp_record: Experiment record from database
            visualize_flag: Whether to show visualizations
            debug_flag: Whether to run in debug mode (no writing/saving)
            **exp_params: Experiment parameters
        """
        self.logger.info(f"Running evaluation system for experiment {exp_nr}")
        
        # Initialize all models before execution
        self._model_exp_initialization(**exp_params)

        # Execute each evaluation model
        for performance_code, eval_model in self.evaluation_models.items():
            self.logger.console_info(f"Running evaluation for '{performance_code}' performance with '{type(eval_model).__name__}' evaluation model...")
            eval_model.run(exp_nr, visualize_flag, debug_flag, **exp_params)
            self.logger.info(f"Finished evaluation for '{performance_code}' with '{type(eval_model).__name__}' model.")

            # Push results to database if not in debug mode
            if not debug_flag:
                self.logger.info(f"Pushing results to database for '{performance_code}'...")
                self.interface.push_to_database(exp_record, performance_code, eval_model.performance_metrics)
            else:
                self.logger.info(f"Debug mode: Skipping database push for '{performance_code}'")

        self.logger.console_info("All evaluations completed successfully.")

        # Update system-wide performance metrics
        if not debug_flag:
            self.logger.info("Updating system performance...")
            self.interface.update_system_performance(exp_record)
        else:
            self.logger.info("Debug mode: Skipping system performance update")

    # === PRIVATE/INTERNAL METHODS (Internal use only) ===
    def _model_exp_initialization(self, **exp_params) -> None:
        """
        Initialize all evaluation models for the current experiment.
        
        Args:
            **exp_params: Experiment parameters
        """
        for eval_model in self.evaluation_models.values():
            self.logger.info(f"Initializing arrays of evaluation model '{eval_model.performance_code}' and its feature model '{type(eval_model.feature_model).__name__}'")

            # Configure models with experiment parameters
            eval_model.set_experiment_parameters(**exp_params)
            eval_model.feature_model.set_experiment_parameters(**exp_params)

            # Initialize arrays with correct dimensions
            dim_sizes = eval_model._compute_dim_sizes()
            eval_model.reset_for_new_experiment(dim_sizes)
            eval_model.feature_model.reset_for_new_experiment(eval_model.performance_code, dim_sizes)
