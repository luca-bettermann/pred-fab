import os
import numpy as np
import pandas as pd
import itertools
from typing import Any, Dict, List, Type, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from numpy.typing import NDArray

from .features import FeatureModel
from ..utils import ParameterHandling, FolderNavigator, LBPLogger


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

        # By default, the evaluation model is deactivated from the system
        self.active: bool = False

        # Feature model configuration
        feature_model_type = self._declare_feature_model_type()
        if not isinstance(feature_model_type, type) or not issubclass(feature_model_type, FeatureModel):
            raise ValueError("Feature model type must be a subclass of FeatureModel.")
        self.feature_model_type: Type[FeatureModel] = feature_model_type
        self.feature_model: Optional[FeatureModel] = None

        # Dimensional configuration
        self.dim_names: List[str] = []
        self.dim_iterator_names: List[str] = []
        self.dim_param_names: List[str] = []
        self._initialize_dimensions()

        # Performance storage and configuration
        self.round_digits: int = round_digits
        self.performance_code = performance_code
        self.performance_array: NDArray[np.float64] = np.empty((0,))
        self.performance_metrics: Dict[str, Optional[np.floating]] = {}

        # Apply dataclass-based parameter handling
        self.set_model_parameters(**study_params)

    # === ABSTRACT METHODS (Must be implemented by subclasses) ===
    @abstractmethod
    def _declare_dimensions(self) -> List[Tuple[str, str, str]]:
        """
        Declare the hierarchical dimension structure for evaluation.
        This defines the dimensions and their iterators for multi-dimensional evaluation.
        Note that the names must represent the naming convention used in the code and database.
        
        Returns:
            List of (dimension_name, iterator_name, parameter_name) tuples
            e.g., [('layers', 'layer_id', 'n_layers'), ('segments', 'segment_id', 'n_segments')]
        """
        ...

    @abstractmethod
    def _declare_feature_model_type(self) -> Type[FeatureModel]:
        """
        Declare the feature model type to use for feature extraction.
        
        Returns:
            Class of the feature model to use
        """
        ...

    @abstractmethod
    def _compute_target_value(self) -> float:
        """
        Compute the target value for performance evaluation.
        
        Returns:
            Target value for the current evaluation context
        """
        ...

    # === PUBLIC API METHODS (Called externally) ===
    def add_feature_model(self, feature_model: FeatureModel) -> None:
        """
        Predefined logic of how feature models are added to evaluation models.
        
        Args:
            feature_model: FeatureModel instance to use for feature extraction
        """
        # Directly set the feature model instance (one-to-one relationship)
        self.feature_model = feature_model

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
            self.feature_model.run(self.performance_code, exp_nr, visualize_flag, debug_flag, **dims_dict)

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
    def _initialize_dimensions(self) -> None:
        """Parse dimension configuration into separate lists."""
        # get the dimension names from the declare_dimensions method
        dim_list = self._declare_dimensions()

        # Validate the dimension list
        if dim_list is None:
            raise ValueError("No dimensions declared in the model. Please implement declare_dimensions method.")
        elif not isinstance(dim_list, list):
            raise ValueError("Declared dimensions must be a list of tuples (dimension_name, iterator_name, parameter_name).")
        elif not all(isinstance(dim, tuple) and len(dim) == 3 for dim in dim_list):
            raise ValueError("Each dimension must be a tuple of (dimension_name, iterator_name, parameter_name).")
        else:
            self.logger.debug(f"Declared dimensions: {dim_list}")

        # Store dimension names in separate lists for easy access
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

        # Ensure performance value is within [0, 1] range
        if performance_value and not 0 <= performance_value <= 1:
            self.logger.warning(f"Performance value {performance_value} out of bounds [0, 1] for dims {dims}. Clamping to [0, 1].")
            performance_value = np.clip(performance_value, 0, 1)
        
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

