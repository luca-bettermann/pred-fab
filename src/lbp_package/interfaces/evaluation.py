import os
import numpy as np
import pandas as pd
import itertools
from typing import Any, Dict, List, Type, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from numpy.typing import NDArray

from .features import FeatureModel
from ..utils import ParameterHandling, LBPLogger


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
            logger: LBPLogger,
            round_digits: int = 3,
            **kwargs) -> None:
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
        self._set_dim_lists()

        # Performance storage and configuration
        self.round_digits: int = round_digits
        self.performance_code = performance_code
        self.performance_array: NDArray[np.float64] = np.empty((0,))

        # Store kwargs so that they can be passed on to the feature models
        self.kwargs = kwargs

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
    def add_feature_model(self, feature_model: FeatureModel, **kwargs) -> None:
        """
        Predefined logic of how feature models are added to evaluation models.
        
        Args:
            feature_model: FeatureModel instance to use for feature extraction
        """
        # Directly set the feature model instance (one-to-one relationship)
        self.feature_model = feature_model

    def run(self, exp_code: str, exp_folder: str, visualize_flag: bool, debug_flag: bool, **exp_params) -> Dict[str, Optional[np.floating]]:
        """
        Execute the evaluation pipeline.
        
        Args:
            exp_code: Experiment code
            visualize_flag: Whether to show visualizations
            debug_flag: Whether to run in debug mode (no saving)
            **exp_params: Experiment parameters

        Returns:
            Dictionary of aggregated performance metrics.
        """
        self.logger.info(f"Starting evaluation for experiment {exp_code}")

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
            self.feature_model.run(self.performance_code, exp_code, exp_folder, visualize_flag, **dims_dict)

            # Compute performance for current dimensions
            self._initialization_step()
            self._compute_performance(dims)
            self._cleanup_step()

        # Save results and aggregate performance
        if not debug_flag:
            self._save_results_locally(exp_code, exp_folder)
        else:
            self.logger.info("Debug mode: Skipping result saving")

        # Unpack performance_array
        target_array, diff_array, performance_array = np.split(self.performance_array, 3, axis=-1)

        # Aggregate targets, diffs, signs and performance
        aggr_metrics = self._compute_default_aggr_metrics(target_array, diff_array, performance_array)

        # Compute interpretable aggregated performance metrics
        custom_aggr_metrics = self._compute_custom_aggr_metrics(target_array, diff_array, performance_array)
        assert isinstance(custom_aggr_metrics, dict), "Aggregation must return a dictionary"
        assert all(key in custom_aggr_metrics for key in self._declare_custom_aggr_metrics()), "Aggregation must include all required metrics"

        # Update custom aggregation metrics
        aggr_metrics.update(custom_aggr_metrics)

        # Log and return aggr_metrics
        self.logger.info(f"Evaluation completed: {aggr_metrics}")
        return aggr_metrics


    def reset_for_new_experiment(self, dim_sizes: List[int]) -> None:
        """
        Reset evaluation model for a new experiment.
        
        Args:
            dim_sizes: Dimensions of the performance array to create
        """
        self.logger.info(f"Resetting evaluation model '{type(self).__name__}' for new experiment")

        # Initialize performance array: [target_value, diff, performance_value]
        self.performance_array = np.empty(dim_sizes + [3,])

    # === OPTIONAL METHODS ===
    def _declare_scaling_factor(self) -> Optional[float]:
        """
        Optionally compute scaling factor for performance normalization.
        
        Returns:
            Scaling factor or None for default scaling
        """
        return None
    
    def _declare_custom_aggr_metrics(self):
        """
        Declare custom performance metrics for the evaluation. 
        
        By default, the custom metrics are set as:
        - "Performance": Maximum performance across dimensions
        - "Robustness": Measure of performance stability across dimensions
        - "Resilience": Measure of performance recovery across dimensions

        Subclasses can override this function to add their own metrics.

        Returns:
            List of custom performance metric names
        """
        return ["Performance", "Robustness", "Resilience"]

    def _compute_custom_aggr_metrics(
        self, 
        target_array: NDArray[np.float64], 
        diff_array: NDArray[np.float64], 
        performance_array: NDArray[np.float64]) -> Dict[str, Optional[np.floating]]:
        """
        This function computes the custom metrics declared in _declare_custom_aggr_metrics.

        Args:
            target_array: Array of target values to aggregate
            diff_array: Array of difference values to aggregate
            performance_array: Array of performance values to aggregate

        Returns:
            Dict: custom_metrics

        custom_metrics needs to contain the declared metrics as keys in its dictionary.
        By default, this function returns None as values.

        Subclasses can override for custom aggregation strategies.
        """
        custom_metrics: Dict[str, Optional[np.floating]] = {}

        # Initialize metrics as None
        for metric in self._declare_custom_aggr_metrics():
            custom_metrics[metric] = None

        return custom_metrics

    def _initialization_step(self) -> None:
        """Optional initialization before performance computation."""
        pass

    def _cleanup_step(self) -> None:
        """Optional cleanup after performance computation."""
        pass

    # === PRIVATE API METHODS (Called internally) ===
    def _set_dim_lists(self) -> None:
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
        scaling_factor = self._declare_scaling_factor()

        # Evaluate performance based on feature and target values
        if (feature_value is None or np.isnan(feature_value)) or target_value is None:
            performance_value = None
            diff = None
            self.logger.warning(f"Feature or target value is None for dims {dims}, skipping performance computation.")
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

    def _compute_default_aggr_metrics(
            self, 
            target_array: NDArray[np.float64], 
            diff_array: NDArray[np.float64], 
            performance_array: NDArray[np.float64]) -> Dict[str, Optional[np.floating]]:
        """
        Computes the default aggregation metrics.
        
        Args:
            target_array: Array of target values to aggregate
            diff_array: Array of difference values to aggregate
            performance_array: Array of performance values to aggregate

        Returns:
            Dictionary of default aggregated metrics.
        """
        performance_metrics: Dict[str, Optional[np.floating]] = {}

        # Default: compute mean target, diff and performance values
        performance_metrics['Target_Avg'] = np.round(np.average(target_array), self.round_digits)
        performance_metrics['Target_Std'] = np.round(np.std(target_array), self.round_digits)
        performance_metrics['Diff_Abs_Avg'] = np.round(np.average(np.abs(diff_array)), self.round_digits) # absolute diff
        performance_metrics['Diff_Sign_Avg'] = np.round(np.average(np.sign(diff_array)), self.round_digits) # sign of diff
        performance_metrics['Performance_Avg'] = np.round(np.average(performance_array), self.round_digits)

        # return the aggregated performance metrics
        return performance_metrics

    def _save_results_locally(self, exp_code: str, exp_folder: str) -> None:
        """Save evaluation results to CSV file."""
        folder_path = os.path.join(exp_folder, 'results')

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

