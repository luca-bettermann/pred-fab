import numpy as np
import itertools
from typing import Any, Dict, List, Type, Tuple, Optional, final
from dataclasses import dataclass
from abc import ABC, abstractmethod
from numpy.typing import NDArray

from .features import IFeatureModel
from ..utils import ParameterHandling, LBPLogger


@dataclass
class IEvaluationModel(ParameterHandling, ABC):
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
            weight: Optional[float] = None,
            **kwargs) -> None:
        """
        Initialize evaluation model.

        Args:
            performance_code: Code identifying the performance metric
            logger: Logger instance for debugging and monitoring
            round_digits: Number of decimal places for rounding results
            calibration_weight: Optional weight for calibration objective function.
                              If None, this model cannot be used in calibration.
            **kwargs: Additional parameters for configuration
        """
        self.logger = logger

        # By default, the evaluation model is deactivated from the system
        self.active: bool = False

        # Feature model validation
        if not isinstance(self.feature_model_type, type) or not issubclass(self.feature_model_type, IFeatureModel):
            raise ValueError("Feature model type must be a subclass of FeatureModel.")
        self.feature_model: Optional[IFeatureModel] = None

        # Dimensional configuration
        self._validate_dim_properties()

        # Performance configuration
        self.round_digits: int = round_digits
        self.performance_code = performance_code
        self.metric_names = ["feature_value", "target_value", "scaling_factor", "performance_value"]

        # Calibration configuration
        self.weight = weight
        if weight is not None:
            self.logger.info(f"EvaluationModel '{performance_code}' set with calibration weight: {weight}")

        # Store kwargs so that they can be passed on to the feature models
        self.kwargs = kwargs

    # === ABSTRACT PROPERTIES ===
    @property
    @abstractmethod
    def feature_model_type(self) -> Type[IFeatureModel]:
        """
        Property to access the feature model class.

        Returns:
            Class of the feature model used for feature extraction
        """
        ...

    @property
    @abstractmethod
    def dim_names(self) -> List[str]:
        """
        Property to access dimension names.

        Returns:
            List of dimension names (e.g. ['layers', 'segments'])
        """
        ...

    @property
    @abstractmethod
    def dim_param_names(self) -> List[str]:
        """
        Property to access dimension parameters. 
        Must align with naming convention in the study and experiment records.

        Returns:
            List of dimension parameter names (e.g. ['n_layers', 'n_segments'])
        """
        ...

    @property
    @abstractmethod
    def dim_iterator_names(self) -> List[str]:
        """
        Property to access dimension iterator names.

        Returns:
            List of dimension iterator names (e.g. ['layer_id', 'segment_id'])
        """
        ...

    @property
    @abstractmethod
    def target_value(self) -> float:
        """
        Compute the target value for performance evaluation.

        Returns:
            Target value for the current evaluation context
        """
        ...

    # === OPTIONAL PROPERTIES ===
    @property
    def scaling_factor(self) -> Optional[float]:
        """
        Optionally define scaling factor for performance normalization.
        
        Returns:
            Scaling factor or None for default scaling
        """
        return None
    
    @property
    def custom_aggr_metrics(self) -> List[str]:
        """
        Declare custom performance metrics for the evaluation. 
        Subclasses can override this function to add their own metrics.

        Returns:
            List of custom performance metric names
        """
        return []

    # === OPTIONAL METHODS ===
    def _compute_custom_aggr_metrics(
        self, 
        feature_array: NDArray[np.float64], 
        performance_array: NDArray[np.float64]) -> Dict[str, Optional[np.floating]]:
        """
        This function computes the custom metrics defined in custom_aggr_metrics.

        Args:
            feature_array: Array of feature values to aggregate
            performance_array: Array of performance values to aggregate

        Returns:
            Dict: custom_metrics

        custom_metrics needs to contain the declared metrics as keys in its dictionary.
        By default, this function returns None as values.

        Subclasses can override for custom aggregation strategies.
        """
        custom_metrics: Dict[str, Optional[np.floating]] = {}

        # Initialize metrics as None
        for metric in self.custom_aggr_metrics:
            custom_metrics[metric] = None

        return custom_metrics

    def _initialization_step(self, exp_code: str, exp_folder: str, visualize_flag: bool, debug_flag: bool) -> None:
        """Optional initialization before performance computation."""
        pass

    def _cleanup_step(self, exp_code: str, exp_folder: str, visualize_flag: bool, debug_flag: bool) -> None:
        """Optional cleanup after performance computation."""
        pass

    # === PUBLIC API METHODS ===
    @final
    def add_feature_model(self, feature_model: IFeatureModel, **kwargs) -> None:
        """Add feature model to evaluation model."""
        # Directly set the feature model instance (one-to-one relationship)
        self.feature_model = feature_model

    @final
    def run(self, exp_code: str, exp_folder: str, visualize_flag: bool, debug_flag: bool) -> Tuple[Dict[str, Optional[np.floating]], np.ndarray, np.ndarray]:
        """Execute the evaluation pipeline."""
        self.logger.info(f"Starting evaluation for experiment {exp_code}")

        # Initialize performance array
        num_dims = len(self.dim_names)
        metrics_array = np.empty(self._get_dim_sizes() + [num_dims + len(self.metric_names),])

        # Process all dimensional combinations
        dim_combinations = list(itertools.product(*self._compute_dim_ranges()))
        total_dims = len(dim_combinations)
        self.logger.info(f"Processing {total_dims} dimensional combinations")

        for i, dims in enumerate(dim_combinations):

            # Create runtime parameters for current dimensions
            dims_dict = dict(zip(self.dim_iterator_names, dims))
            self.logger.debug(f"Processing dimension {i+1}/{total_dims}: {dims_dict}...")
            self.set_dim_parameters(**dims_dict)

            # Extract features
            assert self.feature_model is not None, "Feature model not initialized."
            feature_value = self.feature_model.run(self.performance_code, exp_code, exp_folder, visualize_flag, debug_flag, **dims_dict)

            # Compute performance for current dimensions
            self._initialization_step(exp_code, exp_folder, visualize_flag, debug_flag)

            # Compute evaluation components
            target_value = self.target_value
            scaling_factor = self.scaling_factor

            # Compute performance value
            self.logger.debug(f"Computing performance for dimensions: {dims}...")
            performance_value = self._compute_performance(feature_value, target_value, scaling_factor)

            self._cleanup_step(exp_code, exp_folder, visualize_flag, debug_flag)

            # Store results in performance array
            metrics_array[dims][:num_dims] = dims
            metrics_array[dims][num_dims] = feature_value
            metrics_array[dims][num_dims + 1] = target_value
            metrics_array[dims][num_dims + 2] = scaling_factor
            metrics_array[dims][num_dims + 3] = performance_value

        # Unpack metrics_array
        feature_array = metrics_array[..., num_dims]
        performance_array = metrics_array[..., -1]

        # Aggregate targets, diffs, signs and performance
        aggr_metrics = self._compute_default_aggr_metrics("Feature_Avg", feature_array, "Performance_Avg", performance_array)

        # Prepare dim_arrays
        dim_array = np.array(dim_combinations)

        # Compute custom aggregated performance metrics
        custom_aggr_metrics = self._compute_custom_aggr_metrics(feature_array, performance_array)
        assert isinstance(custom_aggr_metrics, dict), "Aggregation must return a dictionary"
        assert all(key in custom_aggr_metrics for key in self.custom_aggr_metrics), "Aggregation must include all required metrics"

        # Update custom aggregation metrics
        aggr_metrics.update(custom_aggr_metrics)

        # Log and return aggr_metrics
        self.logger.info(f"Evaluation completed: {aggr_metrics}")
        return aggr_metrics, metrics_array, dim_array

    # === PRIVATE METHODS ===
    @final
    def _compute_performance(self, feature_value: np.ndarray, target_value: float, scaling_factor: Optional[float]) -> Optional[NDArray[np.floating]]:
        """Compute performance value for current dimensions."""

        # Evaluate performance based on feature and target values
        if (feature_value is None or np.isnan(feature_value)) or target_value is None:
            performance_value = None
            diff = None
            self.logger.warning(f"Feature or target value is None, skipping performance computation.")
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
            self.logger.warning(f"Performance value {performance_value} out of bounds, clamping to [0, 1].")
            performance_value = np.clip(performance_value, 0, 1)

        self.logger.debug(
            f"Performance computed: "
            f"feature={np.round(feature_value, self.round_digits) if feature_value is not None else None}, "
            f"target={np.round(target_value, self.round_digits) if target_value is not None else None}, "
            f"diff={np.round(diff, self.round_digits) if diff is not None else None}, "
            f"scaling={np.round(scaling_factor, self.round_digits) if scaling_factor is not None else None}, "
            f"performance={np.round(performance_value, self.round_digits) if performance_value is not None else None}"
        )
        return performance_value

    @final
    def _compute_default_aggr_metrics(
            self, 
            key_feature_avg: str,
            feature_array: NDArray[np.float64], 
            key_performance_avg: str,
            performance_array: NDArray[np.float64]) -> Dict[str, Optional[np.floating]]:
        """Computes the default aggregation metrics. """
        default_metrics: Dict[str, Optional[np.floating]] = {}

        # Default: compute mean feature and performance values
        default_metrics[key_feature_avg] = np.round(np.average(feature_array), self.round_digits)
        default_metrics[key_performance_avg] = np.round(np.average(performance_array), self.round_digits)

        # return the aggregated performance metrics
        return default_metrics

    @final
    def _get_dim_sizes(self) -> List[int]:
        """Get sizes of all dimensions from parameter values."""
        return [getattr(self, attr_name) for attr_name in self.dim_param_names]

    @final
    def _compute_dim_ranges(self) -> List[range]:
        """Create range objects for all dimensions."""
        dim_sizes = self._get_dim_sizes()
        return [range(size) for size in dim_sizes]

    @final
    def _validate_dim_properties(self) -> None:
        """Validate that dimension properties are correctly set."""

        # Validate dtype and content of dimension properties
        if not isinstance(self.dim_names, list) or not all(isinstance(name, str) for name in self.dim_names):
            raise ValueError("dim_names property must be a list of strings.")
        if not isinstance(self.dim_iterator_names, list) or not all(isinstance(name, str) for name in self.dim_iterator_names):
            raise ValueError("dim_iterator_names property must be a list of strings.")
        if not isinstance(self.dim_param_names, list) or not all(isinstance(name, str) for name in self.dim_param_names):
            raise ValueError("dim_param_names property must be a list of strings.")
        
        # Validate that all dimension lists have the same length
        if not (len(self.dim_names) == len(self.dim_iterator_names) == len(self.dim_param_names)):
            raise ValueError("dim_names, dim_iterator_names, and dim_param_names must have the same length.")
