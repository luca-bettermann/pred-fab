import numpy as np
import itertools
from typing import Any, Dict, List, Type, Tuple, Optional, final
from dataclasses import dataclass
from abc import ABC, abstractmethod
from numpy.typing import NDArray

from .features import IFeatureModel
from ..core.dataset import ExperimentData
from ..core.data_blocks import MetricArrays, PerformanceAttributes
from ..utils import LBPLogger


@dataclass
class IEvaluationModel(ABC):
    """
    Abstract base class for evaluation models.
    
    Evaluates feature values against target values to compute performance metrics.
    Stores results directly in ExperimentData.
    Models declare their parameters as dataclass fields (DataObjects).
    """
    
    logger: LBPLogger  # Required field for logging
    feature_model: Optional[IFeatureModel] = None  # Set via add_feature_model()
    
    # === ABSTRACT METHODS ===
    
    @abstractmethod
    def _compute_target_value(self, **param_values) -> float:
        """
        Compute target value for performance evaluation at specific parameters.
        
        Args:
            **param_values: Parameter name-value pairs
            
        Returns:
            Target value for these parameters
        """
        ...
    
    @abstractmethod
    def _compute_scaling_factor(self, **param_values) -> Optional[float]:
        """
        Optionally compute scaling factor for performance normalization.
        
        Args:
            **param_values: Parameter name-value pairs
            
        Returns:
            Scaling factor or None for default scaling
        """
        ...
    
    # === OPTIONAL METHODS ===
    def _compute_aggregation(self, exp_data: ExperimentData) -> Dict[str, float]:
        """
        Optionally compute aggregated metrics from performance/metric arrays.
        
        Default implementation returns empty dict. Override to add custom aggregations.
        
        Args:
            exp_data: ExperimentData with populated performance and metric_arrays
            
        Returns:
            Dictionary of aggregated metric name-value pairs
        """
        return {}
    
    # === PUBLIC API ===
    @final
    def add_feature_model(self, feature_model: IFeatureModel) -> None:
        """Connect feature model for evaluation."""
        self.feature_model = feature_model
        self.logger.info(f"Added feature model: {type(feature_model).__name__}")
    
    @final
    def run(self, feature_name: str, performance_attr_name: str, exp_data: ExperimentData, 
            visualize: bool = False) -> None:
        """
        Evaluate feature against target values and store results in ExperimentData.
        
        Args:
            feature_name: Name of feature to evaluate
            performance_attr_name: Name for performance attribute
            exp_data: ExperimentData to populate with results
            visualize: Enable visualizations if True
        """
        if self.feature_model is None:
            raise ValueError("Feature model not set. Call add_feature_model() first.")
        
        if exp_data.dimensions is None:
            raise ValueError("ExperimentData must have dimensions defined")
        
        self.logger.info(f"Starting evaluation for feature '{feature_name}'")
        
        # Get dimension info from exp_data.parameters (DataDimension objects)
        from ..core.data_objects import DataDimension
        dim_objects = [obj for obj in exp_data.parameters.data_objects.values() 
                      if isinstance(obj, DataDimension)]
        dim_names = [obj.name for obj in dim_objects]
        dim_sizes = tuple(exp_data.parameters.get_value(obj.name) for obj in dim_objects)
        
        # Initialize arrays
        num_dims = len(dim_names)
        total_metrics = num_dims + 4  # dims + feature + target + scaling + performance
        metric_array = np.empty(dim_sizes + (total_metrics,))
        performance_array = np.empty(dim_sizes)
        
        # Generate all dimensional combinations
        dim_ranges = [range(size) for size in dim_sizes]
        dim_combinations = list(itertools.product(*dim_ranges))
        total_combos = len(dim_combinations)
        
        self.logger.info(f"Processing {total_combos} dimensional combinations")
        
        # Combine static parameters with dimensional parameters
        static_params = {}
        if exp_data.parameters:
            static_params = dict(exp_data.parameters.values)
        
        for i, dims in enumerate(dim_combinations):
            # Build complete parameter dict
            dim_params = dict(zip(dim_names, dims))
            all_params = {**static_params, **dim_params}
            
            self.logger.debug(f"Processing {i+1}/{total_combos}: {dim_params}")
            
            # Extract feature
            feature_value = self.feature_model.run(feature_name, visualize=visualize, **all_params)
            
            # Compute target and scaling for these parameters
            target_value = self._compute_target_value(**all_params)
            
            # Validate return type
            if not isinstance(target_value, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"_compute_target_value() must return numeric value, "
                    f"got {type(target_value).__name__}"
                )
            
            scaling_factor = self._compute_scaling_factor(**all_params)
            
            # Validate return type
            if scaling_factor is not None and not isinstance(scaling_factor, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"_compute_scaling_factor() must return numeric value or None, "
                    f"got {type(scaling_factor).__name__}"
                )
            
            # Compute performance
            performance_value = self._compute_performance(
                feature_value, target_value, scaling_factor
            )
            
            # Store in arrays
            metric_array[dims][:num_dims] = dims
            metric_array[dims][num_dims] = feature_value
            metric_array[dims][num_dims + 1] = target_value
            metric_array[dims][num_dims + 2] = scaling_factor if scaling_factor is not None else np.nan
            metric_array[dims][num_dims + 3] = performance_value if performance_value is not None else np.nan
            performance_array[dims] = performance_value if performance_value is not None else np.nan
        
        # Store results in ExperimentData
        from ..core.data_objects import DataArray, Performance, DataReal
        
        # Store aggregated feature value in features block
        if exp_data.features is None:
            from ..core.data_blocks import DataBlock
            exp_data.features = DataBlock()
        
        feature_obj = DataReal(feature_name)
        exp_data.features.add(feature_name, feature_obj)
        # Extract feature values from metric_array (at index num_dims)
        feature_values = metric_array[..., num_dims]
        exp_data.features.set_value(feature_name, float(np.nanmean(feature_values)))
        
        # Create performance attribute DataObject
        perf_attr = Performance.real(min_val=0.0, max_val=1.0)
        perf_attr.name = performance_attr_name
        
        # Add to PerformanceAttributes block
        if exp_data.performance is None:
            exp_data.performance = PerformanceAttributes()
        exp_data.performance.add(performance_attr_name, perf_attr)
        exp_data.performance.set_value(performance_attr_name, float(np.nanmean(performance_array)))
        
        # Create metric array names
        metric_names = dim_names + [feature_name, "target", "scaling", performance_attr_name]
        
        # Store metric array DataObject
        metric_data_array = DataArray(name=feature_name, shape=metric_array.shape, dtype=np.dtype(np.float64))
        # Store metric names in constraints for reference
        metric_data_array.constraints["metric_names"] = metric_names
        
        # Add to MetricArrays block
        if exp_data.metric_arrays is None:
            exp_data.metric_arrays = MetricArrays()
        exp_data.metric_arrays.add(feature_name, metric_data_array)
        exp_data.metric_arrays.set_value(feature_name, metric_array)
        
        # Compute aggregations if overridden
        aggr_metrics = self._compute_aggregation(exp_data)
        
        # Validate return type
        if not isinstance(aggr_metrics, dict):
            raise TypeError(
                f"_compute_aggregation() must return dict, "
                f"got {type(aggr_metrics).__name__}"
            )
        
        # Validate dict values are numeric
        for key, value in aggr_metrics.items():
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"_compute_aggregation() dict values must be numeric, "
                    f"got {type(value).__name__} for key '{key}'"
                )
        
        self.logger.info(
            f"Evaluation complete. Avg performance: {np.nanmean(performance_array):.3f}. "
            f"Aggregations: {aggr_metrics}"
        )
    
    # === PRIVATE METHODS ===
    @final
    def _compute_performance(
        self, 
        feature_value: float, 
        target_value: float, 
        scaling_factor: Optional[float]
    ) -> Optional[float]:
        """Compute performance value from feature, target, and scaling."""
        
        if feature_value is None or np.isnan(feature_value) or target_value is None:
            self.logger.warning("Feature or target is None/NaN, returning None")
            return None
        
        # Compute difference from target
        diff = feature_value - target_value
        
        # Apply scaling for normalization
        if scaling_factor is not None and scaling_factor > 0:
            performance_value = 1.0 - np.abs(diff) / scaling_factor
        elif target_value > 0:
            performance_value = 1.0 - np.abs(diff) / target_value
        else:
            performance_value = np.abs(diff)
            self.logger.warning("Performance not scaled (target_value <= 0)")
        
        # Clamp to [0, 1]
        if not 0 <= performance_value <= 1:
            self.logger.warning(f"Performance {performance_value:.3f} out of bounds, clamping")
            performance_value = np.clip(performance_value, 0, 1)
        
        self.logger.debug(
            f"Performance: feature={feature_value:.3f}, target={target_value:.3f}, "
            f"diff={diff:.3f}, scaling={scaling_factor}, perf={performance_value:.3f}"
        )
        
        return float(performance_value)
