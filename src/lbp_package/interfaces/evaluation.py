import numpy as np
import itertools
from typing import Any, Dict, List, Type, Tuple, Optional, final
from dataclasses import dataclass
from abc import ABC, abstractmethod
from numpy.typing import NDArray

from .features import IFeatureModel
from ..core.dataset import ExperimentData
from ..core.data_blocks import MetricArrays, PerformanceAttributes
from ..core.data_objects import DataDimension
from ..utils import LBPLogger


@dataclass
class IEvaluationModel(ABC):
    """
    Abstract base class for evaluation models.
    
    Evaluates feature values against target values to compute performance metrics.
    Stores results directly in ExperimentData.
    
    Models declare their parameters as dataclass fields (DataObjects).
    Optionally implement _get_model_artifacts() and _set_model_artifacts() for export support.
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
            evaluate_from: int = 0, evaluate_to: Optional[int] = None,
            visualize: bool = False) -> None:
        """Evaluate feature against target values and store results in ExperimentData."""
        # Validate preconditions
        if self.feature_model is None:
            raise ValueError("Feature model not set. Call add_feature_model() first")
        if exp_data.dimensions is None:
            raise ValueError("ExperimentData must have dimensions defined")
        
        self.logger.info(f"Starting evaluation for feature '{feature_name}'")
        
        # Extract dimension info
        dim_objects = [obj for obj in exp_data.parameters.data_objects.values() 
                      if isinstance(obj, DataDimension)]
        dim_names = [obj.name for obj in dim_objects]
        dim_sizes = tuple(exp_data.parameters.get_value(obj.name) for obj in dim_objects)
        
        # Initialize metric arrays
        num_dims = len(dim_names)
        total_metrics = num_dims + 4
        metric_array = np.empty(dim_sizes + (total_metrics,))
        performance_array = np.empty(dim_sizes)
        
        # Generate dimensional combinations
        dim_ranges = [range(size) for size in dim_sizes]
        all_dim_combinations = list(itertools.product(*dim_ranges))
        
        # Apply dimensional slice
        if evaluate_to is None:
            dim_combinations = all_dim_combinations[evaluate_from:]
        else:
            dim_combinations = all_dim_combinations[evaluate_from:evaluate_to]
        
        total_combos = len(dim_combinations)
        total_all = len(all_dim_combinations)
        self.logger.info(f"Processing {total_combos}/{total_all} dimensional combinations [{evaluate_from}:{evaluate_to}]")
        
        # Build static parameters
        static_params = dict(exp_data.parameters.values) if exp_data.parameters else {}
        
        # Process each combination
        for i, dims in enumerate(dim_combinations):
            dim_params = dict(zip(dim_names, dims))
            all_params = {**static_params, **dim_params}
            self.logger.debug(f"Processing {i+1}/{total_combos}: {dim_params}")
            
            # Compute feature, target, and performance
            feature_value = self.feature_model.run(feature_name, visualize=visualize, **all_params)
            target_value = self._compute_target_value(**all_params)
            scaling_factor = self._compute_scaling_factor(**all_params)
            
            # Validate outputs from user implementation
            if not isinstance(target_value, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"_compute_target_value() must return numeric. "
                    f"Expected int/float, got {type(target_value).__name__}"
                )
            if scaling_factor is not None and not isinstance(scaling_factor, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"_compute_scaling_factor() must return numeric or None. "
                    f"Expected int/float/None, got {type(scaling_factor).__name__}"
                )
            
            performance_value = self._compute_performance(feature_value, target_value, scaling_factor)
            
            # Store results in arrays
            metric_array[dims][:num_dims] = dims
            metric_array[dims][num_dims] = feature_value
            metric_array[dims][num_dims + 1] = target_value
            metric_array[dims][num_dims + 2] = scaling_factor if scaling_factor is not None else np.nan
            metric_array[dims][num_dims + 3] = performance_value if performance_value is not None else np.nan
            performance_array[dims] = performance_value if performance_value is not None else np.nan
        
        # Store results in ExperimentData
        self._store_results(exp_data, feature_name, performance_attr_name, 
                          metric_array, performance_array, dim_names, num_dims,
                          evaluate_from, evaluate_to)
        
        # Compute aggregations
        aggr_metrics = self._compute_aggregation(exp_data)
        if not isinstance(aggr_metrics, dict):
            raise TypeError(
                f"_compute_aggregation() must return dict. "
                f"Expected dict, got {type(aggr_metrics).__name__}"
            )
        
        # Validate aggregation values
        for key, value in aggr_metrics.items():
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"_compute_aggregation() values must be numeric. "
                    f"Expected int/float for '{key}', got {type(value).__name__}"
                )
        
        self.logger.info(
            f"Evaluation complete. Avg performance: {np.nanmean(performance_array):.3f}. "
            f"Aggregations: {aggr_metrics}"
        )
    
    # === PRIVATE METHODS ===
    @final
    def _compute_performance(
        self, feature_value: float, target_value: float, scaling_factor: Optional[float]
    ) -> Optional[float]:
        """Compute performance value from feature, target, and scaling."""
        # Handle missing values
        if feature_value is None or np.isnan(feature_value) or target_value is None:
            self.logger.warning("Feature or target is None/NaN, returning None")
            return None
        
        # Compute difference and normalize
        diff = feature_value - target_value
        if scaling_factor is not None and scaling_factor > 0:
            performance_value = 1.0 - np.abs(diff) / scaling_factor
        elif target_value > 0:
            performance_value = 1.0 - np.abs(diff) / target_value
        else:
            performance_value = np.abs(diff)
            self.logger.warning("Performance not scaled (target_value <= 0)")
        
        # Clamp to valid range
        if not 0 <= performance_value <= 1:
            self.logger.warning(f"Performance {performance_value:.3f} out of bounds, clamping")
            performance_value = np.clip(performance_value, 0, 1)
        
        self.logger.debug(
            f"Performance: feature={feature_value:.3f}, target={target_value:.3f}, "
            f"diff={diff:.3f}, scaling={scaling_factor}, perf={performance_value:.3f}"
        )
        return float(performance_value)
    
    @final
    def _store_results(
        self, exp_data: ExperimentData, feature_name: str, performance_attr_name: str,
        metric_array: np.ndarray, performance_array: np.ndarray, 
        dim_names: List[str], num_dims: int,
        evaluate_from: int = 0, evaluate_to: Optional[int] = None
    ) -> None:
        """Store evaluation results in ExperimentData blocks."""
        from ..core.data_objects import DataArray, Performance
        from ..core.data_blocks import MetricArrays, PerformanceAttributes
        
        # Phase 10: Store only feature values (dimensions implicit in indices)
        # Extract feature values from metric_array
        feature_values = metric_array[..., num_dims]
        
        # Apply dimensional slicing if requested
        if evaluate_from > 0 or (evaluate_to is not None and evaluate_to < feature_values.size):
            # Create slice for first dimension
            if len(feature_values.shape) > 0:
                sliced_values = feature_values.copy()
                # Flatten to 1D, slice, then reshape back
                flat = sliced_values.flatten()
                if evaluate_to is None:
                    evaluate_to = len(flat)
                sliced_flat = flat[evaluate_from:evaluate_to]
                # For now, store sliced values - full slicing support needs more work
                feature_values = sliced_flat
        
        # Store feature values in metric_arrays (always initialized)
        feature_data_array = DataArray(name=feature_name, shape=feature_values.shape, dtype=np.dtype(np.float64))
        exp_data.metric_arrays.add(feature_name, feature_data_array)
        exp_data.metric_arrays.set_value(feature_name, feature_values)
        
        # Store aggregated performance attribute (always initialized)
        perf_attr = Performance.real(min_val=0.0, max_val=1.0)
        perf_attr.name = performance_attr_name
        exp_data.performance.add(performance_attr_name, perf_attr)
        exp_data.performance.set_value(performance_attr_name, float(np.nanmean(performance_array)))
    
    # === EXPORT/IMPORT SUPPORT (OPTIONAL) ===
    
    def _get_model_artifacts(self) -> Dict[str, Any]:
        """
        Optionally serialize model state for export.
        
        Override only if evaluation model has state to preserve (lookup tables,
        configuration, precomputed data). Default returns empty dict.
        
        Most evaluation models don't need this - they just compute targets/scaling
        via pure functions using parameters.
        
        Returns:
            Dict containing model state (default: empty)
            
        Example:
            def _get_model_artifacts(self):
                return {
                    'target_lookup': self.target_lookup_table,
                    'scaling_config': {'max_deviation': 10.0}
                }
        """
        return {}
    
    def _set_model_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """
        Optionally restore model state from export.
        
        Override only if you override _get_model_artifacts(). Must perfectly
        reverse _get_model_artifacts(). Default does nothing.
        
        Args:
            artifacts: Dict containing model state (from _get_model_artifacts())
            
        Example:
            def _set_model_artifacts(self, artifacts):
                self.target_lookup_table = artifacts['target_lookup']
                self.scaling_config = artifacts['scaling_config']
        """
        pass
