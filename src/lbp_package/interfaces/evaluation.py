import numpy as np
from typing import Any, Dict, List, Type, Tuple, Optional, final
from abc import ABC, abstractmethod
from numpy.typing import NDArray

from .base import BaseInterface
from .features import IFeatureModel
from ..core import ExperimentData, DataArray, DataObject, DataDimension, Dimensions, Parameters, Dataset
from ..utils import LBPLogger


class IEvaluationModel(BaseInterface):
    """
    Abstract base class for evaluation models.
    
    Evaluates feature values against target values to compute performance metrics.
    Stores results directly in ExperimentData.
    """

    def __init__(self, dataset: Dataset, logger: LBPLogger):
        """Initialize evaluation system."""
        super().__init__(dataset, logger)
    
    # === ABSTRACT METHODS ===

    @property
    @abstractmethod
    def performance_code(self) -> str:
        """
        Unique code identifying the performance metric evaluated by this model.
        Used by orchestration system to map evaluation results to performance attributes.
        
        Returns:
            Performance code string (e.g., 'dimensional_accuracy')
        """
        ...

    @property
    @abstractmethod
    def feature_input_code(self) -> str:
        """
        Return the code of the feature model used by this evaluation model.
        Used by the orchestration system to instantiate the correct feature model.

        Important: input code needs to match the feature model's declared output code.
        
        Returns:
            Class type of the feature model (e.g. MyFeatureModel)
        """
        ...

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
    def set_feature_model(self, feature_model: IFeatureModel) -> None:
        """Connect feature model for evaluation."""
        self.feature_model = feature_model
        self.logger.info(f"Added feature model: {type(feature_model).__name__}")
    
    @final
    def run(
        self, 
        parameters: Parameters,
        # dimensions: Dimensions,
        evaluate_from: int = 0, 
        evaluate_to: Optional[int] = None,
        visualize: bool = False
    ) -> Tuple[Dict[str, float], Dict[str, NDArray]]:
        """Evaluate feature against target values and store results in ExperimentData."""
        # Validate preconditions
        if self.feature_model is None:
            raise ValueError("Feature model not set. Call add_feature_model() first")
        
        self.logger.info(f"Starting evaluation for '{self.performance_code}'")
        

        # TODO: input should be dim_combinations,

        # # Extract dimension info
        # dim_objects = [obj for obj in exp_data.parameters.data_objects.values() 
        #               if isinstance(obj, DataDimension)]
        # dim_names = [obj.name for obj in dim_objects]
        # dim_sizes = tuple(exp_data.parameters.get_value(obj.name) for obj in dim_objects)

        # Unnpack DataBlocks
        dims = dimensions.get_values_dict()
        dim_combinations = dimensions.get_dim_combinations()
        params = parameters.get_values_dict()

        # Initialize feature array
        n_rows = len(dim_combinations) if dim_combinations else 1
        num_dims = len(dims)
        feature_array = np.empty((n_rows, num_dims + 1))
        
        # Apply dimensional slice
        if evaluate_to is None:
            dim_process = dim_combinations[evaluate_from:]
        else:
            dim_process = dim_combinations[evaluate_from:evaluate_to]
        self.logger.info(f"Processing {len(dim_process)}/{len(dim_combinations)} dimensional combinations [{evaluate_from}:{evaluate_to}]")
        
        # Process each combination
        i_last = evaluate_to if evaluate_to is not None else len(dim_combinations)
        for i, current_dim in enumerate(dim_process):
            i_global = evaluate_from + i
            self.logger.debug(f"Processing {i_global}/{i_last}: {current_dim}")

            # merge dims and params into single dict
            current_dim_dict = dict(zip(dims.keys(), current_dim))
            combined = {**current_dim_dict, **params}
            
            # Compute feature, target, and performance
            feature_value = self.feature_model.run(self.feature_input_code, combined, visualize=visualize)
            target_value = self._compute_target_value()
            scaling_factor = self._compute_scaling_factor()
            
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
            
            # compute performance value
            performance_value = self._compute_performance(feature_value, target_value, scaling_factor)
            
            # TODO: Inisialized arrays in ExperimentData and pass in this method. OR pass indices back

            # Store results in arrays
            feature_array[current_dim][:num_dims] = current_dim
            feature_array[current_dim][num_dims] = feature_value
        

        # TODO: Compute aggregation on-the-fly for a slice of the array. 
        # TODO: Decide whether we should store performance at all?

        # # Compute aggregations
        # aggr_metrics = self._compute_aggregation(exp_data)
        # if not isinstance(aggr_metrics, dict):
        #     raise TypeError(
        #         f"_compute_aggregation() must return dict. "
        #         f"Expected dict, got {type(aggr_metrics).__name__}"
        #     )
        
        # # Validate aggregation values
        # for key, value in aggr_metrics.items():
        #     if not isinstance(value, (int, float, np.integer, np.floating)):
        #         raise TypeError(
        #             f"_compute_aggregation() values must be numeric. "
        #             f"Expected int/float for '{key}', got {type(value).__name__}"
        #         )
        
        # self.logger.info(
        #     f"Evaluation complete. Avg performance: {np.nanmean(performance_array):.3f}. "
        #     f"Aggregations: {aggr_metrics}"
        # )

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
        metric_array: np.ndarray, performance_array: np.ndarray, num_dims: int,
        evaluate_from: int = 0, evaluate_to: Optional[int] = None
    ) -> None:
        """Store evaluation results in ExperimentData blocks."""
        # TODO: Move store results to orchestration
        # TODO: Document that Interfaces interact with dicts, orchestration with ExperimentData and data blocks
        
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
        feature_data_array = DataArray(code=feature_name, shape=feature_values.shape, dtype=np.dtype(np.float64))
        exp_data.features.add(feature_name, feature_data_array)
        exp_data.features.set_value(feature_name, feature_values)
        
        # Store aggregated performance attribute (always initialized)
        perf_attr = Performance.real(min_val=0.0, max_val=1.0)
        perf_attr.name = performance_attr_name
        exp_data.performance.add(performance_attr_name, perf_attr)
        exp_data.performance.set_value(performance_attr_name, float(np.nanmean(performance_array)))
    
    # # === EXPORT/IMPORT SUPPORT (OPTIONAL) ===
    
    # def _get_model_artifacts(self) -> Dict[str, Any]:
    #     """
    #     Optionally serialize model state for export.
        
    #     Override only if evaluation model has state to preserve (lookup tables,
    #     configuration, precomputed data). Default returns empty dict.
        
    #     Most evaluation models don't need this - they just compute targets/scaling
    #     via pure functions using parameters.
        
    #     Returns:
    #         Dict containing model state (default: empty)
            
    #     Example:
    #         def _get_model_artifacts(self):
    #             return {
    #                 'target_lookup': self.target_lookup_table,
    #                 'scaling_config': {'max_deviation': 10.0}
    #             }
    #     """
    #     return {}
    
    # def _set_model_artifacts(self, artifacts: Dict[str, Any]) -> None:
    #     """
    #     Optionally restore model state from export.
        
    #     Override only if you override _get_model_artifacts(). Must perfectly
    #     reverse _get_model_artifacts(). Default does nothing.
        
    #     Args:
    #         artifacts: Dict containing model state (from _get_model_artifacts())
            
    #     Example:
    #         def _set_model_artifacts(self, artifacts):
    #             self.target_lookup_table = artifacts['target_lookup']
    #             self.scaling_config = artifacts['scaling_config']
    #     """
    #     pass
