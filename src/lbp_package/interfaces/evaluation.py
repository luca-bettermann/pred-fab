import numpy as np
from typing import Tuple, Optional, List, final, Dict
from abc import ABC, abstractmethod
from numpy.typing import NDArray

from .base import BaseInterface
from ..core import Parameters, Dataset
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

    # this includes the input_parameters abstract property from BaseInterface

    @abstractmethod
    def input_feature(self) -> str:
        """
        Unique code identifying the feature that is required for this evaluation.

        Returns:
            Feature code string (e.g., 'feature_1')
        """
        ...

    @abstractmethod
    def output_performance(self) -> str:
        """
        Unique code identifying the performance metric evaluated by this model.
        
        Returns:
            Performance code string (e.g., 'dimensional_accuracy')
        """
        ...

    @abstractmethod
    def _compute_target_value(self, params: Dict, **dimensions) -> float:
        """
        Compute target value for performance evaluation at specific parameters.
        
        Args:
            params: Parameter name-value pairs
            **dimensions: Additional dimension parameters
            
        Returns:
            Target value for these parameters
        """
        ...
    
    def _compute_scaling_factor(self, params: Dict, **dimensions) -> Optional[float]:
        """
        Optionally compute scaling factor for performance normalization.
        
        Args:
            params: Parameter name-value pairs
            **dimensions: Additional dimension parameters
            
        Returns:
            Scaling factor or None for default scaling
        """
        return None
    
    # === PUBLIC API ===

    @final
    def compute_performance(
        self, 
        feature_array: NDArray, 
        parameters: Parameters
        ) -> Tuple[Optional[float], List[Optional[float]]]:
        """Compute average of the performance from the feature array."""
        
        # Unpack DataBlocks
        params = parameters.get_values_dict()
        dim_iterator_codes = [dim.iterator_code for dim in self.get_input_dimensions()]

        # Compute list of performance values
        performance_list = []
        for row in feature_array:
            # Extract current dimension values
            current_dim = row[:-1]
            feature_value = row[-1]
            
            # merge dims and params into single dict
            current_dim_dict = dict(zip(dim_iterator_codes, current_dim))

            # Compute target value, scaling factor
            target_value = self._compute_target_value(params, **current_dim_dict)
            scaling_factor = self._compute_scaling_factor(params, **current_dim_dict)
        
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
        
            # Compute performance value
            performance_value = self._compute_performance_value(feature_value, target_value, scaling_factor)
            performance_list.append(performance_value)

        performance_array = np.array(performance_list)
        avg_performance = float(np.nanmean(performance_array)) if len(performance_array) > 0 else None
        return avg_performance, performance_list

    @final
    def _compute_performance_value(
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

    # === WRAPPERS ===

    @final
    @property
    def input_features(self) -> List[str]:
        """Wrapper for input property."""
        input_feat = self.input_feature()
        if not isinstance(input_feat, str):
            raise TypeError(f"input_feature() must return str, got {type(input_feat).__name__}")
        return [input_feat]

    @final
    @property
    def outputs(self) -> List[str]:
        """Wrapper for output property."""
        perf_code = self.output_performance()
        if not isinstance(perf_code, str):
            raise TypeError(f"performance_code() must return str, got {type(perf_code).__name__}")
        return [perf_code]
