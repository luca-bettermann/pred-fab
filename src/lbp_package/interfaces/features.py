import numpy as np
from typing import Any, Dict, List, Optional, final
from dataclasses import dataclass

from abc import ABC, abstractmethod
from numpy.typing import NDArray
from ..core.dataset import Dataset
from ..utils import LBPLogger


@dataclass
class IFeatureModel(ABC):
    """
    Abstract base class for feature extraction models.
    
    Uses Dataset memoization to avoid redundant feature computation.
    Models declare their parameters as dataclass fields (DataObjects).
    """
    
    dataset: Dataset  # Required field for memoization
    logger: LBPLogger  # Required field for logging
    
    # === ABSTRACT METHODS ===
    @abstractmethod
    def _load_data(self, **param_values) -> Any:
        """
        Load domain-specific data for feature extraction at specific parameter values.
        
        Uses parameter values to locate/load required data. May access external
        databases, files, or other data sources not managed by Dataset.
        
        Args:
            **param_values: Parameter name-value pairs defining the context
            
        Returns:
            Loaded data object (format depends on domain)
        """
        ...

    @abstractmethod
    def _compute_features(self, data: Any, visualize: bool = False) -> float:
        """
        Extract feature value from loaded data.
        
        Args:
            data: Data object from _load_data
            visualize: Enable visualizations if True
            
        Returns:
            Computed feature value
        """
        ...
    
    # === PUBLIC API ===
    
    @final
    def run(self, feature_name: str, visualize: bool = False, **param_values) -> float:
        """
        Extract feature with memoization via Dataset.
        
        Checks Dataset cache first. If not cached, loads data and computes feature.
        
        Args:
            feature_name: Name of feature to extract
            visualize: Enable visualizations if True
            **param_values: Parameter name-value pairs
            
        Returns:
            Feature value
        """
        # Check if already computed
        if self.dataset.has_features_at(**param_values):
            try:
                cached_value = self.dataset.get_feature_value(feature_name, **param_values)
                self.logger.debug(f"Using cached feature '{feature_name}' for {param_values}")
                return cached_value
            except KeyError:
                pass  # Feature name not in cache, compute it
        
        # Load data for these parameters
        self.logger.debug(f"Computing feature '{feature_name}' for {param_values}")
        data = self._load_data(**param_values)
        
        # Compute feature
        feature_value = self._compute_features(data, visualize=visualize)
        
        # Validate return type
        if not isinstance(feature_value, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"_compute_features() must return numeric value, "
                f"got {type(feature_value).__name__}"
            )
        
        # Cache result
        self.dataset.set_feature_value(feature_name, feature_value, **param_values)
        
        return feature_value