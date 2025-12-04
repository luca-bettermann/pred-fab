import numpy as np
from typing import Any, Dict, List, Optional, final
from dataclasses import dataclass

from abc import ABC, abstractmethod
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
    @property
    @abstractmethod
    def feature_codes(self) -> List[str]:
        """List of the feature codes produced by this model."""
        ...

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
    def _compute_features(self, data: Any, visualize: bool = False) -> Dict[str, float]:
        """
        Extract feature value from loaded data.
        
        Args:
            data: Data object from _load_data
            visualize: Enable visualizations if True
            
        Returns:
            Computed feature values as a dict mapping feature codes to numeric values
        """
        ...
    
    # === PUBLIC API ===
    
    @final
    def run(self, feature_name: str, visualize: bool = False, **param_values) -> Dict[str, float]:
        """Extract feature with memoization via Dataset."""
        # TODO: check if **params is equal to dims or not. how are we handling non-dim params?

        # Check cache
        if self.dataset.has_cached_features_at(**param_values):
            try:
                cached_value = self.dataset.get_cached_feature_value(feature_name, **param_values)
                self.logger.debug(f"Using cached feature '{feature_name}' for {param_values}")
                return cached_value
            except KeyError:
                pass
        
        # Load and compute
        self.logger.debug(f"Computing feature '{feature_name}' for {param_values}")
        data = self._load_data(**param_values)
        feature_dict = self._compute_features(data, visualize=visualize)
        
        # Validate output from user implementation
        for feature_code, feature_value in feature_dict.items():
            if not isinstance(feature_value, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"_compute_features() must return numeric values, got {type(feature_value).__name__} for feature '{feature_code}'"
                )
        
            # Cache and return
            self.dataset.cache_feature_value(feature_code, feature_value, **param_values)
        
        return feature_dict