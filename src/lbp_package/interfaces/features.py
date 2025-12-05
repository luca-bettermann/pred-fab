import numpy as np
from typing import Any, Dict, List, Optional, final
from dataclasses import dataclass

from abc import ABC, abstractmethod
from .base import BaseInterface
from ..core import Dataset, DataObject
from ..utils import LBPLogger


class IFeatureModel(BaseInterface):
    """
    Abstract base class for feature extraction models.
    
    Uses Dataset memoization to avoid redundant feature computation.
    Models declare their parameters as dataclass fields (DataObjects).
    """

    def __init__(self, dataset: Dataset, logger: LBPLogger):
        """Initialize evaluation system."""
        super().__init__(dataset, logger)
    
    # === ABSTRACT METHODS ===
    @property
    @abstractmethod
    def feature_codes(self) -> List[str]:
        """List of the feature codes produced by this model."""
        ...

    @property
    @abstractmethod
    def required_parameters(self) -> List[DataObject]:
        """
        Define the parameters and dimensions this model needs from the experiment.
        
        Returns:
            List of DataObjects defining the schema.
            Example: [Parameter.real("speed", ...), Dimension.int("layers", ...)]
        """
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
    def run(
        self, 
        feature_name: str,
        params: Dict[str, Any], 
        visualize: bool = False, 
        ) -> float:
        """Extract feature with memoization via Dataset."""
        # Check cache
        if self.dataset.has_cached_features_at(params):
            try:
                cached_value = self.dataset.get_cached_feature_value(feature_name, params)
                self.logger.debug(f"Using cached feature '{feature_name}' for {params}")
                return cached_value
            except KeyError:
                raise KeyError(
                    f"Feature '{feature_name}' not found in cache for parameters {params} despite has_cached_features_at() returning True"
                )
        
        # Load and compute
        self.logger.debug(f"Computing feature '{feature_name}' for {params}")
        data = self._load_data(**params)
        feature_dict = self._compute_features(data, visualize=visualize, **params)
        
        # Validate output from user implementation
        for feature_code, feature_value in feature_dict.items():
            if not isinstance(feature_value, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"_compute_features() must return numeric values, got {type(feature_value).__name__} for feature '{feature_code}'"
                )
        
            # Cache and return
            self.dataset.cache_feature_value(feature_code, feature_value, params)
        
        return feature_dict[feature_name]