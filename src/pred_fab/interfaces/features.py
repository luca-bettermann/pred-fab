import numpy as np
from typing import Any, Dict, List, Optional, final
from numpy.typing import NDArray
from dataclasses import dataclass

from abc import ABC, abstractmethod
from .base_interface import BaseInterface
from ..core import Dataset, Parameters
from ..utils import PfabLogger


class IFeatureModel(BaseInterface):
    """Abstract base for feature extraction models that iterate over dimension combinations."""

    def __init__(self, logger: PfabLogger):
        super().__init__(logger)
    
    # === ABSTRACT METHODS ===

    # abstract methods from BaseInterface:
    # - input_parameters
    # - input_features
    # - outputs

    @abstractmethod
    def _load_data(self, params: Dict, **dimensions) -> Any:
        """Load domain-specific raw data for the given parameter context (files, DB, etc.)."""
        ...

    @abstractmethod
    def _compute_feature_logic(
        self,
        data: Any,
        params: Dict,
        visualize: bool = False,
        **dimensions
        ) -> Dict[str, float]:
        """Extract feature values from loaded data; returns dict mapping feature codes to numeric values."""
        ...
    
    # Pre-define input features as empty. Features can not have other features as inputs.
    @property
    def input_features(self) -> List[str]:
        return []
        
    # === PUBLIC API ===

    @final
    def compute_features(
        self,
        parameters: Parameters,
        evaluate_from: int,
        evaluate_to: Optional[int] = None,
        visualize: bool = False
        ) -> NDArray:
        """Iterate over every dimension combination in [evaluate_from, evaluate_to) and call _load_data + _compute_feature_logic.

        Returns a 2-D array of shape (n_combinations, n_dims + n_outputs) where the first
        n_dims columns are the dimension iterator values and the remaining columns are feature values.
        """

        self.logger.info(f"Starting evaluation for '{self.outputs}'")
        
        # Prepare dimension combinations
        dim_objs = parameters.get_dim_objects(self.input_parameters)
        num_dims = len(dim_objs)
        dim_iterator_codes = [dim.iterator_code for dim in dim_objs]
        dim_combinations = parameters.get_dim_combinations(
            dim_codes=[dim.code for dim in dim_objs], 
            evaluate_from=evaluate_from, 
            evaluate_to=evaluate_to
            )
        
        # Initialize 2D-array
        shape = (len(dim_combinations), num_dims + len(self.outputs))
        feature_array = np.empty(shape)

        # Unpack DataBlocks
        params = parameters.get_values_dict()

        # Process each combination
        i_end = evaluate_to if evaluate_to is not None else len(dim_combinations)
        for i, current_dim in enumerate(dim_combinations):
            i_global = evaluate_from + i

            # merge dims and params into single dict
            current_dim_dict = dict(zip(dim_iterator_codes, current_dim))
            self.logger.debug(f"Processing {i_global}/{i_end}: {current_dim_dict}")
            
            # Compute feature, target, and performance
            feature_values = self._compute_feature_values(
                params,
                current_dim_dict,
                visualize=visualize,
                )

            # Store in array
            feature_array[i, :num_dims] = current_dim
            feature_array[i, num_dims:] = feature_values
    
        # return 2d array
        return feature_array

    @final
    def _compute_feature_values(
        self, 
        params,
        dimensions,
        visualize: bool = False, 
        ) -> List[float]:
        """Extract feature with memoization via Dataset."""
        
        # Load and compute
        self.logger.debug(f"Computing features '{self.outputs}' for {params}")
        data = self._load_data(params, **dimensions)
        feature_dict = self._compute_feature_logic(data, params, visualize=visualize, **dimensions)
        
        # Validate output from user implementation
        for feature_code, feature_value in feature_dict.items():
            if not isinstance(feature_value, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"_compute_features() must return numeric values, got {type(feature_value).__name__} for feature '{feature_code}'"
                )
        
        # Get the correct order of feature values
        feature_values = [feature_dict[code] for code in self.outputs] # type: ignore
        return feature_values
