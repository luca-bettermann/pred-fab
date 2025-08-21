import numpy as np
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from abc import ABC, abstractmethod
from numpy.typing import NDArray
from ..utils import ParameterHandling, LBPLogger


@dataclass
class FeatureModel(ParameterHandling, ABC):
    """
    Abstract base class for feature extraction models.
    
    Provides a standardized interface for extracting features from experimental data.
    Uses dataclass-based parameter handling for clean configuration management.
    """
    
    def __init__(self, 
                 performance_code: str,
                 logger: LBPLogger,
                 study_params: Dict[str, Any],
                 round_digits: int,
                 **kwargs) -> None:
        """
        Initialize feature extraction model.

        Args:
            performance_code: Code identifying the performance metric
            folder_navigator: File system navigation utility
            logger: Logger instance for debugging and monitoring
            **study_params: Study parameters for configuration
        """
        self.logger = logger
        self.round_digits = round_digits

        # Feature storage - supports multiple performance codes per model
        self.features: Dict[str, NDArray[np.float64]] = {}
        self.associated_codes: List[str] = []
        self.initialize_for_code(performance_code)

        # Track processed dimensions to avoid duplicate computation
        self.processed_dims: List = []
        self.is_processed_state: bool = False

        # Temporary storage for current feature computation
        self.current_feature: Dict[str, float] = {}

        # Apply dataclass-based parameter handling
        self.set_model_parameters(**study_params)

    # === ABSTRACT METHODS (Must be implemented by subclasses) ===
    @abstractmethod
    def _load_data(self, exp_code: str, exp_folder: str) -> Any:
        """
        Load domain-specific, unstructured data for feature extraction. Potentially requires
        a database connection to access raw data files or streams.

        Data Responsibility Boundary: This method handles complex, domain-specific data
        that the DataInterface doesn't manage (geometry files, sensor streams, images,
        environmental data, etc.). The DataInterface handles structured metadata only.

        Args:
            exp_code: Experiment code
            exp_folder: Experiment folder path

        Returns:
            Loaded data object (format depends on domain requirements)
        """
        ...

    @abstractmethod
    def _compute_features(self, data: Any, visualize_flag: bool) -> Dict[str, float]:
        """
        Extract features from loaded data.
        
        Args:
            data: Data object from _load_data
            visualize_flag: Whether to show visualizations
            
        Returns:
            Dictionary mapping performance codes to feature values
        """
        ...

    # === PUBLIC API METHODS (Called externally) ===
    def initialize_for_code(self, associated_code: str) -> None:
        """
        Initialize feature storage for a new performance code.
        
        Args:
            performance_code: Code identifying the performance metric
        """
        self.associated_codes.append(associated_code)
        self.features[associated_code] = np.empty([])

    def run(self, performance_code: str, exp_code: str, exp_folder: str, visualize_flag: bool, **dims_dict) -> None:
        """
        Execute the feature extraction pipeline.

        Args:
            performance_code: Code identifying the performance metric
            exp_nr: Experiment number
            visualize_flag: Whether to show visualizations
            **dims_dict: Runtime parameters for dimensional indexing
        """
        # Set runtime parameters for current extraction
        self.set_runtime_parameters(**dims_dict)

        # Optional initialization step
        self._initialization_step(performance_code, exp_code)

        # Check if dimensions already processed
        self._set_processed_state(**dims_dict)

        if not self.is_processed_state:

            # Load data for feature extraction
            current_data = self._load_data(exp_code, exp_folder)

            # Compute features
            feature_dict = self._compute_features(current_data, visualize_flag)
            
            # Store results in feature arrays
            for code, value in feature_dict.items():
                indices = tuple(dims_dict.values())

                assert code in self.associated_codes, f"Associated code '{code}' not initialized in feature model."
                assert isinstance(self.features[code], np.ndarray), f"Feature storage for '{code}' is not initialized as numpy array."
                self.features[code][indices] = value
                self.logger.debug(f"Extracted feature '{code}': {round(value, self.round_digits) if value is not None else value}")

        else:
            # Skip and reset
            self.logger.info("Data already processed for these dimensions, skipping")
            self.is_processed_state = False

        # Optional cleanup step
        self._cleanup_step(performance_code, exp_code)

    def reset_for_new_experiment(self, performance_code: str, dim_sizes: List[int]) -> None:
        """
        Reset feature storage for a new experiment.
        
        Args:
            performance_code: Code identifying the performance metric
            dim_sizes: Dimensions of the feature array to create
        """
        self.logger.info(f"Resetting '{type(self).__name__}' feature model for new experiment")
        self.features[performance_code] = np.empty(dim_sizes)
        self.processed_dims = []
        
    # === OPTIONAL METHODS ===
    def _initialization_step(self, performance_code: str, exp_code: str) -> None:
        """
        Optional initialization logic before feature extraction.
        
        Args:
            performance_code: Code identifying the associated metric
            exp_code: Experiment code
        """
        pass

    def _cleanup_step(self, performance_code: str, exp_code: str) -> None:
        """
        Optional cleanup logic after feature extraction.
        
        Args:
            performance_code: Code identifying the associated metric
            exp_code: Experiment code
        """
        pass
    
    # === PRIVATE API METHODS (Called internally) ===
    def _set_processed_state(self, **dims_indices) -> None:
        """Check if current dimensions have been processed."""
        if dims_indices in self.processed_dims:
            self.is_processed_state = True
        else:
            self.processed_dims.append(dims_indices)
            self.is_processed_state = False