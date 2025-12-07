from typing import Any, Dict, Tuple, Type, Optional, List
import numpy as np


from ..utils import LBPLogger
from ..core import Dataset, ExperimentData, Parameters
from .base import BaseOrchestrationSystem


class FeatureSystem(BaseOrchestrationSystem):
    """
    Orchestrates multiple evaluation models using Dataset.
    
    Manages evaluation model execution and stores results in ExperimentData.
    """
    
    def __init__(self, dataset: Dataset, logger: LBPLogger):
        """Initialize evaluation system."""
        super().__init__(dataset, logger)
    
    # === FEATURE EXTRACTION ===

    def compute_exp_features(
        self,
        exp_data: ExperimentData,
        evaluate_from: int = 0,
        evaluate_to: Optional[int] = None,
        visualize: bool = False,
        recompute: bool = False
    ) -> Dict[str, np.ndarray]:
        """Execute all evaluations for an experiment and mutate exp_data with results."""
        # Handle recompute logic
        if recompute:
            self.logger.info(f"Recompute flag set - clearing cache")

        # Check if the features and performance are already computed
        skip_for_code = {code: exp_data.is_complete(code, evaluate_from, evaluate_to) 
                         for code in exp_data.features.keys() if not recompute}

        # Get evaluation results from core logic
        feature_dict = self._compute_features_from_params(
            parameters=exp_data.parameters,
            evaluate_from=evaluate_from,
            evaluate_to=evaluate_to,
            visualize=visualize,
            skip_feature_code=skip_for_code
        )

        # Update exp_data with results
        exp_data.features.set_values(feature_dict)

        return feature_dict

    def _compute_features_from_params(
        self,
        parameters: Parameters,
        evaluate_from: int = 0,
        evaluate_to: Optional[int] = None,
        visualize: bool = False,
        skip_feature_code: Dict[str, bool] = {}
    ) -> Dict[str, np.ndarray]:
        """Core evaluation logic from raw parameters."""
        
        # Prepare result dictionaries
        feature_dict: Dict[str, np.ndarray] = {}

        # Run feature extraction for each feature code
        for feature_model in self.models:
            # Skip if already loaded
            if skip_feature_code.get(feature_model.feature_codes, False):
                self.logger.info(f"Skipping feature extraction for '{feature_model.feature_input_code}' as feature already complete")
                continue

            # Run feature extraction
            feature_array = feature_model.compute_features(
                parameters=parameters,
                evaluate_from=evaluate_from,
                evaluate_to=evaluate_to,
                visualize=visualize
            )

            # Collect results
            feature_dict[feature_model.feature_input_code] = feature_array

        return feature_dict
