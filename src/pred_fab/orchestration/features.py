from typing import Any, Dict, Tuple, Type, Optional, List
import numpy as np

from pred_fab.core.data_objects import DataArray
from pred_fab.interfaces.features import IFeatureModel


from ..utils import PfabLogger
from ..core import DatasetSchema, ExperimentData, Parameters
from .base_system import BaseOrchestrationSystem


class FeatureSystem(BaseOrchestrationSystem):
    """
    Orchestrates multiple feature models.
    
    Manages evaluation model execution and stores results in ExperimentData.
    """
    
    def __init__(self, logger: PfabLogger):
        """Initialize evaluation system."""
        super().__init__(logger)
        self.models: List[IFeatureModel] = []

    def set_dim_codes_for_arrays(self, schema: DatasetSchema) -> None:
        """Set dimension codes for all metric arrays based on dataset parameters."""
        dim_codes = schema.parameters.get_dim_names()
        
        # Iterate over all feature models to set dim codes
        for model in self.models:
            for output_code in model.outputs:
                data_array = schema.features.data_objects[output_code]
                if isinstance(data_array, DataArray):
                    model_dim_codes = [code for code in model.input_parameters if code in dim_codes]
                    data_array.set_dim_codes(model_dim_codes)

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
            if all(skip_feature_code.get(code, False) for code in feature_model.outputs):
                self.logger.info(f"Skipping feature extraction for '{feature_model.outputs}' as features already complete")
                continue

            # Run feature extraction and return 3d feature array
            feature_array = feature_model.compute_features(
                parameters=parameters,
                evaluate_from=evaluate_from,
                evaluate_to=evaluate_to,
                visualize=visualize
            )

            # Collect results
            for i, code in enumerate(feature_model.outputs):
                feature_dict[code] = feature_array[i]

        return feature_dict

