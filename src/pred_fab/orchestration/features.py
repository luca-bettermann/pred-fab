from typing import Any, Dict, Tuple, Type, Optional, List, Set
import numpy as np

from pred_fab.core.data_objects import DataArray
from pred_fab.interfaces.features import IFeatureModel


from ..utils import PfabLogger
from ..core import DatasetSchema, ExperimentData, Parameters, DataDimension
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

    def _set_feature_column_names(self, schema: DatasetSchema) -> None:
        """Set dimension codes for all metric arrays based on dataset parameters."""        
        # Iterate over all feature models to set dim codes
        for model in self.models:
            for output_code in model.outputs:

                # get data array
                if not output_code in schema.features.data_objects:
                    raise ValueError(f"Output '{output_code}' from model '{model.__class__}' is not in schema.")
                data_array = schema.features.data_objects[output_code]
                if not isinstance(data_array, DataArray):
                    raise ValueError(f"Expected obj of type 'DataArray', got {data_array.__class__} instead.")

                # set column names in data array
                dim_objs = schema.parameters.get_dim_objects(model.input_parameters)
                model_column_names = [dim.iterator_code for dim in dim_objs]
                model_column_names.append(output_code)
                data_array.set_columns(model_column_names)

    # === FEATURE EXTRACTION ===

    def run_feature_extraction(
        self,
        exp_data: ExperimentData,
        evaluate_from: int = 0,
        evaluate_to: Optional[int] = None,
        visualize: bool = False,
        recompute: bool = False
    ) -> Dict[str, np.ndarray]:
        """Execute all feature extractions for an experiment and mutate exp_data with results."""
        # Handle recompute logic
        if recompute:
            self.logger.info(f"Recompute flag set - clearing cache")

        # Check if the features are already computed
        skip_for_code = {code: exp_data.is_complete(code, evaluate_from, evaluate_to) 
                         for code in exp_data.features.keys() if not recompute}

        # Get feature extraction results from core logic
        feature_dict = self._compute_features_from_params(
            parameters=exp_data.parameters,
            evaluate_from=evaluate_from,
            evaluate_to=evaluate_to,
            visualize=visualize,
            skip_feature_code=skip_for_code
        )

        # Update exp_data with results
        exp_data.features.set_values_from_dict(feature_dict, self.logger)

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

            # Collect results (dim + feature value)
            num_dims = len(feature_model.get_input_dimensions())
            for i, code in enumerate(feature_model.outputs):
                # Directly slice [rows, (dims columns + specific feature column)]
                feature_dict[code] = feature_array[:, list(range(num_dims)) + [num_dims+i]]

        return feature_dict
