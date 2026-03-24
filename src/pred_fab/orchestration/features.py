from typing import Dict, Optional, List
import numpy as np

from pred_fab.core.data_objects import DataArray
from pred_fab.interfaces.features import IFeatureModel


from ..utils import PfabLogger
from ..core import DatasetSchema, ExperimentData, Parameters, Features
from .base_system import BaseOrchestrationSystem


class FeatureSystem(BaseOrchestrationSystem):
    """Orchestrates feature extraction across all registered feature models."""

    def __init__(self, logger: PfabLogger):
        super().__init__(logger)
        self.models: List[IFeatureModel] = []
        self._schema: Optional[DatasetSchema] = None

    def _set_feature_column_names(self, schema: DatasetSchema) -> None:
        """Set domain axis iterator column names on each model's DataArray outputs, and store schema ref."""
        self._schema = schema
        for model in self.models:
            for output_code in model.outputs:

                # get data array
                if output_code not in schema.features.data_objects:
                    raise ValueError(f"Output '{output_code}' from model '{model.__class__}' is not in schema.")
                data_array = schema.features.data_objects[output_code]
                if not isinstance(data_array, DataArray):
                    raise ValueError(f"Expected obj of type 'DataArray', got {data_array.__class__} instead.")

                # Set column names only if not already explicitly set.
                if not data_array.columns:
                    domain_code = data_array.domain_code
                    if domain_code is None:
                        # Scalar feature: no iterator columns
                        data_array.set_columns([output_code])
                    else:
                        domain = schema.domains.get(domain_code)
                        feature_depth = data_array.feature_depth
                        axes = domain.axes if feature_depth is None else domain.axes[:feature_depth]
                        col_names = [ax.iterator_code for ax in axes] + [output_code]
                        data_array.set_columns(col_names)

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
            features=exp_data.features,
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
        features: Features,
        evaluate_from: int = 0,
        evaluate_to: Optional[int] = None,
        visualize: bool = False,
        skip_feature_code: Dict[str, bool] = {}
    ) -> Dict[str, np.ndarray]:
        """Run all feature models and return {code: tensor} dict."""

        # Prepare result dictionaries
        feature_dict: Dict[str, np.ndarray] = {}

        # Run feature extraction for each feature code
        for feature_model in self.models:
            # Skip if already loaded
            if all(skip_feature_code.get(code, False) for code in feature_model.outputs):
                self.logger.info(f"Skipping feature extraction for '{feature_model.outputs}' as features already complete")
                continue

            # Resolve domain for this model
            domain = None
            if feature_model.input_domain is not None:
                # domain will be injected by agent via schema — here we get it from the schema ref
                # via the _ref_features or caller must pass it; use _schema_ref if available
                if hasattr(self, '_schema') and self._schema is not None:
                    schema: DatasetSchema = self._schema  # type: ignore[assignment]
                    if schema.domains.has(feature_model.input_domain):
                        domain = schema.domains.get(feature_model.input_domain)

            # Run feature extraction and return 2d feature array
            feature_array = feature_model.compute_features(
                parameters=parameters,
                domain=domain,
                evaluate_from=evaluate_from,
                evaluate_to=evaluate_to,
                visualize=visualize
            )

            # Determine number of dimension columns from domain
            num_dims = 0
            if domain is not None:
                depth = feature_model.depth
                max_depth = len(domain.axes) if depth is None else min(depth, len(domain.axes))
                num_dims = max_depth

            for i, code in enumerate(feature_model.outputs):
                # Slice [iterators..., selected-feature] from model output table.
                table = feature_array[:, list(range(num_dims)) + [num_dims + i]]
                # Convert to canonical tensor via shared Features transformation.
                feature_dict[code] = features.table_to_tensor(code, table, parameters)

        return feature_dict
