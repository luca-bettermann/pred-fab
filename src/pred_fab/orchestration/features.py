from typing import Any, Callable, Dict, Optional, List, Tuple
import numpy as np

from pred_fab.core.data_objects import DataArray, Domain
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
        self._model_domain_map: Dict[int, Tuple[Optional[Domain], Optional[int]]] = {}

    def _set_feature_column_names(self, schema: DatasetSchema) -> None:
        """Derive and validate domain+depth from schema outputs, then set column names on DataArrays.

        For each feature model, all outputs must share the same ``domain_code`` and ``feature_depth``
        in the schema; raises ValueError if they diverge.  The resolved Domain object and depth are
        stored in ``_model_domain_map`` keyed by ``id(model)`` for use during feature extraction.
        """
        self._schema = schema
        for model in self.models:
            domain_codes: List[Optional[str]] = []
            feature_depths: List[Optional[int]] = []

            for output_code in model.outputs:
                # get data array
                if output_code not in schema.features.data_objects:
                    raise ValueError(f"Output '{output_code}' from model '{model.__class__}' is not in schema.")
                data_array = schema.features.data_objects[output_code]
                if not isinstance(data_array, DataArray):
                    raise ValueError(f"Expected obj of type 'DataArray', got {data_array.__class__} instead.")
                domain_codes.append(data_array.domain_code)
                feature_depths.append(data_array.feature_depth)

            # Validate all outputs share the same domain_code
            if len(set(domain_codes)) > 1:
                raise ValueError(
                    f"Feature model '{model.__class__.__name__}' has outputs with mixed domain_codes: "
                    f"{dict(zip(model.outputs, domain_codes))}. All outputs must share the same domain_code."
                )

            # Validate all outputs share the same feature_depth
            if len(set(feature_depths)) > 1:
                raise ValueError(
                    f"Feature model '{model.__class__.__name__}' has outputs with mixed feature_depths: "
                    f"{dict(zip(model.outputs, feature_depths))}. All outputs must share the same feature_depth."
                )

            derived_domain_code = domain_codes[0] if domain_codes else None
            derived_depth = feature_depths[0] if feature_depths else None

            # Resolve the Domain object from the schema
            domain: Optional[Domain] = None
            if derived_domain_code is not None:
                domain = schema.domains.get(derived_domain_code)
                if domain is None:
                    raise ValueError(
                        f"Feature model '{model.__class__.__name__}' outputs reference domain_code "
                        f"'{derived_domain_code}', but this domain is not registered in the schema."
                    )

            # Store derived (domain, depth) for use during feature extraction
            self._model_domain_map[id(model)] = (domain, derived_depth)

            # Set column names only if not already explicitly set.
            for output_code in model.outputs:
                data_array = schema.features.data_objects[output_code]  # type: ignore[assignment]
                if not data_array.columns:  # type: ignore[union-attr]
                    if domain is None:
                        data_array.set_columns([output_code])  # type: ignore[union-attr]
                    else:
                        axes = domain.axes if derived_depth is None else domain.axes[:derived_depth]
                        col_names = [ax.iterator_code for ax in axes] + [output_code]
                        data_array.set_columns(col_names)  # type: ignore[union-attr]

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

        # Provide per-row effective parameter resolution so that runtime parameter updates
        # recorded on the experiment (e.g. adapted print_speed during online adaptation) are
        # reflected in feature extraction.  ExperimentData.get_effective_parameters_for_row
        # applies all recorded ParameterUpdateEvents that start at or before the given row.
        get_params_for_row: Optional[Callable[[int], Dict[str, Any]]] = None
        if exp_data.parameter_updates:
            get_params_for_row = exp_data.get_effective_parameters_for_row

        # Get feature extraction results from core logic
        feature_dict = self._compute_features_from_params(
            parameters=exp_data.parameters,
            features=exp_data.features,
            evaluate_from=evaluate_from,
            evaluate_to=evaluate_to,
            visualize=visualize,
            skip_feature_code=skip_for_code,
            get_params_for_row=get_params_for_row,
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
        skip_feature_code: Dict[str, bool] = {},
        get_params_for_row: Optional[Callable[[int], Dict[str, Any]]] = None,
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

            # Look up domain and depth derived during _set_feature_column_names.
            if id(feature_model) not in self._model_domain_map:
                raise RuntimeError(
                    f"FeatureSystem has no domain mapping for '{feature_model.__class__.__name__}'. "
                    f"Ensure PfabAgent is fully initialized before running feature extraction."
                )
            domain, depth = self._model_domain_map[id(feature_model)]

            # Run feature extraction and return 2d feature array
            feature_array = feature_model.compute_features(
                parameters=parameters,
                domain=domain,
                evaluate_from=evaluate_from,
                evaluate_to=evaluate_to,
                visualize=visualize,
                depth=depth,
                get_params_for_row=get_params_for_row,
            )

            # Determine number of dimension columns from domain
            num_dims = 0
            if domain is not None:
                max_depth = len(domain.axes) if depth is None else min(depth, len(domain.axes))
                num_dims = max_depth

            for i, code in enumerate(feature_model.outputs):
                # Slice [iterators..., selected-feature] from model output table.
                table = feature_array[:, list(range(num_dims)) + [num_dims + i]]
                # Convert to canonical tensor via shared Features transformation.
                feature_dict[code] = features.table_to_tensor(code, table, parameters)

        return feature_dict
