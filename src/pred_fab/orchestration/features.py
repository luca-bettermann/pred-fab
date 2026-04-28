from typing import Any, Callable
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
        self.models: list[IFeatureModel] = []
        self._schema: DatasetSchema | None = None
        self._model_domain_map: dict[int, tuple[Domain | None, int | None]] = {}
        self._recursive_features: list[DataArray] = []

    def _set_feature_column_names(self, schema: DatasetSchema) -> None:
        """Derive and validate domain+depth from schema outputs, then set column names on DataArrays.

        For each feature model, all outputs must share the same ``domain_code`` and ``feature_depth``
        in the schema; raises ValueError if they diverge.  The resolved Domain object and depth are
        stored in ``_model_domain_map`` keyed by ``id(model)`` for use during feature extraction.
        """
        self._schema = schema
        for model in self.models:
            domain_codes: list[str | None] = []
            feature_depths: list[int | None] = []

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
            domain: Domain | None = None
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

        # Collect recursive features from the schema for post-hoc tensor derivation.
        self._recursive_features = []
        for feat_code, feat_obj in schema.features.items():
            if isinstance(feat_obj, DataArray) and feat_obj.is_recursive:
                self._recursive_features.append(feat_obj)

    # === FEATURE EXTRACTION ===

    def run_feature_extraction(
        self,
        exp_data: ExperimentData,
        evaluate_from: int = 0,
        evaluate_to: int | None = None,
        visualize: bool = False,
        recompute: bool = False
    ) -> dict[str, np.ndarray]:
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
        get_params_for_row: Callable[[int], dict[str, Any]] | None = None
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
        evaluate_to: int | None = None,
        visualize: bool = False,
        skip_feature_code: dict[str, bool] = {},
        get_params_for_row: Callable[[int], dict[str, Any]] | None = None,
    ) -> dict[str, np.ndarray]:
        """Run all feature models and return {code: tensor} dict."""

        # Prepare result dictionaries
        feature_dict: dict[str, np.ndarray] = {}

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

        # Derive recursive features by shifting source tensors.
        for rec_feat in self._recursive_features:
            source_code = rec_feat.recursive_source
            if source_code is None or rec_feat.recursive_dimensions is None:
                continue
            if source_code not in feature_dict:
                # Source may already be stored on the Features block (e.g. partial eval).
                if features.has_value(source_code):
                    source_tensor = features.get_value(source_code)
                else:
                    self.logger.warning(
                        f"Recursive feature '{rec_feat.code}' references source '{source_code}' "
                        f"which has not been computed."
                    )
                    continue
            else:
                source_tensor = feature_dict[source_code]

            shifted = self._shift_tensor(
                source_tensor,
                rec_feat.recursive_dimensions,
                rec_feat.recursive_depth or 1,
                features,
                parameters,
            )
            feature_dict[rec_feat.code] = shifted

        return feature_dict

    def _shift_tensor(
        self,
        tensor: np.ndarray,
        dimension_iterator_codes: tuple[str, ...],
        step: int,
        features: Features,
        parameters: 'Parameters',
    ) -> np.ndarray:
        """Shift a source tensor backward along specified dimensions, padding boundaries with NaN."""
        result = tensor.copy()
        for iter_code in dimension_iterator_codes:
            axis = self._resolve_axis_index(iter_code, features, parameters)
            if axis is None:
                continue
            result = self._shift_along_axis(result, axis, step)
        return result

    def _resolve_axis_index(
        self,
        iterator_code: str,
        features: Features,
        parameters: 'Parameters',
    ) -> int | None:
        """Map a dimension iterator code (e.g. 'layer_idx') to a tensor axis index."""
        dim_objs = parameters._get_domain_axis_objects()
        iterator_codes = [d.iterator_code for d in dim_objs]
        if iterator_code in iterator_codes:
            return iterator_codes.index(iterator_code)
        self.logger.warning(f"Iterator code '{iterator_code}' not found in domain axes.")
        return None

    @staticmethod
    def _shift_along_axis(tensor: np.ndarray, axis: int, step: int) -> np.ndarray:
        """Shift tensor values backward along an axis by ``step``, padding leading positions with NaN."""
        result = np.full_like(tensor, np.nan)
        src = [slice(None)] * tensor.ndim
        dst = [slice(None)] * tensor.ndim
        src[axis] = slice(None, tensor.shape[axis] - step)
        dst[axis] = slice(step, None)
        result[tuple(dst)] = tensor[tuple(src)]
        return result
