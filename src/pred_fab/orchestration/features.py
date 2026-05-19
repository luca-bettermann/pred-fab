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

    def _set_feature_column_names(self, schema: DatasetSchema) -> None:
        """Derive and validate domain+depth from schema outputs, then set column names on DataArrays.

        Also validates:
          - each model implements at least one of ``_load_data`` / ``_load_from_features``
          - ``input_features`` reference outputs of another registered model
          - the dependency graph is acyclic
        """
        self._schema = schema

        # --- Per-model domain/depth derivation (unchanged) ---
        for model in self.models:
            domain_codes: list[str | None] = []
            feature_depths: list[int | None] = []

            for output_code in model.outputs:
                if output_code not in schema.features.data_objects:
                    raise ValueError(f"Output '{output_code}' from model '{model.__class__}' is not in schema.")
                data_array = schema.features.data_objects[output_code]
                if not isinstance(data_array, DataArray):
                    raise ValueError(f"Expected obj of type 'DataArray', got {data_array.__class__} instead.")
                domain_codes.append(data_array.domain_code)
                feature_depths.append(data_array.feature_depth)

            if len(set(domain_codes)) > 1:
                raise ValueError(
                    f"Feature model '{model.__class__.__name__}' has outputs with mixed domain_codes: "
                    f"{dict(zip(model.outputs, domain_codes))}. All outputs must share the same domain_code."
                )

            if len(set(feature_depths)) > 1:
                raise ValueError(
                    f"Feature model '{model.__class__.__name__}' has outputs with mixed feature_depths: "
                    f"{dict(zip(model.outputs, feature_depths))}. All outputs must share the same feature_depth."
                )

            derived_domain_code = domain_codes[0] if domain_codes else None
            derived_depth = feature_depths[0] if feature_depths else None

            domain: Domain | None = None
            if derived_domain_code is not None:
                domain = schema.domains.get(derived_domain_code)
                if domain is None:
                    raise ValueError(
                        f"Feature model '{model.__class__.__name__}' outputs reference domain_code "
                        f"'{derived_domain_code}', but this domain is not registered in the schema."
                    )

            self._model_domain_map[id(model)] = (domain, derived_depth)

            for output_code in model.outputs:
                data_array = schema.features.data_objects[output_code]  # type: ignore[assignment]
                if not data_array.columns:  # type: ignore[union-attr]
                    if domain is None:
                        data_array.set_columns([output_code])  # type: ignore[union-attr]
                    else:
                        axes = domain.axes if derived_depth is None else domain.axes[:derived_depth]
                        col_names = [ax.iterator_code for ax in axes] + [output_code]
                        data_array.set_columns(col_names)  # type: ignore[union-attr]

        # --- Validate load method implementation ---
        for model in self.models:
            if not model.uses_raw_data and not model.uses_feature_input:
                raise TypeError(
                    f"{type(model).__name__} must implement _load_data or "
                    f"_load_from_features"
                )

        # --- Validate input_features references and acyclicity ---
        self._validate_feature_dependencies()

    def _validate_feature_dependencies(self) -> None:
        """Check that input_features reference valid outputs and the graph is acyclic."""
        all_outputs: set[str] = set()
        for m in self.models:
            all_outputs.update(m.outputs)

        for m in self.models:
            for feat in m.input_features:
                if feat not in all_outputs:
                    raise ValueError(
                        f"{type(m).__name__} declares input_feature '{feat}' but "
                        f"no registered feature model produces it."
                    )

        # Cycle detection via topo sort
        self._topo_sort_models()

    def _build_dependency_graph(self) -> dict[int, set[int]]:
        """Build {model_id: set of upstream model_ids} from input_features."""
        output_to_model: dict[str, IFeatureModel] = {}
        for m in self.models:
            for out in m.outputs:
                output_to_model[out] = m

        deps: dict[int, set[int]] = {id(m): set() for m in self.models}
        for m in self.models:
            for feat_code in m.input_features:
                producer = output_to_model.get(feat_code)
                if producer is not None and producer is not m:
                    deps[id(m)].add(id(producer))
        return deps

    def _topo_sort_models(self) -> list[IFeatureModel]:
        """Order models so dependencies run first. Raises on cycles."""
        deps = self._build_dependency_graph()
        in_degree = {id(m): len(deps[id(m)]) for m in self.models}
        sorted_list: list[IFeatureModel] = []
        queue = [m for m in self.models if in_degree[id(m)] == 0]
        while queue:
            m = queue.pop(0)
            sorted_list.append(m)
            for other in self.models:
                if id(m) in deps[id(other)]:
                    in_degree[id(other)] -= 1
                    if in_degree[id(other)] == 0:
                        queue.append(other)
        if len(sorted_list) != len(self.models):
            raise ValueError(
                "Feature model dependency cycle detected — "
                "input_features form a circular dependency."
            )
        return sorted_list

    # === FEATURE EXTRACTION ===

    def run_feature_extraction(
        self,
        exp_data: ExperimentData,
        evaluate_from: int = 0,
        evaluate_to: int | None = None,
        visualize: bool = False,
        recompute: bool = False,
        feature: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Execute feature extractions for an experiment and mutate exp_data with results.

        When ``feature`` is provided, only models whose class name contains
        that string (case-insensitive) are run. ``None`` runs all models.
        """
        if recompute:
            self.logger.info(f"Recompute flag set - clearing cache")

        skip_for_code = {code: exp_data.is_complete(code, evaluate_from, evaluate_to)
                         for code in exp_data.features.keys() if not recompute}

        get_params_for_row: Callable[[int], dict[str, Any]] | None = None
        if exp_data.parameter_updates:
            get_params_for_row = exp_data.get_effective_parameters_for_row

        feature_dict = self._compute_features_from_params(
            parameters=exp_data.parameters,
            features=exp_data.features,
            evaluate_from=evaluate_from,
            evaluate_to=evaluate_to,
            visualize=visualize,
            skip_feature_code=skip_for_code,
            get_params_for_row=get_params_for_row,
            feature_filter=feature,
        )

        exp_data.features.set_values_from_dict(feature_dict, self.logger)
        for code in feature_dict:
            exp_data.mark_dirty(code)

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
        feature_filter: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Run feature models in dependency order and return {code: tensor} dict."""

        feature_dict: dict[str, np.ndarray] = {}

        features_so_far: dict[str, np.ndarray] = {}
        for code, tensor in features.get_values_dict().items():
            if tensor is not None:
                features_so_far[code] = features.tensor_to_table(code, tensor, parameters)

        ordered_models = self._topo_sort_models()

        for feature_model in ordered_models:
            if feature_filter is not None:
                name = type(feature_model).__name__.lower()
                if feature_filter.lower() not in name:
                    continue
            if all(skip_feature_code.get(code, False) for code in feature_model.outputs):
                self.logger.info(f"Skipping feature extraction for '{feature_model.outputs}' as features already complete")
                continue

            if id(feature_model) not in self._model_domain_map:
                raise RuntimeError(
                    f"FeatureSystem has no domain mapping for '{feature_model.__class__.__name__}'. "
                    f"Ensure PfabAgent is fully initialized before running feature extraction."
                )
            domain, depth = self._model_domain_map[id(feature_model)]

            feature_array = feature_model.compute_features(
                parameters=parameters,
                domain=domain,
                evaluate_from=evaluate_from,
                evaluate_to=evaluate_to,
                visualize=visualize,
                depth=depth,
                get_params_for_row=get_params_for_row,
                features_so_far=features_so_far if feature_model.input_features else None,
            )

            num_dims = 0
            if domain is not None:
                max_depth = len(domain.axes) if depth is None else min(depth, len(domain.axes))
                num_dims = max_depth

            for i, code in enumerate(feature_model.outputs):
                table = feature_array[:, list(range(num_dims)) + [num_dims + i]]
                tensor = features.table_to_tensor(code, table, parameters)
                feature_dict[code] = tensor
                features_so_far[code] = table

        return feature_dict
