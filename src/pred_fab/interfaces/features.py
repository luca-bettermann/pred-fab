import numpy as np
from typing import Any, Callable, final
from numpy.typing import NDArray
from dataclasses import dataclass

from abc import ABC, abstractmethod
from .base_interface import BaseInterface
from ..core import Dataset, Parameters
from ..core.data_objects import Domain
from ..utils import PfabLogger


class IFeatureModel(BaseInterface):
    """Abstract base for feature extraction models that iterate over domain axis combinations.

    Domain and depth are NOT declared on the model — they are derived from the schema during
    FeatureSystem initialization (``_set_feature_column_names``).  The constraint is that all
    outputs of a single feature model must share the same ``domain_code`` and ``feature_depth``
    in the schema; this is a structural requirement because a single ``compute_features`` call
    iterates one domain at one depth.

    Subclasses must implement exactly one of:
      - ``_load_data`` — for models that consume raw data (files, sensors, databases)
      - ``_load_from_features`` — for models that consume other features' outputs

    Both default to ``NotImplementedError``. FeatureSystem validates at init
    that at least one is overridden.
    """

    def __init__(self, logger: PfabLogger):
        super().__init__(logger)

    # === DATA LOADING (implement one) ===

    def _load_data(self, params: dict, **dimensions) -> Any:
        """Load domain-specific raw data for the given parameter context (files, DB, etc.)."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _load_data or _load_from_features"
        )

    def _load_from_features(
        self,
        features: dict[str, np.ndarray],
        params: dict,
        **dimensions,
    ) -> Any:
        """Load data from upstream feature arrays.

        ``features`` maps feature code → full 2D table ``(n_rows, n_dims + 1)``
        for each declared ``input_features``. Use ``slice_feature_at`` to
        extract values at the current dimension context.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _load_data or _load_from_features"
        )

    # === FEATURE LOGIC (always implement) ===

    @abstractmethod
    def _compute_feature_logic(
        self,
        data: Any,
        params: dict,
        visualize: bool = False,
        **dimensions
        ) -> dict[str, float]:
        """Extract feature values from loaded data; returns dict mapping feature codes to numeric values."""
        ...

    @property
    def input_features(self) -> list[str]:
        """Feature codes from other models that this model consumes.

        Override to declare dependencies — the FeatureSystem will run
        upstream models first and pass their computed arrays via
        ``_load_from_features``.  Default: no dependencies.
        """
        return []

    @property
    def uses_raw_data(self) -> bool:
        """True if this model implements ``_load_data`` (loads from raw sources)."""
        return type(self)._load_data is not IFeatureModel._load_data

    @property
    def uses_feature_input(self) -> bool:
        """True if this model implements ``_load_from_features`` (consumes upstream features)."""
        return type(self)._load_from_features is not IFeatureModel._load_from_features

    # === PUBLIC API ===

    @final
    def compute_features(
        self,
        parameters: Parameters,
        domain: Domain | None,
        evaluate_from: int,
        evaluate_to: int | None = None,
        visualize: bool = False,
        depth: int | None = None,
        get_params_for_row: Callable[[int], dict[str, Any]] | None = None,
        features_so_far: dict[str, np.ndarray] | None = None,
        ) -> NDArray:
        """Iterate over every domain axis combination in [evaluate_from, evaluate_to) and call the appropriate load + compute methods.

        Returns a 2-D array of shape (n_combinations, n_dims + n_outputs) where the first
        n_dims columns are the domain axis iterator values and the remaining columns are feature values.
        """
        self.logger.info(f"Starting evaluation for '{self.outputs}'")

        # Resolve iteration axes from domain
        if domain is None or len(domain.axes) == 0:
            axes = []
        else:
            max_depth = len(domain.axes) if depth is None else min(depth, len(domain.axes))
            axes = domain.axes[:max_depth]

        num_dims = len(axes)
        dim_codes = [a.code for a in axes]
        dim_iterator_codes = [a.iterator_code for a in axes]

        if axes:
            dim_combinations = parameters.get_dim_combinations(
                dim_codes=dim_codes,
                evaluate_from=evaluate_from,
                evaluate_to=evaluate_to
            )
        else:
            dim_combinations = [()]

        # Initialize 2D-array
        shape = (len(dim_combinations), num_dims + len(self.outputs))
        feature_array = np.empty(shape)

        # Snapshot of initial parameters used when no per-row override is provided.
        params_snapshot = parameters.get_values_dict()

        # Expose row-resolver so subclasses can look up params at other positions
        # (e.g. previous layer). Temporary — will be replaced by SequenceContext.
        self._get_params_for_row = get_params_for_row

        # Process each combination
        i_end = evaluate_to if evaluate_to is not None else len(dim_combinations)
        for i, current_dim in enumerate(dim_combinations):
            i_global = evaluate_from + i

            params = get_params_for_row(i_global) if get_params_for_row is not None else params_snapshot

            current_dim_dict = dict(zip(dim_iterator_codes, current_dim)) if axes else {}
            self.logger.debug(f"Processing {i_global}/{i_end}: {current_dim_dict}")

            feature_values = self._compute_feature_values(
                params,
                current_dim_dict,
                visualize=visualize,
                features_so_far=features_so_far,
                )

            if axes:
                feature_array[i, :num_dims] = current_dim
            feature_array[i, num_dims:] = feature_values

        return feature_array

    @final
    def _compute_feature_values(
        self,
        params,
        dimensions,
        visualize: bool = False,
        features_so_far: dict[str, np.ndarray] | None = None,
        ) -> list[float]:
        """Load data via the appropriate method, then compute features."""

        self.logger.debug(f"Computing features '{self.outputs}' for {params}")

        if self.uses_feature_input and features_so_far is not None:
            relevant = {k: v for k, v in features_so_far.items() if k in self.input_features}
            data = self._load_from_features(relevant, params, **dimensions)
        else:
            data = self._load_data(params, **dimensions)

        feature_dict = self._compute_feature_logic(data, params, visualize=visualize, **dimensions)

        for feature_code, feature_value in feature_dict.items():
            if not isinstance(feature_value, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"_compute_features() must return numeric values, got {type(feature_value).__name__} for feature '{feature_code}'"
                )

        feature_values = [feature_dict[code] for code in self.outputs]  # type: ignore
        return feature_values


def slice_feature_at(
    feature_table: np.ndarray,
    axis_code: str,
    axis_value: int,
    dim_codes: list[str],
) -> np.ndarray:
    """Filter a feature table by axis code and return the value column.

    ``feature_table`` has shape ``(n_rows, n_dims + 1)`` where the first
    ``n_dims`` columns are dimension indices and the last column is the
    feature value.

    ``dim_codes`` is the ordered list of dimension code names matching
    the columns (from the schema domain axes).

    Returns the 1-D array of values where ``axis_code == axis_value``.
    """
    if axis_code not in dim_codes:
        raise ValueError(
            f"axis_code {axis_code!r} not found in dim_codes {dim_codes}. "
            f"Check that the upstream feature's domain matches."
        )
    col_idx = dim_codes.index(axis_code)
    mask = feature_table[:, col_idx] == axis_value
    return feature_table[mask, -1]
