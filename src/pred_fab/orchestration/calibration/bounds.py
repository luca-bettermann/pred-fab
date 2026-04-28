from typing import Any

import numpy as np

from ...core import DataModule, DatasetSchema
from ...core import DataInt, DataReal, DataObject, DataBool, DataCategorical, DataDomainAxis
from ...utils import PfabLogger


# ======================================================================
# BoundsManager — schema-aware bounds computation
# ======================================================================

class BoundsManager:
    """Schema-aware parameter bounds, fixed params, trust regions, and schedule configs."""

    def __init__(self, schema: DatasetSchema, logger: PfabLogger):
        self.schema = schema
        self.logger = logger

        self.data_objects: dict[str, DataObject] = {}
        self.schema_bounds: dict[str, tuple[float, float]] = {}
        self.param_bounds: dict[str, tuple[float, float]] = {}
        self.fixed_params: dict[str, Any] = {}
        self.trust_regions: dict[str, float] = {}
        self.schedule_configs: dict[str, str] = {}  # param_code -> dimension_code

        self._set_param_constraints_from_schema(schema)

    def _set_param_constraints_from_schema(self, schema: DatasetSchema) -> None:
        """Extract parameter constraints from dataset schema."""
        for code, data_obj in schema.parameters.data_objects.items():
            if isinstance(data_obj, (DataBool, DataCategorical)):
                min_val, max_val = 0.0, 1.0
            elif issubclass(type(data_obj), DataObject):
                min_val = data_obj.constraints.get("min", -np.inf)
                max_val = data_obj.constraints.get("max", np.inf)
            else:
                raise TypeError(f"Expected DataObject type for parameter '{code}', got {type(data_obj).__name__}")

            self.data_objects[code] = data_obj
            self.schema_bounds[code] = (min_val, max_val)

    def configure_param_bounds(self, bounds: dict[str, tuple[float, float]], force: bool = False) -> None:
        """Configure parameter ranges for offline calibration."""
        for code, (low, high) in bounds.items():
            if not self._validate_and_clean_config(
                code, (DataReal, DataInt), ['fixed_params'], force
            ):
                continue

            schema_min, schema_max = self.schema_bounds[code]
            if low < schema_min or high > schema_max:
                raise ValueError(
                    f"Bounds for object '{code}' exceed schema constraints: "
                    f"[{low}, {high}] vs schema [{schema_min}, {schema_max}]"
                )

            self.param_bounds[code] = (low, high)
            self.logger.debug(f"Set parameter bounds: {code} -> [{low}, {high}]")

    def configure_fixed_params(self, fixed_params: dict[str, Any], force: bool = False) -> None:
        """Configure fixed parameter values."""
        for code, value in (fixed_params or {}).items():
            if not self._validate_and_clean_config(
                code, None, ['param_bounds', 'trust_regions'], force
            ):
                continue

            self.fixed_params[code] = value
            self.logger.debug(f"Set fixed parameter: {code} -> {value}")

    def configure_adaptation_delta(self, deltas: dict[str, float], force: bool = False) -> None:
        """Configure trust region deltas for online calibration."""
        for code, delta in deltas.items():
            if not self._validate_and_clean_config(
                code, (DataReal, DataInt), ['fixed_params'], force
            ):
                continue

            obj = self.data_objects[code]
            if not obj.runtime_adjustable:
                raise ValueError(
                    f"Parameter '{code}' is not runtime-adjustable. Trust regions can only be "
                    f"configured for parameters declared with runtime=True in the schema. "
                    f"Either mark '{code}' as runtime=True in the schema definition, or remove "
                    f"this configure_adaptation_delta() call."
                )

            self.trust_regions[code] = delta

    def configure_schedule_parameter(self, code: str, dimension_code: str, force: bool = False) -> None:
        """Declare that a runtime-adjustable parameter should be re-optimised at each step of the given dimension."""
        if code not in self.data_objects:
            self.logger.console_warning(
                f"Object '{code}' not found in schema; ignoring configure_schedule_parameter."
            )
            return

        obj = self.data_objects[code]

        if not obj.runtime_adjustable:
            raise ValueError(
                f"Parameter '{code}' is not runtime-adjustable. configure_schedule_parameter() "
                f"requires a parameter declared with runtime=True in the schema."
            )

        if not isinstance(obj, (DataReal, DataInt)):
            raise ValueError(
                f"Parameter '{code}' type {type(obj).__name__} is not supported for "
                f"dimension stepping. Only DataReal and DataInt parameters can be step parameters."
            )

        if dimension_code not in self.data_objects:
            raise ValueError(
                f"Dimension '{dimension_code}' not found in schema."
            )
        dim_obj = self.data_objects[dimension_code]
        if not isinstance(dim_obj, DataDomainAxis):
            raise ValueError(
                f"'{dimension_code}' is not a DataDomainAxis parameter "
                f"(got {type(dim_obj).__name__})."
            )

        if code in self.schedule_configs and not force:
            self.logger.console_warning(
                f"Parameter '{code}' already has a schedule configuration for "
                f"'{self.schedule_configs[code]}'; ignoring. Use force=True to overwrite."
            )
            return

        self.schedule_configs[code] = dimension_code
        if code not in self.trust_regions:
            lo, hi = self.schema_bounds.get(code, (0.0, 1.0))
            if lo != -np.inf and hi != np.inf:
                self.trust_regions[code] = (hi - lo) / 10.0
        self.logger.debug(
            f"Configured schedule for '{code}' stepping through '{dimension_code}'."
        )

    def _validate_and_clean_config(
        self,
        code: str,
        allowed_types: tuple[type, ...] | None,
        conflicting_collections: list[str],
        force: bool
    ) -> bool:
        """Validate parameter against schema and check for conflicting configurations."""
        if code not in self.data_objects:
            self.logger.console_warning(f"Object '{code}' not found in schema; ignoring.")
            return False

        if allowed_types:
            obj = self.data_objects[code]
            if not isinstance(obj, allowed_types):
                 self.logger.console_warning(
                     f"Object '{code}' type {type(obj).__name__} not supported for this configuration; ignoring."
                )
                 return False

        for collection_name in conflicting_collections:
            collection = getattr(self, collection_name)
            if code in collection:
                if force:
                    del collection[code]
                    self.logger.debug(f"Removed '{code}' from {collection_name} due to force=True.")
                else:
                    self.logger.console_warning(
                        f"Object '{code}' is already configured in {collection_name}; ignoring. Use force=True to overwrite."
                    )
                    return False
        return True

    def get_tunable_params(self, datamodule: DataModule) -> list[str]:
        """Return codes of parameters the optimizer can actually vary.

        Excludes: context features, fixed params, features (lag values),
        and one-hot columns. Only returns the original parameter codes
        that have non-zero-width bounds.
        """
        context_codes = set(datamodule.context_feature_codes)
        col_map = datamodule.get_onehot_column_map()
        schema_params = set(datamodule.dataset.schema.parameters.data_objects.keys())
        tunable = []
        for code in datamodule.input_columns:
            if code in context_codes:
                continue
            if code in col_map:
                continue
            if code not in schema_params:
                continue
            if code in self.fixed_params:
                continue
            lo, hi = self._get_hierarchical_bounds_for_code(code)
            if hi - lo < 1e-12:
                continue
            tunable.append(code)
        return tunable

    def _get_global_bounds(self, datamodule: DataModule) -> np.ndarray:
        """Return normalized optimization bounds over the full parameter space."""
        bounds_list = []
        col_map = datamodule.get_onehot_column_map()
        context_codes = set(datamodule.context_feature_codes)

        for code in datamodule.input_columns:
            if code in context_codes:
                bounds_list.append(self._normalize_bounds(code, 0.0, 0.0, datamodule))
                continue

            if code in col_map:
                parent_param, cat_val = col_map[code]
                if parent_param in self.fixed_params:
                    val = 1.0 if self.fixed_params[parent_param] == cat_val else 0.0
                    low, high = val, val
                else:
                    low, high = 0.0, 1.0
            else:
                low, high = self._get_hierarchical_bounds_for_code(code)

            n_low, n_high = self._normalize_bounds(code, low, high, datamodule)
            bounds_list.append((n_low, n_high))

        return np.array(bounds_list)

    def _get_trust_region_bounds(self, datamodule: DataModule, current_params: dict[str, Any]) -> np.ndarray:
        """Return normalized trust-region bounds centred on current_params."""
        bounds_list = []
        col_map = datamodule.get_onehot_column_map()

        for code in datamodule.input_columns:
            curr = 0.0
            is_one_hot = code in col_map

            if is_one_hot:
                parent_param, cat_val = col_map[code]
                if parent_param and parent_param in current_params:
                     curr = 1.0 if current_params[parent_param] == cat_val else 0.0
                elif code in current_params:
                    curr = current_params[code]
            else:
                if code in current_params:
                    curr = current_params[code]

            if code in self.trust_regions:
                delta = self.trust_regions[code]
                low, high = curr - delta, curr + delta
            else:
                low, high = curr, curr

            bounds_list.append(self._normalize_bounds(code, low, high, datamodule))

        return np.array(bounds_list)

    def _get_hierarchical_bounds_for_code(self, code: str) -> tuple[float, float]:
        if code in self.fixed_params:
            val = self.fixed_params[code]
            low, high = val, val
        elif code in self.param_bounds:
            low, high = self.param_bounds[code]
        elif code in self.schema_bounds:
            low, high = self.schema_bounds[code]
        else:
            raise ValueError(f"No bounds found for '{code}'. Cannot determine optimization bounds.")
        return low, high

    def _normalize_bounds(self, col: str, low: float, high: float, datamodule: DataModule) -> tuple[float, float]:
        """Normalize bounds to [0, 1] based on schema constraints."""
        n_low, n_high = datamodule.normalize_parameter_bounds(col, low, high)
        if (n_low, n_high) != (low, high):
            self.logger.debug(f"Processed bounds for '{col}': raw [{low}, {high}] -> normalized [{n_low}, {n_high}]")
        else:
            self.logger.debug(f"No normalization stats for '{col}'. Using raw bounds [{low}, {high}].")
        return n_low, n_high
