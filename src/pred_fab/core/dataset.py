"""Dataset — in-memory experiment container validated against a DatasetSchema; persistence delegated to LocalData."""

import json
import numpy as np
import pandas as pd
import torch
import os
from dataclasses import dataclass, field
from typing import Callable, Any, Literal
import functools

from .schema import DatasetSchema, assert_blocks_compatible


@dataclass(frozen=True)
class ExportedTensorDict:
    """Tensor-native dataset export.

    Each row corresponds to one (experiment, cell) pair. ``cell_meta[i]``
    gives the ``(exp_idx, cell_idx)`` for row ``i`` — useful for Phase C
    SS substitution (looking up prior-cell predictions per row).

    - ``X``: ``dict[col_name, (n_rows,) tensor]`` — categoricals are long
      indices into their ``categorical_mappings`` list; numerics are float.
    - ``y``: ``dict[col_name, (n_rows,) float tensor]`` — NaN where the
      feature value is missing for that cell.
    - ``cell_meta``: ``(n_rows, 2)`` long tensor with ``[exp_idx, cell_idx]``.
    """
    X: dict[str, torch.Tensor]
    y: dict[str, torch.Tensor]
    cell_meta: torch.Tensor

    @property
    def n_rows(self) -> int:
        return int(self.cell_meta.shape[0])

    def is_empty(self) -> bool:
        return self.n_rows == 0
from ..core import DataBlock, Parameters, Features, PerformanceAttributes, DataDomainAxis
from .data_objects import DataArray

from ..interfaces.external_data import IExternalData
from ..utils import LocalData, PfabLogger
from ..utils.enum import BlockType, Loaders


@dataclass(frozen=True)
class ParameterProposal:
    """Lightweight value object carrying proposed parameter values."""

    values: dict[str, Any]
    source_step: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow copy of proposed values."""
        return dict(self.values)

    @classmethod
    def from_dict(cls, values: dict[str, Any], source_step: str | None = None) -> 'ParameterProposal':
        """Build proposal from a plain dictionary."""
        return cls(values=dict(values), source_step=source_step)

    # Mapping-like compatibility for existing call-sites.
    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __contains__(self, key: object) -> bool:
        return key in self.values

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def items(self):
        return self.values.items()

    def keys(self):
        return self.values.keys()

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class ParameterUpdateEvent:
    """Immutable record of an applied parameter update at a specific fabrication step."""

    updates: dict[str, Any]
    iterator_code: str | None = None
    step_index: int | None = None
    source_step: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to plain dictionary."""
        return {
            "updates": dict(self.updates),
            "iterator_code": self.iterator_code,
            "step_index": self.step_index,
            "source_step": self.source_step,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ParameterUpdateEvent':
        """Deserialize event from plain dictionary."""
        return cls(
            updates=dict(data.get("updates", {})),
            iterator_code=data.get("iterator_code") or data.get("dimension"),
            step_index=data.get("step_index"),
            source_step=data.get("source_step"),
        )

@dataclass
class ParameterTrajectory:
    """Sparse ordered trajectory of runtime parameter changes for one dimension level."""
    dimension: str
    entries: list[tuple[int, 'ParameterProposal']] = field(default_factory=list)

    def apply(self, experiment: 'ExperimentData') -> None:
        """Record all trajectory entries as ParameterUpdateEvents on the experiment.

        Pure conversion goes through :func:`trajectory_to_events`; experiment-
        side delta + sanitize + append is then handled by
        ``ExperimentData.record_parameter_update`` per event.
        """
        for event in trajectory_to_events(self):
            proposal = ParameterProposal(
                values=dict(event.updates),
                source_step=event.source_step,
            )
            experiment.record_parameter_update(
                proposal,
                iterator_code=event.iterator_code,
                step_index=event.step_index,
            )


def trajectory_to_events(trajectory: 'ParameterTrajectory') -> list[ParameterUpdateEvent]:
    """Flatten a ``ParameterTrajectory`` into a list of ``ParameterUpdateEvent``s.

    Each ``(step_index, proposal)`` entry becomes one event tagged with the
    trajectory's dimension. Pure conversion — no experiment context required.
    Inverse of :func:`events_to_trajectory`.
    """
    return [
        ParameterUpdateEvent(
            updates=dict(proposal.values),
            iterator_code=trajectory.dimension,
            step_index=step_index,
            source_step=proposal.source_step,
        )
        for step_index, proposal in trajectory.entries
    ]


def events_to_trajectory(
    events: list[ParameterUpdateEvent],
    dimension: str,
) -> 'ParameterTrajectory':
    """Build a ``ParameterTrajectory`` for ``dimension`` from a flat event list.

    Filters events to those whose ``dimension`` matches; sorts ascending by
    ``step_index``; wraps each event's ``updates`` into a ``ParameterProposal``.
    Events without a ``step_index`` (i.e. initial-state events not bound to a
    trajectory step) are skipped. Inverse of :func:`trajectory_to_events`.
    """
    matched = [e for e in events if e.iterator_code == dimension and e.step_index is not None]
    matched.sort(key=lambda e: e.step_index)  # type: ignore[arg-type, return-value]
    entries: list[tuple[int, ParameterProposal]] = [
        (
            int(e.step_index),  # type: ignore[arg-type]
            ParameterProposal(values=dict(e.updates), source_step=e.source_step),
        )
        for e in matched
    ]
    return ParameterTrajectory(dimension=dimension, entries=entries)


@dataclass
class ExperimentSpec:
    """Initial parameter proposal plus optional per-dimension runtime trajectories."""
    initial_params: 'ParameterProposal'
    trajectories: dict[str, 'ParameterTrajectory'] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def apply_trajectories(self, experiment: 'ExperimentData') -> None:
        """Apply all dimensional trajectories to the experiment as ParameterUpdateEvents."""
        for trajectory in self.trajectories.values():
            trajectory.apply(experiment)

    # dict-like delegation to initial_params for backward compatibility.

    def __getitem__(self, key: str) -> Any:
        return self.initial_params[key]

    def __contains__(self, key: object) -> bool:
        return key in self.initial_params

    def get(self, key: str, default: Any = None) -> Any:
        return self.initial_params.get(key, default)

    def keys(self):
        return self.initial_params.keys()

    def items(self):
        return self.initial_params.items()

    def __iter__(self):
        return iter(self.initial_params)

    def __len__(self) -> int:
        return len(self.initial_params)


class ExperimentData:
    """All data for a single experiment: parameters, features, performance, and parameter update log."""

    def __init__(self,
                 exp_code: str,
                 parameters: Parameters,
                 performance: PerformanceAttributes,
                 features: Features,
                 dataset_code: str | None = None,
                 ):
        self.code = exp_code
        self.parameters = parameters
        self.performance = performance
        self.features = features
        # Optional dataset grouping label — opaque to pred-fab core. External
        # systems (e.g. pred-fab-nocodb) use this to tag experiments as belonging
        # to a named dataset (baseline / reference / test / exploration / etc.).
        self.dataset_code: str | None = dataset_code
        self.parameter_updates: list[ParameterUpdateEvent] = []
        self.config_snapshot: dict[str, Any] | None = None

    # === Helper Methods for Validation ===

    def is_valid(self, schema: 'DatasetSchema') -> bool:
        """Check structural compatibility of exp with schema (raises on mismatch)."""
        assert_blocks_compatible([
            (self.parameters, schema.parameters),
            (self.performance, schema.performance_attrs),
            (self.features, schema.features),
        ])
        return True
    
    def is_complete(self, feature_code: str, evaluate_from: int, evaluate_to: int | None) -> bool:
        """Check if feature array is non-empty in specified range."""
        if not self.features.has(feature_code):
            raise KeyError(f"Feature code '{feature_code}' not found in experiment '{self.code}'")

        array = self.features.get_value(feature_code)
        flat = np.asarray(array).reshape(-1)
        end_index = evaluate_to if evaluate_to is not None else len(flat)
        
        # Check if all values in the specified range are NaN
        if np.all(~np.isnan(flat[evaluate_from:end_index])):
            return True
        return False

    def get_effective_parameters_for_row(self, row_index: int) -> dict[str, Any]:
        """Get effective parameter values at a flattened row index, including applied updates.

        Uses the full domain strides to determine which events have taken
        effect by the given row.
        """
        effective = self.parameters.get_values_dict().copy()
        if not self.parameter_updates:
            return effective

        strides = self.parameters._get_dimension_strides()
        for event in self.parameter_updates:
            if event.iterator_code is None or event.step_index is None:
                coerced = self.parameters.sanitize_values(event.updates, ignore_unknown=True)
                effective.update(coerced)
                continue
            stride = strides.get(event.iterator_code, 1)
            event_start = event.step_index * stride
            if row_index >= event_start:
                coerced = self.parameters.sanitize_values(event.updates, ignore_unknown=True)
                effective.update(coerced)
        return effective

    def get_effective_parameters_for_context(
        self, iterator_ctx: dict[str, int],
    ) -> dict[str, Any]:
        """Get effective parameter values for a given iterator context.

        Matches events by iterator_code and step_index against the context.
        Works at any depth — no flat index computation needed.
        """
        effective = self.parameters.get_values_dict().copy()
        for event in self.parameter_updates:
            if event.iterator_code is None or event.step_index is None:
                coerced = self.parameters.sanitize_values(event.updates, ignore_unknown=True)
                effective.update(coerced)
                continue
            ctx_value = iterator_ctx.get(event.iterator_code)
            if ctx_value is not None and ctx_value >= event.step_index:
                coerced = self.parameters.sanitize_values(event.updates, ignore_unknown=True)
                effective.update(coerced)
        return effective

    def get_num_rows(self) -> int:
        """Return flattened row count implied by dimensional parameters."""
        dim_names = self.parameters.get_dim_names()
        if not dim_names:
            return 1
        dim_sizes = self.parameters.get_dim_values(dim_names)
        return int(np.prod(dim_sizes))

    def get_effective_parameters_at_step(
        self,
        iterator_code: str | None = None,
        step_index: int | None = None,
    ) -> dict[str, Any]:
        """Get effective parameters at the start of the specified step context."""
        if iterator_code is None and step_index is None:
            return self.get_effective_parameters_for_row(0)
        if iterator_code is None or step_index is None:
            raise ValueError("Both iterator_code and step_index must be provided together.")
        start, _ = self.parameters.get_start_and_end_indices(iterator_code, step_index)
        return self.get_effective_parameters_for_row(start)

    def record_parameter_update(
        self,
        proposal: ParameterProposal,
        iterator_code: str | None = None,
        step_index: int | None = None,
    ) -> ParameterUpdateEvent | None:
        """Record an applied parameter proposal for later reconstruction of effective training rows."""
        if iterator_code is None and step_index is not None:
            raise ValueError("step_index can only be provided with iterator_code.")
        if iterator_code is not None and step_index is None:
            raise ValueError("iterator_code can only be provided with step_index.")
        if not proposal.values:
            return None

        before = self.get_effective_parameters_at_step(iterator_code=iterator_code, step_index=step_index)

        delta = {
            code: value
            for code, value in proposal.values.items()
            if code not in before or before[code] != value
        }
        if not delta:
            return None

        for code in delta:
            obj = self.parameters.get(code)
            if isinstance(obj, DataDomainAxis):
                raise ValueError(f"Recording updates for domain axis parameter '{code}' is not supported.")

        sanitized_delta = self.parameters.sanitize_values(delta, ignore_unknown=False)
        event = ParameterUpdateEvent(
            updates=sanitized_delta,
            iterator_code=iterator_code,
            step_index=step_index,
            source_step=proposal.source_step,
        )
        self.parameter_updates.append(event)
        return event

    # === Helper Methods for Data Access ===

    def set_data(self, values: Any, block_type: BlockType, logger: PfabLogger) -> None:
        """Set values for a block type via its handler."""
        self._block_handler(block_type).set(self, values, logger)

    def get_data_dict(self, block_type: BlockType) -> dict[str, Any]:
        """Get a block type's values as a dict via its handler."""
        return self._block_handler(block_type).get(self)

    def has_data(self, block_type: BlockType) -> bool:
        """Whether a block type has data, via its handler."""
        return self._block_handler(block_type).has(self)

    @staticmethod
    def _block_handler(block_type: BlockType) -> "_BlockHandler":
        handler = _BLOCK_HANDLERS.get(block_type)
        if handler is None:
            raise ValueError(f"Unknown block type: {block_type}")
        return handler
            
    def is_feature_populated(self, feature_name: str) -> bool:
        """Check if a specific feature is populated (not just initialized)."""
        return self.features.is_populated(feature_name)


# ---------------------------------------------------------------------------
# BlockType handlers — one set/get/has strategy per block, keyed in the registry
# ---------------------------------------------------------------------------

class _BlockHandler:
    """Strategy for one BlockType on an ExperimentData (set / get-dict / has)."""

    def set(self, exp: "ExperimentData", values: Any, logger: PfabLogger) -> None:
        raise NotImplementedError

    def get(self, exp: "ExperimentData") -> dict[str, Any]:
        raise NotImplementedError

    def has(self, exp: "ExperimentData") -> bool:
        raise NotImplementedError


class _ParametersHandler(_BlockHandler):
    def set(self, exp, values, logger):
        exp.parameters.set_values_from_dict(values, logger)

    def get(self, exp):
        return exp.parameters.get_values_dict()

    def has(self, exp):
        return bool(exp.parameters.get_values_dict())


class _PerfAttrsHandler(_BlockHandler):
    def set(self, exp, values, logger):
        exp.performance.set_values_from_dict(values, logger)

    def get(self, exp):
        return exp.performance.get_values_dict()

    def has(self, exp):
        return bool(exp.performance.get_values_dict())


class _FeaturesHandler(_BlockHandler):
    def set(self, exp, values, logger):
        exp.features.set_values_from_df(values, logger, parameters=exp.parameters)

    def get(self, exp):
        return exp.features.get_values_dict()

    def has(self, exp):
        return bool(exp.features.get_values_dict())


class _ParamUpdatesHandler(_BlockHandler):
    def set(self, exp, values, logger):
        if isinstance(values, dict):
            events_raw = values.get("events", [])
        elif isinstance(values, list):
            events_raw = values
        else:
            raise TypeError(
                f"Expected list/dict for parameter updates in experiment '{exp.code}', "
                f"got {type(values).__name__}"
            )
        events = [ParameterUpdateEvent.from_dict(v) for v in events_raw]
        exp.parameter_updates = [
            ParameterUpdateEvent(
                updates=exp.parameters.sanitize_values(e.updates, ignore_unknown=True),
                iterator_code=e.iterator_code,
                step_index=e.step_index,
                source_step=e.source_step,
            )
            for e in events
        ]

    def get(self, exp):
        return {"events": [e.to_dict() for e in exp.parameter_updates]}

    def has(self, exp):
        return bool(exp.parameter_updates)


class _MetadataHandler(_BlockHandler):
    def set(self, exp, values, logger):
        if not isinstance(values, dict):
            raise TypeError(
                f"Expected dict for metadata in experiment '{exp.code}', "
                f"got {type(values).__name__}"
            )
        exp.dataset_code = values.get("dataset_code")

    def get(self, exp):
        if exp.dataset_code is None:
            return {}
        return {"dataset_code": exp.dataset_code}

    def has(self, exp):
        return exp.dataset_code is not None


_BLOCK_HANDLERS: dict[BlockType, _BlockHandler] = {
    BlockType.PARAMETERS: _ParametersHandler(),
    BlockType.PARAM_UPDATES: _ParamUpdatesHandler(),
    BlockType.FEATURES: _FeaturesHandler(),
    BlockType.PERF_ATTRS: _PerfAttrsHandler(),
    BlockType.METADATA: _MetadataHandler(),
}


class Dataset:
    """Schema-validated experiment container with hierarchical load/save (memory → local → external)."""

    def __init__(self,
                 schema: DatasetSchema,
                 external_data: IExternalData | None = None,
                 debug_flag: bool = False):
        self.schema = schema
        self.local_data = schema.local_data
        self.external_data = external_data
        self.debug_flag = debug_flag
        self._external_push_made = False  # set in _hierarchical_save, summarised in save_experiments

        # Initialize local data handler and logger
        self.logger = PfabLogger.get_logger(schema.local_data.get_log_folder('logs'))
        
        # Master storage using ExperimentData
        self._experiments: dict[str, ExperimentData] = {}  # exp_code → ExperimentData
        
        # Feature column names
        # self.feature_columns: dict[str, list[str]] | None = None

    def get_experiment(self, exp_code: str) -> ExperimentData:
        """Get complete ExperimentData for an exp_code."""
        if exp_code not in self._experiments:
            raise KeyError(f"Experiment {exp_code} not found")
        return self._experiments[exp_code]

    def get_all_experiments(self) -> list[ExperimentData]:
        """Get list of all ExperimentData objects."""
        return list(self._experiments.values())
    
    # === Create ExperimentData Objects ===

    def _init_from_schema(self, block_class: Any, schema_dict: DataBlock) -> Any:
        block = block_class()
        for _, data_obj in schema_dict.items():
            block.add(data_obj)
        return block
        
    def _create_experiment_shell(
        self,
        exp_code: str,
        dataset_code: str | None = None,
    ) -> ExperimentData:
        """Create new empty experiment shell with all blocks initialized."""
        params_block = self._init_from_schema(Parameters, self.schema.parameters)
        perf_block = self._init_from_schema(PerformanceAttributes, self.schema.performance_attrs)
        arrays_block = self._init_from_schema(Features, self.schema.features)

        return ExperimentData(
            exp_code=exp_code,
            parameters=params_block,
            performance=perf_block,
            features=arrays_block,
            dataset_code=dataset_code,
        )
    
    def _build_experiment_data(
        self,
        exp_code: str,
        parameters: dict[str, Any],
        performance: dict[str, Any] | None,
        metric_arrays: dict[str, np.ndarray] | None,
        parameter_updates: list[dict[str, Any]] | None = None,
        dataset_code: str | None = None,
    ) -> ExperimentData:
        """Build ExperimentData from loaded components."""
        # 1. Create shell with schema structure
        exp_data = self._create_experiment_shell(exp_code, dataset_code=dataset_code)
        
        # 2. Set parameters and dimensions
        exp_data.parameters.set_values_from_dict(parameters, self.logger)

        # 3. Initialize feature arrays based on parameters
        exp_data.features.initialize_arrays(exp_data.parameters)

        # 4. Set optional blocks
        if performance:
            exp_data.set_data(performance, BlockType.PERF_ATTRS, self.logger)

        if metric_arrays:
            exp_data.set_data(metric_arrays, BlockType.FEATURES, self.logger)

        if parameter_updates:
            exp_data.set_data({"events": parameter_updates}, BlockType.PARAM_UPDATES, self.logger)

        # 5. Validate against schema
        exp_data.is_valid(self.schema)

        return exp_data
    
    def create_experiment(
        self,
        exp_code: str,
        parameters: dict[str, Any],
        performance: dict[str, Any] | None = None,
        features: dict[str, np.ndarray] | None = None,
        parameter_updates: list[dict[str, Any]] | None = None,
        dataset_code: str | None = None,
        config_snapshot: dict[str, Any] | None = None,
        recompute: bool = False
    ) -> ExperimentData:
        """Create and register a new experiment; raises ValueError if it already exists and recompute=False.

        ``dataset_code`` (optional) tags the experiment as belonging to a named
        dataset group (baseline / reference / test / etc.). Pred-fab core uses
        it only as opaque metadata; ``DataModule.set_split_dataset`` reads it
        for sugar-style split assignment.
        """
        # Check memory
        if exp_code in self._experiments and not recompute:
            raise ValueError(f"Experiment {exp_code} already exists in memory")

        # Check local storage
        if not recompute:
            # Check if folder exists
            exp_folder = self.local_data.get_experiment_folder(exp_code)
            if os.path.exists(exp_folder):
                 raise ValueError(f"Experiment {exp_code} already exists locally")

        # Build and store
        exp_data = self._build_experiment_data(
            exp_code, parameters, performance, features, parameter_updates,
            dataset_code=dataset_code,
        )
        if config_snapshot:
            exp_data.config_snapshot = config_snapshot
        self._experiments[exp_code] = exp_data
        return exp_data
    
    def add_experiment(self, exp_data: ExperimentData, recompute: bool = False) -> None:
        """Manually add an existing ExperimentData to the dataset."""        
        # Check memory
        if exp_data.code in self._experiments and not recompute:
            raise ValueError(f"Experiment {exp_data.code} already exists in memory")
        
        # Validate against schema
        exp_data.is_valid(self.schema)
        
        # Store
        self._experiments[exp_data.code] = exp_data
    
    def get_experiment_codes(self) -> list[str]:
        """Get list of all experiment codes in dataset."""
        return list(self._experiments.keys())

    def select(self, predicate: Callable[[ExperimentData], bool]) -> list[str]:
        """Experiment codes whose ``ExperimentData`` satisfies ``predicate``, in dataset order.

        The **selection primitive**: a dataset / split is a *query* over experiments,
        not a stored partition. ``dataset_code`` matching, provenance (``source_step``
        on the initial ``parameter_updates`` event), parameter-region, and completeness
        filters are all just predicates over ``ExperimentData``. This is the building
        block the split helpers (``DataModule.set_split_*``) and the first-class dataset
        layer delegate to — see the KB note *First-class dataset concept in pred-fab*.

        Returns codes (the lingua franca of splits and folds); call
        :meth:`get_experiment` for the data behind a code.
        """
        return [code for code, exp in self._experiments.items() if predicate(exp)]

    def list_dataset_codes(self) -> list[str]:
        """Return distinct ``dataset_code`` values across loaded experiments, preserving insertion order."""
        seen: dict[str, None] = {}
        for exp in self._experiments.values():
            if exp.dataset_code is not None and exp.dataset_code not in seen:
                seen[exp.dataset_code] = None
        return list(seen)
    
    def get_experiment_params(self, exp_code: str) -> dict[str, Any]:
        """Get experiment parameters as dictionary."""
        exp_data = self.get_experiment(exp_code)
        params = {}
        for name in exp_data.parameters.keys():
            if exp_data.parameters.has_value(name):
                params[name] = exp_data.parameters.get_value(name)
        return params
    
    def has_experiment(self, exp_code: str) -> bool:
        """Check if experiment exists in dataset."""
        return exp_code in self._experiments

    def get_populated_experiment_codes(self) -> list[str]:
        """Get list of experiments that have all measured features populated.

        Iterator-style positional inputs (``f"{ic}_pos"``) live on Domain,
        not Features — they're not stored, so they aren't checked here.
        """
        feature_names = list(self.schema.features.keys())
        return [
            code for code in self.get_experiment_codes()
            if all(self.get_experiment(code).is_feature_populated(f) for f in feature_names)
        ]

    def state_report(self) -> None:
        """Log an overview of the current dataset to the console."""
        exp_codes = self.get_experiment_codes()
        total = len(exp_codes)

        # Count parameter and performance presence
        count_params = sum(1 for c in exp_codes if self.get_experiment(c).has_data(BlockType.PARAMETERS))
        count_perf = sum(1 for c in exp_codes if self.get_experiment(c).has_data(BlockType.PERF_ATTRS))
        
        # Count completely populated features (using existing helper)
        count_features = len(self.get_populated_experiment_codes())

        summary = [
            f"===== 'Dataset State Report' =====",
            f"\nSchema: \t\t{self.schema.name}",
            f"Experiments: \t\t{total}",
            f"  - Parameters: \t{count_params}/{total}",
            f"  - Features: \t\t{count_features}/{total}",
            f"  - Performance: \t{count_perf}/{total}",
        ]

        self.logger.console_new_line()
        self.logger.console_info("\n".join(summary))
        self.logger.console_new_line()
    
    # === Feature completeness validation ===

    def validate_completeness(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Check whether all experiments have all expected feature values.

        Returns a nested dict: ``{exp_code: {feat_code: {expected, actual, nan, ok}}}``.
        ``expected`` is derived from the experiment's axis sizes and the feature's depth.
        """
        report: dict[str, dict[str, dict[str, Any]]] = {}

        for exp_code in self.get_experiment_codes():
            exp = self.get_experiment(exp_code)
            exp_report: dict[str, dict[str, Any]] = {}

            for feat_code in self.schema.features.keys():
                feat_obj = self.schema.features.get(feat_code)
                if not isinstance(feat_obj, DataArray):
                    continue

                expected = self._expected_feature_count(exp, feat_obj)

                if not exp.features.has(feat_code):
                    exp_report[feat_code] = {"expected": expected, "actual": 0, "nan": 0, "ok": False}
                    continue

                tensor = exp.features.get_value(feat_code)
                flat = np.asarray(tensor).reshape(-1)
                n_nan = int(np.isnan(flat).sum())
                n_actual = len(flat) - n_nan

                exp_report[feat_code] = {
                    "expected": expected,
                    "actual": n_actual,
                    "nan": n_nan,
                    "ok": n_actual == expected and n_nan == 0,
                }

            report[exp_code] = exp_report

        return report

    def is_all_complete(self) -> bool:
        """True if every experiment has all expected feature values with no NaN."""
        report = self.validate_completeness()
        return all(
            entry["ok"]
            for exp in report.values()
            for entry in exp.values()
        )

    def get_incomplete(self) -> list[tuple[str, str]]:
        """Return ``(exp_code, feat_code)`` pairs that are incomplete or have NaN."""
        report = self.validate_completeness()
        return [
            (exp_code, feat_code)
            for exp_code, feats in report.items()
            for feat_code, entry in feats.items()
            if not entry["ok"]
        ]

    def _expected_feature_count(self, exp: ExperimentData, feat_obj: DataArray) -> int:
        """Compute expected number of values for a feature on an experiment.

        Uses the experiment's actual axis sizes (from parameters), not schema max.
        """
        domain_code = feat_obj.domain_code
        depth = feat_obj.feature_depth
        if domain_code is None or depth is None or depth == 0:
            return 1

        domain = self.schema.domains.get(domain_code)
        if domain is None:
            return 1

        params = exp.parameters.get_values_dict()
        count = 1
        for axis in domain.axes[:depth]:
            axis_size = int(params.get(axis.code, 1))
            count *= axis_size
        return count

    # === Helper Methods for Hierarchical Loading/Saving ===

    def _set_exp_data(self, code: str, data: Any, block_type: BlockType) -> None:
        if code in self._experiments:
            exp = self._experiments[code]
            exp.set_data(data, block_type, self.logger)

    def _has_exp_data(self, code: str, block_type: BlockType) -> bool:
        if code not in self._experiments:
            return False
        exp = self._experiments[code]
        return exp.has_data(block_type)
    
    def _get_exp_data(self, code: str, block_type: BlockType) -> Any:
        if code not in self._experiments:
            return None
        exp = self._experiments[code]
        return exp.get_data_dict(block_type)
    
    def _get_exp_feature_array(self, code: str, feature_name: str, block_type: BlockType) -> np.ndarray | None:
        exp = self.get_experiment(code)
        if feature_name not in exp.features.values:
            raise KeyError(f"{block_type} '{feature_name}' not found for experiment '{code}'")
        return exp.features.tensor_to_table(feature_name, exp.features.get_value(feature_name), exp.parameters)
    
    def _get_array_column_names(self, feature_name: str) -> list[str]:
        """Get column names for a specific feature array."""
        if feature_name not in self.schema.features.data_objects:
            raise KeyError(f"Feature '{feature_name}' not found in schema")
        
        return self.schema.features.get(feature_name).columns # type: ignore

    # === Hierarchical Load/Save Methods ===
    
    def populate(
        self,
        source: Loaders = Loaders.LOCAL,
        verbose_flag: bool = False,
        dataset: str | None = None,
        exclude: list[str] | None = ["failed"],
    ) -> int:
        """Load experiments from storage hierarchically.

        When ``dataset`` is provided (e.g. ``"discovery"``), only experiments
        whose code contains that substring are loaded. ``None`` loads all.

        ``exclude`` filters out experiments whose code contains any of the
        given substrings (e.g. ``["failed"]`` skips failed experiments).
        """
        if source != Loaders.LOCAL:
            raise NotImplementedError(f"Only {source.value} source is currently supported")

        local_codes = self.local_data.list_experiments()
        external_codes = (self.external_data.list_codes(dataset=dataset)
                          if self.external_data is not None else [])
        exp_codes = sorted(set(local_codes) | set(external_codes))
        if dataset is not None:
            exp_codes = [c for c in exp_codes if f"/{dataset}/" in c or c.startswith(f"{dataset}/")]
        if exclude:
            exp_codes = [c for c in exp_codes if not any(x in c for x in exclude)]

        missing = self.load_experiments(exp_codes, verbose=verbose_flag)
        loaded_count = len(exp_codes) - len(missing)

        return loaded_count

    def load_experiment(self, exp_code: str, verbose: bool = False) -> ExperimentData:
        """Add experiment by loading it hierarchically."""
        # 1. Hierarchical load
        missing = self.load_experiments([exp_code], verbose=verbose)
        
        if exp_code in missing:
            raise KeyError(f"Experiment {exp_code} could not be loaded")
        
        return self.get_experiment(exp_code)
    
    def load_experiments(self, exp_codes: list[str], recompute_flag: bool = False, verbose: bool = False) -> list[str]:
        """Load multiple experiments using hierarchical pattern with progress tracking."""
        
        self.logger.console_new_line()
        self._logging(f"Loading experiments {exp_codes}...", self.logger.console_execute, verbose)

        # 1. Ensure shells exist
        for code in exp_codes:
            if code not in self._experiments:
                self._experiments[code] = self._create_experiment_shell(code)

        # 2. Load Experiment Parameters
        missing_params = self._hierarchical_load(
            BlockType.PARAMETERS,
            exp_codes,
            loader=self.local_data.load_parameters,
            setter=functools.partial(self._set_exp_data, block_type=BlockType.PARAMETERS),
            in_memory=functools.partial(self._has_exp_data, block_type=BlockType.PARAMETERS),
            external_loader=self.external_data.pull_parameters if self.external_data else None,
            recompute_flag=recompute_flag,
            verbose=verbose
        )
        
        if missing_params and len(self.schema.parameters.data_objects):
            raise ValueError(f"No parameters found for any of the following experiments: {missing_params}")

        # 2b. Load parameter update logs (optional provenance state).
        self._hierarchical_load(
            BlockType.PARAM_UPDATES,
            exp_codes,
            loader=self.local_data.load_parameter_updates,
            setter=functools.partial(self._set_exp_data, block_type=BlockType.PARAM_UPDATES),
            in_memory=functools.partial(self._has_exp_data, block_type=BlockType.PARAM_UPDATES),
            external_loader=self.external_data.pull_parameter_updates if self.external_data else None,
            recompute_flag=recompute_flag,
            verbose=verbose
        )

        # 2c. Load metadata (optional — dataset_code etc.). Missing file is fine.
        self._hierarchical_load(
            BlockType.METADATA,
            exp_codes,
            loader=self.local_data.load_metadata,
            setter=functools.partial(self._set_exp_data, block_type=BlockType.METADATA),
            in_memory=functools.partial(self._has_exp_data, block_type=BlockType.METADATA),
            external_loader=None,
            recompute_flag=recompute_flag,
            verbose=verbose
        )

        # Filter codes that were actually found and validate (parameters are mandatory)
        for code in exp_codes:
            exp_data = self.get_experiment(code)
            exp_data.is_valid(self.schema)

            # Initialize feature arrays based on loaded parameters
            exp_data.features.initialize_arrays(exp_data.parameters, recompute_flag)

        # 3. Load Performance Metrics (side-effecting: populates exp blocks in memory).
        self._hierarchical_load(
            BlockType.PERF_ATTRS, exp_codes,
            loader=self.local_data.load_performance,
            setter=functools.partial(self._set_exp_data, block_type=BlockType.PERF_ATTRS),
            in_memory=functools.partial(self._has_exp_data, block_type=BlockType.PERF_ATTRS),
            external_loader=self.external_data.pull_performance if self.external_data else None,
            recompute_flag=recompute_flag,
            verbose=verbose
        )

        # 4. Load Features (side-effecting per feature; missing sets are not surfaced).
        for name in self.schema.features.keys():
            self._hierarchical_load(
                name, exp_codes,
                loader=self.local_data.load_features,
                setter=functools.partial(self._set_exp_data, block_type=BlockType.FEATURES),
                in_memory=lambda code: self.get_experiment(code).is_feature_populated(name),
                external_loader=self.external_data.pull_features if self.external_data else None,
                recompute_flag=recompute_flag,
                verbose=verbose,
                feature_name=name # Passed to kwargs
            )

        self.logger.console_success(f"Successfully loaded {len(exp_codes)} experiments.")
        self.logger.console_new_line()
        return missing_params
    
    def save_all(self, recompute_flag: bool = False, verbose_flag: bool = False) -> None:
        """Save all experiments currently in memory."""
        exp_codes = list(self._experiments.keys())
        self.save_experiments(exp_codes, recompute=recompute_flag, verbose=verbose_flag)

    def save_experiment(self, exp_code: str, recompute: bool = False, verbose: bool = False) -> None:
        """Save a single experiment hierarchically."""
        self.save_experiments([exp_code], recompute=recompute, verbose=verbose)
    
    def save_experiments(self, exp_codes: list[str], recompute: bool = False, verbose=False) -> None:
        """Save multiple experiments hierarchically with progress tracking."""        
        # Filter to experiments that exist in dataset
        codes_to_save = [code for code in exp_codes if self.has_experiment(code)]

        self._external_push_made = False
        self.logger.console_new_line(force=True)
        self._logging(f"Saving experiments {codes_to_save}...", self.logger.console_execute, verbose)
        
        if not codes_to_save:
            self.logger.console_warning(f"None of {exp_codes} exist in dataset - skipping save operation")
            return
        elif exp_codes != codes_to_save:
            missing = set(exp_codes) - set(codes_to_save)
            self.logger.console_warning(f"Experiments {missing} do not exist in dataset - skipping save for these")
        
        # 1. Save Schema
        self.save_schema(recompute=recompute, verbose=verbose)
        
        # 2. Save Parameters
        self._hierarchical_save(
            BlockType.PARAMETERS, codes_to_save,
            getter=functools.partial(self._get_exp_data, block_type=BlockType.PARAMETERS),
            saver=self.local_data.save_parameters,
            external_saver=self.external_data.push_parameters if self.external_data else None,
            recompute=recompute,
            verbose=verbose
        )

        # 2b. Save parameter update logs.
        self._hierarchical_save(
            BlockType.PARAM_UPDATES, codes_to_save,
            getter=functools.partial(self._get_exp_data, block_type=BlockType.PARAM_UPDATES),
            saver=self.local_data.save_parameter_updates,
            external_saver=self.external_data.push_parameter_updates if self.external_data else None,
            recompute=recompute,
            verbose=verbose
        )

        # 2c. Save metadata (dataset_code etc.). Skipped per-experiment if empty.
        self._hierarchical_save(
            BlockType.METADATA, codes_to_save,
            getter=functools.partial(self._get_exp_data, block_type=BlockType.METADATA),
            saver=self.local_data.save_metadata,
            external_saver=None,
            recompute=recompute,
            verbose=verbose
        )

        # 2d. Save config snapshot (proposal settings at time of creation).
        for code in codes_to_save:
            exp = self.get_experiment(code)
            if exp.config_snapshot:
                path = self.local_data.get_experiment_file_path(code, "config.json")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    json.dump(exp.config_snapshot, f, indent=2)

        # 3. Save Performance
        self._hierarchical_save(
            BlockType.PERF_ATTRS, codes_to_save,
            getter=functools.partial(self._get_exp_data, block_type=BlockType.PERF_ATTRS),
            saver=self.local_data.save_performance,
            external_saver=self.external_data.push_performance if self.external_data else None,
            recompute=recompute,
            verbose=verbose
        )

        # 4. Save Features
        for name in self.schema.features.keys():
            self._hierarchical_save(
                name, codes_to_save,
                getter=functools.partial(self._get_exp_feature_array, feature_name=name, block_type=BlockType.FEATURES),
                saver=self.local_data.save_features,
                external_saver=self.external_data.push_features if self.external_data else None,
                recompute=recompute,
                column_names=self._get_array_column_names(name),
                verbose=verbose,
                feature_name=name # pass to kwargs
            )

        # One console summary for the whole external push (per-block detail is logged).
        if self.external_data is not None and not self.debug_flag:
            self.logger.console_status_clear()
            if self._external_push_made:
                count = len(codes_to_save)
                self.logger.console_pushed(
                    f"Pushed {count} experiment{'s' if count != 1 else ''} to external source."
                )
        self.logger.info(f"Successfully saved experiments {codes_to_save}.")

    def save_schema(self, recompute: bool = False, verbose: bool = True) -> None:
        """Save schema hierarchically."""        
        # 1. Save locally
        saved = self.local_data.save_schema(self.schema.to_dict(), recompute=recompute)
        if saved:
            self._logging(f"Saved dataset schema '{self.schema.name}' as local file.", self.logger.console_saved, verbose)
        else:
            self.logger.info(f"Dataset schema '{self.schema.name}' already exists as local file.")

        # 2. Save externally
        if self.external_data and not self.debug_flag:
            pushed = self.external_data.push_schema(self.schema.name, self.schema.to_dict())
            if pushed:
                self._logging(f"Pushed dataset schema '{self.schema.name}' to external source.", self.logger.console_pushed, verbose)
            else:
                self.logger.info(f"Skipped pushing schema to external source (check ExternalData logic).")
        elif verbose:
            self.logger.info(f"Skipped external push for schema '{self.schema.name}' (debug={self.debug_flag}, has_ext_data={self.external_data is not None}).")
    
    def _check_for_retrieved_codes(self, target_pre: list[str], target_post: list[str], dtype: str, source: Loaders, verbose: bool) -> list[str]:
        """Check which codes were successfully retrieved and log to console."""
        retrieved_codes = [code for code in target_pre if code not in target_post]
        if retrieved_codes:
            message = f"Retrieved from {source.value}: {dtype} for {len(retrieved_codes)} experiments."
            if source == Loaders.MEMORY:
                self.logger.info(message)
            elif source == Loaders.LOCAL:
                self._logging(message, self.logger.console_loaded, verbose)
            elif source == Loaders.EXTERNAL:
                self._logging(message, self.logger.console_pulled, verbose)
            else:
                raise ValueError(f"Unknown Loaders source '{source}', check enum.")
        return retrieved_codes
    
    def _hierarchical_load(self, 
                        dtype: str,
                        target_codes: list[str],
                        loader: Callable[..., tuple[list[str], dict[str, Any]]],
                        setter: Callable[[str, Any], None],
                        in_memory: Callable[[str], bool],
                        external_loader: Callable[..., tuple[list[str], Any]] | None = None,
                        recompute_flag: bool = False,
                        verbose: bool = False,
                        **kwargs) -> list[str]:
        """Universal hierarchical data loading: Memory → Local Files → External Source"""
        # 1. Check memory
        memory_missing = [code for code in target_codes if not in_memory(code)]
        self._check_for_retrieved_codes(target_codes, memory_missing, dtype, Loaders.MEMORY, verbose)

        if not memory_missing:
            return []
        
        # 2. Load from local files
        if not recompute_flag:
            local_missing, local_data = loader(memory_missing, **kwargs)
            # directly store retrieved data in ExpData object
            for code, data in local_data.items():
                setter(code, data)
            self._check_for_retrieved_codes(memory_missing, local_missing, dtype, Loaders.LOCAL, verbose)
        else:
            local_missing = memory_missing
            self.logger.info(f"Recompute mode: Skipping loading {dtype} from local files")

        if not local_missing:
            return []

        # 3. Load from external sources
        if not self.debug_flag and external_loader:
            external_missing, external_data = external_loader(local_missing, **kwargs)
            # directly store retrieved data in ExpData object
            for code, data in external_data.items():
                if isinstance(data, np.ndarray) and "feature_name" in kwargs:
                    col_names = self._get_array_column_names(kwargs["feature_name"])
                    # pandas-stubs types `columns` too narrowly (Axes | None); a
                    # plain list[str] is valid at runtime. False positive.
                    data = pd.DataFrame(data, columns=col_names)  # type: ignore[arg-type]
                setter(code, data)
            self._check_for_retrieved_codes(local_missing, external_missing, dtype, Loaders.EXTERNAL, verbose)
        elif self.debug_flag:
            external_missing = local_missing 
            self.logger.info(f"Debug mode: Skipping loading {dtype} from external source")
        else:
            external_missing = local_missing
            self.logger.warning(f"No external data interface provided: Skipping loading {dtype} from external source")

        return external_missing

    def _hierarchical_save(self, 
                           dtype: str,
                           target_codes: list[str],
                           getter: Callable[[str], Any],
                           saver: Callable[..., bool],
                           external_saver: Callable[..., bool] | None = None,
                           column_names: list[str] | None = None,
                           recompute: bool = False,
                           verbose: bool = True,
                           **kwargs) -> None:
        """Universal hierarchical data saving: Memory → Local Files → External Source"""
        # 1. Filter to codes that exist in memory (and have data)
        data_to_save = {}
        for code in target_codes:
            val = getter(code)
            # Check for non-empty data (handling dicts and arrays)
            if isinstance(val, (dict, list)) and len(val) > 0:
                data_to_save[code] = val
            elif isinstance(val, np.ndarray) and val.size > 0 and not np.all(np.isnan(val[:, -1])):
                data_to_save[code] = val
            elif not isinstance(val, (dict, list, np.ndarray)):
                raise ValueError(f"Unsupported data type for saving: {type(val)}")
            else:
                self.logger.warning(f"No data to save for {dtype} '{code}'")
                    
        if not data_to_save:
            self.logger.info(f"No data in memory ({len(target_codes)} exps): {dtype}")
            return
        
        codes_to_save = list(data_to_save.keys())
        
        # 2. Save to local files
        saved = saver(codes_to_save, data_to_save, recompute, column_names=column_names, **kwargs)
        if saved:
            self._logging(f"Saved to local files: {dtype} for {len(codes_to_save)} experiments.", self.logger.console_saved, verbose)
        else:
            self.logger.info(f"{dtype.capitalize()} already exist as local files.")

        # 3. Save to external source (skip if local didn't write anything new).
        # Live, in-place status per block; save_experiments prints one summary
        # line so a full-dataset push doesn't flood the terminal.
        if not self.debug_flag and external_saver and saved:
            label = str(getattr(dtype, "name", dtype)).lower()
            self.logger.console_status(f"Pushing {label}…")
            pushed = external_saver(codes_to_save, data_to_save, recompute, **kwargs)
            if pushed:
                self._external_push_made = True
                self.logger.info(f"Pushed to external source: {dtype} for {len(codes_to_save)} experiments.")
            else:
                self.logger.info(f"Skipped pushing {dtype} to external source due to missing implementation in ExternalData.")
        elif self.debug_flag:
            self.logger.info(f"Skipped pushing {dtype} to external source due to debug mode.")
        else:
            self.logger.warning(f"Skipped pushing {dtype} to external source due missing ExternalData source.")

    def _build_export_rows(
        self, experiment_codes: list[str], max_depth: int | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[int, int]]]:
        """Shared row builder used by ``export_to_dataframe`` and ``export_to_tensor_dict``.

        . Returns ``(X_rows, y_rows, cell_meta)`` where
        ``cell_meta[i] = (exp_idx, flat_cell_idx)`` for row ``i`` — the
        (experiment-code-index, flat-cell-index) pair used by Phase C SS
        substitution to identify each row's prior-cell coordinates.
        """
        X_rows: list[dict[str, Any]] = []
        y_rows: list[dict[str, Any]] = []
        cell_meta: list[tuple[int, int]] = []

        for exp_idx, code in enumerate(experiment_codes):
            exp_data = self.get_experiment(code)
            if exp_data.features is None:
                continue

            dim_names = exp_data.parameters.get_dim_names()
            if max_depth is not None and len(dim_names) > max_depth:
                dim_names = dim_names[:max_depth]

            if not dim_names:
                # Case 1: scalar experiment (no dimensions).
                y_dict: dict[str, Any] = {}
                for feature_name in exp_data.features.keys():
                    value = exp_data.features.get_value(feature_name)
                    if isinstance(value, np.ndarray):
                        y_dict[feature_name] = float(value.flat[0])
                    else:
                        y_dict[feature_name] = float(value)
                X_rows.append(exp_data.get_effective_parameters_for_row(0))
                y_rows.append(y_dict)
                cell_meta.append((exp_idx, 0))
                continue

            # Case 2: multi-dimensional.
            dim_combinations = exp_data.parameters.get_dim_combinations(dim_names)
            dim_iterators = exp_data.parameters.get_dim_iterator_codes(codes=dim_names)

            # Iterator-style positional inputs are implicit on every Domain.
            # For each axis in the experiment's domain, expose f"{ic}_pos"
            # populated as idx / (size - 1).
            iterator_features: list[tuple[str, str, int]] = []
            for i, dim_name in enumerate(dim_names):
                ic = dim_iterators[i]
                size = int(exp_data.parameters.get_value(dim_name))
                iterator_features.append((f"{ic}_pos", ic, size))

            for row_idx, idx_tuple in enumerate(dim_combinations):
                # Key by both dimension name and iterator code: recorded updates
                # are keyed by name (the row-based path matches on stride name),
                # while the positional columns below look up by iterator code.
                iterator_ctx: dict[str, int] = {}
                for i in range(len(dim_names)):
                    iterator_ctx[dim_names[i]] = idx_tuple[i]
                    iterator_ctx[dim_iterators[i]] = idx_tuple[i]
                row_dict = exp_data.get_effective_parameters_for_context(iterator_ctx)

                for feat_code, axis_code, size in iterator_features:
                    raw_idx = iterator_ctx.get(axis_code)
                    if raw_idx is not None:
                        row_dict[feat_code] = float(raw_idx) / max(size - 1, 1)

                X_rows.append(row_dict)

                y_dict = {}
                for feature_name in exp_data.features.keys():
                    val = exp_data.features.value_at(feature_name, exp_data.parameters, iterator_ctx)
                    if val is not None and not np.isnan(val):
                        y_dict[feature_name] = val

                y_rows.append(y_dict)
                cell_meta.append((exp_idx, row_idx))

        return X_rows, y_rows, cell_meta

    def export_to_dataframe(
        self, experiment_codes: list[str], max_depth: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Export experiments to (X_params, y_features) DataFrames, expanding dimension combinations into rows.

        now a thin wrapper around ``_build_export_rows``;
        the same row-builder backs ``export_to_tensor_dict``.
        """
        if not experiment_codes:
            return pd.DataFrame(), pd.DataFrame()
        X_rows, y_rows, _ = self._build_export_rows(experiment_codes, max_depth=max_depth)
        if not X_rows:
            return pd.DataFrame(), pd.DataFrame()
        return pd.DataFrame(X_rows), pd.DataFrame(y_rows)

    def export_to_tensor_dict(
        self,
        experiment_codes: list[str],
        x_columns: list[str] | None = None,
        y_columns: list[str] | None = None,
        categorical_mappings: dict[str, list[str]] | None = None,
        max_depth: int | None = None,
    ) -> "ExportedTensorDict":
        """Tensor-native export.

        Per-column tensors with dtype-aware encoding (cats → long, numerics →
        float). Returns ``ExportedTensorDict`` with:
          - ``X``: ``dict[col, torch.Tensor]`` of length ``n_rows`` per column
          - ``y``: ``dict[col, torch.Tensor]`` of length ``n_rows`` per column
            (NaN for cells where the feature value is missing)
          - ``cell_meta``: ``(n_rows, 2)`` long tensor — ``[exp_idx, cell_idx]``
            per row, used by Phase C SS substitution

        ``x_columns`` / ``y_columns`` (when supplied) restrict + order the
        output dict keys; ``categorical_mappings`` (parent_col → category
        list) determines which X columns get encoded as long-tensor cat
        indices instead of float values.
        """
        if not experiment_codes:
            return ExportedTensorDict({}, {}, torch.zeros((0, 2), dtype=torch.long))

        X_rows, y_rows, cell_meta_list = self._build_export_rows(experiment_codes, max_depth=max_depth)
        if not X_rows:
            return ExportedTensorDict({}, {}, torch.zeros((0, 2), dtype=torch.long))

        cat_maps = categorical_mappings or {}
        n_rows = len(X_rows)

        def _x_cols() -> list[str]:
            if x_columns is not None:
                return x_columns
            seen: set[str] = set()
            ordered: list[str] = []
            for row in X_rows:
                for k in row.keys():
                    if k not in seen:
                        seen.add(k)
                        ordered.append(k)
            return ordered

        def _y_cols() -> list[str]:
            if y_columns is not None:
                return y_columns
            seen: set[str] = set()
            ordered: list[str] = []
            for row in y_rows:
                for k in row.keys():
                    if k not in seen:
                        seen.add(k)
                        ordered.append(k)
            return ordered

        X_dict: dict[str, torch.Tensor] = {}
        for col in _x_cols():
            if col in cat_maps:
                cats = cat_maps[col]
                cat_to_idx = {c: i for i, c in enumerate(cats)}
                idxs = [cat_to_idx.get(str(row.get(col, "")), 0) for row in X_rows]
                X_dict[col] = torch.tensor(idxs, dtype=torch.long)
            else:
                vals = []
                for row in X_rows:
                    v = row.get(col)
                    vals.append(0.0 if v is None else float(v))
                X_dict[col] = torch.tensor(vals, dtype=torch.float32)

        y_dict_t: dict[str, torch.Tensor] = {}
        for col in _y_cols():
            vals = []
            for row in y_rows:
                v = row.get(col)
                vals.append(float('nan') if v is None else float(v))
            y_dict_t[col] = torch.tensor(vals, dtype=torch.float32)

        cell_meta_tensor = torch.tensor(cell_meta_list, dtype=torch.long) \
            if cell_meta_list else torch.zeros((0, 2), dtype=torch.long)

        return ExportedTensorDict(X_dict, y_dict_t, cell_meta_tensor)

    def _logging(self, msg: str, verbose_func: Callable[[str], None], verbose: bool):
        if verbose:
            verbose_func(msg)
        else:
            self.logger.info(msg)
