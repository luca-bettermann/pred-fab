"""Core data model + ML data-prep.

The model surface (types + `Dataset`/`ExperimentData` + the per-position traversal) imports
torch/pandas-free — torch/pandas are confined to the export/ML methods (lazy). `DataModule`
is the one genuinely torch-bound member, so it's loaded **lazily** via PEP 562 ``__getattr__``
to keep ``from pred_fab.core import …`` torch-free for model-only consumers.
"""
from typing import TYPE_CHECKING

from .data_objects import (
    DataObject,
    DataReal,
    DataInt,
    DataBool,
    DataCategorical,
    DataDomainAxis,
    DataArray,
    Dimension,
    Domain,
    Parameter,
    Feature,
    PerformanceAttribute
)

from .data_blocks import (
    DataBlock,
    Parameters,
    PerformanceAttributes,
    Features,
    Domains,
)

from .schema import DatasetSchema, SchemaRegistry
from .dataset import (
    Dataset,
    ExperimentData,
    ParameterProposal,
    ParameterUpdateEvent,
    ParameterTrajectory,
    ExperimentSpec,
    events_to_trajectory,
    trajectory_to_events,
)
from .experiment_set import ExperimentSet, FitPart, Fit
from .provenance import Provenance


def __getattr__(name: str):
    """Lazily expose `DataModule` (torch-bound) without pulling torch at package import."""
    if name == "DataModule":
        from .datamodule import DataModule
        return DataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .datamodule import DataModule

__all__ = [
    'ExperimentSet',
    'FitPart',
    'Fit',
    'Provenance',
    'DataObject',
    'DataReal',
    'DataInt',
    'DataBool',
    'DataCategorical',
    'DataDomainAxis',
    'DataArray',
    'Dimension',
    'Domain',

    'Parameter',
    'Feature',
    'PerformanceAttribute',

    'DataBlock',
    'Parameters',
    'PerformanceAttributes',
    'Features',
    'Domains',

    'DatasetSchema',
    'SchemaRegistry',
    'Dataset',
    'ExperimentData',
    'DataModule',
    'ParameterProposal',
    'ParameterUpdateEvent',
    'ParameterTrajectory',
    'ExperimentSpec',
]
