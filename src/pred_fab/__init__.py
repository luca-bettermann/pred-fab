"""pred-fab — a predictive layer for digital fabrication systems.

The top-level convenience names (`PfabAgent`, the interfaces, …) pull the ML stack
(torch), so they are imported **lazily** via PEP 562 ``__getattr__``: ``import pred_fab``
and ``from pred_fab.core import …`` (the torch-free model + traversal surface) do not pull
torch/pandas; touching ``pred_fab.PfabAgent`` loads the ML stack on demand. ML deps live in
the ``pred-fab[ml]`` extra.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# name -> (module, attribute) — loaded on first access (keeps the ML stack out of import).
_LAZY: dict[str, tuple[str, str]] = {
    "PfabAgent": ("pred_fab.orchestration", "PfabAgent"),
    "InferenceBundle": ("pred_fab.orchestration", "InferenceBundle"),
    "IExternalData": ("pred_fab.interfaces", "IExternalData"),
    "IFeatureModel": ("pred_fab.interfaces", "IFeatureModel"),
    "IEvaluationModel": ("pred_fab.interfaces", "IEvaluationModel"),
    "IPredictionModel": ("pred_fab.interfaces", "IPredictionModel"),
    "DeterministicModel": ("pred_fab.interfaces", "DeterministicModel"),
    "combined_score": ("pred_fab.utils.metrics", "combined_score"),
}

__all__ = [
    "PfabAgent",
    "InferenceBundle",
    "IExternalData",
    "IFeatureModel",
    "IEvaluationModel",
    "IPredictionModel",
    "DeterministicModel",
    "combined_score",
]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # static resolution without importing at runtime
    from .interfaces import (
        DeterministicModel,
        IEvaluationModel,
        IExternalData,
        IFeatureModel,
        IPredictionModel,
    )
    from .orchestration import InferenceBundle, PfabAgent
    from .utils.metrics import combined_score
