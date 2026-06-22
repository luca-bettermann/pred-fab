"""Storage port — the contract ``Dataset`` needs from an external-data backend.

``core`` owns the *requirement* (what the data layer asks of storage); the
concrete implementations live outside (``interfaces.IExternalData`` and the
lbp/nocodb adapters) and structurally satisfy this ``Protocol``. Defining the
port here keeps ``core`` the dependency root: it names what it needs without
importing upward into ``interfaces`` (resolves the former core↔interfaces
import cycle — see [[PFAB - Repo Strategy]]).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
if TYPE_CHECKING:
    import pandas as pd  # annotation-only (PEP 563) — port stays pandas-free at import


@runtime_checkable
class ExternalDataPort(Protocol):
    """The pull/push surface ``Dataset`` calls on an external-data backend."""

    def list_codes(self, dataset: str | None = None) -> list[str]: ...

    def pull_parameters(
        self, exp_codes: list[str]
    ) -> tuple[list[str], dict[str, dict[str, Any]]]: ...

    def pull_parameter_updates(
        self, exp_codes: list[str]
    ) -> tuple[list[str], dict[str, Any]]: ...

    def pull_performance(
        self, exp_codes: list[str]
    ) -> tuple[list[str], dict[str, dict[str, Any]]]: ...

    def pull_features(
        self, exp_codes: list[str], feature_name: str = "default", **kwargs: Any
    ) -> tuple[list[str], dict[str, np.ndarray | pd.DataFrame]]: ...

    def pull_provenance(
        self, exp_codes: list[str]
    ) -> tuple[list[str], dict[str, dict[str, Any]]]: ...

    def pull_experiment_sets(self) -> list[dict[str, Any]]: ...

    def push_parameters(
        self, exp_codes: list[str], parameters: dict[str, dict[str, Any]], recompute: bool = False
    ) -> bool: ...

    def push_parameter_updates(
        self, exp_codes: list[str], updates: dict[str, dict[str, Any]],
        recompute: bool = False, **kwargs: Any,
    ) -> bool: ...

    def push_performance(
        self, exp_codes: list[str], performance: dict[str, dict[str, Any]], recompute: bool = False
    ) -> bool: ...

    def push_provenance(
        self, exp_codes: list[str], provenance: dict[str, dict[str, Any]], recompute: bool = False
    ) -> bool: ...

    def push_features(
        self, exp_codes: list[str], features: dict[str, np.ndarray],
        recompute: bool = False, feature_name: str = "default", **kwargs: Any,
    ) -> bool: ...

    def push_experiment_sets(self, sets: list[dict[str, Any]], recompute: bool = False) -> bool: ...

    def push_schema(self, schema_id: str, schema_data: dict[str, Any]) -> bool: ...
