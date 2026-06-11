"""Provenance — the typed, per-experiment reproducibility record.

Provenance answers *how / from what* an experiment was made: its design (:class:`Strategy`),
the generative settings (kappa, seed, bounds, trust, fixed, ...), the fit it was generated
under (its origin set + position), and the schema version. It is a typed view over the
``config_snapshot`` dict pred-fab already stamps and persists (local ``config.json`` +
NocoDB), kept separate from grouping (:class:`ExperimentSet`) per the design — provenance is
reproducibility, sets are grouping. See the KB note *ExperimentSet data model refactor*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .experiment_set import ExperimentSet, Fit, Strategy

_STRATEGY_VALUES = {s.value for s in Strategy}


@dataclass
class Provenance:
    """Typed view over a ``config_snapshot`` — round-trips losslessly via to/from_dict."""
    strategy: Strategy | None = None
    seed: int | None = None
    settings: dict[str, Any] = field(default_factory=dict)   # generative knobs
    origin: tuple[str, int | None] | None = None             # (origin set code, position)
    schema_version: str | None = None

    @classmethod
    def from_dict(cls, snapshot: dict[str, Any] | None) -> "Provenance":
        """Parse a ``config_snapshot`` dict into a typed Provenance."""
        d = dict(snapshot or {})
        design = d.pop("design", None)
        seed = d.pop("seed", None)
        origin = d.pop("origin", None)
        schema_version = d.pop("schema_version", None)
        strategy = Strategy(design) if design in _STRATEGY_VALUES else None
        origin_t: tuple[str, int | None] | None = None
        if isinstance(origin, (list, tuple)) and len(origin) == 2:
            origin_t = (str(origin[0]), origin[1])
        return cls(
            strategy=strategy, seed=seed, settings=d,
            origin=origin_t, schema_version=schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Reconstruct the flat ``config_snapshot`` dict (settings + the typed top-level keys)."""
        out: dict[str, Any] = dict(self.settings)
        if self.strategy is not None:
            out["design"] = self.strategy.value
        if self.seed is not None:
            out["seed"] = self.seed
        if self.origin is not None:
            out["origin"] = list(self.origin)
        if self.schema_version is not None:
            out["schema_version"] = self.schema_version
        return out

    def fit(self, origin_set: ExperimentSet) -> Fit:
        """The fit this experiment was generated under: the origin set windowed to the
        member's position (whole for a batch origin) plus its parent chain. ``origin_set`` is
        the resolved :class:`ExperimentSet` named by ``origin`` — looked up by the caller."""
        position = self.origin[1] if self.origin is not None else None
        return Fit.of(origin_set, window=position)
