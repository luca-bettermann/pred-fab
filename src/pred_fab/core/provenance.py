"""Provenance — the typed, per-experiment reproducibility record.

Provenance answers *how / from what* an experiment was made: its design (:class:`Strategy`),
the generative settings (kappa, seed, bounds, trust, fixed, ...), the fit it was generated
under (its origin set + position), and the schema version. It is a typed view over the
``config_snapshot`` dict pred-fab already stamps and persists (local ``config.json`` +
NocoDB), kept separate from grouping (:class:`ExperimentSet`) per the design — provenance is
reproducibility, sets are grouping. See the KB note *ExperimentSet data model refactor*.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from .experiment_set import ExperimentSet, Fit, Strategy

_STRATEGY_VALUES = {s.value for s in Strategy}


@dataclass
class Provenance:
    """Typed view over a ``config_snapshot`` — round-trips losslessly via to/from_dict.

    No schema version: under one-schema-per-stack (singleton, SSOT in code, snapshot in the
    DB config record) the schema is constant within a study, so per-experiment versioning is
    redundant — see the KB note *ExperimentSet data model refactor* → studies & schema.
    """
    strategy: Strategy | None = None
    seed: int | None = None
    settings: dict[str, Any] = field(default_factory=dict)   # generative knobs
    origin: tuple[str, int | None] | None = None             # (origin set code, position)

    @classmethod
    def from_dict(cls, snapshot: dict[str, Any] | None) -> "Provenance":
        """Parse a ``config_snapshot`` dict into a typed Provenance.

        Lossless: a ``design`` not in :class:`Strategy` or a malformed ``origin``
        is preserved verbatim in ``settings`` (and warned about) rather than
        silently dropped, so ``to_dict(from_dict(x)) == x``.
        """
        d = dict(snapshot or {})
        seed = d.pop("seed", None)

        # design → strategy; keep an unrecognised value in settings (lossless).
        strategy: Strategy | None = None
        design = d.get("design", None)
        if design is not None:
            if design in _STRATEGY_VALUES:
                strategy = Strategy(design)
                d.pop("design")
            else:
                warnings.warn(
                    f"Provenance: unknown design {design!r} not in Strategy; "
                    f"preserved in settings.",
                    stacklevel=2,
                )

        # origin → typed tuple; keep a malformed value in settings (lossless).
        origin_t: tuple[str, int | None] | None = None
        origin = d.get("origin", None)
        if origin is not None:
            if isinstance(origin, (list, tuple)) and len(origin) == 2:
                origin_t = (str(origin[0]), origin[1])
                d.pop("origin")
            else:
                warnings.warn(
                    f"Provenance: malformed origin {origin!r}; preserved in settings.",
                    stacklevel=2,
                )

        return cls(strategy=strategy, seed=seed, settings=d, origin=origin_t)

    def to_dict(self) -> dict[str, Any]:
        """Reconstruct the flat ``config_snapshot`` dict (settings + the typed top-level keys)."""
        out: dict[str, Any] = dict(self.settings)
        if self.strategy is not None:
            out["design"] = self.strategy.value
        if self.seed is not None:
            out["seed"] = self.seed
        if self.origin is not None:
            out["origin"] = list(self.origin)
        return out

    def fit(self, origin_set: ExperimentSet) -> Fit:
        """The fit this experiment was generated under: the origin set windowed to the
        member's position (whole for a batch origin) plus its parent chain. ``origin_set`` is
        the resolved :class:`ExperimentSet` named by ``origin`` — looked up by the caller."""
        position = self.origin[1] if self.origin is not None else None
        return Fit.of(origin_set, window=position)
