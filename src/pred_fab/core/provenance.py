"""Provenance — the typed, per-experiment reproducibility record.

Provenance answers *how / from what* an experiment was made: its source method
(:class:`SourceStep`), the generative settings (kappa, seed, bounds, trust, fixed, ...),
and the fit it was generated under (origin set + position). A typed view over the
``config_snapshot`` dict pred-fab stamps and persists (local ``config.json`` + NocoDB),
kept separate from grouping (:class:`ExperimentSet`) — provenance is reproducibility, sets
are grouping. See the KB note *ExperimentSet data model refactor*.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from ..utils.enum import SourceStep
from .experiment_set import ExperimentSet, Fit

_SOURCE_VALUES = {s.value for s in SourceStep}


@dataclass
class Provenance:
    """Typed view over a ``config_snapshot`` — round-trips losslessly via to/from_dict.

    No schema version: under one-schema-per-stack the schema is constant within a study, so
    per-experiment versioning is redundant — see the KB note *ExperimentSet data model
    refactor* → studies & schema. ``kappa`` and the other generative knobs live in
    ``settings``; ``source`` is the typed classifier.
    """
    source: SourceStep | None = None
    seed: int | None = None
    settings: dict[str, Any] = field(default_factory=dict)   # generative knobs incl. kappa
    origin: tuple[str, int | None] | None = None             # (origin set code, position)

    @classmethod
    def from_dict(cls, snapshot: dict[str, Any] | None) -> "Provenance":
        """Parse a ``config_snapshot`` dict into a typed Provenance.

        Lossless: a ``source`` not in :class:`SourceStep` or a malformed ``origin``
        is preserved verbatim in ``settings`` (and warned about) rather than silently
        dropped, so ``to_dict(from_dict(x)) == x``.
        """
        d = dict(snapshot or {})
        seed = d.pop("seed", None)

        # source → SourceStep; keep an unrecognised value in settings (lossless).
        source: SourceStep | None = None
        raw_source = d.get("source", None)
        if raw_source is not None:
            if raw_source in _SOURCE_VALUES:
                source = SourceStep(raw_source)
                d.pop("source")
            else:
                warnings.warn(
                    f"Provenance: unknown source {raw_source!r} not in SourceStep; "
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

        return cls(source=source, seed=seed, settings=d, origin=origin_t)

    def to_dict(self) -> dict[str, Any]:
        """Reconstruct the flat ``config_snapshot`` dict (settings + the typed top-level keys)."""
        out: dict[str, Any] = dict(self.settings)
        if self.source is not None:
            out["source"] = self.source.value
        if self.seed is not None:
            out["seed"] = self.seed
        if self.origin is not None:
            out["origin"] = list(self.origin)
        return out

    def fit(self, origin_set: ExperimentSet) -> Fit:
        """The within-origin-set prefix this experiment was generated under (whole for a batch
        origin). ``origin_set`` is the resolved :class:`ExperimentSet` named by ``origin`` —
        looked up by the caller. Compose any base sets explicitly if the run used them."""
        position = self.origin[1] if self.origin is not None else None
        return Fit.of(origin_set, window=position)
