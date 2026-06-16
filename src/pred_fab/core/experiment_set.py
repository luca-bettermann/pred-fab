"""ExperimentSet — a named group of experiments, and the Fit that composes groups
into a model's training set.

The grouping primitive of the first-class data model — distinct from the ``Dataset``
*container* of all experiments. An ``ExperimentSet`` names *which* experiments belong
together: a discovery batch (unordered), a sequential exploration run (ordered). A ``Fit``
composes sets (each optionally windowed) into the experiment codes a model trained on,
so "evaluate at stage k" is ``Fit([base, run.window(k)]).experiment_codes()``.

How an experiment was *generated* (κ, the source method) is flat per-experiment provenance
(see :class:`Provenance` / ``ExperimentData.source`` / ``.kappa``), never carried on the
set — the set is just membership + order. Pure objects, no persistence, no model. See the
KB note *ExperimentSet data model refactor*.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentSet:
    """A named group of experiment codes, optionally ordered.

    ``members`` are experiment codes (in sequence if ``ordered``). ``ordered=True`` means the
    sequence carries meaning (an exploration run) so ``window`` / stage-k slicing apply;
    ``False`` is a batch (discovery / sobol / grid). Membership is many-to-many at the
    storage layer; this object is one group. Generation metadata lives in each member's
    provenance, not here.
    """
    code: str
    members: list[str] = field(default_factory=list)
    ordered: bool = False

    def codes(self) -> list[str]:
        """Member experiment codes (in order if ordered)."""
        return list(self.members)

    def window(self, k: int) -> ExperimentSet:
        """The first ``k`` members as an ordered sub-set. Requires an ordered set."""
        self._require_ordered("window")
        return ExperimentSet(code=f"{self.code}[:{k}]", members=self.members[:k], ordered=True)

    def fit_at(self, index: int) -> Fit:
        """The within-set prefix before member ``index`` (0-based) — what this set's own
        sequence had produced before that member. Requires an ordered set. Compose any base
        sets explicitly (``Fit([base, run.window(index)])``) if the run built on them."""
        self._require_ordered("fit_at")
        return Fit.of(self, window=index)

    def _require_ordered(self, op: str) -> None:
        if not self.ordered:
            raise ValueError(f"{op} requires an ordered set; '{self.code}' is a batch.")

    def __len__(self) -> int:
        return len(self.members)

    def __contains__(self, code: object) -> bool:
        return code in self.members

    def __iter__(self):
        return iter(self.members)

    def to_dict(self) -> dict:
        return {"code": self.code, "members": list(self.members), "ordered": bool(self.ordered)}

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentSet:
        return cls(
            code=d["code"], members=list(d.get("members", [])), ordered=bool(d.get("ordered", False)),
        )


@dataclass
class FitPart:
    """One set's contribution to a fit — windowed (first ``window`` members) where ordered."""
    experiment_set: ExperimentSet
    window: int | None = None

    def codes(self) -> list[str]:
        codes = self.experiment_set.codes()
        return codes if self.window is None else codes[: self.window]


@dataclass
class Fit:
    """What a model trains on — a composition of ``(set, window?)`` parts.

    ``experiment_codes`` resolves the parts to the deduped union of their (windowed) members.
    Compose several sets directly — ``Fit([FitPart(disc), FitPart(expl, window=k)])`` for
    ``disc ∪ expl[:k]``; ``Fit.of`` is the single-set shorthand.
    """
    parts: list[FitPart] = field(default_factory=list)

    @classmethod
    def of(cls, eset: ExperimentSet, window: int | None = None) -> Fit:
        """A single-set fit: ``eset`` windowed to ``window`` (whole if ``None``)."""
        return cls([FitPart(eset, window=window)])

    def experiment_codes(self) -> list[str]:
        """Union of every part's (windowed) members — deduped, parts in order."""
        seen: dict[str, None] = {}
        for part in self.parts:
            for c in part.codes():
                seen.setdefault(c, None)
        return list(seen)
