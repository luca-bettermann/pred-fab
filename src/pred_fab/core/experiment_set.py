"""ExperimentSet — a named, strategy-tagged group of experiments, and the Fit that
composes such groups into a model's training set.

The grouping primitive of the first-class data model — distinct from the ``Dataset``
*container* of all experiments. An ``ExperimentSet`` names *which* experiments belong
together: a discovery batch, a sequential exploration run, a Sobol probe. A ``Fit``
composes sets (each optionally windowed) into the experiment codes a model trained on,
so "evaluate the model at stage k" is ``Fit.of(run, window=k).experiment_codes()`` and the
nested-supersets of an acquisition loop collapse to *one ordered set + an index*.

Pure objects — no persistence, no model. See the KB note *ExperimentSet data model refactor*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Strategy(str, Enum):
    """How an ExperimentSet was generated — the design (the queryable provenance axis).

    Ordered strategies (exploration, inference) carry a sequence + window; batch strategies
    (discovery, sobol, grid, adaptation) are always used whole.
    """
    DISCOVERY = "discovery"
    EXPLORATION = "exploration"
    INFERENCE = "inference"
    SOBOL = "sobol"
    GRID = "grid"
    ADAPTATION = "adaptation"

    @property
    def ordered(self) -> bool:
        """Whether sets of this strategy are sequential (carry order + window) by default."""
        return self in (Strategy.EXPLORATION, Strategy.INFERENCE)


@dataclass
class ExperimentSet:
    """A named group of experiments with a strategy and optional structure.

    ``members`` are experiment codes (in sequence if ``ordered``). ``parent`` is the only
    inter-set link — an exploration run points at the discovery set it was trained on top
    of. Membership is many-to-many at the storage layer; this object is one group.
    """
    code: str
    strategy: Strategy
    members: list[str] = field(default_factory=list)
    ordered: bool | None = None          # None → default from the strategy
    parent: ExperimentSet | None = None

    def __post_init__(self) -> None:
        if self.ordered is None:
            self.ordered = self.strategy.ordered

    def codes(self) -> list[str]:
        """Member experiment codes (in order if ordered)."""
        return list(self.members)

    def window(self, k: int) -> ExperimentSet:
        """The first ``k`` members as a sub-set (ordered). Requires an ordered set."""
        self._require_ordered("window")
        return ExperimentSet(
            code=f"{self.code}[:{k}]", strategy=self.strategy,
            members=self.members[:k], ordered=True, parent=self.parent,
        )

    def fit_at(self, index: int) -> Fit:
        """The fit that *generated* the member at ``index`` (0-based): the parent chain
        (whole) plus this set's first ``index`` members — everything the proposing model
        had seen before that member. Requires an ordered set."""
        self._require_ordered("fit_at")
        return Fit.of(self, window=index)

    def _require_ordered(self, op: str) -> None:
        if not self.ordered:
            raise ValueError(
                f"{op} requires an ordered set; '{self.code}' ({self.strategy.value}) is a batch."
            )

    def __len__(self) -> int:
        return len(self.members)

    def __contains__(self, code: object) -> bool:
        return code in self.members

    def __iter__(self):
        return iter(self.members)

    def to_dict(self) -> dict:
        """Serialize (``parent`` as its code — the registry resolves the link on load)."""
        return {
            "code": self.code,
            "strategy": self.strategy.value,
            "members": list(self.members),
            "ordered": bool(self.ordered),
            "parent": self.parent.code if self.parent is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict, parent: ExperimentSet | None = None) -> ExperimentSet:
        """Deserialize one set; ``parent`` is the resolved set named by ``d['parent']``."""
        return cls(
            code=d["code"], strategy=Strategy(d["strategy"]),
            members=list(d.get("members", [])), ordered=d.get("ordered"), parent=parent,
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

    ``experiment_codes`` resolves the parts to the union of their (windowed) members — the
    training set. ``Fit.of`` builds the canonical fit for a set: its whole parent chain plus
    the set itself (windowed), e.g. ``discovery ∪ exploration[:k]``. Feed
    ``experiment_codes()`` to ``DataModule.set_split_codes`` to train/evaluate at any stage.
    """
    parts: list[FitPart] = field(default_factory=list)

    @classmethod
    def of(cls, eset: ExperimentSet, window: int | None = None) -> Fit:
        chain: list[ExperimentSet] = []
        p = eset.parent
        while p is not None:
            chain.append(p)
            p = p.parent
        parts = [FitPart(s) for s in reversed(chain)]   # root ancestor (e.g. discovery) first
        parts.append(FitPart(eset, window=window))
        return cls(parts)

    def experiment_codes(self) -> list[str]:
        """Union of every part's (windowed) members — deduped, stable order (parents first)."""
        seen: dict[str, None] = {}
        for part in self.parts:
            for c in part.codes():
                seen.setdefault(c, None)
        return list(seen)
