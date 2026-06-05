"""Cross-validation instrument — the free, re-fit generalisation probe.

CV is the cheap evaluation instrument of the first-class dataset design: it reads the
model's **error field** where coverage is *high* (the active-learning loop already put
dense data in the operating region), complementing the paid Sobol probe that guards the
*unexplored* domain. See the KB note *First-class dataset concept in pred-fab*.

Mechanics (per that note):
  * Folds are taken at the **experiment level** — a whole experiment (all its rows /
    layers) is held out together, never split, or its own rows would leak into training.
  * Each fold trains a **fresh model + normaliser**, built from the deployed model
    *factories* (the ``(class, kwargs)`` specs), not the trained instances — so the
    held-out estimate is honest.
  * The **deployed model still fits on all data**: this is a re-fit diagnostic that never
    touches it (every fold builds a throwaway :class:`PredictionSystem`).
  * Output is the **located error field** — held-out error per experiment, tagged with
    that experiment's parameters (its position in the space) — not just an aggregate.
    That field is what the coverage-gated signal reads against the coverage field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core import DataModule, Dataset, DatasetSchema
from ..utils import LocalData, PfabLogger
from .prediction import PredictionSystem


def make_experiment_folds(
    codes: list[str], k: int | None = None, seed: int = 0,
) -> list[list[str]]:
    """Partition experiment ``codes`` into held-out **test** groups.

    ``k=None`` (or ``k >= n``) → leave-one-out: one experiment per fold (no shuffle —
    order is irrelevant). Otherwise → ``k`` balanced folds by round-robin over a
    deterministic shuffle. The *train* set for each fold is every code not in it.
    Folds are experiment-level; this never splits an experiment's rows.
    """
    n = len(codes)
    if n == 0:
        return []
    if k is None or k >= n:
        return [[c] for c in codes]
    k = max(2, k)
    order = np.random.RandomState(seed).permutation(n)
    folds: list[list[str]] = [[] for _ in range(k)]
    for pos, idx in enumerate(order):
        folds[pos % k].append(codes[int(idx)])
    return [f for f in folds if f]


@dataclass
class HeldOutError:
    """One held-out experiment's prediction error, located by its parameters.

    ``metrics`` is the per-feature ``{r2, mae, n_samples, [r2_inf]}`` from validating a
    fold model (which never saw this experiment) against it; ``params`` is the
    experiment's position in the parameter space — the field's coordinate.
    """
    exp_code: str
    params: dict[str, Any]
    metrics: dict[str, dict[str, float]]


@dataclass
class CVResult:
    """The located error field plus convenience reads over it."""
    held_out: list[HeldOutError]
    mode: str
    n_folds: int

    def features(self) -> list[str]:
        """Feature codes present in the field (insertion order across experiments)."""
        seen: dict[str, None] = {}
        for h in self.held_out:
            for feat in h.metrics:
                seen.setdefault(feat, None)
        return list(seen)

    def error_field(
        self, feature: str, metric: str = "mae",
    ) -> list[tuple[dict[str, Any], float]]:
        """``(params, error)`` pairs for one feature — the field read against coverage.

        Each pair locates a held-out error in parameter space. The coverage-gated signal
        (next slice) reads high error at high coverage as a *model* problem, high error at
        low coverage as an *acquisition* problem.
        """
        out: list[tuple[dict[str, Any], float]] = []
        for h in self.held_out:
            m = h.metrics.get(feature)
            if m is not None and metric in m:
                out.append((h.params, float(m[metric])))
        return out

    def aggregate(self, metric: str = "mae") -> dict[str, dict[str, float]]:
        """Per-feature sample-weighted mean of ``metric`` across held-out experiments.

        ``mae`` is poolable (a sample-weighted mean is a valid CV estimate); ``r2`` is
        not (a mean of per-experiment R² is not a CV R²), so prefer ``mae`` here and read
        ``r2`` per experiment from the field for diagnostics.
        """
        acc: dict[str, list[tuple[float, float]]] = {}
        for h in self.held_out:
            for feat, m in h.metrics.items():
                if metric in m:
                    w = float(m.get("n_samples", 1.0))
                    acc.setdefault(feat, []).append((float(m[metric]), w))
        out: dict[str, dict[str, float]] = {}
        for feat, pairs in acc.items():
            tw = sum(w for _, w in pairs)
            mean = sum(v * w for v, w in pairs) / tw if tw > 0 else float("nan")
            out[feat] = {
                f"{metric}_mean": mean,
                "n_experiments": float(len(pairs)),
                "n_samples": tw,
            }
        return out


class CrossValidator:
    """Runs experiment-level CV from model *factories*, building a fresh system per fold.

    Construct from the deployed pieces (schema, storage, the prediction-model specs) — or
    :meth:`from_agent` to lift them off a configured agent. ``run`` returns the located
    error field; nothing here mutates the deployed model.
    """

    def __init__(
        self,
        schema: DatasetSchema,
        local_data: LocalData,
        logger: PfabLogger,
        prediction_specs: list[tuple[type, dict]],
        eval_system: Any | None = None,
    ):
        if not prediction_specs:
            raise ValueError("CrossValidator needs at least one prediction-model spec (factory).")
        self.schema = schema
        self.local_data = local_data
        self.logger = logger
        self.prediction_specs = prediction_specs
        self.eval_system = eval_system

    @classmethod
    def from_agent(cls, agent: Any) -> "CrossValidator":
        """Lift the prediction-model factories and context off a configured agent."""
        eval_system = getattr(agent, "eval_system", None)
        if eval_system is not None and not getattr(eval_system, "is_initialized", False):
            eval_system = None
        return cls(
            schema=agent.schema,
            local_data=agent.local_data,
            logger=agent.logger,
            prediction_specs=agent._prediction_model_specs,
            eval_system=eval_system,
        )

    def _fresh_system(self) -> PredictionSystem:
        """A throwaway PredictionSystem with fresh, untrained model instances."""
        system = PredictionSystem(logger=self.logger, schema=self.schema, local_data=self.local_data)
        for model_class, kwargs in self.prediction_specs:
            system.models.append(model_class(logger=self.logger, **kwargs))
        system.set_ref_objects(self.schema)
        system._initialized = True
        return system

    def run(
        self,
        dataset: Dataset,
        base_datamodule: DataModule,
        *,
        k: int | None = None,
        seed: int = 0,
        codes: list[str] | None = None,
        verbose: bool = False,
    ) -> CVResult:
        """Cross-validate over populated experiments and return the located error field.

        ``k=None`` → leave-one-out (one experiment per fold — gives the field at point
        granularity); else ``k``-fold. ``base_datamodule`` supplies the input/output
        column configuration; each fold deep-copies it, re-points the splits, and re-fits
        normalisation on its own training set. ``codes`` restricts the experiment set
        (default: all populated experiments).
        """
        all_codes = codes if codes is not None else dataset.get_populated_experiment_codes()
        if len(all_codes) < 2:
            raise ValueError(
                f"Cross-validation needs >= 2 populated experiments, got {len(all_codes)}."
            )
        folds = make_experiment_folds(all_codes, k=k, seed=seed)

        held: list[HeldOutError] = []
        prev_console = self.logger._console_output_enabled
        if not verbose:
            self.logger.set_console_output(False)  # one diagnostic, not N metric tables
        try:
            for test_codes in folds:
                test_set = set(test_codes)
                train_codes = [c for c in all_codes if c not in test_set]
                if not train_codes:
                    continue
                dm = base_datamodule.copy()
                dm.set_split_codes(train_codes, [], test_codes)
                system = self._fresh_system()
                system.train(dm)  # re-fits normaliser on this fold's train split
                for exp_code in test_codes:
                    # Locate each held-out experiment individually (point granularity for
                    # the field; for LOO this is the fold itself). No retrain — only the
                    # test split moves; the fold model already excluded all of test_codes.
                    dm.set_split_codes(train_codes, [], [exp_code])
                    metrics = system.validate(use_test=True, eval_system=self.eval_system)
                    held.append(HeldOutError(
                        exp_code=exp_code,
                        params=dataset.get_experiment_params(exp_code),
                        metrics=metrics,
                    ))
        finally:
            self.logger.set_console_output(prev_console)

        mode = "loo" if (k is None or k >= len(all_codes)) else f"{len(folds)}-fold"
        self.logger.info(
            f"Cross-validation ({mode}): {len(held)} held-out experiments across {len(folds)} folds."
        )
        return CVResult(held_out=held, mode=mode, n_folds=len(folds))
