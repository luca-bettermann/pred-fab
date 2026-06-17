"""Shared input-transform decisions for the model input pipeline.

Single home for the parts of "raw values → model input" that **must** be
applied identically at training time (`DataModule` / `PredictionSystem`) and
at inference time (`InferenceBundle`): which columns a model receives and how
categoricals are index-encoded. The inference bundle is deliberately
dependency-free of `Dataset`/training, so these live in `core` (depending on
nothing heavier) rather than on `DataModule` — that way the bundle reuses the
exact same decisions instead of re-deriving them and drifting (the source of
the per-model-slice and silent-categorical bugs).

Backend-specific array assembly (numpy vs torch stacking) stays with each
caller; only the *decisions* are shared here.
"""
from __future__ import annotations

from typing import Any


def column_indices(
    input_columns: list[str],
    codes: list[str],
    *,
    skip_missing: bool = False,
) -> list[int]:
    """Indices into ``input_columns`` for ``codes``, in the order given.

    Each schema code maps to exactly one column. Raises on a missing code
    unless ``skip_missing`` — so a model never silently receives the wrong
    columns.
    """
    idx_of = {c: i for i, c in enumerate(input_columns)}
    indices: list[int] = []
    for code in codes:
        if code in idx_of:
            indices.append(idx_of[code])
        elif not skip_missing:
            raise ValueError(f"Column '{code}' not found in input_columns.")
    return indices


def categorical_to_index(value: Any, categories: list[str], *, code: str) -> int:
    """Index of a category label in its trained vocabulary.

    Raises on an unknown value — never silently maps to index 0 (which would
    yield wrong predictions for a deployed model).
    """
    try:
        return categories.index(value)
    except ValueError:
        raise ValueError(
            f"Unknown categorical value {value!r} for '{code}'. "
            f"Trained categories: {categories}"
        ) from None
