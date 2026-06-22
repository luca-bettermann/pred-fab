"""Pure parameter resolution — the effective-params apply logic as a plain-dict function.

Single home (SSOT) for "apply sparse dimensional updates onto a base, given a position
context": :meth:`ExperimentData.get_effective_parameters_for_context` delegates here, and
external consumers (e.g. rtde's per-dim recompute) call it directly with **plain dicts** —
no `DataObject` / `Roles` / schema / sanitisation, so they don't couple to pred-fab's
internal data model across version bumps.
"""
from __future__ import annotations

from typing import Any


def effective_parameters(
    base: dict[str, Any],
    updates: list[dict[str, Any]],
    ctx: dict[str, int],
) -> dict[str, Any]:
    """Resolve effective parameter values at position ``ctx`` from a ``base`` + sparse ``updates``.

    Starts from ``base`` and applies each update's ``updates`` mapping when it is in effect at
    ``ctx``. An update ``{"updates": {code: value}, "iterator_code": str, "step_index": int}``
    is in effect when ``ctx.get(iterator_code) >= step_index``; an update whose
    ``iterator_code``/``step_index`` is absent or ``None`` is unconditional initial-state
    (always applied). Updates are applied in list order — later ones override earlier ones
    and the base. Pure: plain dicts in, plain dict out; callers pass already-typed values
    (no coercion here).
    """
    effective = dict(base)
    for update in updates:
        iterator_code = update.get("iterator_code")
        step_index = update.get("step_index")
        if iterator_code is None or step_index is None:
            effective.update(update.get("updates", {}))
            continue
        ctx_value = ctx.get(iterator_code)
        if ctx_value is not None and ctx_value >= step_index:
            effective.update(update.get("updates", {}))
    return effective
