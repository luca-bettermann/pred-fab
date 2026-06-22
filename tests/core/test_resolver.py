"""Tests for the pure `effective_parameters` resolver (the plain-dict external entry point).

This is the SSOT apply logic that `ExperimentData.get_effective_parameters_for_context`
delegates to and that external consumers (rtde recompute) call directly."""
from pred_fab.core import effective_parameters


def test_base_only_when_no_updates():
    assert effective_parameters({"a": 1, "b": 2}, [], {}) == {"a": 1, "b": 2}


def test_update_applies_at_or_after_step():
    base = {"speed": 0.05}
    updates = [{"updates": {"speed": 0.09}, "iterator_code": "layer", "step_index": 3}]
    assert effective_parameters(base, updates, {"layer": 2}) == {"speed": 0.05}   # before
    assert effective_parameters(base, updates, {"layer": 3}) == {"speed": 0.09}   # at
    assert effective_parameters(base, updates, {"layer": 5}) == {"speed": 0.09}   # after


def test_unconditional_initial_state_update():
    base = {"mode": "clay"}
    updates = [{"updates": {"mode": "concrete"}}]   # no iterator_code/step_index → always
    assert effective_parameters(base, updates, {}) == {"mode": "concrete"}


def test_later_update_overrides_earlier_and_base():
    base = {"x": 0}
    updates = [
        {"updates": {"x": 1}, "iterator_code": "d", "step_index": 1},
        {"updates": {"x": 2}, "iterator_code": "d", "step_index": 2},
    ]
    assert effective_parameters(base, updates, {"d": 5})["x"] == 2   # last applicable wins
    assert effective_parameters(base, updates, {"d": 1})["x"] == 1   # only the first is in effect


def test_non_matching_iterator_is_ignored():
    base = {"x": 0}
    updates = [{"updates": {"x": 9}, "iterator_code": "other", "step_index": 1}]
    assert effective_parameters(base, updates, {"layer": 5}) == {"x": 0}   # ctx has no 'other'


def test_does_not_mutate_inputs():
    base = {"x": 0}
    updates = [{"updates": {"x": 1}, "iterator_code": "d", "step_index": 0}]
    effective_parameters(base, updates, {"d": 0})
    assert base == {"x": 0}   # base untouched
