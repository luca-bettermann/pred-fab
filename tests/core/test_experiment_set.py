"""ExperimentSet / Fit — the pure grouping + fit-composition object model."""
import pytest

from pred_fab.core import ExperimentSet, Fit, FitPart


# ===== ExperimentSet =====

def test_ordered_is_explicit_and_defaults_false():
    assert ExperimentSet("D1", members=["d1", "d2"]).ordered is False
    assert ExperimentSet("E1", members=["e1", "e2"], ordered=True).ordered is True


def test_codes_len_contains_iter():
    s = ExperimentSet("E1", members=["e1", "e2", "e3"], ordered=True)
    assert s.codes() == ["e1", "e2", "e3"]
    assert len(s) == 3
    assert "e2" in s and "zz" not in s
    assert list(s) == ["e1", "e2", "e3"]


def test_window_returns_prefix_as_ordered_subset():
    s = ExperimentSet("E1", members=["e1", "e2", "e3", "e4"], ordered=True)
    w = s.window(2)
    assert w.codes() == ["e1", "e2"]
    assert w.ordered is True


def test_window_on_batch_raises():
    disc = ExperimentSet("D1", members=["d1", "d2"])
    with pytest.raises(ValueError, match="requires an ordered set"):
        disc.window(1)


# ===== Fit composition =====

def test_fit_of_single_set_is_its_members():
    grid = ExperimentSet("G1", members=["g1", "g2", "g3"])
    assert Fit.of(grid).experiment_codes() == ["g1", "g2", "g3"]


def test_fit_of_windowed_set():
    expl = ExperimentSet("E1", members=["e1", "e2", "e3", "e4", "e5"], ordered=True)
    assert Fit.of(expl, window=3).experiment_codes() == ["e1", "e2", "e3"]


def test_fit_composes_multiple_sets_in_order():
    disc = ExperimentSet("D1", members=["d1", "d2", "d3"])
    expl = ExperimentSet("E1", members=["e1", "e2", "e3", "e4", "e5"], ordered=True)
    fit = Fit([FitPart(disc), FitPart(expl, window=3)])      # disc ∪ expl[:3]
    assert fit.experiment_codes() == ["d1", "d2", "d3", "e1", "e2", "e3"]


def test_experiment_codes_dedupes_overlap():
    a = ExperimentSet("A", members=["x1", "x2"])
    b = ExperimentSet("B", members=["x2", "x3"])
    fit = Fit([FitPart(a), FitPart(b)])
    assert fit.experiment_codes() == ["x1", "x2", "x3"]      # x2 once, stable order


def test_fit_at_is_within_set_prefix():
    expl = ExperimentSet("E1", members=["e1", "e2", "e3"], ordered=True)
    assert expl.fit_at(2).experiment_codes() == ["e1", "e2"]   # what preceded member index 2
    assert expl.fit_at(0).experiment_codes() == []             # nothing precedes the first


def test_fit_at_on_batch_raises():
    disc = ExperimentSet("D1", members=["d1"])
    with pytest.raises(ValueError, match="requires an ordered set"):
        disc.fit_at(0)


# ===== serialization =====

def test_to_dict_from_dict_round_trip():
    expl = ExperimentSet("E1", members=["e1", "e2"], ordered=True)
    d = expl.to_dict()
    assert d == {"code": "E1", "members": ["e1", "e2"], "ordered": True}
    back = ExperimentSet.from_dict(d)
    assert back.code == "E1" and back.members == ["e1", "e2"] and back.ordered is True
