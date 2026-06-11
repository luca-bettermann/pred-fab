"""ExperimentSet / Fit — the pure grouping + fit-composition object model."""
import pytest

from pred_fab.core import Strategy, ExperimentSet, Fit, FitPart


# ===== Strategy =====

def test_strategy_ordered_defaults():
    assert Strategy.EXPLORATION.ordered and Strategy.INFERENCE.ordered
    assert not Strategy.DISCOVERY.ordered
    assert not Strategy.SOBOL.ordered
    assert not Strategy.GRID.ordered
    assert not Strategy.ADAPTATION.ordered


# ===== ExperimentSet =====

def test_ordered_defaults_from_strategy_and_can_be_overridden():
    disc = ExperimentSet("D1", Strategy.DISCOVERY, members=["d1", "d2"])
    expl = ExperimentSet("E1", Strategy.EXPLORATION, members=["e1", "e2"])
    assert disc.ordered is False
    assert expl.ordered is True
    # explicit override wins
    forced = ExperimentSet("X", Strategy.DISCOVERY, members=["x"], ordered=True)
    assert forced.ordered is True


def test_codes_len_contains_iter():
    s = ExperimentSet("E1", Strategy.EXPLORATION, members=["e1", "e2", "e3"])
    assert s.codes() == ["e1", "e2", "e3"]
    assert len(s) == 3
    assert "e2" in s and "zz" not in s
    assert list(s) == ["e1", "e2", "e3"]


def test_window_returns_prefix_as_ordered_subset():
    s = ExperimentSet("E1", Strategy.EXPLORATION, members=["e1", "e2", "e3", "e4"])
    w = s.window(2)
    assert w.codes() == ["e1", "e2"]
    assert w.ordered is True
    assert w.strategy is Strategy.EXPLORATION


def test_window_on_batch_raises():
    disc = ExperimentSet("D1", Strategy.DISCOVERY, members=["d1", "d2"])
    with pytest.raises(ValueError, match="requires an ordered set"):
        disc.window(1)


# ===== Fit composition =====

def test_fit_of_batch_set_is_just_its_members():
    grid = ExperimentSet("G1", Strategy.GRID, members=["g1", "g2", "g3"])
    assert Fit.of(grid).experiment_codes() == ["g1", "g2", "g3"]


def test_fit_of_exploration_unions_parent_discovery_with_window():
    disc = ExperimentSet("D1", Strategy.DISCOVERY, members=["d1", "d2", "d3"])
    expl = ExperimentSet("E1", Strategy.EXPLORATION, members=["e1", "e2", "e3", "e4", "e5"], parent=disc)

    fit_k = Fit.of(expl, window=3)                       # discovery(whole) ∪ exploration[:3]
    assert fit_k.experiment_codes() == ["d1", "d2", "d3", "e1", "e2", "e3"]  # parents first


def test_fit_includes_whole_parent_chain():
    root = ExperimentSet("R", Strategy.DISCOVERY, members=["r1"])
    mid = ExperimentSet("M", Strategy.EXPLORATION, members=["m1", "m2"], parent=root)
    leaf = ExperimentSet("L", Strategy.EXPLORATION, members=["l1", "l2"], parent=mid)
    # whole root + whole mid + leaf[:1], deepest ancestor first
    assert Fit.of(leaf, window=1).experiment_codes() == ["r1", "m1", "m2", "l1"]


def test_experiment_codes_dedupes_overlap():
    a = ExperimentSet("A", Strategy.DISCOVERY, members=["x1", "x2"])
    b = ExperimentSet("B", Strategy.GRID, members=["x2", "x3"])
    fit = Fit([FitPart(a), FitPart(b)])
    assert fit.experiment_codes() == ["x1", "x2", "x3"]   # x2 once, stable order


def test_fit_at_generates_what_the_proposing_model_saw():
    disc = ExperimentSet("D1", Strategy.DISCOVERY, members=["d1", "d2"])
    expl = ExperimentSet("E1", Strategy.EXPLORATION, members=["e1", "e2", "e3"], parent=disc)
    # member at index 2 (e3) was proposed by a model trained on discovery + e1, e2
    assert expl.fit_at(2).experiment_codes() == ["d1", "d2", "e1", "e2"]
    # the first exploration member saw only the discovery base
    assert expl.fit_at(0).experiment_codes() == ["d1", "d2"]


def test_fit_at_on_batch_raises():
    disc = ExperimentSet("D1", Strategy.DISCOVERY, members=["d1"])
    with pytest.raises(ValueError, match="requires an ordered set"):
        disc.fit_at(0)


# ===== serialization =====

def test_to_dict_from_dict_round_trip():
    disc = ExperimentSet("D1", Strategy.DISCOVERY, members=["d1", "d2"])
    expl = ExperimentSet("E1", Strategy.EXPLORATION, members=["e1", "e2"], parent=disc)
    d = expl.to_dict()
    assert d == {
        "code": "E1", "strategy": "exploration",
        "members": ["e1", "e2"], "ordered": True, "parent": "D1",
    }
    back = ExperimentSet.from_dict(d, parent=disc)
    assert back.code == "E1" and back.strategy is Strategy.EXPLORATION
    assert back.members == ["e1", "e2"] and back.ordered is True and back.parent is disc
