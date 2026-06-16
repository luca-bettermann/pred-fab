"""Provenance — typed reproducibility view over config_snapshot, and fit() derivation."""
from pred_fab.core import Provenance, ExperimentSet
from pred_fab.utils.enum import SourceStep


def test_from_dict_parses_source_seed_and_settings():
    p = Provenance.from_dict(
        {"source": "exploration_step", "seed": 7, "kappa": 0.4, "param_bounds": {"p": [0, 10]}}
    )
    assert p.source is SourceStep.EXPLORATION
    assert p.seed == 7
    assert p.settings == {"kappa": 0.4, "param_bounds": {"p": [0, 10]}}
    assert p.origin is None


def test_round_trips_losslessly():
    snap = {
        "source": "sobol_step", "seed": 1, "kappa": None,
        "param_bounds": {"p": [0, 1]}, "origin": ["E1", 3],
    }
    p = Provenance.from_dict(snap)
    assert p.origin == ("E1", 3)
    out = p.to_dict()
    assert out["source"] == "sobol_step" and out["seed"] == 1
    assert out["origin"] == ["E1", 3] and out["kappa"] is None
    assert Provenance.from_dict(out) == p          # stable round-trip


def test_unknown_source_yields_none():
    assert Provenance.from_dict({"source": "weird", "seed": 0}).source is None


def test_empty_snapshot_is_safe():
    p = Provenance.from_dict(None)
    assert p.source is None and p.seed is None and p.settings == {} and p.origin is None


def test_fit_is_within_origin_set_prefix():
    expl = ExperimentSet("E1", members=["e1", "e2", "e3"], ordered=True)
    # origin = exploration position 2 → the within-set prefix it was generated under is e1, e2
    p = Provenance.from_dict({"source": "exploration_step", "seed": 0, "origin": ["E1", 2]})
    assert p.fit(expl).experiment_codes() == ["e1", "e2"]


def test_fit_batch_origin_is_whole_set():
    batch = ExperimentSet("S1", members=["s1", "s2"])
    p = Provenance.from_dict({"source": "sobol_step", "origin": ["S1", None]})
    assert p.fit(batch).experiment_codes() == ["s1", "s2"]
