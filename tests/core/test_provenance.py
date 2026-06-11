"""Provenance — typed reproducibility view over config_snapshot, and fit() derivation."""
from pred_fab.core import Provenance, Strategy, ExperimentSet


def test_from_dict_parses_design_seed_and_settings():
    p = Provenance.from_dict(
        {"design": "exploration", "seed": 7, "kappa": 0.4, "param_bounds": {"p": [0, 10]}}
    )
    assert p.strategy is Strategy.EXPLORATION
    assert p.seed == 7
    assert p.settings == {"kappa": 0.4, "param_bounds": {"p": [0, 10]}}
    assert p.origin is None


def test_round_trips_losslessly():
    snap = {
        "design": "sobol", "seed": 1, "kappa": None,
        "param_bounds": {"p": [0, 1]}, "origin": ["E1", 3], "schema_version": "v2",
    }
    p = Provenance.from_dict(snap)
    assert p.origin == ("E1", 3)
    assert p.schema_version == "v2"
    out = p.to_dict()
    assert out["design"] == "sobol" and out["seed"] == 1
    assert out["origin"] == ["E1", 3] and out["kappa"] is None
    assert Provenance.from_dict(out) == p          # stable round-trip


def test_unknown_design_yields_none_strategy():
    assert Provenance.from_dict({"design": "weird", "seed": 0}).strategy is None


def test_empty_snapshot_is_safe():
    p = Provenance.from_dict(None)
    assert p.strategy is None and p.seed is None and p.settings == {} and p.origin is None


def test_fit_derives_the_generating_training_set():
    disc = ExperimentSet("D1", Strategy.DISCOVERY, members=["d1", "d2"])
    expl = ExperimentSet("E1", Strategy.EXPLORATION, members=["e1", "e2", "e3"], parent=disc)
    # origin = exploration position 2 → trained on discovery + e1, e2
    p = Provenance.from_dict({"design": "exploration", "seed": 0, "origin": ["E1", 2]})
    assert p.fit(expl).experiment_codes() == ["d1", "d2", "e1", "e2"]


def test_fit_batch_origin_is_whole_set():
    grid = ExperimentSet("G1", Strategy.GRID, members=["g1", "g2"])
    p = Provenance.from_dict({"design": "grid", "origin": ["G1", None]})
    assert p.fit(grid).experiment_codes() == ["g1", "g2"]
