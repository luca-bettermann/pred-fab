"""Cross-validation instrument: experiment-level folds + the located error field.

Fold construction is tested pure; the runner is exercised on the workflow stack
(3 loaded + evaluated experiments) — the same fixture chain as test_validate_r2_inf,
since each fold reuses PredictionSystem.train + validate.
"""
import numpy as np
import pytest

from pred_fab.orchestration.cross_validation import (
    CrossValidator,
    CVResult,
    HeldOutError,
    make_experiment_folds,
    diagnose_error_coverage,
    DiagnosedPoint,
)
from tests.utils.builders import (
    build_workflow_stack,
    evaluate_loaded_workflow_experiments,
    configure_default_workflow_calibration,
    build_prepared_workflow_datamodule,
)


# ===== make_experiment_folds (pure) =====

def test_make_folds_loo_is_one_experiment_per_fold():
    codes = ["a", "b", "c", "d"]
    folds = make_experiment_folds(codes, k=None)
    assert folds == [["a"], ["b"], ["c"], ["d"]]


def test_make_folds_kfold_is_balanced_disjoint_and_covering():
    codes = [f"e{i}" for i in range(10)]
    folds = make_experiment_folds(codes, k=3, seed=0)
    assert len(folds) == 3
    sizes = sorted(len(f) for f in folds)
    assert sizes == [3, 3, 4]                      # balanced
    flat = [c for f in folds for c in f]
    assert sorted(flat) == sorted(codes)           # covering
    assert len(flat) == len(set(flat))             # disjoint


def test_make_folds_k_ge_n_falls_back_to_loo():
    codes = ["a", "b", "c"]
    assert make_experiment_folds(codes, k=99) == [["a"], ["b"], ["c"]]


def test_make_folds_empty():
    assert make_experiment_folds([], k=None) == []


def test_make_folds_kfold_is_deterministic_with_seed():
    codes = [f"e{i}" for i in range(12)]
    assert make_experiment_folds(codes, k=4, seed=7) == make_experiment_folds(codes, k=4, seed=7)


# ===== CrossValidator runner =====

@pytest.fixture
def cv_stack(tmp_path):
    """Workflow stack with 3 evaluated experiments + a base datamodule (all in train)."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    configure_default_workflow_calibration(agent)
    base_dm = build_prepared_workflow_datamodule(agent, dataset, val_size=0.0, test_size=0.0)
    return agent, dataset, base_dm, codes


def test_cross_validate_loo_locates_every_experiment(cv_stack):
    agent, dataset, base_dm, codes = cv_stack
    result = CrossValidator.from_agent(agent).run(dataset, base_dm)

    assert isinstance(result, CVResult)
    assert result.mode == "loo"
    assert result.n_folds == len(codes)
    assert len(result.held_out) == len(codes)
    # Every loaded experiment is located exactly once, with its parameters attached.
    located = {h.exp_code for h in result.held_out}
    assert located == set(codes)
    for h in result.held_out:
        assert isinstance(h, HeldOutError)
        assert h.params, "held-out experiment must carry its parameters (its field coordinate)"
        assert h.metrics, "validate() should yield per-feature metrics for the held-out experiment"
        for feat_metrics in h.metrics.values():
            assert "mae" in feat_metrics and np.isfinite(feat_metrics["mae"])


def test_cv_error_field_and_aggregate(cv_stack):
    agent, dataset, base_dm, codes = cv_stack
    result = CrossValidator.from_agent(agent).run(dataset, base_dm)

    feats = result.features()
    assert feats, "the field should expose at least one predicted feature"
    for feat in feats:
        field = result.error_field(feat, metric="mae")
        assert len(field) == len(codes)                       # one located point per experiment
        for params, err in field:
            assert isinstance(params, dict) and params
            assert np.isfinite(err)

    agg = result.aggregate(metric="mae")
    for feat in feats:
        assert np.isfinite(agg[feat]["mae_mean"])
        assert agg[feat]["n_experiments"] == float(len(codes))


def test_cv_kfold_still_locates_all_experiments(cv_stack):
    agent, dataset, base_dm, codes = cv_stack
    result = CrossValidator.from_agent(agent).run(dataset, base_dm, k=2)

    assert result.mode == "2-fold"
    assert result.n_folds == 2
    # Each experiment is still held out exactly once across the folds.
    assert {h.exp_code for h in result.held_out} == set(codes)


def test_cv_does_not_touch_the_deployed_system(cv_stack):
    """CV is a re-fit diagnostic: it builds throwaway systems, never the deployed one."""
    agent, dataset, base_dm, codes = cv_stack
    deployed_models = agent.pred_system.models
    deployed_ids = [id(m) for m in deployed_models]

    CrossValidator.from_agent(agent).run(dataset, base_dm)

    assert agent.pred_system.models is deployed_models          # same list
    assert [id(m) for m in agent.pred_system.models] == deployed_ids  # same instances


def test_cv_requires_at_least_two_experiments(cv_stack):
    agent, dataset, base_dm, codes = cv_stack
    with pytest.raises(ValueError, match="2 populated experiments"):
        CrossValidator.from_agent(agent).run(dataset, base_dm, codes=["exp_001"])


# ===== error-vs-coverage diagnostic (concept 3) =====

def _craft_cv(points: list[tuple[str, float, float]]) -> CVResult:
    """CVResult from (code, mae, x) triples — x is a coverage proxy on params['x']."""
    held = [
        HeldOutError(exp_code=c, params={"x": x}, metrics={"f": {"mae": mae, "n_samples": 1.0}})
        for c, mae, x in points
    ]
    return CVResult(held_out=held, mode="loo", n_folds=len(held))


def test_diagnose_labels_the_four_quadrants():
    # errors [1,1,0,0] → median 0.5; coverages [.9,.1,.9,.1] → median 0.5
    cv = _craft_cv([("a", 1.0, 0.9), ("b", 1.0, 0.1), ("c", 0.0, 0.9), ("d", 0.0, 0.1)])
    diag = diagnose_error_coverage(cv, coverage_fn=lambda p: p["x"])
    label = {pt.exp_code: pt.label for pt in diag.points}
    assert label == {
        "a": "model_problem",     # high error, high coverage
        "b": "under_explored",    # high error, low coverage
        "c": "trustworthy",       # low error, high coverage
        "d": "sparse_ok",         # low error, low coverage
    }
    assert diag.summary() == {
        "model_problem": 1, "under_explored": 1, "trustworthy": 1, "sparse_ok": 1
    }
    assert [p.exp_code for p in diag.model_problems()] == ["a"]
    assert [p.exp_code for p in diag.under_explored()] == ["b"]


def test_diagnose_thresholds_default_to_medians():
    cv = _craft_cv([("a", 1.0, 0.9), ("b", 1.0, 0.1), ("c", 0.0, 0.9), ("d", 0.0, 0.1)])
    diag = diagnose_error_coverage(cv, coverage_fn=lambda p: p["x"])
    assert diag.error_threshold == 0.5
    assert diag.coverage_threshold == 0.5


def test_diagnose_explicit_thresholds_override_medians():
    cv = _craft_cv([("a", 0.4, 0.9), ("b", 0.6, 0.2)])
    # Fix coverage cut at 0.5 (D=1) and error cut at 0.5: a low-err/high-cov, b high-err/low-cov.
    diag = diagnose_error_coverage(
        cv, coverage_fn=lambda p: p["x"], error_threshold=0.5, coverage_threshold=0.5,
    )
    label = {pt.exp_code: pt.label for pt in diag.points}
    assert label == {"a": "trustworthy", "b": "under_explored"}


def test_diagnose_empty_field_is_safe():
    diag = diagnose_error_coverage(CVResult([], "loo", 0), coverage_fn=lambda p: 0.0)
    assert diag.points == []
    assert diag.summary() == {
        "model_problem": 0, "under_explored": 0, "trustworthy": 0, "sparse_ok": 0
    }


def test_diagnose_runs_over_real_cv_field(cv_stack):
    """End-to-end: diagnose a real CV field via CVResult.diagnose (stub coverage proxy;
    CalibrationSystem.evidence is the production coverage_fn, exercised elsewhere)."""
    agent, dataset, base_dm, codes = cv_stack
    result = CrossValidator.from_agent(agent).run(dataset, base_dm)

    diag = result.diagnose(coverage_fn=lambda p: float(p.get("param_1", 0.0)))
    assert diag.points
    valid = {"model_problem", "under_explored", "trustworthy", "sparse_ok"}
    assert all(isinstance(p, DiagnosedPoint) and p.label in valid for p in diag.points)
    assert sum(diag.summary().values()) == len(diag.points)
    # one diagnosed point per (held-out experiment, feature)
    assert len(diag.points) == len(result.held_out) * len(result.features())
