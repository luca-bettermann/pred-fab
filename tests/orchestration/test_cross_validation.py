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
