import pytest
import numpy as np

from tests.utils.builders import (
    build_workflow_stack,
    evaluate_loaded_workflow_experiments,
    configure_default_workflow_calibration,
    build_prepared_workflow_datamodule,
)


@pytest.fixture
def trained_stack(tmp_path):
    """Build, evaluate, configure, train a workflow stack for validation tests."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    configure_default_workflow_calibration(agent)
    datamodule = build_prepared_workflow_datamodule(agent, dataset, val_size=0.5)
    agent.train(datamodule, validate=False)

    # The WorkflowPredictionModel declares input_parameters=["param_1", "param_2"]
    # but ships with weights shaped (4, 2) — a leftover from an earlier test layout.
    # validate() correctly slices X to the 2 declared input columns, so we resize
    # weights to (2, 2) so forward_pass works during validation.
    for model in agent.pred_system.models:
        if hasattr(model, "weights") and model.weights.shape[0] != 2:
            model.weights = model.weights[:2, :]

    return agent, dataset, datamodule


def test_validate_returns_per_feature_dict(trained_stack):
    """validate() should return {feature_name: {metric: value}} dict."""
    agent, dataset, datamodule = trained_stack
    results = agent.pred_system.validate()
    assert isinstance(results, dict)
    # Should have entries for each predicted feature
    for key, metrics in results.items():
        assert isinstance(metrics, dict)
        assert "r2" in metrics
        assert "mae" in metrics
        assert "n_samples" in metrics


def test_validate_includes_r2_inf_when_performances_loaded(trained_stack):
    """R²_inf is computed automatically from stored performance scores (the
    fixture evaluates the experiments + configures performance weights)."""
    agent, dataset, datamodule = trained_stack
    results = agent.pred_system.validate(eval_system=agent.eval_system)
    assert len(results) > 0
    for key, metrics in results.items():
        assert "r2_inf" in metrics
        assert isinstance(metrics["r2_inf"], float)


def test_agent_train_validate_returns_per_feature_with_r2_inf(trained_stack):
    """agent.train(validate=True) should return per-feature results with R2_inf."""
    agent, dataset, datamodule = trained_stack
    # Re-prepare to get a fresh split
    datamodule.prepare(val_size=0.5, recompute=True)
    results = agent.train(datamodule, validate=True)
    assert results is not None
    assert isinstance(results, dict)
    for key, metrics in results.items():
        assert "r2" in metrics
        # R2_inf should be present because calibration is configured with performance_weights
        assert "r2_inf" in metrics


def test_agent_train_no_validate_returns_none(trained_stack):
    """agent.train(validate=False) should return None."""
    agent, dataset, datamodule = trained_stack
    result = agent.train(datamodule, validate=False)
    assert result is None


def test_validate_r2_inf_is_finite(trained_stack):
    """R2_inf values should be finite numbers."""
    agent, dataset, datamodule = trained_stack
    results = agent.pred_system.validate(eval_system=agent.eval_system)
    for key, metrics in results.items():
        assert np.isfinite(metrics["r2"])
        assert np.isfinite(metrics["r2_inf"])


def test_expand_per_experiment_handles_skipped_experiment():
    """The identity-keyed expansion must survive an experiment contributing
    zero rows — the old np.unique/zip alignment crashed or mis-weighted."""
    from pred_fab.orchestration.prediction import PredictionSystem
    per_exp = [0.1, 0.5, 0.9]  # weights for experiments 0, 1, 2
    # Experiment 1 contributed no rows; rows come from exp 0 (×2) and exp 2 (×3).
    row_exp_ids = np.array([0, 0, 2, 2, 2])
    out = PredictionSystem._expand_per_experiment(per_exp, row_exp_ids)
    np.testing.assert_array_equal(out, [0.1, 0.1, 0.9, 0.9, 0.9])


def test_expand_per_experiment_empty_and_out_of_range():
    from pred_fab.orchestration.prediction import PredictionSystem
    assert PredictionSystem._expand_per_experiment([0.3], np.array([], dtype=int)).size == 0
    with pytest.raises(ValueError, match="out of range"):
        PredictionSystem._expand_per_experiment([0.3, 0.4], np.array([0, 2]))
