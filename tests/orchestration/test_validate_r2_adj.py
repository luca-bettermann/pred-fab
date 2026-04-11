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
        assert "rmse" in metrics
        assert "n_samples" in metrics


def test_validate_without_weights_has_no_r2_adj(trained_stack):
    """When no performance_weights are passed, R2_adj should not appear."""
    agent, dataset, datamodule = trained_stack
    results = agent.pred_system.validate(performance_weights=None)
    for key, metrics in results.items():
        assert "r2_adj" not in metrics


def test_validate_with_weights_includes_r2_adj(trained_stack):
    """When performance_weights are passed, each feature should have R2_adj."""
    agent, dataset, datamodule = trained_stack
    weights = {"performance_1": 2.0, "performance_2": 1.3}
    results = agent.pred_system.validate(performance_weights=weights)
    assert len(results) > 0
    for key, metrics in results.items():
        assert "r2_adj" in metrics
        assert isinstance(metrics["r2_adj"], float)


def test_agent_train_validate_returns_per_feature_with_r2_adj(trained_stack):
    """agent.train(validate=True) should return per-feature results with R2_adj."""
    agent, dataset, datamodule = trained_stack
    # Re-prepare to get a fresh split
    datamodule.prepare(val_size=0.5, recompute=True)
    results = agent.train(datamodule, validate=True)
    assert results is not None
    assert isinstance(results, dict)
    for key, metrics in results.items():
        assert "r2" in metrics
        # R2_adj should be present because calibration is configured with performance_weights
        assert "r2_adj" in metrics


def test_agent_train_no_validate_returns_none(trained_stack):
    """agent.train(validate=False) should return None."""
    agent, dataset, datamodule = trained_stack
    result = agent.train(datamodule, validate=False)
    assert result is None


def test_validate_r2_adj_is_finite(trained_stack):
    """R2_adj values should be finite numbers."""
    agent, dataset, datamodule = trained_stack
    weights = {"performance_1": 2.0, "performance_2": 1.3}
    results = agent.pred_system.validate(performance_weights=weights)
    for key, metrics in results.items():
        assert np.isfinite(metrics["r2"])
        assert np.isfinite(metrics["r2_adj"])
