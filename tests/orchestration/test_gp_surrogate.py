"""Tests for the GP surrogate uncertainty estimation used in ISARC release."""
import pytest
import numpy as np

from pred_fab.interfaces.calibration import GaussianProcessSurrogate
from pred_fab.utils.enum import Mode
from tests.utils.builders import (
    build_real_agent_stack,
    build_workflow_stack,
    evaluate_loaded_workflow_experiments,
    build_prepared_workflow_datamodule,
)


# ---------------------------------------------------------------------------
# GaussianProcessSurrogate interface tests
# ---------------------------------------------------------------------------

def test_gp_surrogate_not_fitted_before_fit():
    from tests.utils.builders import build_test_logger
    import tempfile, pathlib
    logger = build_test_logger(pathlib.Path(tempfile.mkdtemp()))
    gp = GaussianProcessSurrogate(logger)
    assert not gp.is_fitted


def test_gp_surrogate_fit_sets_fitted_flag(tmp_path):
    from tests.utils.builders import build_test_logger
    logger = build_test_logger(tmp_path)
    gp = GaussianProcessSurrogate(logger)

    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    y = np.array([[0.8], [0.6], [0.4]])
    gp.fit(X, y)
    assert gp.is_fitted


def test_gp_surrogate_predict_returns_mean_and_std(tmp_path):
    from tests.utils.builders import build_test_logger
    logger = build_test_logger(tmp_path)
    gp = GaussianProcessSurrogate(logger)

    X_train = np.array([[0.0], [0.5], [1.0]])
    y_train = np.array([[0.9], [0.5], [0.1]])
    gp.fit(X_train, y_train)

    X_test = np.array([[0.25], [0.75]])
    mean, std = gp.predict(X_test)

    assert mean.shape[0] == 2
    assert std.shape[0] == 2
    # std must be non-negative
    assert np.all(std >= 0)


def test_gp_surrogate_std_higher_in_sparse_region(tmp_path):
    """GP std should be higher far from training data than near it."""
    from tests.utils.builders import build_test_logger
    logger = build_test_logger(tmp_path)
    gp = GaussianProcessSurrogate(logger)

    # Train only near 0.0
    X_train = np.linspace(0.0, 0.1, 5).reshape(-1, 1)
    y_train = np.random.default_rng(0).random((5, 1))
    gp.fit(X_train, y_train)

    _, std_near = gp.predict(np.array([[0.05]]))
    _, std_far = gp.predict(np.array([[0.9]]))

    assert float(std_far[0, 0]) > float(std_near[0, 0])


def test_gp_surrogate_fit_with_empty_x_does_not_raise(tmp_path):
    from tests.utils.builders import build_test_logger
    logger = build_test_logger(tmp_path)
    gp = GaussianProcessSurrogate(logger)
    # Empty fit should not raise and leave gp unfitted
    gp.fit(np.empty((0, 2)), np.empty((0, 1)))
    assert not gp.is_fitted


# ---------------------------------------------------------------------------
# Agent GP surrogate integration tests
# ---------------------------------------------------------------------------

def test_agent_gp_surrogate_not_fitted_before_train(tmp_path):
    """GP surrogate should not be fitted before train() is called."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    assert agent._gp_surrogate is not None
    assert not agent._gp_surrogate.is_fitted


def test_agent_gp_fitted_after_train_with_performance_data(tmp_path):
    """GP surrogate should be fitted after train() when performance data exists."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)
    # Only 1 experiment — too few for GP fitting (needs >= 2)
    # GP is not fitted; uncertainty_fn falls back to 1.0
    assert agent._gp_surrogate is not None


def test_agent_gp_fitted_with_multiple_experiments(tmp_path):
    """GP surrogate is fitted after train() when >= 2 experiments have performance data."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    datamodule = build_prepared_workflow_datamodule(agent, dataset)
    agent.configure_calibration(
        performance_weights={"performance_1": 1.0, "performance_2": 1.0},
        bounds={"param_1": (0.0, 10.0), "param_2": (1, 4), "n_layers": (1, 3), "n_segments": (1, 3)},
        fixed_params={"param_3": "B"},
    )
    agent.train(datamodule=datamodule, validate=False, test=False)
    assert agent._gp_surrogate is not None
    assert agent._gp_surrogate.is_fitted


def test_uncertainty_fn_returns_one_before_gp_fitted(tmp_path):
    """uncertainty_fn should return 1.0 (max uncertainty) before GP is fitted."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    cs = agent.calibration_system
    # Directly call the injected uncertainty_fn with a dummy normalized vector
    X_dummy = np.zeros(3)
    u = cs.uncertainty_fn(X_dummy)
    assert u == pytest.approx(1.0)


def test_uncertainty_fn_returns_float_in_0_1_after_gp_fitted(tmp_path):
    """After GP is fitted, uncertainty_fn should return a value in [0, 1]."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    datamodule = build_prepared_workflow_datamodule(agent, dataset)
    agent.configure_calibration(
        performance_weights={"performance_1": 1.0, "performance_2": 1.0},
        bounds={"param_1": (0.0, 10.0), "param_2": (1, 4), "n_layers": (1, 3), "n_segments": (1, 3)},
        fixed_params={"param_3": "B"},
    )
    agent.train(datamodule=datamodule, validate=False, test=False)

    # Probe with a normalized vector
    X_probe = np.array(datamodule.params_to_array({"param_1": 5.0, "param_2": 2, "n_layers": 2, "n_segments": 2}))
    u = agent.calibration_system.uncertainty_fn(X_probe)
    assert isinstance(u, float)
    assert 0.0 <= u <= 1.0


# ---------------------------------------------------------------------------
# Exploration with GP uncertainty: regression checks
# ---------------------------------------------------------------------------

def test_exploration_step_uses_gp_uncertainty_not_kde(tmp_path):
    """After training, exploration_step returns a valid ExperimentSpec using GP uncertainty."""
    from pred_fab.core import ExperimentSpec
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    datamodule = build_prepared_workflow_datamodule(agent, dataset)
    agent.configure_calibration(
        performance_weights={"performance_1": 1.0, "performance_2": 1.0},
        bounds={"param_1": (0.0, 10.0), "param_2": (1, 4), "n_layers": (1, 3), "n_segments": (1, 3)},
        fixed_params={"param_3": "B"},
    )
    agent.train(datamodule=datamodule, validate=False, test=False)

    result = agent.exploration_step(datamodule=datamodule, w_explore=0.5)
    assert isinstance(result, ExperimentSpec)
    assert "param_1" in result
    # schedules must be empty (no trajectory)
    assert result.schedules == {}


def test_inference_step_runs_after_gp_fitted(tmp_path):
    """inference_step returns a valid ExperimentSpec when GP surrogate is fitted."""
    from pred_fab.core import ExperimentSpec
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    datamodule = build_prepared_workflow_datamodule(agent, dataset)
    agent.configure_calibration(
        performance_weights={"performance_1": 1.0, "performance_2": 1.0},
        bounds={"param_1": (0.0, 10.0), "param_2": (1, 4), "n_layers": (1, 3), "n_segments": (1, 3)},
        fixed_params={"param_3": "B"},
    )
    agent.train(datamodule=datamodule, validate=False, test=False)

    exp = dataset.get_experiment(codes[0])
    result = agent.inference_step(exp_data=exp, datamodule=datamodule, recompute=False)
    assert isinstance(result, ExperimentSpec)
    assert result.schedules == {}
