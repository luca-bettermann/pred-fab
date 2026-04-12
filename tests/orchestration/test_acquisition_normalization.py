"""
Tests for acquisition function min-max normalization and configure() extensions
(mpc_lookahead, mpc_discount).
"""
import pytest
import numpy as np

from pred_fab.utils.enum import Mode
from pred_fab.orchestration.calibration import Optimizer
from tests.utils.builders import (
    build_real_agent_stack,
    build_workflow_stack,
    evaluate_loaded_workflow_experiments,
    build_prepared_workflow_datamodule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_trained_agent(tmp_path):
    """Return (agent, exp, datamodule) trained and ready for calibration."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)
    return agent, exp, datamodule


# ===========================================================================
# Acquisition range estimation
# ===========================================================================

class TestAcquisitionRangeEstimation:
    """_get_acquisition_ranges returns perf range from training data; uncertainty is not normalized."""

    def test_returns_perf_range_and_none_unc(self, tmp_path):
        agent, exp, datamodule = _setup_trained_agent(tmp_path)
        cal = agent.calibration_system
        perf_range, unc_range = cal._get_acquisition_ranges()
        assert perf_range is not None
        assert unc_range is None  # uncertainty is inherently [0,1], not renormalized

    def test_perf_range_min_leq_max(self, tmp_path):
        agent, exp, datamodule = _setup_trained_agent(tmp_path)
        cal = agent.calibration_system
        perf_range, _ = cal._get_acquisition_ranges()
        assert perf_range is not None
        assert perf_range[0] <= perf_range[1]

    def test_perf_range_is_finite(self, tmp_path):
        agent, exp, datamodule = _setup_trained_agent(tmp_path)
        cal = agent.calibration_system
        perf_range, _ = cal._get_acquisition_ranges()
        assert perf_range is not None
        assert np.isfinite(perf_range[0]) and np.isfinite(perf_range[1])


# ===========================================================================
# Normalized acquisition function
# ===========================================================================

class TestNormalizedAcquisitionFunc:
    """The acquisition function with ranges normalizes perf and uncertainty."""

    def test_acquisition_with_ranges_returns_finite(self, tmp_path):
        agent, exp, datamodule = _setup_trained_agent(tmp_path)
        cal = agent.calibration_system
        cal._active_datamodule = datamodule
        bounds = cal._get_global_bounds(datamodule)
        X = np.random.uniform(bounds[:, 0], bounds[:, 1])
        result = cal._acquisition_func(X, w_explore=0.5,
                                        perf_range=(0.2, 0.8),
                                        unc_range=(0.0, 1.0))
        assert np.isfinite(result)

    def test_acquisition_without_ranges_returns_finite(self, tmp_path):
        """Fallback: no normalization when ranges are None."""
        agent, exp, datamodule = _setup_trained_agent(tmp_path)
        cal = agent.calibration_system
        cal._active_datamodule = datamodule
        bounds = cal._get_global_bounds(datamodule)
        X = np.random.uniform(bounds[:, 0], bounds[:, 1])
        result = cal._acquisition_func(X, w_explore=0.5,
                                        perf_range=None,
                                        unc_range=None)
        assert np.isfinite(result)

    def test_normalized_acquisition_is_negative(self, tmp_path):
        """Acquisition returns -score (for minimization)."""
        agent, exp, datamodule = _setup_trained_agent(tmp_path)
        cal = agent.calibration_system
        cal._active_datamodule = datamodule
        bounds = cal._get_global_bounds(datamodule)
        X = np.random.uniform(bounds[:, 0], bounds[:, 1])
        result = cal._acquisition_func(X, w_explore=0.5,
                                        perf_range=(0.2, 0.8),
                                        unc_range=(0.0, 1.0))
        assert result <= 0.0


# ===========================================================================
# build_objective passes bounds for exploration
# ===========================================================================

class TestBuildObjective:
    """_build_objective creates normalized acquisition for exploration mode."""

    def test_exploration_objective_callable(self, tmp_path):
        agent, exp, datamodule = _setup_trained_agent(tmp_path)
        cal = agent.calibration_system
        cal._active_datamodule = datamodule
        bounds = cal._get_global_bounds(datamodule)
        obj = cal._build_objective(Mode.EXPLORATION, w_explore=0.7, bounds=bounds)
        assert callable(obj)

    def test_inference_objective_callable(self, tmp_path):
        agent, exp, datamodule = _setup_trained_agent(tmp_path)
        cal = agent.calibration_system
        cal._active_datamodule = datamodule
        obj = cal._build_objective(Mode.INFERENCE, w_explore=0.0)
        assert callable(obj)


# ===========================================================================
# configure() — mpc_lookahead and mpc_discount
# ===========================================================================

class TestConfigureMPC:
    """agent.configure() can set mpc_lookahead and mpc_discount."""

    def test_mpc_lookahead_default_is_zero(self, tmp_path):
        agent, exp, datamodule = _setup_trained_agent(tmp_path)
        assert agent.calibration_system.default_mpc_lookahead == 0

    def test_mpc_discount_default_is_0_9(self, tmp_path):
        agent, exp, datamodule = _setup_trained_agent(tmp_path)
        assert agent.calibration_system.default_mpc_discount == 0.9

    def test_configure_sets_mpc_lookahead(self, tmp_path):
        agent, exp, datamodule = _setup_trained_agent(tmp_path)
        agent.configure(mpc_lookahead=3)
        assert agent.calibration_system.default_mpc_lookahead == 3

    def test_configure_sets_mpc_discount(self, tmp_path):
        agent, exp, datamodule = _setup_trained_agent(tmp_path)
        agent.configure(mpc_discount=0.8)
        assert agent.calibration_system.default_mpc_discount == 0.8

    def test_configure_mpc_does_not_affect_other_settings(self, tmp_path):
        agent, exp, datamodule = _setup_trained_agent(tmp_path)
        agent.configure(mpc_lookahead=5, mpc_discount=0.7)
        # Other settings should remain default
        assert agent.calibration_system.optimizer == Optimizer.DE
