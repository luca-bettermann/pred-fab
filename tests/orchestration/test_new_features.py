"""Tests for new features: optimizer config, virtual KDE points, trajectory smoothing, perf range."""

import pytest
import numpy as np

from pred_fab.orchestration.calibration import Optimizer
from tests.utils.builders import build_real_agent_stack


def _setup_trained_agent(tmp_path):
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)
    return agent, exp, datamodule


# ===========================================================================
# Optimizer configuration
# ===========================================================================

class TestOptimizerConfig:
    """agent.configure() sets optimizer parameters."""

    def test_default_optimizer_is_de(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        assert agent.calibration_system.optimizer == Optimizer.DE

    def test_default_online_optimizer_is_lbfgsb(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        assert agent.calibration_system.online_optimizer == Optimizer.LBFGSB

    def test_configure_de_maxiter(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        agent.configure(de_maxiter=200)
        assert agent.calibration_system.de_maxiter == 200

    def test_configure_de_popsize(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        agent.configure(de_popsize=20)
        assert agent.calibration_system.de_popsize == 20

    def test_configure_lbfgsb_maxfun(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        agent.configure(lbfgsb_maxfun=500)
        assert agent.calibration_system.lbfgsb_maxfun == 500

    def test_configure_lbfgsb_eps(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        agent.configure(lbfgsb_eps=0.01)
        assert agent.calibration_system.lbfgsb_eps == 0.01

    def test_configure_online_optimizer(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        agent.configure(online_optimizer=Optimizer.DE)
        assert agent.calibration_system.online_optimizer == Optimizer.DE

    def test_configure_trajectory_smoothing(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        agent.configure(trajectory_smoothing=0.2)
        assert agent.calibration_system.trajectory_smoothing == 0.2


# ===========================================================================
# Performance range tracking
# ===========================================================================

class TestPerfRange:
    """Running perf range is updated after train()."""

    def test_perf_range_set_after_train(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        cal = agent.calibration_system
        assert cal._perf_range_min is not None
        assert cal._perf_range_max is not None

    def test_perf_range_min_leq_max(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        cal = agent.calibration_system
        assert cal._perf_range_min <= cal._perf_range_max

    def test_get_acquisition_ranges_uses_perf_range(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        cal = agent.calibration_system
        perf_range, unc_range = cal._get_acquisition_ranges()
        assert perf_range == (cal._perf_range_min, cal._perf_range_max)
        assert unc_range is None  # uncertainty not renormalized


# ===========================================================================
# Virtual KDE points
# ===========================================================================

class TestVirtualKDEPoints:
    """Virtual points modify KDE and are cleared after use."""

    def test_add_virtual_point_changes_uncertainty(self, tmp_path):
        agent, _, dm = _setup_trained_agent(tmp_path)
        params = {"param_1": 2.5, "dim_1": 2, "dim_2": 3}
        X = dm.params_to_array(params)

        u_before = agent.pred_system.uncertainty(X)
        agent.pred_system.add_virtual_point(params, dm)
        u_after = agent.pred_system.uncertainty(X)

        # Uncertainty should decrease at the virtual point
        assert u_after <= u_before

    def test_clear_virtual_points_restores_uncertainty(self, tmp_path):
        agent, _, dm = _setup_trained_agent(tmp_path)
        params = {"param_1": 2.5, "dim_1": 2, "dim_2": 3}
        X = dm.params_to_array(params)

        u_before = agent.pred_system.uncertainty(X)
        agent.pred_system.add_virtual_point(params, dm)
        agent.pred_system.clear_virtual_points()
        u_after = agent.pred_system.uncertainty(X)

        assert abs(u_after - u_before) < 1e-10

    def test_multiple_virtual_points_cleared(self, tmp_path):
        agent, _, dm = _setup_trained_agent(tmp_path)
        params1 = {"param_1": 1.0, "dim_1": 2, "dim_2": 3}
        params2 = {"param_1": 4.0, "dim_1": 2, "dim_2": 3}
        X = dm.params_to_array(params1)

        u_before = agent.pred_system.uncertainty(X)
        agent.pred_system.add_virtual_point(params1, dm)
        agent.pred_system.add_virtual_point(params2, dm)
        agent.pred_system.clear_virtual_points()
        u_after = agent.pred_system.uncertainty(X)

        assert abs(u_after - u_before) < 1e-10


# ===========================================================================
# Trajectory smoothing
# ===========================================================================

class TestTrajectorySmoothing:
    """Trajectory smoothing parameter is stored and defaults to 0."""

    def test_default_smoothing_is_zero(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        assert agent.calibration_system.trajectory_smoothing == 0.0

    def test_smoothing_configurable(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        agent.configure(trajectory_smoothing=0.15)
        assert agent.calibration_system.trajectory_smoothing == 0.15


# ===========================================================================
# Console output (no emojis)
# ===========================================================================

class TestConsoleCleanOutput:
    """Console output uses ANSI checkmarks, not emojis."""

    def test_console_success_no_emoji(self, tmp_path, capsys):
        agent, _, _ = _setup_trained_agent(tmp_path)
        agent.logger.console_success("test message")
        captured = capsys.readouterr().out
        assert "✓" in captured
        assert "✅" not in captured

    def test_console_warning_no_emoji(self, tmp_path, capsys):
        agent, _, _ = _setup_trained_agent(tmp_path)
        agent.logger.console_warning("test warning")
        captured = capsys.readouterr().out
        assert "!" in captured
        assert "⚠" not in captured
