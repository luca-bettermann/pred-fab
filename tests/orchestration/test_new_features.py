"""Tests for new features: optimizer config, virtual KDE points, schedule smoothing, perf range."""

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
    """agent.configure_optimizer() sets optimizer parameters."""

    def test_default_optimizer_is_de(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        assert agent.calibration_system.optimizer == Optimizer.DE

    def test_default_online_optimizer_is_lbfgsb(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        assert agent.calibration_system.online_optimizer == Optimizer.LBFGSB

    def test_configure_de_maxiter(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        agent.configure_optimizer(de_maxiter=200)
        assert agent.calibration_system.de_maxiter == 200

    def test_configure_de_popsize(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        agent.configure_optimizer(de_popsize=20)
        assert agent.calibration_system.de_popsize == 20

    def test_configure_lbfgsb_maxfun(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        agent.configure_optimizer(lbfgsb_maxfun=500)
        assert agent.calibration_system.lbfgsb_maxfun == 500

    def test_configure_lbfgsb_eps(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        agent.configure_optimizer(lbfgsb_eps=0.01)
        assert agent.calibration_system.lbfgsb_eps == 0.01

    def test_configure_online_optimizer(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        agent.configure_optimizer(online_backend=Optimizer.DE)
        assert agent.calibration_system.online_optimizer == Optimizer.DE

    def test_configure_schedule_smoothing(self, tmp_path):
        agent, _, _ = _setup_trained_agent(tmp_path)
        agent.calibration_system.schedule_smoothing = 0.2
        assert agent.calibration_system.schedule_smoothing == 0.2


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
# Virtual stacking (push/pop) for sequential Phase-2 mode
# ===========================================================================

class TestVirtualStacking:
    """push_virtual_points temporarily adds weighted kernel points; pop restores."""

    def test_push_and_pop_round_trip(self, tmp_path):
        agent, _, dm = _setup_trained_agent(tmp_path)
        pred = agent.pred_system
        if not pred._model_kdes:
            pytest.skip("KDE not fitted")

        # Snapshot counts
        counts_before = {id_: (kde.latent_points.shape, kde.point_weights.shape)
                         for id_, kde in pred._model_kdes.items()}

        params_list = [{"param_1": 2.5, "dim_1": 2, "dim_2": 3},
                       {"param_1": 4.0, "dim_1": 2, "dim_2": 3}]
        pred.push_virtual_points(params_list, weights_list=[5.0, 3.0])

        for kde in pred._model_kdes.values():
            # Two points added with weights 5.0 and 3.0.
            assert kde.point_weights[-1] == pytest.approx(3.0)
            assert kde.point_weights[-2] == pytest.approx(5.0)

        pred.pop_virtual_points()
        for id_, kde in pred._model_kdes.items():
            assert kde.latent_points.shape == counts_before[id_][0]
            assert kde.point_weights.shape == counts_before[id_][1]


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
