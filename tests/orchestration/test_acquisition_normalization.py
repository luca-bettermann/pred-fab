"""
Tests for acquisition function min-max normalization.
"""
import pytest
import numpy as np

from pred_fab.utils.enum import Mode
from tests.utils.builders import (
    build_real_agent_stack,
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
        result = cal._acquisition_objective(X, kappa=0.5, perf_range=(0.2, 0.8))
        assert np.isfinite(result)

    def test_acquisition_without_ranges_returns_finite(self, tmp_path):
        """Fallback: no normalization when ranges are None."""
        agent, exp, datamodule = _setup_trained_agent(tmp_path)
        cal = agent.calibration_system
        cal._active_datamodule = datamodule
        bounds = cal._get_global_bounds(datamodule)
        X = np.random.uniform(bounds[:, 0], bounds[:, 1])
        result = cal._acquisition_objective(X, kappa=0.5, perf_range=None)
        assert np.isfinite(result)

    def test_normalized_acquisition_is_negative(self, tmp_path):
        """Acquisition returns -(non-negative score) for minimization.

        Sampling in the normalized [0,1] domain (where kernels and perfs both
        produce non-negative scores) should give a negative objective value.
        Sampling outside the normalized domain is undefined behaviour for the
        integrated evidence model — we exclude that case here.
        """
        agent, exp, datamodule = _setup_trained_agent(tmp_path)
        cal = agent.calibration_system
        cal._active_datamodule = datamodule
        # Sample in normalized [0,1]^D.
        n_dm = len(datamodule.input_columns)
        np.random.seed(0)
        X = np.random.uniform(0.0, 1.0, size=n_dm)
        # No perf_range renormalization (raw perf ∈ [0,1] already).
        result = cal._acquisition_objective(X, kappa=0.5, perf_range=None)
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
        obj = cal._build_objective(Mode.EXPLORATION, kappa=0.7)
        assert callable(obj)

    def test_inference_objective_callable(self, tmp_path):
        agent, exp, datamodule = _setup_trained_agent(tmp_path)
        cal = agent.calibration_system
        cal._active_datamodule = datamodule
        obj = cal._build_objective(Mode.INFERENCE, kappa=0.0)
        assert callable(obj)
