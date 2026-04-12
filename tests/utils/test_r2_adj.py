"""Tests for performance-weighted R²_adj."""

import numpy as np
import pytest

from pred_fab.utils import Metrics


def test_adjusted_r2_returns_expected_keys():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    result = Metrics.calculate_adjusted_r2(y, y, importance=np.array([0.5, 0.5, 0.5, 0.5]))
    assert set(result.keys()) == {"r2", "r2_adj", "n_samples"}
    assert result["n_samples"] == 4


def test_adjusted_r2_uniform_importance_equals_r2():
    """With constant importance, all weights are equal -> r2_adj equals r2."""
    rng = np.random.default_rng(7)
    y_true = rng.normal(5, 2, 30)
    y_pred = y_true + rng.normal(0, 0.5, 30)
    importance = np.full(30, 0.6)

    result = Metrics.calculate_adjusted_r2(y_true, y_pred, importance)
    assert result["r2_adj"] == pytest.approx(result["r2"], abs=1e-10)


def test_adjusted_r2_higher_when_high_importance_predicted_better():
    """If high-importance samples are predicted well, r2_adj > r2."""
    n = 100
    rng = np.random.default_rng(123)
    y_true = rng.uniform(0, 10, n)
    importance = rng.uniform(0, 1, n)

    noise = np.where(importance > 0.5, 0.0, rng.normal(0, 3.0, n))
    y_pred = y_true + noise

    result = Metrics.calculate_adjusted_r2(y_true, y_pred, importance)
    assert result["r2_adj"] > result["r2"]


def test_adjusted_r2_lower_when_high_importance_predicted_worse():
    """If high-importance samples are predicted poorly, r2_adj < r2."""
    n = 100
    rng = np.random.default_rng(456)
    y_true = rng.uniform(0, 10, n)
    importance = rng.uniform(0, 1, n)

    noise = np.where(importance > 0.5, rng.normal(0, 3.0, n), 0.0)
    y_pred = y_true + noise

    result = Metrics.calculate_adjusted_r2(y_true, y_pred, importance)
    assert result["r2_adj"] < result["r2"]


def test_adjusted_r2_empty_arrays():
    result = Metrics.calculate_adjusted_r2(np.array([]), np.array([]), np.array([]))
    assert result["r2"] == 0.0
    assert result["r2_adj"] == 0.0
    assert result["n_samples"] == 0


def test_adjusted_r2_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        Metrics.calculate_adjusted_r2(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([0.5])
        )
