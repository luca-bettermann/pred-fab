import numpy as np
import pytest

from pred_fab.utils import LocalData, Metrics


def test_local_data_roundtrip_parameters_and_features(tmp_path):
    local = LocalData(str(tmp_path))
    local.set_schema("schema_test")

    saved_params = local.save_parameters(["exp_001"], {"exp_001": {"a": 1, "b": 2.0}}, recompute=True)
    saved_feats = local.save_features(
        ["exp_001"],
        {"exp_001": np.array([[0, 0, 1.1], [0, 1, 2.2]])},  # type: ignore[arg-type]
        recompute=True,
        feature_name="feature_x",
        column_names=["d1", "d2", "feature_x"],
    )
    assert saved_params is True
    assert saved_feats is True

    missing_p, params = local.load_parameters(["exp_001", "exp_999"])
    missing_f, feats = local.load_features(["exp_001", "exp_999"], feature_name="feature_x")

    assert missing_p == ["exp_999"]
    assert missing_f == ["exp_999"]
    assert params["exp_001"]["a"] == 1
    assert list(feats["exp_001"].columns) == ["d1", "d2", "feature_x"]  # type: ignore[union-attr]
    assert float(feats["exp_001"].iloc[1]["feature_x"]) == 2.2  # type: ignore[union-attr]


def test_metrics_handles_constant_targets_and_shape_mismatch():
    perfect = Metrics.calculate_regression_metrics(np.array([3.0, 3.0]), np.array([3.0, 3.0]))
    imperfect = Metrics.calculate_regression_metrics(np.array([3.0, 3.0]), np.array([2.0, 4.0]))

    assert perfect["r2"] == 1.0
    assert imperfect["r2"] == 0.0

    with pytest.raises(ValueError):
        Metrics.calculate_regression_metrics(np.array([1.0, 2.0]), np.array([1.0]))


def test_local_data_save_load_performance_roundtrip(tmp_path):
    local = LocalData(str(tmp_path))
    local.set_schema("schema_test")

    saved = local.save_performance(
        ["exp_001"],
        {"exp_001": {"performance_1": 0.85, "performance_2": 0.42}},
        recompute=True,
    )
    assert saved is True

    missing, perf = local.load_performance(["exp_001"])
    assert missing == []
    assert perf["exp_001"]["performance_1"] == pytest.approx(0.85)
    assert perf["exp_001"]["performance_2"] == pytest.approx(0.42)


def test_local_data_save_with_recompute_false_skips_existing_file(tmp_path):
    local = LocalData(str(tmp_path))
    local.set_schema("schema_test")

    local.save_parameters(["exp_001"], {"exp_001": {"a": 1}}, recompute=True)
    result = local.save_parameters(["exp_001"], {"exp_001": {"a": 99}}, recompute=False)
    assert result is False

    _, params = local.load_parameters(["exp_001"])
    assert params["exp_001"]["a"] == 1  # Original value preserved


def test_metrics_returns_all_expected_keys():
    result = Metrics.calculate_regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.1]))
    assert set(result.keys()) == {"mae", "rmse", "r2", "n_samples"}
    assert result["n_samples"] == 3


def test_metrics_empty_arrays_return_zeros():
    result = Metrics.calculate_regression_metrics(np.array([]), np.array([]))
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["r2"] == 0.0
    assert result["n_samples"] == 0


# ── Performance-adjusted R² ─────────────────────────────────────────────────

def test_adjusted_r2_returns_expected_keys():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    result = Metrics.calculate_adjusted_r2(y, y, importance=np.array([0.5, 0.5, 0.5, 0.5]))
    assert set(result.keys()) == {"r2", "r2_adj", "n_samples"}
    assert result["n_samples"] == 4


def test_adjusted_r2_equals_r2_when_alpha_is_one():
    """When alpha=1, weights are uniform → r2_adj must equal r2."""
    rng = np.random.default_rng(42)
    y_true = rng.normal(0, 1, 50)
    y_pred = y_true + rng.normal(0, 0.3, 50)
    importance = rng.uniform(0, 1, 50)

    result = Metrics.calculate_adjusted_r2(y_true, y_pred, importance, alpha=1.0)
    assert result["r2_adj"] == pytest.approx(result["r2"], abs=1e-10)


def test_adjusted_r2_uniform_importance_equals_r2():
    """With constant importance, weights are constant → r2_adj equals r2."""
    rng = np.random.default_rng(7)
    y_true = rng.normal(5, 2, 30)
    y_pred = y_true + rng.normal(0, 0.5, 30)
    importance = np.full(30, 0.6)

    result = Metrics.calculate_adjusted_r2(y_true, y_pred, importance, alpha=0.0)
    assert result["r2_adj"] == pytest.approx(result["r2"], abs=1e-10)


def test_adjusted_r2_higher_when_high_importance_predicted_better():
    """If high-importance samples are predicted well, up-weighting them raises R²_adj."""
    n = 100
    rng = np.random.default_rng(123)
    y_true = rng.uniform(0, 10, n)
    importance = rng.uniform(0, 1, n)

    # Make predictions perfect for high-importance, noisy for low-importance
    noise = np.where(importance > 0.5, 0.0, rng.normal(0, 3.0, n))
    y_pred = y_true + noise

    result = Metrics.calculate_adjusted_r2(y_true, y_pred, importance, alpha=0.0)
    # Up-weighting well-predicted important samples → r2_adj > r2
    assert result["r2_adj"] > result["r2"]


def test_adjusted_r2_lower_when_high_importance_predicted_worse():
    """If high-importance samples are predicted poorly, up-weighting them lowers R²_adj."""
    n = 100
    rng = np.random.default_rng(456)
    y_true = rng.uniform(0, 10, n)
    importance = rng.uniform(0, 1, n)

    # Make predictions noisy for high-importance, perfect for low-importance
    noise = np.where(importance > 0.5, rng.normal(0, 3.0, n), 0.0)
    y_pred = y_true + noise

    result = Metrics.calculate_adjusted_r2(y_true, y_pred, importance, alpha=0.0)
    # Up-weighting poorly-predicted important samples → r2_adj < r2
    assert result["r2_adj"] < result["r2"]


def test_adjusted_r2_gap_scales_with_alpha():
    """The gap |r2_adj - r2| should shrink as alpha increases toward 1."""
    n = 80
    rng = np.random.default_rng(789)
    y_true = rng.uniform(0, 10, n)
    importance = rng.uniform(0, 1, n)
    noise = np.where(importance > 0.5, rng.normal(0, 2.0, n), 0.0)
    y_pred = y_true + noise

    gaps = []
    for a in [0.0, 0.3, 0.7, 1.0]:
        r = Metrics.calculate_adjusted_r2(y_true, y_pred, importance, alpha=a)
        gaps.append(abs(r["r2_adj"] - r["r2"]))

    # Gaps should be monotonically non-increasing
    for i in range(len(gaps) - 1):
        assert gaps[i] >= gaps[i + 1] - 1e-10


def test_adjusted_r2_empty_arrays():
    result = Metrics.calculate_adjusted_r2(np.array([]), np.array([]), np.array([]))
    assert result["r2"] == 0.0
    assert result["r2_adj"] == 0.0
    assert result["n_samples"] == 0


def test_adjusted_r2_symmetric_flag_ignored_when_importance_precomputed():
    """The symmetric flag is a caller hint; the method itself just uses the importance array.
    Verify it doesn't change the result when the same importance is passed."""
    rng = np.random.default_rng(99)
    y_true = rng.normal(5, 2, 30)
    y_pred = y_true + rng.normal(0, 0.5, 30)
    importance = rng.uniform(0, 1, 30)

    r_sym = Metrics.calculate_adjusted_r2(y_true, y_pred, importance, symmetric=True)
    r_asym = Metrics.calculate_adjusted_r2(y_true, y_pred, importance, symmetric=False)
    # Same importance → same result regardless of flag
    assert r_sym["r2_adj"] == pytest.approx(r_asym["r2_adj"], abs=1e-10)


def test_adjusted_r2_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        Metrics.calculate_adjusted_r2(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([0.5])
        )
