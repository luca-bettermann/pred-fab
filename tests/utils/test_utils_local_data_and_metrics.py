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


# R²_adj tests moved to tests/utils/test_r2_adj.py
