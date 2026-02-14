import numpy as np
import pytest

from pred_fab.utils import LocalData, Metrics


def test_local_data_roundtrip_parameters_and_features(tmp_path):
    local = LocalData(str(tmp_path))
    local.set_schema("schema_test")

    saved_params = local.save_parameters(["exp_001"], {"exp_001": {"a": 1, "b": 2.0}}, recompute=True)
    saved_feats = local.save_features(
        ["exp_001"],
        {"exp_001": np.array([[0, 0, 1.1], [0, 1, 2.2]])},
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
    assert list(feats["exp_001"].columns) == ["d1", "d2", "feature_x"]
    assert float(feats["exp_001"].iloc[1]["feature_x"]) == 2.2


def test_metrics_handles_constant_targets_and_shape_mismatch():
    perfect = Metrics.calculate_regression_metrics(np.array([3.0, 3.0]), np.array([3.0, 3.0]))
    imperfect = Metrics.calculate_regression_metrics(np.array([3.0, 3.0]), np.array([2.0, 4.0]))

    assert perfect["r2"] == 1.0
    assert imperfect["r2"] == 0.0

    with pytest.raises(ValueError):
        Metrics.calculate_regression_metrics(np.array([1.0, 2.0]), np.array([1.0]))
