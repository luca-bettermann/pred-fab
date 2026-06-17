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
    assert set(result.keys()) == {"mae", "r2", "n_samples"}
    assert result["n_samples"] == 3


def test_metrics_empty_arrays_return_zeros():
    result = Metrics.calculate_regression_metrics(np.array([]), np.array([]))
    assert result["mae"] == 0.0
    assert result["r2"] == 0.0
    assert result["n_samples"] == 0


# R²_inf tests moved to tests/utils/test_r2_inf.py


def test_combined_score_renormalises_over_present_keys():
    from pred_fab.utils import combined_score
    weights = {"a": 1.0, "b": 1.0, "c": 2.0}
    # full set: (1*0.5 + 1*0.5 + 2*1.0) / 4 = 0.75
    assert combined_score({"a": 0.5, "b": 0.5, "c": 1.0}, weights) == pytest.approx(0.75)
    # 'c' missing: must renormalise over a,b only -> (0.5+0.5)/2 = 0.5, NOT /4 = 0.25
    assert combined_score({"a": 0.5, "b": 0.5}, weights) == pytest.approx(0.5)
    # 'c' present but None: same as missing
    assert combined_score({"a": 0.5, "b": 0.5, "c": None}, weights) == pytest.approx(0.5)


def test_combined_score_ignores_unweighted_keys():
    from pred_fab.utils import combined_score
    weights = {"a": 1.0}
    # 'b' has no weight: contributes nothing to numerator or denominator
    assert combined_score({"a": 0.8, "b": 0.2}, weights) == pytest.approx(0.8)


def test_combined_score_preserves_gradient_with_tensors():
    import torch
    from pred_fab.utils import combined_score
    a = torch.tensor(0.4, requires_grad=True)
    b = torch.tensor(0.6, requires_grad=True)
    out = combined_score({"a": a, "b": b}, {"a": 1.0, "b": 3.0})
    out.backward()
    assert a.grad is not None and b.grad is not None
    # weight-normalised gradients: dout/da = 1/4, dout/db = 3/4
    assert float(a.grad) == pytest.approx(0.25)
    assert float(b.grad) == pytest.approx(0.75)


def test_combined_score_zero_total_weight_returns_zero():
    from pred_fab.utils import combined_score
    assert combined_score({"a": 0.5}, {"a": 0.0}) == 0.0
    assert combined_score({}, {"a": 1.0}) == 0.0


def test_importance_weight_matches_sigmoid_and_is_shared():
    import numpy as np
    from pred_fab.utils import importance_weight
    from pred_fab.utils.metrics import IMPORTANCE_FLOOR, IMPORTANCE_STEEPNESS
    scores = np.array([0.2, 0.5, 0.8])
    w = importance_weight(scores)
    # bounded in [floor, 1]; monotonic in score
    assert np.all(w >= IMPORTANCE_FLOOR - 1e-9) and np.all(w <= 1.0 + 1e-9)
    assert w[0] < w[1] < w[2]
    # ref_scores anchors k/mean: evaluating at the mean gives the sigmoid midpoint
    mid = importance_weight(np.array([scores.mean()]), ref_scores=scores)[0]
    assert mid == pytest.approx(IMPORTANCE_FLOOR + (1 - IMPORTANCE_FLOOR) * 0.5)


def test_importance_weight_zero_std_is_flat():
    import numpy as np
    from pred_fab.utils import importance_weight
    from pred_fab.utils.metrics import IMPORTANCE_FLOOR
    w = importance_weight(np.array([0.5, 0.5, 0.5]))
    # k=0 → sigmoid=0.5 everywhere → constant weight
    assert np.allclose(w, IMPORTANCE_FLOOR + (1 - IMPORTANCE_FLOOR) * 0.5)


def test_combined_score_zero_weight_preserves_tensor_grad():
    import torch
    from pred_fab.utils import combined_score
    v = torch.tensor(0.7, requires_grad=True)
    out = combined_score({"a": v}, {"a": 0.0})  # total weight 0
    assert torch.is_tensor(out) and out.requires_grad  # grad-bearing zero, not bare float
    assert float(out) == 0.0


def test_regression_metrics_rejects_2d_shape_mismatch():
    import numpy as np
    from pred_fab.utils import Metrics
    yt = np.zeros((4, 2))
    yp = np.zeros((4, 3))  # same len (4), different shape
    with pytest.raises(ValueError, match="Shape mismatch"):
        Metrics.calculate_regression_metrics(yt, yp)
