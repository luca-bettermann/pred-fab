"""Tests for tensor-typed evaluation with the unified interface.

Guarantees:
  1. _score_tensor produces correct values and gradients.
  2. _evaluate_feature_dict_tensor returns dict[perf_code, (S,) tensor]
     with NaN for candidates missing a required feature.
"""

from typing import Any
import numpy as np
import pytest
import torch

from pred_fab.core.data_blocks import Parameters as ParametersBlock
from pred_fab.core.data_objects import Parameter
from pred_fab.interfaces import IEvaluationModel
from pred_fab.orchestration import EvaluationSystem
from pred_fab.utils import PfabLogger


class _SimpleEval(IEvaluationModel):
    """target=0, scaling=1: perf = clamp(1 - |feat|, 0, 1)."""

    @property
    def input_parameters(self) -> list[str]: return []
    @property
    def input_features(self) -> list[str]: return ["feat_a"]
    @property
    def output_performance(self) -> str: return "perf_a"

    def _score_row(self, feature_values, params, **dims):
        return float(np.clip(1.0 - abs(feature_values["feat_a"]), 0.0, 1.0))

    def _score_tensor(self, feature_tensors, parameters_list):
        feat = feature_tensors["feat_a"]
        perfs = 1.0 - feat.abs()
        perfs = torch.clamp(perfs, 0.0, 1.0)
        nan_mask = torch.isnan(feat)
        perfs_safe = torch.where(nan_mask, torch.zeros_like(perfs), perfs)
        valid = (~nan_mask).sum(dim=1).to(perfs.dtype)
        safe = torch.where(valid > 0, valid, torch.ones_like(valid))
        avgs = perfs_safe.sum(dim=1) / safe
        return torch.where(valid > 0, avgs, torch.full_like(avgs, float('nan')))


def _params_block(values: dict[str, float]) -> ParametersBlock:
    block = ParametersBlock.from_list([
        Parameter.real(code, min_val=-100.0, max_val=100.0)
        for code in values
    ])
    block.set_values_from_dict(values, PfabLogger.get_logger("/tmp/eval_tensor_test"))
    return block


# ── Numerical correctness ────────────────────────────────────────────────


def test_score_tensor_basic(tmp_path):
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    model = _SimpleEval(logger=logger)

    feat = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.5, 0.5, 0.5],
    ], dtype=torch.float32)
    params = [_params_block({"x": 1.0}), _params_block({"x": 2.0})]

    avgs = model.compute_performance_tensor({"feat_a": feat}, params)
    expected_0 = (0.9 + 0.8 + 0.7) / 3
    expected_1 = 0.5
    np.testing.assert_allclose(avgs.detach().numpy(), [expected_0, expected_1], atol=1e-5)


def test_score_tensor_nan_handling(tmp_path):
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    model = _SimpleEval(logger=logger)

    feat = torch.tensor([
        [0.1, 0.2, float('nan'), 0.4],
        [0.5, 0.5, 0.5, 0.5],
    ], dtype=torch.float32)
    params = [_params_block({"x": 1.0}), _params_block({"x": 2.0})]

    avgs = model.compute_performance_tensor({"feat_a": feat}, params)
    expected_0 = (0.9 + 0.8 + 0.6) / 3
    expected_1 = 0.5
    np.testing.assert_allclose(avgs.detach().numpy(), [expected_0, expected_1], atol=1e-5)


def test_score_tensor_clamp(tmp_path):
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    model = _SimpleEval(logger=logger)

    feat = torch.tensor([[0.0, 0.5, 1.0, 2.0, -1.0, -2.0]], dtype=torch.float32)
    params = [_params_block({"x": 1.0})]

    avgs = model.compute_performance_tensor({"feat_a": feat}, params)
    expected = (1.0 + 0.5 + 0.0 + 0.0 + 0.0 + 0.0) / 6
    np.testing.assert_allclose(avgs.detach().numpy(), [expected], atol=1e-5)


# ── Gradient flow ────────────────────────────────────────────────────────


def test_grad_flows_through_eval(tmp_path):
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    model = _SimpleEval(logger=logger)

    feat = torch.tensor([[0.3, 0.4, 0.2]], requires_grad=True, dtype=torch.float32)
    params = [_params_block({"x": 1.0})]

    avg = model.compute_performance_tensor({"feat_a": feat}, params)
    avg.sum().backward()

    assert feat.grad is not None
    assert torch.isfinite(feat.grad).all()
    expected = -1.0 / 3.0
    np.testing.assert_allclose(feat.grad.numpy(), [[expected, expected, expected]], atol=1e-5)


def test_grad_zero_at_clamp_boundary(tmp_path):
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    model = _SimpleEval(logger=logger)

    feat = torch.tensor([[2.0, 3.0, -2.0]], requires_grad=True, dtype=torch.float32)
    params = [_params_block({"x": 1.0})]

    avg = model.compute_performance_tensor({"feat_a": feat}, params)
    avg.sum().backward()

    assert feat.grad is not None
    np.testing.assert_allclose(feat.grad.numpy(), [[0.0, 0.0, 0.0]], atol=1e-7)


# ── EvaluationSystem dispatch ────────────────────────────────────────────


def test_evaluate_feature_dict_tensor_dispatches(tmp_path):
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    model = _SimpleEval(logger=logger)
    sys = EvaluationSystem(logger=logger)
    sys.models.append(model)

    features_dicts_S = [
        {"feat_a": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)},
        {"feat_a": torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)},
    ]
    params = [_params_block({"x": 1.0}), _params_block({"x": 2.0})]

    result = sys._evaluate_feature_dict_tensor(features_dicts_S, params)
    assert "perf_a" in result
    assert isinstance(result["perf_a"], torch.Tensor)
    assert result["perf_a"].shape == (2,)


def test_evaluate_tensor_missing_feature_yields_nan(tmp_path):
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    model = _SimpleEval(logger=logger)
    sys = EvaluationSystem(logger=logger)
    sys.models.append(model)

    features_dicts_S = [
        {"feat_a": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)},
        {},
    ]
    params = [_params_block({"x": 1.0}), _params_block({"x": 2.0})]

    result = sys._evaluate_feature_dict_tensor(features_dicts_S, params)
    assert torch.isfinite(result["perf_a"][0])
    assert torch.isnan(result["perf_a"][1])


def test_evaluate_tensor_empty_list_returns_empty(tmp_path):
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    model = _SimpleEval(logger=logger)
    sys = EvaluationSystem(logger=logger)
    sys.models.append(model)

    result = sys._evaluate_feature_dict_tensor([], [])
    assert result == {}
