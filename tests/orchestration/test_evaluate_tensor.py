"""Tests for tensor-typed evaluation.

Three guarantees this commit promises:
  1. Numerical equivalence with compute_performance_batched (numpy path)
     when TARGETS_CONSTANT=True — same affine arithmetic, same clamp.
  2. Gradient flow: feature_values_S with requires_grad=True produces
     finite gradients on backward of the (S,) perf output.
  3. _evaluate_feature_dict_tensor returns a flat
     dict[perf_code, torch.Tensor (S,)] mirroring the per-candidate axis,
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


class _ConstantTargetEval(IEvaluationModel):
    """Constant target = 0, scaling = 1 — minimal model for unit testing."""
    TARGETS_CONSTANT = True

    @property
    def input_parameters(self) -> list[str]: return []
    @property
    def input_feature(self) -> str: return "feat_a"
    @property
    def output_performance(self) -> str: return "perf_a"

    def _compute_target_value(self, params: dict, **dimensions: Any) -> float:
        return 0.0
    def _compute_scaling_factor(self, params: dict, **dimensions: Any) -> float | None:
        return 1.0


def _params_block(values: dict[str, float]) -> ParametersBlock:
    block = ParametersBlock.from_list([
        Parameter.real(code, min_val=-100.0, max_val=100.0)
        for code in values
    ])
    block.set_values_from_dict(values, PfabLogger.get_logger("/tmp/eval_tensor_test"))
    return block


# ── Numerical equivalence ─────────────────────────────────────────────────


def test_tensor_matches_batched_numerical(tmp_path):
    """compute_performance_tensor matches compute_performance_batched at 1e-6."""
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    model = _ConstantTargetEval(logger=logger)

    rng = np.random.default_rng(0)
    S = 4
    n_rows = 6
    feat_values = rng.uniform(-0.5, 0.5, size=(S, n_rows))

    # Numpy batched path expects (n_rows, n_iter_dims+1) with iterator-index columns.
    # Build trivial-iterator (1 iter dim) tabular form for a fair comparison.
    indices = np.arange(n_rows).reshape(-1, 1)
    feat_arrays_numpy = [
        np.concatenate([indices, feat_values[s].reshape(-1, 1)], axis=1)
        for s in range(S)
    ]
    params_blocks = [_params_block({"x": float(s)}) for s in range(S)]

    avgs_numpy = model.compute_performance_batched(feat_arrays_numpy, params_blocks)
    avgs_tensor = model.compute_performance_tensor(
        torch.from_numpy(feat_values).float(), params_blocks,
    )

    np.testing.assert_allclose(
        avgs_tensor.detach().cpu().numpy(),
        np.array([float(a) for a in avgs_numpy], dtype=np.float32),
        atol=1e-5, rtol=1e-5,
    )


def test_tensor_handles_nan_feature_values(tmp_path):
    """NaN feature values are excluded from the mean (gradient-aware)."""
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    model = _ConstantTargetEval(logger=logger)

    feat_values = torch.tensor([
        [0.1, 0.2, float('nan'), 0.4],
        [0.5, 0.5, 0.5, 0.5],
    ], dtype=torch.float32)
    params = [_params_block({"x": 1.0}), _params_block({"x": 2.0})]

    avgs = model.compute_performance_tensor(feat_values, params)
    # Row 0 has 3 valid values: perf = 1 - |feat|, mean of (0.9, 0.8, 0.6) = 0.7666...
    expected_0 = (0.9 + 0.8 + 0.6) / 3
    # Row 1: all 0.5 → perf = 0.5 each, mean = 0.5
    expected_1 = 0.5
    np.testing.assert_allclose(avgs.detach().cpu().numpy(), [expected_0, expected_1], atol=1e-5)


def test_tensor_clamps_at_unit_interval(tmp_path):
    """1 - |feat - target| / scaling clamps to [0, 1]."""
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    model = _ConstantTargetEval(logger=logger)

    feat_values = torch.tensor([
        [0.0, 0.5, 1.0, 2.0, -1.0, -2.0],  # feat=0 → perf=1; feat=2 → perf=−1 → clamp 0
    ], dtype=torch.float32)
    params = [_params_block({"x": 1.0})]

    avgs = model.compute_performance_tensor(feat_values, params)
    # perfs = [1.0, 0.5, 0.0, 0.0 (clamped), 0.0, 0.0 (clamped)]
    expected = (1.0 + 0.5 + 0.0 + 0.0 + 0.0 + 0.0) / 6
    np.testing.assert_allclose(avgs.detach().cpu().numpy(), [expected], atol=1e-5)


# ── Gradient flow ─────────────────────────────────────────────────────────


def test_grad_flows_through_eval(tmp_path):
    """feature_values_S.requires_grad_(True) → eval → backward propagates."""
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    model = _ConstantTargetEval(logger=logger)

    # Pick feature values inside the [0, 1] perf-clamp range so gradient is non-zero.
    feat_values = torch.tensor([[0.3, 0.4, 0.2]], requires_grad=True, dtype=torch.float32)
    params = [_params_block({"x": 1.0})]

    avg = model.compute_performance_tensor(feat_values, params)
    avg.sum().backward()

    assert feat_values.grad is not None
    assert torch.isfinite(feat_values.grad).all()
    # d/dfeat [1 - |feat| / 1] = -sign(feat) / n_rows for in-clamp feat values.
    # All positive feat → all gradients = -1/3.
    expected = -1.0 / 3.0
    np.testing.assert_allclose(feat_values.grad.numpy(), [[expected, expected, expected]], atol=1e-5)


def test_grad_zero_at_clamp_boundary(tmp_path):
    """Saturated perf (feat outside [target ± scaling]) has zero gradient."""
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    model = _ConstantTargetEval(logger=logger)

    # |feat| > scaling=1 → clamp 0 → grad 0.
    feat_values = torch.tensor([[2.0, 3.0, -2.0]], requires_grad=True, dtype=torch.float32)
    params = [_params_block({"x": 1.0})]

    avg = model.compute_performance_tensor(feat_values, params)
    avg.sum().backward()

    # All gradients should be zero (clamp saturation).
    assert feat_values.grad is not None
    np.testing.assert_allclose(feat_values.grad.numpy(), [[0.0, 0.0, 0.0]], atol=1e-7)


# ── EvaluationSystem dispatch ─────────────────────────────────────────────


def test_evaluate_feature_dict_tensor_dispatches(tmp_path):
    """_evaluate_feature_dict_tensor returns dict[perf_code, (S,) tensor]."""
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    model = _ConstantTargetEval(logger=logger)
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
    """Candidate missing the required feature gets NaN in the output."""
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    model = _ConstantTargetEval(logger=logger)
    sys = EvaluationSystem(logger=logger)
    sys.models.append(model)

    features_dicts_S = [
        {"feat_a": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)},
        {},  # missing feat_a
    ]
    params = [_params_block({"x": 1.0}), _params_block({"x": 2.0})]

    result = sys._evaluate_feature_dict_tensor(features_dicts_S, params)
    assert torch.isfinite(result["perf_a"][0])
    assert torch.isnan(result["perf_a"][1])


def test_evaluate_tensor_empty_list_returns_empty(tmp_path):
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    model = _ConstantTargetEval(logger=logger)
    sys = EvaluationSystem(logger=logger)
    sys.models.append(model)

    result = sys._evaluate_feature_dict_tensor([], [])
    assert result == {}
