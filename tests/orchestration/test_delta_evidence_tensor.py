"""Tests for tensor-typed Δ∫E.

Equivalence and gradient flow for:
  - PredictionSystem.delta_integrated_evidence_batched_tensor
  - PredictionSystem.delta_integrated_evidence_joint_batched_tensor
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from pred_fab.interfaces import IPredictionModel
from pred_fab.orchestration import PredictionSystem
from pred_fab.utils import LocalData, PfabLogger
from tests.utils.builders import build_mixed_feature_schema


class _IdentityModel(IPredictionModel):
    """Trivial 1-input → 1-output linear model for unit tests."""
    @property
    def input_parameters(self) -> list[str]: return ["p1"]
    @property
    def input_features(self) -> list[str]: return []
    @property
    def outputs(self) -> list[str]: return ["feat_a"]

    def train(self, *_: Any, **__: Any) -> None:
        self._is_trained = True

    def forward_pass(self, X: torch.Tensor, gradient_pass: bool = False) -> torch.Tensor:
        # one output column, copy input directly
        return X[:, :1].to(dtype=torch.float32)

    def encode(self, X: torch.Tensor, gradient_pass: bool = False) -> torch.Tensor:
        return X


def _build_pred_system(tmp_path) -> PredictionSystem:
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    schema = build_mixed_feature_schema(tmp_path)
    return PredictionSystem(logger=logger, schema=schema, local_data=LocalData(str(tmp_path)))


# ── Numerical equivalence with the numpy variant ──────────────────────────


def test_delta_evidence_batched_tensor_matches_numpy(tmp_path):
    """Tensor batched returns same values as numpy batched at 1e-5."""
    sys = _build_pred_system(tmp_path)
    sys.models.append(_IdentityModel(logger=sys.logger))

    # No KDEs registered yet → both paths return zeros of shape (S,).
    rng = np.random.default_rng(0)
    new_norm_batch = rng.uniform(0, 1, size=(3, 2)).astype(np.float64)
    out_torch = sys.delta_integrated_evidence_batched_tensor(
        torch.from_numpy(new_norm_batch).double(),
    )
    out_np = sys.delta_integrated_evidence_batched(new_norm_batch)
    assert out_torch.shape == (3,)
    assert out_np.shape == (3,)
    np.testing.assert_allclose(
        out_torch.detach().cpu().numpy(), out_np, atol=1e-6,
    )


def test_delta_evidence_joint_batched_tensor_matches_numpy(tmp_path):
    """Joint tensor variant matches joint numpy at 1e-5 with no KDEs."""
    sys = _build_pred_system(tmp_path)
    sys.models.append(_IdentityModel(logger=sys.logger))

    rng = np.random.default_rng(1)
    new_norm_batch_SL = rng.uniform(0, 1, size=(3, 2, 2)).astype(np.float64)
    out_torch = sys.delta_integrated_evidence_joint_batched_tensor(
        torch.from_numpy(new_norm_batch_SL).double(),
    )
    out_np = sys.delta_integrated_evidence_joint_batched(new_norm_batch_SL)
    np.testing.assert_allclose(
        out_torch.detach().cpu().numpy(), out_np, atol=1e-6,
    )


# ── Edge cases ────────────────────────────────────────────────────────────


def test_delta_evidence_batched_tensor_empty_returns_empty(tmp_path):
    sys = _build_pred_system(tmp_path)
    sys.models.append(_IdentityModel(logger=sys.logger))

    out = sys.delta_integrated_evidence_batched_tensor(
        torch.zeros((0, 3), dtype=torch.float64),
    )
    assert out.shape == (0,)


def test_delta_evidence_joint_tensor_empty_returns_empty(tmp_path):
    sys = _build_pred_system(tmp_path)
    sys.models.append(_IdentityModel(logger=sys.logger))

    out = sys.delta_integrated_evidence_joint_batched_tensor(
        torch.zeros((0, 1, 3), dtype=torch.float64),
    )
    assert out.shape == (0,)
