"""Tests for tensor-typed Δ∫E.

Equivalence and gradient flow for:
  - PredictionSystem.delta_integrated_evidence_batched
  - PredictionSystem.delta_integrated_evidence_joint_batched
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

    @property
    def domain_spec(self) -> tuple[str | None, int | list[int]]: return None, 0

    def train(self, *_: Any, **__: Any) -> None:
        self._is_trained = True

    def forward_pass(self, X: torch.Tensor, gradient_pass: bool = False) -> dict[str, torch.Tensor]:
        return {"feat_a": X[:, 0].to(dtype=torch.float32)}

    def encode(self, X: torch.Tensor, gradient_pass: bool = False) -> torch.Tensor:
        return X


def _build_pred_system(tmp_path) -> PredictionSystem:
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    schema = build_mixed_feature_schema(tmp_path)
    return PredictionSystem(logger=logger, schema=schema, local_data=LocalData(str(tmp_path)))


# ── Edge cases ────────────────────────────────────────────────────────────


def test_delta_evidence_batched_tensor_empty_returns_empty(tmp_path):
    sys = _build_pred_system(tmp_path)
    sys.models.append(_IdentityModel(logger=sys.logger))

    out = sys.delta_integrated_evidence_batched(
        torch.zeros((0, 3), dtype=torch.float64),
    )
    assert out.shape == (0,)


def test_delta_evidence_joint_tensor_empty_returns_empty(tmp_path):
    sys = _build_pred_system(tmp_path)
    sys.models.append(_IdentityModel(logger=sys.logger))

    out = sys.delta_integrated_evidence_joint_batched(
        torch.zeros((0, 1, 3), dtype=torch.float64),
    )
    assert out.shape == (0,)
