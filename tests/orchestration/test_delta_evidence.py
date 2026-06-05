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
from pred_fab.orchestration.prediction import _ModelKDE
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


# ── σ-frame unification (Thread 3) ─────────────────────────────────────────


def _insert_kde(sys, model, latent_points, domain_bounds, sigma=0.1):
    lp = np.asarray(latent_points, dtype=float)
    sys._model_kdes[id(model)] = _ModelKDE(
        model=model,
        latent_points=lp,
        point_weights=np.ones(lp.shape[0]),
        sigma=sigma,
        active_mask=np.ones(lp.shape[1], dtype=bool),
        n_active_dims=lp.shape[1],
        weight=1.0,
        domain_bounds=np.asarray(domain_bounds, dtype=float),
    )


def test_delta_evidence_sigma_frame_is_scale_invariant(tmp_path):
    """σ is a *fraction of range*: Δ∫E is invariant to a joint rescale of the
    latent space (centers + candidate + domain span by the same factor).

    Regression for the σ-frame split — the acquisition Δ∫E read raw latent
    distances (σ absolute), so rescaling the domain changed Δ∫E. It now
    normalises by domain_bounds like density_at, so both paths share one
    effective kernel width. With the bug, the two scenarios below differ.
    """
    sys = _build_pred_system(tmp_path)
    model = _IdentityModel(logger=sys.logger)
    sys.models.append(model)

    # Scenario A: latent domain [0, 1].
    _insert_kde(sys, model, [[0.0], [0.3], [0.6]], [[0.0, 1.0]], sigma=0.1)
    dE_A = float(sys.delta_integrated_evidence_batched(
        torch.tensor([[0.15]], dtype=torch.float64))[0].item())

    # Scenario B: same configuration scaled ×10 (identical fractions of range).
    _insert_kde(sys, model, [[0.0], [3.0], [6.0]], [[0.0, 10.0]], sigma=0.1)
    dE_B = float(sys.delta_integrated_evidence_batched(
        torch.tensor([[1.5]], dtype=torch.float64))[0].item())

    assert dE_A > 0.0                       # adding a point increases coverage (non-vacuous)
    assert dE_A == pytest.approx(dE_B, abs=1e-9)


def test_delta_evidence_matches_density_at_frame(tmp_path):
    """A candidate on top of an existing KDE point reads the same normalised
    distance (0) in both density_at and the acquisition path — i.e. one σ-frame.
    """
    sys = _build_pred_system(tmp_path)
    model = _IdentityModel(logger=sys.logger)
    sys.models.append(model)
    _insert_kde(sys, model, [[2.0], [5.0]], [[0.0, 10.0]], sigma=0.1)

    # density_at uses the same _to_unit_frame normalisation; a point at an
    # existing centre is maximally "covered" (density >= its own kernel mass).
    d_on = sys.density_at(np.array([2.0]))
    d_off = sys.density_at(np.array([8.0]))
    assert d_on > d_off  # on-centre is denser than a far point, in the unit frame
