"""Tests for tensor-typed KDE evaluation (Strategy D commit 4).

Three guarantees this commit promises:
  1. Numerical equivalence between
     ``KernelFieldEstimator.integrated_evidence_perturbed_batched_joint_torch``
     and the existing numpy ``integrated_evidence_perturbed_batched_joint``
     (same math, same shapes, just torch ops).
  2. Gradient flow: ``new_centers_SL`` with ``requires_grad=True`` produces
     finite gradients on backward of the ``(S,)`` E_new output.
  3. Regime dispatcher ``_choose_kde_regime(n, σ, D)`` returns expected
     labels — dense for small N or full-coverage σ; knn / cluster for the
     scale-aware cases that trigger follow-up commits.
"""

import numpy as np
import pytest
import torch

from pred_fab.orchestration.evidence import (
    EstimatorConfig,
    KernelFieldEstimator,
    KernelIndex,
    _choose_kde_regime,
    _in_unit_cube_torch,
    make_estimator,
)


@pytest.fixture
def kf_estimator() -> KernelFieldEstimator:
    cfg = EstimatorConfig(type="kernel_field")
    est = make_estimator(cfg)
    assert isinstance(est, KernelFieldEstimator)
    return est


# ── _in_unit_cube_torch ───────────────────────────────────────────────────


def test_in_unit_cube_torch_basic():
    pts = torch.tensor([
        [0.5, 0.5, 0.5],   # in
        [0.0, 0.0, 0.0],   # in (closed boundary)
        [1.0, 1.0, 1.0],   # in (closed boundary)
        [-0.1, 0.5, 0.5],  # out (negative)
        [0.5, 1.5, 0.5],   # out (>1)
    ])
    expected = torch.tensor([True, True, True, False, False])
    assert torch.equal(_in_unit_cube_torch(pts), expected)


# ── Regime dispatcher ─────────────────────────────────────────────────────


def test_regime_dense_for_small_n():
    """n_kernels < 100 → dense regardless of σ/D."""
    assert _choose_kde_regime(10, sigma=0.05, D=4) == "dense"
    assert _choose_kde_regime(50, sigma=0.5, D=2) == "dense"
    assert _choose_kde_regime(99, sigma=0.001, D=10) == "dense"


def test_regime_dense_when_5sigma_covers_unit_cube():
    """High σ in low D → 5σ ball covers cube, n_active ≈ n_kernels — dense correct."""
    # σ=0.5, D=2 → V(5σ-ball) = π·(2.5)² ≈ 19.6 → capped at 1.0 → n_active = n.
    assert _choose_kde_regime(500, sigma=0.5, D=2) == "dense"
    assert _choose_kde_regime(1000, sigma=0.4, D=3) == "dense"


def test_regime_knn_for_sparse_high_dim():
    """Small σ in high D → tiny 5σ-ball → n_active ≪ n → KNN should fire."""
    # σ=0.05, D=10 → V(5σ ball) = π⁵·(0.25)¹⁰ / 5! ≈ negligible → n_active≪n.
    assert _choose_kde_regime(1000, sigma=0.05, D=10) == "knn"
    assert _choose_kde_regime(5000, sigma=0.03, D=8) == "knn"


def test_regime_cluster_for_huge_n():
    """n_kernels > 100k → cluster regime regardless of σ."""
    assert _choose_kde_regime(200_000, sigma=0.01, D=4) == "cluster"
    assert _choose_kde_regime(150_000, sigma=0.05, D=4) == "cluster"


# ── Numerical equivalence with the numpy variant ──────────────────────────


def test_torch_matches_numpy_smoke_scale(kf_estimator):
    """3 old kernels, D=4, S=8 — no truncation regime."""
    rng = np.random.default_rng(42)
    sigma = 0.075
    D = 4
    n_old = 3
    S = 8

    old_centers = rng.uniform(0.1, 0.9, size=(n_old, D))
    old_weights = np.ones(n_old)
    new_centers = rng.uniform(0.1, 0.9, size=(S, D))
    new_weights = np.ones(S)

    index_old = KernelIndex(old_centers, old_weights, sigma)

    expected_np = kf_estimator.integrated_evidence_perturbed_batched_joint(
        index_old, new_centers[:, None, :], new_weights[:, None],
    )
    got_torch = kf_estimator.integrated_evidence_perturbed_batched_joint_torch(
        index_old,
        torch.from_numpy(new_centers[:, None, :]).double(),
        torch.from_numpy(new_weights[:, None]).double(),
    )

    np.testing.assert_allclose(
        got_torch.detach().cpu().numpy(), expected_np, atol=1e-6, rtol=1e-6,
    )


def test_torch_matches_numpy_l_3(kf_estimator):
    """L=3 — typical schedule trajectory length."""
    rng = np.random.default_rng(7)
    sigma = 0.075
    D = 4
    n_old = 5
    S = 4
    L = 3

    old_centers = rng.uniform(0.1, 0.9, size=(n_old, D))
    old_weights = np.ones(n_old)
    new_centers_SL = rng.uniform(0.1, 0.9, size=(S, L, D))
    new_weights_SL = np.ones((S, L))

    index_old = KernelIndex(old_centers, old_weights, sigma)

    expected_np = kf_estimator.integrated_evidence_perturbed_batched_joint(
        index_old, new_centers_SL, new_weights_SL,
    )
    got_torch = kf_estimator.integrated_evidence_perturbed_batched_joint_torch(
        index_old,
        torch.from_numpy(new_centers_SL).double(),
        torch.from_numpy(new_weights_SL).double(),
    )

    np.testing.assert_allclose(
        got_torch.detach().cpu().numpy(), expected_np, atol=1e-6, rtol=1e-6,
    )


def test_torch_matches_numpy_empty_old(kf_estimator):
    """No old kernels — only the candidate's own L points contribute."""
    rng = np.random.default_rng(3)
    sigma = 0.075
    D = 3
    S = 3
    L = 2

    new_centers_SL = rng.uniform(0.1, 0.9, size=(S, L, D))
    new_weights_SL = np.ones((S, L))

    index_old = KernelIndex(np.zeros((0, D)), np.zeros(0), sigma)

    expected_np = kf_estimator.integrated_evidence_perturbed_batched_joint(
        index_old, new_centers_SL, new_weights_SL,
    )
    got_torch = kf_estimator.integrated_evidence_perturbed_batched_joint_torch(
        index_old,
        torch.from_numpy(new_centers_SL).double(),
        torch.from_numpy(new_weights_SL).double(),
    )

    np.testing.assert_allclose(
        got_torch.detach().cpu().numpy(), expected_np, atol=1e-6, rtol=1e-6,
    )


def test_torch_matches_numpy_nonuniform_weights(kf_estimator):
    """Non-uniform weights for both old and joint-added new kernels."""
    rng = np.random.default_rng(55)
    sigma = 0.075
    D = 4
    n_old = 4
    S = 5
    L = 3

    old_centers = rng.uniform(0.1, 0.9, size=(n_old, D))
    old_weights = rng.uniform(0.4, 1.4, size=n_old)
    new_centers_SL = rng.uniform(0.1, 0.9, size=(S, L, D))
    new_weights_SL = rng.uniform(0.5, 1.8, size=(S, L))

    index_old = KernelIndex(old_centers, old_weights, sigma)

    expected_np = kf_estimator.integrated_evidence_perturbed_batched_joint(
        index_old, new_centers_SL, new_weights_SL,
    )
    got_torch = kf_estimator.integrated_evidence_perturbed_batched_joint_torch(
        index_old,
        torch.from_numpy(new_centers_SL).double(),
        torch.from_numpy(new_weights_SL).double(),
    )

    np.testing.assert_allclose(
        got_torch.detach().cpu().numpy(), expected_np, atol=1e-6, rtol=1e-6,
    )


# ── Gradient flow ─────────────────────────────────────────────────────────


def test_grad_flows_through_kde(kf_estimator):
    """new_centers_SL.requires_grad=True → backward produces finite gradients."""
    sigma = 0.075
    D = 3
    n_old = 2
    S = 4
    L = 2

    rng = np.random.default_rng(0)
    old_centers = rng.uniform(0.2, 0.8, size=(n_old, D))
    old_weights = np.ones(n_old)
    index_old = KernelIndex(old_centers, old_weights, sigma)

    new_centers_SL = torch.tensor(
        rng.uniform(0.2, 0.8, size=(S, L, D)),
        requires_grad=True, dtype=torch.float64,
    )
    new_weights_SL = torch.ones((S, L), dtype=torch.float64)

    e_new = kf_estimator.integrated_evidence_perturbed_batched_joint_torch(
        index_old, new_centers_SL, new_weights_SL,
    )
    e_new.sum().backward()

    assert new_centers_SL.grad is not None
    assert torch.isfinite(new_centers_SL.grad).all()
    # Gradients shouldn't all be zero — moving a kernel changes the integral.
    assert (new_centers_SL.grad.abs() > 0).any()


def test_grad_flows_with_no_old_kernels(kf_estimator):
    """Empty index_old branch must also propagate gradients."""
    sigma = 0.075
    D = 3
    S = 3
    L = 2

    rng = np.random.default_rng(1)
    index_old = KernelIndex(np.zeros((0, D)), np.zeros(0), sigma)

    new_centers_SL = torch.tensor(
        rng.uniform(0.2, 0.8, size=(S, L, D)),
        requires_grad=True, dtype=torch.float64,
    )
    new_weights_SL = torch.ones((S, L), dtype=torch.float64)

    e_new = kf_estimator.integrated_evidence_perturbed_batched_joint_torch(
        index_old, new_centers_SL, new_weights_SL,
    )
    e_new.sum().backward()

    assert new_centers_SL.grad is not None
    assert torch.isfinite(new_centers_SL.grad).all()
    assert (new_centers_SL.grad.abs() > 0).any()


def test_grad_flows_through_weights(kf_estimator):
    """new_weights_SL.requires_grad=True → backward propagates."""
    sigma = 0.075
    D = 3
    n_old = 2
    S = 3
    L = 2

    rng = np.random.default_rng(2)
    old_centers = rng.uniform(0.2, 0.8, size=(n_old, D))
    old_weights = np.ones(n_old)
    index_old = KernelIndex(old_centers, old_weights, sigma)

    new_centers_SL = torch.tensor(
        rng.uniform(0.2, 0.8, size=(S, L, D)),
        dtype=torch.float64,
    )
    new_weights_SL = torch.ones((S, L), dtype=torch.float64, requires_grad=True)

    e_new = kf_estimator.integrated_evidence_perturbed_batched_joint_torch(
        index_old, new_centers_SL, new_weights_SL,
    )
    e_new.sum().backward()

    assert new_weights_SL.grad is not None
    assert torch.isfinite(new_weights_SL.grad).all()


# ── Edge cases ────────────────────────────────────────────────────────────


def test_torch_empty_s_returns_empty(kf_estimator):
    sigma = 0.075
    D = 4
    old_centers = np.array([[0.5] * D])
    old_weights = np.array([1.0])
    index_old = KernelIndex(old_centers, old_weights, sigma)

    got = kf_estimator.integrated_evidence_perturbed_batched_joint_torch(
        index_old,
        torch.zeros((0, 1, D), dtype=torch.float64),
        torch.zeros((0, 1), dtype=torch.float64),
    )
    assert got.shape == (0,)


def test_torch_dispatcher_logs_when_non_dense_regime(kf_estimator, caplog):
    """When regime would be 'knn' or 'cluster', dispatcher logs INFO and
    falls back to dense — gives empirical signal for follow-up commits.

    Picked (n_old=120, σ=0.005, D=4) to trigger 'knn' while keeping the
    dense fallback's per-old broadcast tensor small (~few MB, well within
    the 1 GB RAM budget on this server).
    """
    import logging

    rng = np.random.default_rng(0)
    sigma = 0.005
    D = 4
    n_old = 120

    old_centers = rng.uniform(0.0, 1.0, size=(n_old, D))
    old_weights = np.ones(n_old)
    index_old = KernelIndex(old_centers, old_weights, sigma)

    new_centers_SL = torch.from_numpy(rng.uniform(0.0, 1.0, size=(2, 1, D))).double()
    new_weights_SL = torch.ones((2, 1), dtype=torch.float64)

    with caplog.at_level(logging.INFO, logger="pred_fab.orchestration.evidence"):
        kf_estimator.integrated_evidence_perturbed_batched_joint_torch(
            index_old, new_centers_SL, new_weights_SL,
        )

    assert any("knn" in rec.message for rec in caplog.records)
