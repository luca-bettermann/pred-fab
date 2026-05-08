"""Tests for tensor-typed ANOVA marginal-joint evidence integration.

Three guarantees:
  1. ANOVA torch path produces well-behaved positive evidence values,
     both marginal and joint components contribute, and the combined
     estimator is deterministic.
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


# ── ANOVA torch path: well-behavedness and consistency ───────────────────


def test_torch_positive_evidence_smoke(kf_estimator):
    """E_new values are strictly positive for well-placed candidates."""
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

    got = kf_estimator.integrated_evidence_perturbed_batched_joint_torch(
        index_old,
        torch.from_numpy(new_centers[:, None, :]).double(),
        torch.from_numpy(new_weights[:, None]).double(),
    )
    assert (got > 0).all(), f"E_new should be positive, got {got}"


def test_torch_deterministic(kf_estimator):
    """Same inputs produce identical outputs on repeated calls."""
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
    t_centers = torch.from_numpy(new_centers_SL).double()
    t_weights = torch.from_numpy(new_weights_SL).double()

    r1 = kf_estimator.integrated_evidence_perturbed_batched_joint_torch(
        index_old, t_centers, t_weights,
    )
    r2 = kf_estimator.integrated_evidence_perturbed_batched_joint_torch(
        index_old, t_centers, t_weights,
    )
    np.testing.assert_allclose(
        r1.detach().cpu().numpy(), r2.detach().cpu().numpy(),
        atol=0, rtol=0,
    )


def test_torch_empty_old_positive(kf_estimator):
    """No old kernels -- candidate self-evidence is positive."""
    rng = np.random.default_rng(3)
    sigma = 0.075
    D = 3
    S = 3
    L = 2

    new_centers_SL = rng.uniform(0.1, 0.9, size=(S, L, D))
    new_weights_SL = np.ones((S, L))

    index_old = KernelIndex(np.zeros((0, D)), np.zeros(0), sigma)

    got = kf_estimator.integrated_evidence_perturbed_batched_joint_torch(
        index_old,
        torch.from_numpy(new_centers_SL).double(),
        torch.from_numpy(new_weights_SL).double(),
    )
    assert (got > 0).all()


def test_torch_nonuniform_weights_positive(kf_estimator):
    """Non-uniform weights -- evidence still positive and finite."""
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

    got = kf_estimator.integrated_evidence_perturbed_batched_joint_torch(
        index_old,
        torch.from_numpy(new_centers_SL).double(),
        torch.from_numpy(new_weights_SL).double(),
    )
    assert (got > 0).all()
    assert torch.isfinite(got).all()


def test_torch_marginal_joint_both_contribute(kf_estimator):
    """Verify that both _marginal and _joint produce nonzero evidence."""
    rng = np.random.default_rng(10)
    sigma = 0.075
    D = 3
    n_old = 3
    S = 4
    L = 1

    old_centers = rng.uniform(0.2, 0.8, size=(n_old, D))
    old_weights = np.ones(n_old)

    index_old = KernelIndex(old_centers, old_weights, sigma)
    t_centers = torch.from_numpy(rng.uniform(0.2, 0.8, size=(S, L, D))).double()
    t_weights = torch.ones((S, L), dtype=torch.float64)

    e_marginal = kf_estimator._marginal_evidence_torch(index_old, t_centers, t_weights)
    e_joint = kf_estimator._joint_evidence_torch(index_old, t_centers, t_weights)

    assert (e_marginal > 0).all(), "Marginal evidence should be positive"
    assert (e_joint > 0).all(), "Joint evidence should be positive"
    # Combined = (marginal + joint) / 2, both contribute
    combined = kf_estimator.integrated_evidence_perturbed_batched_joint_torch(
        index_old, t_centers, t_weights,
    )
    D = t_centers.shape[2]
    alpha_m = D / (D + 1)
    alpha_j = 1.0 / (D + 1)
    np.testing.assert_allclose(
        combined.detach().cpu().numpy(),
        (alpha_m * e_marginal + alpha_j * e_joint).detach().cpu().numpy(),
        atol=1e-10,
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
