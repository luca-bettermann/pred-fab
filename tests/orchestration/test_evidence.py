"""Tests for tensor-typed ANOVA marginal-joint evidence integration.

Two guarantees:
  1. ANOVA torch path produces well-behaved positive evidence values,
     both marginal and joint components contribute, and the combined
     estimator is deterministic.
  2. Gradient flow: ``new_centers_SL`` with ``requires_grad=True`` produces
     finite gradients on backward of the ``(S,)`` E_new output.
"""

import numpy as np
import pytest
import torch

from pred_fab.orchestration.evidence import (
    EstimatorConfig,
    KernelFieldEstimator,
    KernelIndex,
    _in_unit_cube,
    make_estimator,
)


@pytest.fixture
def kf_estimator() -> KernelFieldEstimator:
    cfg = EstimatorConfig(type="kernel_field")
    est = make_estimator(cfg)
    assert isinstance(est, KernelFieldEstimator)
    return est


# ── _in_unit_cube ───────────────────────────────────────────────────


def test_in_unit_cube_torch_basic():
    pts = torch.tensor([
        [0.5, 0.5, 0.5],   # in
        [0.0, 0.0, 0.0],   # in (closed boundary)
        [1.0, 1.0, 1.0],   # in (closed boundary)
        [-0.1, 0.5, 0.5],  # out (negative)
        [0.5, 1.5, 0.5],   # out (>1)
    ])
    expected = torch.tensor([True, True, True, False, False])
    assert torch.equal(_in_unit_cube(pts), expected)


# ── Regime dispatcher ─────────────────────────────────────────────────────


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

    got = kf_estimator.integrated_evidence_perturbed_batched_joint(
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

    r1 = kf_estimator.integrated_evidence_perturbed_batched_joint(
        index_old, t_centers, t_weights,
    )
    r2 = kf_estimator.integrated_evidence_perturbed_batched_joint(
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

    got = kf_estimator.integrated_evidence_perturbed_batched_joint(
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

    got = kf_estimator.integrated_evidence_perturbed_batched_joint(
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

    e_marginal = kf_estimator._marginal_evidence(index_old, t_centers, t_weights)
    e_joint = kf_estimator._joint_evidence(index_old, t_centers, t_weights)

    assert (e_marginal > 0).all(), "Marginal evidence should be positive"
    assert (e_joint > 0).all(), "Joint evidence should be positive"
    # Combined = (marginal + joint) / 2, both contribute
    combined = kf_estimator.integrated_evidence_perturbed_batched_joint(
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

    e_new = kf_estimator.integrated_evidence_perturbed_batched_joint(
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

    e_new = kf_estimator.integrated_evidence_perturbed_batched_joint(
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

    e_new = kf_estimator.integrated_evidence_perturbed_batched_joint(
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

    got = kf_estimator.integrated_evidence_perturbed_batched_joint(
        index_old,
        torch.zeros((0, 1, D), dtype=torch.float64),
        torch.zeros((0, 1), dtype=torch.float64),
    )
    assert got.shape == (0,)
