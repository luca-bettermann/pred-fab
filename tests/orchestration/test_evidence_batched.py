"""Equivalence tests for KernelFieldEstimator.integrated_evidence_perturbed_batched.

The batched API replaces the S-times rebuild-index + scalar integrated_evidence
loop with one broadcast tensor reduction. Result must match the scalar path
within floating-point summation noise (different reduction orders).
"""

import numpy as np
import pytest

from pred_fab.orchestration.evidence import (
    KernelIndex,
    KernelFieldEstimator,
    EstimatorConfig,
    make_estimator,
)


@pytest.fixture
def kf_estimator() -> KernelFieldEstimator:
    """Default-config KernelField estimator (matches what PredictionSystem uses)."""
    cfg = EstimatorConfig(type="kernel_field")
    est = make_estimator(cfg)
    assert isinstance(est, KernelFieldEstimator)
    return est


def _scalar_e_new(
    estimator: KernelFieldEstimator,
    index_old: KernelIndex,
    new_centers: np.ndarray,
    new_weights: np.ndarray,
) -> np.ndarray:
    """Loop the scalar API to produce E_new[s] for each s — the reference."""
    S = new_centers.shape[0]
    out = np.zeros(S)
    for s in range(S):
        if index_old.is_empty:
            all_centers = new_centers[s:s + 1]
            all_weights = new_weights[s:s + 1]
        else:
            all_centers = np.vstack([index_old.centers, new_centers[s:s + 1]])
            all_weights = np.concatenate([index_old.weights, new_weights[s:s + 1]])
        index_new = KernelIndex(
            all_centers, all_weights, index_old.sigma,
            cutoff_sigmas=index_old.cutoff_sigmas,
            truncation_threshold=index_old.truncation_threshold,
        )
        out[s] = estimator.integrated_evidence(index_new)
    return out


# ── Smoke-scale: small n_old, no truncation ───────────────────────────────


def test_batched_matches_scalar_smoke_scale(kf_estimator):
    """3 old kernels, D=4, S=8 — no truncation regime (n_old < 10)."""
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

    expected = _scalar_e_new(kf_estimator, index_old, new_centers, new_weights)
    got = kf_estimator.integrated_evidence_perturbed_batched(
        index_old, new_centers, new_weights,
    )

    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_batched_matches_scalar_low_d(kf_estimator):
    """D=2, common in baseline Process phase."""
    rng = np.random.default_rng(7)
    sigma = 0.075
    D = 2
    n_old = 4
    S = 16

    old_centers = rng.uniform(0.0, 1.0, size=(n_old, D))
    old_weights = np.array([1.0, 0.7, 1.2, 0.9])
    new_centers = rng.uniform(0.0, 1.0, size=(S, D))
    new_weights = rng.uniform(0.5, 1.5, size=S)

    index_old = KernelIndex(old_centers, old_weights, sigma)

    expected = _scalar_e_new(kf_estimator, index_old, new_centers, new_weights)
    got = kf_estimator.integrated_evidence_perturbed_batched(
        index_old, new_centers, new_weights,
    )

    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


# ── Truncation regime: n_old >= 10 ─────────────────────────────────────────


def test_batched_matches_scalar_truncation_regime(kf_estimator):
    """n_old=12 — triggers cKDTree truncation in scalar; batched mirrors with mask."""
    rng = np.random.default_rng(123)
    sigma = 0.05
    D = 4
    n_old = 12
    S = 8

    old_centers = rng.uniform(0.0, 1.0, size=(n_old, D))
    old_weights = np.ones(n_old)
    new_centers = rng.uniform(0.0, 1.0, size=(S, D))
    new_weights = np.ones(S)

    index_old = KernelIndex(old_centers, old_weights, sigma)

    expected = _scalar_e_new(kf_estimator, index_old, new_centers, new_weights)
    got = kf_estimator.integrated_evidence_perturbed_batched(
        index_old, new_centers, new_weights,
    )

    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


# ── Edge cases ────────────────────────────────────────────────────────────


def test_batched_empty_old_kernels(kf_estimator):
    """No old kernels: each candidate's E is just its own self-integral."""
    sigma = 0.075
    D = 3
    S = 4

    old_centers = np.zeros((0, D))
    old_weights = np.zeros(0)
    new_centers = np.array([[0.5] * D, [0.2] * D, [0.8] * D, [0.5, 0.5, 0.0]])
    new_weights = np.ones(S)

    index_old = KernelIndex(old_centers, old_weights, sigma)
    expected = _scalar_e_new(kf_estimator, index_old, new_centers, new_weights)
    got = kf_estimator.integrated_evidence_perturbed_batched(
        index_old, new_centers, new_weights,
    )
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_batched_empty_new_returns_empty(kf_estimator):
    """S=0: returns empty array."""
    sigma = 0.075
    D = 4
    old_centers = np.array([[0.5] * D])
    old_weights = np.array([1.0])
    index_old = KernelIndex(old_centers, old_weights, sigma)

    got = kf_estimator.integrated_evidence_perturbed_batched(
        index_old, np.zeros((0, D)), np.zeros(0),
    )
    assert got.shape == (0,)


def test_batched_with_nonuniform_weights(kf_estimator):
    """Mixed weights for both old and new — the affine multiplier must propagate."""
    rng = np.random.default_rng(99)
    sigma = 0.075
    D = 4
    n_old = 5
    S = 6

    old_centers = rng.uniform(0.1, 0.9, size=(n_old, D))
    old_weights = rng.uniform(0.3, 1.5, size=n_old)
    new_centers = rng.uniform(0.1, 0.9, size=(S, D))
    new_weights = rng.uniform(0.5, 2.0, size=S)

    index_old = KernelIndex(old_centers, old_weights, sigma)

    expected = _scalar_e_new(kf_estimator, index_old, new_centers, new_weights)
    got = kf_estimator.integrated_evidence_perturbed_batched(
        index_old, new_centers, new_weights,
    )

    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)
