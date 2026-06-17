"""Value-correctness tests for the evidence integral ∫_{[0,1]^D} D/(1+D) dz.

Complements ``test_evidence.py`` (KernelField shape/grad/determinism smoke
tests) and ``test_delta_evidence.py`` (PredictionSystem-level Δ∫E framing)
with the missing *numerical* coverage: the integral validated against a
closed-form D=1 reference, and ``SobolLocalEstimator`` exercised for value,
determinism, convergence, and domain masking.

D=1 closed form. For a single peak-1 Gaussian kernel
``D(z) = w·exp(-(z-c)²/2σ²)`` with the kernel mass well inside [0,1]
(so the unit-cube mask captures ~all of it), expanding ``D/(1+D)`` as a
geometric series and integrating term-by-term over ℝ gives

    ∫ D/(1+D) dz = σ·√(2π) · Σ_{n≥1} (-1)^(n+1) · wⁿ / √n   (converges for w < 1)

This is the reference the Sobol estimator must reproduce.
"""

import math

import numpy as np
import pytest
import torch

from pred_fab.orchestration.evidence import (
    EstimatorConfig,
    KernelFieldEstimator,
    KernelIndex,
    SobolLocalEstimator,
    evidence_from_density,
    make_estimator,
)


def _closed_form_d1(w: float, sigma: float, terms: int = 200) -> float:
    """∫_ℝ D/(1+D) dz for one peak-1 Gaussian kernel of weight ``w`` (w < 1)."""
    series = sum(((-1) ** (n + 1)) * (w ** n) / math.sqrt(n) for n in range(1, terms + 1))
    return sigma * math.sqrt(2.0 * math.pi) * series


def _single_kernel_integral(estimator, w: float, sigma: float, center: float) -> float:
    """Self-integral of one new D=1 kernel (no old kernels) via the estimator."""
    empty = KernelIndex(np.empty((0, 1)), np.empty(0), sigma)
    new_centers = torch.tensor([[[center]]], dtype=torch.float64)  # (S=1, L=1, D=1)
    new_weights = torch.tensor([[w]], dtype=torch.float64)
    return float(
        estimator.integrated_evidence_perturbed_batched_joint(empty, new_centers, new_weights)[0]
    )


def test_evidence_from_density_saturating_transform():
    """E = D/(1+D): 0→0, monotone increasing, →1 as D→∞."""
    assert evidence_from_density(0.0) == 0.0
    assert evidence_from_density(1.0) == pytest.approx(0.5)
    assert evidence_from_density(1e6) == pytest.approx(1.0, abs=1e-5)
    assert evidence_from_density(0.1) < evidence_from_density(10.0) < 1.0


@pytest.mark.parametrize("w", [0.3, 0.5, 0.8])
def test_sobol_matches_closed_form_d1(w):
    """SobolLocal reproduces the analytic D=1 integral to high precision."""
    sigma, center = 0.05, 0.5  # kernel ~10σ from each boundary → mask captures all mass
    estimator = SobolLocalEstimator(box=8.0, n_samples=16384, seed=0)
    estimated = _single_kernel_integral(estimator, w, sigma, center)
    assert estimated == pytest.approx(_closed_form_d1(w, sigma), rel=1e-4)


@pytest.mark.parametrize("w", [0.3, 0.5, 0.8])
def test_kernel_field_matches_closed_form_d1(w):
    """KernelField reproduces the same Lebesgue integral (looser tolerance — coarse
    shell quadrature). Guards against the σ√(2π) measure mismatch that previously
    made its marginal term ~8x too large (and the estimator ~3.4x overall)."""
    sigma, center = 0.05, 0.5
    estimated = _single_kernel_integral(KernelFieldEstimator(), w, sigma, center)
    assert estimated == pytest.approx(_closed_form_d1(w, sigma), rel=5e-2)


def test_sobol_deterministic_for_fixed_seed():
    """Same seed → bit-identical estimate (the scrambled-Sobol determinism knob)."""
    sigma, center, w = 0.05, 0.5, 0.5
    a = _single_kernel_integral(SobolLocalEstimator(n_samples=2048, seed=7), w, sigma, center)
    b = _single_kernel_integral(SobolLocalEstimator(n_samples=2048, seed=7), w, sigma, center)
    assert a == b


def test_sobol_converges_with_more_samples():
    """Estimation error shrinks as the sample budget grows (QMC convergence)."""
    sigma, center, w = 0.05, 0.5, 0.5
    reference = _closed_form_d1(w, sigma)
    err_coarse = abs(
        _single_kernel_integral(SobolLocalEstimator(box=8.0, n_samples=64, seed=0), w, sigma, center)
        - reference
    )
    err_fine = abs(
        _single_kernel_integral(SobolLocalEstimator(box=8.0, n_samples=8192, seed=0), w, sigma, center)
        - reference
    )
    assert err_fine < err_coarse
    assert err_fine < 1e-3


def test_sobol_unit_cube_mask_halves_boundary_kernel():
    """A kernel centred on the boundary integrates ~half the interior value.

    With ``domain_bounds=None`` the integrand is masked to [0,1]; a kernel at
    z=0 has only its z≥0 half inside, so the integral is ~half of the same
    kernel placed in the interior.
    """
    sigma, w = 0.05, 0.5
    estimator = SobolLocalEstimator(box=8.0, n_samples=16384, seed=0)
    interior = _single_kernel_integral(estimator, w, sigma, center=0.5)
    boundary = _single_kernel_integral(estimator, w, sigma, center=0.0)
    assert boundary == pytest.approx(0.5 * interior, rel=5e-3)


def test_sobol_empty_batch_returns_empty():
    """S=0 → shape (0,) (mirrors the KernelField empty-batch contract)."""
    estimator = SobolLocalEstimator(seed=0)
    empty = KernelIndex(np.empty((0, 2)), np.empty(0), 0.1)
    out = estimator.integrated_evidence_perturbed_batched_joint(
        empty,
        torch.empty((0, 1, 2), dtype=torch.float64),
        torch.empty((0, 1), dtype=torch.float64),
    )
    assert out.shape == (0,)


def test_make_estimator_selects_type():
    """The factory dispatches on the config type discriminator."""
    assert isinstance(make_estimator(EstimatorConfig(type="kernel_field")), KernelFieldEstimator)
    assert isinstance(make_estimator(EstimatorConfig(type="sobol_local")), SobolLocalEstimator)


def _delta_evidence(estimator, idx_old, idx_empty, candidate):
    """ΔE = E(old ∪ candidate) − E(old) for one D=2 candidate."""
    new_c = torch.tensor([[candidate]], dtype=torch.float64)  # (1, 1, 2)
    new_w = torch.tensor([[1.0]], dtype=torch.float64)
    e_new = estimator.integrated_evidence_perturbed_batched_joint(idx_old, new_c, new_w)[0].item()
    old_c = idx_old.centers.unsqueeze(0)
    old_w = idx_old.weights.unsqueeze(0)
    e_old = estimator.integrated_evidence_perturbed_batched_joint(idx_empty, old_c, old_w)[0].item()
    return e_new - e_old


@pytest.mark.parametrize(
    "estimator",
    [
        SobolLocalEstimator(box=4.0, n_samples=8192, seed=0),
        KernelFieldEstimator(),
    ],
    ids=["sobol", "kernel_field"],
)
def test_delta_evidence_rewards_unexplored_regions(estimator):
    """Both estimators must agree on the *behaviour* the acquisition relies on:
    a candidate far from existing data yields a larger, positive ΔE than one
    sitting on top of the cluster.

    Both estimators compute the same Lebesgue integral, but KernelField's ANOVA
    decomposition is only an approximation in D>1, so magnitudes still differ by
    quadrature/approximation error; sign and ranking are the robust shared
    contract, so that is what this asserts.
    """
    sigma = 0.1
    old_centers = np.array([[0.3, 0.3], [0.32, 0.28], [0.28, 0.31]])
    old_weights = np.ones(3)
    idx_old = KernelIndex(old_centers, old_weights, sigma)
    idx_empty = KernelIndex(np.empty((0, 2)), np.empty(0), sigma)

    dE_near = _delta_evidence(estimator, idx_old, idx_empty, [0.30, 0.30])
    dE_far = _delta_evidence(estimator, idx_old, idx_empty, [0.80, 0.80])

    assert dE_near > 0.0
    assert math.isfinite(dE_far)
    assert dE_far > dE_near
