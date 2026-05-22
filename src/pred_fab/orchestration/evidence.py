"""Evidence-integration estimators for the Δ∫E acquisition objective.

Two deterministic/stochastic variants selectable at runtime:

    KernelFieldEstimator  — radial shell quadrature. Per-kernel probes on
                             shells at σ · `radii`, with quadrature weights
                             derived from the Gaussian radial measure.
    SobolLocalEstimator   — scrambled Sobol inside a `[center ± box·σ]^D`
                             cube, volume-weighted.

Both implement the identity
    I = ∫_{[0,1]^D} D/(1+D) dz = Σⱼ wⱼ · 𝔼_{z~N(zⱼ, σ²I)}[1/(1+D(z))]
— a sum of per-kernel self-integrals. Kernels outside the unit cube leak
naturally because the integrand is masked to the unit cube.

A `KernelIndex` wraps the existing kernel set and answers D(z) queries.
For small kernel sets (`< truncation_threshold`, default 10) the index
sums all kernels directly — guaranteeing a non-zero density everywhere
the Gaussian tail reaches, so plots and gradients stay smooth on early
samples. Above the threshold it falls back to cKDTree neighbour lookup
within `cutoff_sigmas · σ` (default 5σ; exp(−12.5) ≈ 4×10⁻⁶) for cost.
"""
from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import gamma, pi
from typing import Literal

import numpy as np
import torch

_logger = logging.getLogger(__name__)


# Module-level KernelField defaults — change here, propagates everywhere.
DEFAULT_RADII: tuple[float, ...] = (0.5, 1.0, 2.0)
DEFAULT_ANGULAR_GAP_DEG: float = 45.0


# ---------------------------------------------------------------------------
# Scale-aware regime dispatcher
# ---------------------------------------------------------------------------

# Thresholds for regime selection. The right metric is the expected number of
# kernels contributing to a typical probe's density — n_active = N · V(5σ-ball)
# capped at the unit cube — not n_kernels alone. The dense regime is always
# correct; knn / cluster are approximations that pay off only at scale.
_REGIME_DENSE_FLOOR = 100        # below: dense; KNN overhead doesn't pay
_REGIME_CLUSTER_N_ACTIVE = 10000  # above n_active: cluster
_REGIME_CLUSTER_N_KERNELS = 100000  # above n_kernels: cluster
_REGIME_KNN_N_ACTIVE = 50         # below n_active: KNN
_REGIME_KNN_ACTIVE_FRAC = 0.5     # n_active < frac · n_kernels: KNN

# knn / cluster are not yet implemented — when selected, the dispatcher logs
# at INFO and returns "dense" so the caller's code stays simple.
_IMPLEMENTED_REGIMES: frozenset[str] = frozenset({"dense"})


def _choose_kde_regime(n_kernels: int, sigma: float, D: int) -> str:
    """Ideal KDE regime for the given ``(n_kernels, sigma, D)``.

    Pure threshold logic — returns the regime label the dispatcher *would
    pick*, ignoring whether it's implemented. Use ``_resolve_kde_regime``
    to get the regime that will actually run.
    """
    v_5sigma = min((pi ** (D / 2)) * ((5.0 * sigma) ** D) / gamma(D / 2 + 1), 1.0)
    n_active = n_kernels * v_5sigma

    if n_kernels < _REGIME_DENSE_FLOOR:
        return "dense"
    if n_active > _REGIME_CLUSTER_N_ACTIVE or n_kernels > _REGIME_CLUSTER_N_KERNELS:
        return "cluster"
    if n_active < _REGIME_KNN_N_ACTIVE or n_active < n_kernels * _REGIME_KNN_ACTIVE_FRAC:
        return "knn"
    return "dense"


def _resolve_kde_regime(n_kernels: int, sigma: float, D: int) -> str:
    """Regime to actually execute — falls back to "dense" for unimplemented choices.

    Logs at INFO when the ideal regime is not yet implemented.
    """
    ideal = _choose_kde_regime(n_kernels, sigma, D)
    if ideal not in _IMPLEMENTED_REGIMES:
        _logger.info(
            "KDE regime %r preferred (n_kernels=%d, σ=%.4f, D=%d) — not yet "
            "implemented; falling back to dense.",
            ideal, n_kernels, sigma, D,
        )
        return "dense"
    return ideal


# ---------------------------------------------------------------------------
# Kernel index — O(M · log K) density evaluation via neighbour search
# ---------------------------------------------------------------------------

class KernelIndex:
    """Spatial index over Gaussian kernel centres for fast D(z) evaluation.

    For n_kernels < ``truncation_threshold`` (default 10), the full distance
    is summed directly. Above the threshold, a 5σ-radius mask via
    ``torch.cdist + torch.where`` keeps the contributing kernels only.
    """

    def __init__(
        self,
        centers: np.ndarray | torch.Tensor,
        weights: np.ndarray | torch.Tensor,
        sigma: float,
        cutoff_sigmas: float = 5.0,
        truncation_threshold: int = 10,
    ):
        self.centers = torch.as_tensor(
            centers if isinstance(centers, torch.Tensor) else np.asarray(centers, dtype=float),
            dtype=torch.float64,
        )
        self.weights = torch.as_tensor(
            weights if isinstance(weights, torch.Tensor) else np.asarray(weights, dtype=float),
            dtype=torch.float64,
        )
        self.sigma = float(sigma)
        self.cutoff_sigmas = float(cutoff_sigmas)
        self.truncation_threshold = int(truncation_threshold)
        self._n = len(self.centers)
        self._D = int(self.centers.shape[1]) if self.centers.ndim == 2 and self._n else 0

    @property
    def is_empty(self) -> bool:
        return self._n == 0

    @property
    def cutoff(self) -> float:
        return self.cutoff_sigmas * self.sigma



# ---------------------------------------------------------------------------
# Radial shell quadrature (Gaussian measure)
# ---------------------------------------------------------------------------

def _surface_area_unit_sphere(D: int) -> float:
    return 2.0 * pi ** (D / 2.0) / gamma(D / 2.0)


def _gaussian_radial_pdf(r: float, sigma: float, D: int) -> float:
    """Peak-1 Gaussian density at radius r — ρ(0) = 1.

    Used as the integration measure for radial shell quadrature, kept
    consistent with the peak-1 kernel definition in `KernelIndex`.
    """
    del D  # peak-1 normalization is dimension-agnostic
    return float(np.exp(-r * r / (2.0 * sigma * sigma)))


def _radial_shell_weights(
    radii: np.ndarray, sigma: float, D: int,
) -> tuple[np.ndarray, float]:
    """Per-shell quadrature weights + centre weight for the N(0, σ²I) measure.

    w_k ≈ P(|z| ∈ shell_k) where shell_k has thickness Δr_k (midpoint rule).
    Centre weight ≈ probability mass in a ball of radius r_0/2.
    """
    K = len(radii)
    surface = _surface_area_unit_sphere(D)
    r_ext = np.concatenate([[0.0], radii, [2.0 * radii[-1] - radii[-2]]])

    weights = np.zeros(K)
    for k in range(K):
        r_k = radii[k]
        dr_k = 0.5 * (r_ext[k + 2] - r_ext[k])
        weights[k] = surface * (r_k ** (D - 1)) * dr_k * _gaussian_radial_pdf(r_k, sigma, D)

    r0_half = 0.5 * r_ext[1]
    vol_ball = (pi ** (D / 2.0) / gamma(D / 2.0 + 1.0)) * (r0_half ** D)
    center_weight = vol_ball * _gaussian_radial_pdf(0.0, sigma, D)
    return weights, center_weight


def _unit_sphere_directions(D: int, n: int) -> np.ndarray:
    """Quasi-uniform unit vectors on S^{D−1}. Fibonacci in 3-D, great-circle in 2-D, Gaussian-normalised elsewhere."""
    if D == 1:
        return np.array([[-1.0], [1.0]])
    if D == 2:
        phi = np.linspace(0.0, 2.0 * pi, n, endpoint=False)
        return np.stack([np.cos(phi), np.sin(phi)], axis=-1)
    if D == 3:
        i = np.arange(n) + 0.5
        phi = np.arccos(1.0 - 2.0 * i / n)
        theta = pi * (1.0 + 5.0 ** 0.5) * i
        return np.stack([
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ], axis=-1)
    rng = np.random.default_rng(0)
    g = rng.standard_normal((n, D))
    return g / np.linalg.norm(g, axis=-1, keepdims=True)


def _angular_gap_n_dirs(D: int, angular_gap_deg: float) -> int:
    """Directions on S^{D−1} to maintain the requested maximum angular gap."""
    if D == 1:
        return 2
    theta = np.deg2rad(angular_gap_deg)
    surface = _surface_area_unit_sphere(D)
    return int(np.ceil(surface / theta ** (D - 1)))


def kernel_field_probe_count(
    D: int,
    radii: tuple[float, ...] = DEFAULT_RADII,
    angular_gap_deg: float = DEFAULT_ANGULAR_GAP_DEG,
) -> int:
    """Total probe count for a KernelField configuration: centre + shells × directions.

    Used by SobolLocalEstimator's `n_samples=None` default to match KernelField's
    sample budget at the same dimensionality (fair compute / accuracy comparison).
    """
    return 1 + len(radii) * _angular_gap_n_dirs(D, angular_gap_deg)


def _in_unit_cube_torch(points: torch.Tensor) -> torch.Tensor:
    """Boolean mask: all dims in [0, 1]."""
    return ((points >= 0.0) & (points <= 1.0)).all(dim=-1)


# ---------------------------------------------------------------------------
# Estimator abstraction
# ---------------------------------------------------------------------------

class EvidenceEstimator(ABC):
    """Estimator for ∫_{[0,1]^D} D/(1+D) dz via per-kernel self-integrals."""

    @abstractmethod
    def integrated_evidence_perturbed_batched_joint_torch(
        self,
        index_old: KernelIndex,
        new_centers_SL: torch.Tensor,
        new_weights_SL: torch.Tensor,
    ) -> torch.Tensor:
        """Per-candidate joint ∫E ``(S,)`` with autograd through ``new_centers_SL``."""
        ...


# ---------------------------------------------------------------------------
# KernelField — radial shell quadrature
# ---------------------------------------------------------------------------

@dataclass
class KernelFieldEstimator(EvidenceEstimator):
    """Deterministic shell quadrature: origin + shells × directions on S^{D−1}."""

    radii: tuple[float, ...] = DEFAULT_RADII
    angular_gap_deg: float = DEFAULT_ANGULAR_GAP_DEG

    _cache: dict = field(default_factory=dict, repr=False, compare=False)
    _cache_torch: dict = field(default_factory=dict, repr=False, compare=False)

    def _probes_weights_self(
        self, D: int, sigma: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Probe offsets, quadrature weights, and self-density at each probe.

        Self-density `ρ_self[k] = exp(−‖offsets[k]‖²/2σ²)` is peak-1 at the
        kernel centre, matching `KernelIndex.density_at`. Same constant
        for every kernel of weight 1 — cached once per (D, σ) and scaled by
        the kernel's weight at use-time.
        """
        key = (D, sigma)
        if key in self._cache:
            return self._cache[key]

        n_dirs = _angular_gap_n_dirs(D, self.angular_gap_deg)
        dirs = _unit_sphere_directions(D, n_dirs)
        abs_radii = np.asarray(self.radii, dtype=float) * sigma
        shell_w, center_w = _radial_shell_weights(abs_radii, sigma, D)

        offsets_list: list[np.ndarray] = [np.zeros(D)]
        weights_list: list[float] = [center_w]
        for k, r in enumerate(abs_radii):
            w_per = shell_w[k] / dirs.shape[0]
            for d_vec in dirs:
                offsets_list.append(r * d_vec)
                weights_list.append(w_per)
        offsets = np.stack(offsets_list)
        weights = np.asarray(weights_list, dtype=float)

        inv_2sig2 = 1.0 / (2.0 * sigma ** 2)
        offset_sq = np.sum(offsets ** 2, axis=-1)
        self_density = np.exp(-offset_sq * inv_2sig2)

        self._cache[key] = (offsets, weights, self_density)
        return offsets, weights, self_density

    def _probes_weights_self_torch(
        self, D: int, sigma: float, dtype: torch.dtype, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Torch-cached version of ``_probes_weights_self``."""
        key = (D, sigma, dtype, device)
        if key in self._cache_torch:
            return self._cache_torch[key]
        offsets_np, weights_np, self_density_np = self._probes_weights_self(D, sigma)
        offsets = torch.from_numpy(offsets_np).to(device=device, dtype=dtype)
        weights = torch.from_numpy(weights_np).to(device=device, dtype=dtype)
        self_density = torch.from_numpy(self_density_np).to(device=device, dtype=dtype)
        self._cache_torch[key] = (offsets, weights, self_density)
        return offsets, weights, self_density

    def integrated_evidence_perturbed_batched_joint_torch(
        self,
        index_old: KernelIndex,
        new_centers_SL: torch.Tensor,
        new_weights_SL: torch.Tensor,
    ) -> torch.Tensor:
        """Per-candidate joint Δ∫E ``(S,)`` with autograd flowing through ``new_centers_SL``.

        ``index_old.centers`` / ``.weights`` come from the numpy-typed
        ``KernelIndex`` and are converted at call entry; old kernels are
        treated as constants (no grad), new candidates carry whatever
        ``requires_grad`` they were given. Regime dispatch via
        ``_choose_kde_regime`` selects dense / knn / cluster; only dense is
        implemented — knn / cluster log at INFO and fall back.
        """
        S = int(new_centers_SL.shape[0])
        if S == 0:
            return torch.zeros(0, dtype=new_centers_SL.dtype)
        L = int(new_centers_SL.shape[1])
        D = int(new_centers_SL.shape[2])
        sigma = index_old.sigma
        n_old = len(index_old.centers) if not index_old.is_empty else 0

        # Resolve the regime (logs + falls back to dense for unimplemented choices).
        # Only "dense" is implemented today; the call still routes correctly when
        # knn / cluster land.
        _resolve_kde_regime(n_old, sigma, D)
        return self._integrated_evidence_joint_dense_torch(
            index_old, new_centers_SL, new_weights_SL,
        )

    def _integrated_evidence_joint_dense_torch(
        self,
        index_old: KernelIndex,
        new_centers_SL: torch.Tensor,
        new_weights_SL: torch.Tensor,
    ) -> torch.Tensor:
        """Marginal-joint evidence with D/(D+1) weighting.

        Marginal: D independent 1D integrals — weighted D/(D+1).
        Joint: D-dimensional shell-probe integral — weighted 1/(D+1).

        Each marginal integral is one independent piece of per-dimension
        information; the joint is one piece of full-dimensional information.
        Equal weight per piece gives D/(D+1) for marginals. At D=1 this
        reduces to 0.5/0.5; as D→∞ marginals dominate (correct — joint
        becomes less informative in high-D).
        """
        D = int(new_centers_SL.shape[2])
        alpha_marginal = D / (D + 1)
        alpha_joint = 1.0 / (D + 1)
        e_marginal = self._marginal_evidence_torch(index_old, new_centers_SL, new_weights_SL)
        e_joint = self._joint_evidence_torch(index_old, new_centers_SL, new_weights_SL)
        return alpha_marginal * e_marginal + alpha_joint * e_joint

    def _marginal_evidence_torch(
        self,
        index_old: KernelIndex,
        new_centers_SL: torch.Tensor,
        new_weights_SL: torch.Tensor,
    ) -> torch.Tensor:
        """D independent 1D integrals — per-dimension evidence with 1D Gaussian density."""
        S = int(new_centers_SL.shape[0])
        L = int(new_centers_SL.shape[1])
        D = int(new_centers_SL.shape[2])
        dtype = new_centers_SL.dtype
        device = new_centers_SL.device
        sigma = index_old.sigma
        inv_2sig2 = 1.0 / (2.0 * sigma ** 2)
        n_old = len(index_old.centers) if not index_old.is_empty else 0

        # 1D probes: centre + ±r for each radius
        abs_radii = torch.tensor([r * sigma for r in self.radii], dtype=dtype, device=device)
        probes_1d = torch.cat([torch.zeros(1, dtype=dtype, device=device),
                               torch.cat([-abs_radii, abs_radii])])
        probes_1d = probes_1d.sort().values  # sorted for cleanliness
        P = probes_1d.shape[0]

        # 1D quadrature weights (normalised Gaussian measure)
        w_raw = torch.exp(-probes_1d ** 2 * inv_2sig2)
        quad_w = w_raw / w_raw.sum()

        # 1D self-density
        self_dens_1d = torch.exp(-probes_1d ** 2 * inv_2sig2)

        # Convert old centres once
        old_d_all = index_old.centers.to(device=device, dtype=dtype) if n_old > 0 else None
        old_w = index_old.weights.to(device=device, dtype=dtype) if n_old > 0 else None

        e_marginal = torch.zeros(S, dtype=dtype, device=device)

        for d in range(D):
            new_d = new_centers_SL[:, :, d]                          # (S, L)
            probes_d = new_d[:, :, None] + probes_1d[None, None, :]  # (S, L, P)
            in_dom_d = ((probes_d >= 0.0) & (probes_d <= 1.0)).to(dtype=dtype)

            # 1D cross-influence between new kernels
            diff_nn = probes_d[:, :, :, None] - new_d[:, None, None, :]  # (S, L, P, L)
            rho_nn = torch.exp(-diff_nn ** 2 * inv_2sig2)
            eye_LL = torch.eye(L, dtype=dtype, device=device)
            rho_other = (rho_nn * new_weights_SL[:, None, None, :] * (1.0 - eye_LL)[None, :, None, :]).sum(dim=-1)

            # 1D density from old kernels
            rho_old = torch.zeros(S, L, P, dtype=dtype, device=device)
            if n_old > 0:
                old_d = old_d_all[:, d]  # type: ignore[index]
                diff_on = probes_d[:, :, :, None] - old_d[None, None, None, :]
                rho_old = (torch.exp(-diff_on ** 2 * inv_2sig2) * old_w[None, None, None, :]).sum(dim=-1)  # type: ignore[index]

            total_rho = self_dens_1d[None, None, :] * new_weights_SL[:, :, None] + rho_other + rho_old
            integrand = 1.0 / (1.0 + total_rho)
            integral_d = (quad_w[None, None, :] * integrand * in_dom_d).sum(dim=-1)
            e_marginal = e_marginal + (new_weights_SL * integral_d).sum(dim=-1)

            # Old kernels' marginal integrals perturbation
            if n_old > 0:
                old_d = old_d_all[:, d]  # type: ignore[index]
                probes_old_d = old_d[:, None] + probes_1d[None, :]    # (n_old, P)
                in_dom_old = ((probes_old_d >= 0.0) & (probes_old_d <= 1.0)).to(dtype=dtype)
                # Old-to-old
                diff_oo = probes_old_d[:, :, None] - old_d[None, None, :]
                rho_oo = torch.exp(-diff_oo ** 2 * inv_2sig2) * old_w[None, None, :]  # type: ignore[index]
                eye_oo = torch.eye(n_old, dtype=dtype, device=device)
                rho_other_old = (rho_oo * (1.0 - eye_oo)[:, None, :]).sum(dim=-1)
                # New-to-old
                diff_no = probes_old_d[:, :, None, None] - new_d[None, None, :, :]
                delta = (torch.exp(-diff_no ** 2 * inv_2sig2) * new_weights_SL[None, None, :, :]).sum(dim=-1)
                self_old = self_dens_1d[None, :] * old_w[:, None]  # type: ignore[index]
                total_old = (self_old + rho_other_old)[:, :, None] + delta
                integrand_old = 1.0 / (1.0 + total_old)
                integral_old = (quad_w[None, :, None] * integrand_old * in_dom_old[:, :, None]).sum(dim=1)
                e_marginal = e_marginal + (old_w[:, None] * integral_old).sum(dim=0)  # type: ignore[index]

        return e_marginal / D

    def _joint_evidence_torch(
        self,
        index_old: KernelIndex,
        new_centers_SL: torch.Tensor,
        new_weights_SL: torch.Tensor,
    ) -> torch.Tensor:
        """D-dimensional shell-probe integral — isotropic kernel (interaction evidence)."""
        S = int(new_centers_SL.shape[0])
        L = int(new_centers_SL.shape[1])
        D = int(new_centers_SL.shape[2])
        dtype = new_centers_SL.dtype
        device = new_centers_SL.device
        sigma = index_old.sigma
        inv_2sig2 = 1.0 / (2.0 * sigma ** 2)

        offsets, quad_weights, self_density = self._probes_weights_self_torch(D, sigma, dtype, device)
        M = offsets.shape[0]

        n_old = len(index_old.centers) if not index_old.is_empty else 0

        probes_new_SL = offsets[None, None, :, :] + new_centers_SL[:, :, None, :]
        in_domain_new_SL = _in_unit_cube_torch(probes_new_SL.reshape(-1, D)).reshape(
            S, L, M
        ).to(dtype=dtype)

        diff_self = probes_new_SL[:, :, :, None, :] - new_centers_SL[:, None, None, :, :]
        d2_self = (diff_self * diff_self).sum(dim=-1)
        kernels_self = torch.exp(-d2_self * inv_2sig2)
        eye_LL = torch.eye(L, dtype=dtype, device=device)
        keep_LL = 1.0 - eye_LL
        weighted_self = kernels_self * new_weights_SL[:, None, None, :]
        rho_other_self_SL = (weighted_self * keep_LL[None, :, None, :]).sum(dim=-1)

        if n_old == 0:
            self_contribution_new = self_density[None, None, :] * new_weights_SL[:, :, None]
            total_density_new = self_contribution_new + rho_other_self_SL
            integrand_new = 1.0 / (1.0 + total_density_new)
            integral_new_SL = (
                quad_weights[None, None, :] * integrand_new * in_domain_new_SL
            ).sum(dim=-1)
            return (new_weights_SL * integral_new_SL).sum(dim=-1)

        old_centers = index_old.centers.to(device=device, dtype=dtype)
        old_weights = index_old.weights.to(device=device, dtype=dtype)

        probes_per_old = offsets[None, :, :] + old_centers[:, None, :]
        in_domain_per_old = _in_unit_cube_torch(probes_per_old.reshape(-1, D)).reshape(
            n_old, M
        ).to(dtype=dtype)

        diff_old = probes_per_old[:, :, None, :] - old_centers[None, None, :, :]
        d2_old = (diff_old * diff_old).sum(dim=-1)
        kernels_old = torch.exp(-d2_old * inv_2sig2)
        weighted_old = kernels_old * old_weights[None, None, :]
        eye_jk = torch.eye(n_old, dtype=dtype, device=device)
        keep_jk = 1.0 - eye_jk
        rho_other_per_old = (weighted_old * keep_jk[:, None, :]).sum(dim=-1)
        self_contribution_per_old = self_density[None, :] * old_weights[:, None]

        diff_new_to_old = (
            probes_per_old[:, :, None, None, :] - new_centers_SL[None, None, :, :, :]
        )
        d2_new_to_old = (diff_new_to_old * diff_new_to_old).sum(dim=-1)
        kernels_new_to_old = torch.exp(-d2_new_to_old * inv_2sig2)
        delta_density = (
            kernels_new_to_old * new_weights_SL[None, None, :, :]
        ).sum(dim=-1)

        total_density_old = (
            (self_contribution_per_old + rho_other_per_old)[:, :, None] + delta_density
        )
        integrand_old = 1.0 / (1.0 + total_density_old)
        integral_old_per_s = (
            integrand_old * (quad_weights[None, :, None] * in_domain_per_old[:, :, None])
        ).sum(dim=1)

        diff_old_to_new = (
            probes_new_SL[:, :, :, None, :] - old_centers[None, None, None, :, :]
        )
        d2_old_to_new = (diff_old_to_new * diff_old_to_new).sum(dim=-1)
        kernels_old_to_new = torch.exp(-d2_old_to_new * inv_2sig2)
        rho_old_at_new_probes = (
            kernels_old_to_new * old_weights[None, None, None, :]
        ).sum(dim=-1)

        self_contribution_new = self_density[None, None, :] * new_weights_SL[:, :, None]
        total_density_new = (
            self_contribution_new + rho_old_at_new_probes + rho_other_self_SL
        )
        integrand_new = 1.0 / (1.0 + total_density_new)
        integral_new_SL = (
            quad_weights[None, None, :] * integrand_new * in_domain_new_SL
        ).sum(dim=-1)

        return (
            (old_weights[:, None] * integral_old_per_s).sum(dim=0)
            + (new_weights_SL * integral_new_SL).sum(dim=-1)
        )


# ---------------------------------------------------------------------------
# Sobol-local — QMC cube around each kernel
# ---------------------------------------------------------------------------

@dataclass
class SobolLocalEstimator(EvidenceEstimator):
    """Volume-weighted QMC in a [center ± box·σ]^D cube.

    Probe count is fixed at ``n_samples`` regardless of D — the high-D
    escape hatch when KernelField's probe count (which grows like
    ``D · π^((D−1)/2)``) becomes intractable. Accuracy degrades with D
    at fixed ``n_samples``, but the compute stays bounded.

    ``n_samples=None`` ties the count to KernelField's probe count at
    the same D — gives matched compute and matched extent (with
    ``box=2.0``) for fair comparison.
    """

    box: float = 2.0
    n_samples: int | None = None
    seed: int = 0

    def _resolve_n_samples(self, D: int) -> int:
        if self.n_samples is not None:
            return int(self.n_samples)
        return kernel_field_probe_count(D)

    def _sobol_offsets_torch(
        self,
        D: int,
        sigma: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Sobol-distributed offsets in ``[-box·σ, +box·σ]^D``."""
        n = self._resolve_n_samples(D)
        box_side = 2.0 * self.box * sigma
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            engine = torch.quasirandom.SobolEngine(
                dimension=D, scramble=True, seed=self.seed,
            )
            unit = engine.draw(n).to(dtype=dtype, device=device)
        return box_side * (unit - 0.5)  # (n, D)

    def integrated_evidence_perturbed_batched_joint_torch(
        self,
        index_old: KernelIndex,
        new_centers_SL: torch.Tensor,
        new_weights_SL: torch.Tensor,
    ) -> torch.Tensor:
        """Per-candidate joint Δ∫E ``(S,)``, gradient-traversable through ``new_centers_SL``.

        Sobol-cube self-integral per kernel; old kernels are constants from
        ``index_old`` (numpy → tensor at call entry, no grad). New kernels'
        Gaussian probes are placed as ``new_centers_SL[s, l] + offsets``,
        so gradient flows through the centre into both the probe positions
        and the per-probe density evaluations.
        """
        S = int(new_centers_SL.shape[0])
        if S == 0 or new_centers_SL.numel() == 0:
            return torch.zeros(0, dtype=new_centers_SL.dtype, device=new_centers_SL.device)
        L = int(new_centers_SL.shape[1])
        D = int(new_centers_SL.shape[2])
        dtype = new_centers_SL.dtype
        device = new_centers_SL.device

        sigma = index_old.sigma
        inv_2sig2 = 1.0 / (2.0 * sigma ** 2)
        box_side = 2.0 * self.box * sigma
        volume = box_side ** D

        offsets = self._sobol_offsets_torch(D, sigma, dtype, device)  # (n, D)
        n = offsets.shape[0]

        # Peak-1 Gaussian density at offset distance — same for every kernel
        # since rho_j(z_j + offset) = exp(-||offset||²/2σ²) is centre-invariant.
        rho = torch.exp(-(offsets ** 2).sum(dim=-1) * inv_2sig2)  # (n,)

        # Old kernels as torch tensors (constants).
        if index_old.is_empty:
            old_centers = torch.zeros((0, D), dtype=dtype, device=device)
            old_weights = torch.zeros(0, dtype=dtype, device=device)
        else:
            old_centers = index_old.centers.to(dtype=dtype, device=device)
            old_weights = index_old.weights.to(dtype=dtype, device=device)
        n_old = old_centers.shape[0]

        # ---------- self-integrals for OLD kernels (per S) ----------
        if n_old > 0:
            probes_old = old_centers[:, None, :] + offsets[None, :, :]      # (n_old, n, D)

            # D contribution from old kernels at old probes (S-independent).
            diff_oo = probes_old.unsqueeze(2) - old_centers[None, None, :, :]  # (n_old, n, n_old, D)
            d2_oo = (diff_oo ** 2).sum(dim=-1)
            D_old_at_old = (
                torch.exp(-d2_oo * inv_2sig2) * old_weights[None, None, :]
            ).sum(dim=-1)                                                      # (n_old, n)

            # D contribution from new kernels at old probes (per S).
            diff_on = probes_old[:, :, None, None, :] - new_centers_SL[None, None, :, :, :]  # (n_old, n, S, L, D)
            d2_on = (diff_on ** 2).sum(dim=-1)
            D_new_at_old = (
                torch.exp(-d2_on * inv_2sig2) * new_weights_SL[None, None, :, :]
            ).sum(dim=-1)                                                      # (n_old, n, S)

            D_total_at_old = D_old_at_old.unsqueeze(-1) + D_new_at_old        # (n_old, n, S)
            in_cube_old = _in_unit_cube_torch(
                probes_old.reshape(-1, D),
            ).reshape(n_old, n).to(dtype=dtype)                               # (n_old, n)

            integrand_old = (
                rho[None, :, None] / (1.0 + D_total_at_old)
                * in_cube_old[:, :, None]
            )                                                                  # (n_old, n, S)
            self_int_old = volume * integrand_old.mean(dim=1)                 # (n_old, S)
            E_old_contrib = (old_weights[:, None] * self_int_old).sum(dim=0)  # (S,)
        else:
            E_old_contrib = torch.zeros(S, dtype=dtype, device=device)

        # ---------- self-integrals for NEW kernels (per S) ----------
        probes_new = new_centers_SL[:, :, None, :] + offsets[None, None, :, :]  # (S, L, n, D)

        if n_old > 0:
            diff_no = probes_new.unsqueeze(3) - old_centers[None, None, None, :, :]  # (S, L, n, n_old, D)
            d2_no = (diff_no ** 2).sum(dim=-1)
            D_old_at_new = (
                torch.exp(-d2_no * inv_2sig2) * old_weights[None, None, None, :]
            ).sum(dim=-1)                                                      # (S, L, n)
        else:
            D_old_at_new = torch.zeros((S, L, n), dtype=dtype, device=device)

        diff_nn = probes_new.unsqueeze(3) - new_centers_SL[:, None, None, :, :]  # (S, L, n, L, D)
        d2_nn = (diff_nn ** 2).sum(dim=-1)
        D_new_at_new = (
            torch.exp(-d2_nn * inv_2sig2) * new_weights_SL[:, None, None, :]
        ).sum(dim=-1)                                                          # (S, L, n)

        D_total_at_new = D_old_at_new + D_new_at_new                          # (S, L, n)
        in_cube_new = _in_unit_cube_torch(
            probes_new.reshape(-1, D),
        ).reshape(S, L, n).to(dtype=dtype)                                     # (S, L, n)

        integrand_new = rho[None, None, :] / (1.0 + D_total_at_new) * in_cube_new  # (S, L, n)
        self_int_new = volume * integrand_new.mean(dim=-1)                    # (S, L)
        E_new_contrib = (new_weights_SL * self_int_new).sum(dim=-1)           # (S,)

        return E_old_contrib + E_new_contrib


# ---------------------------------------------------------------------------
# Config + factory
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EstimatorConfig:
    """Config for the evidence estimator selected at runtime."""

    type: Literal["kernel_field", "sobol_local"] = "kernel_field"
    # KernelField
    radii: tuple[float, ...] = DEFAULT_RADII
    angular_gap_deg: float = DEFAULT_ANGULAR_GAP_DEG
    # Sobol-local
    box: float = 2.0
    n_samples: int | None = None  # None → match KernelField probe count at runtime
    seed: int = 0
    # Shared
    cutoff_sigmas: float = 5.0
    truncation_threshold: int = 10


def make_estimator(config: EstimatorConfig) -> EvidenceEstimator:
    if config.type == "kernel_field":
        return KernelFieldEstimator(
            radii=config.radii, angular_gap_deg=config.angular_gap_deg,
        )
    if config.type == "sobol_local":
        return SobolLocalEstimator(
            box=config.box, n_samples=config.n_samples, seed=config.seed,
        )
    raise ValueError(f"unknown estimator type: {config.type!r}")


def _evidence_gain_grid_from_centers(
    centers: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    x_bounds: tuple[float, float] = (0.0, 1.0),
    y_bounds: tuple[float, float] = (0.0, 1.0),
    resolution: int = 80,
    estimator: EvidenceEstimator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Low-level: 2D ΔE grid from pre-normalized centers in [0,1]^D."""
    kf = estimator or KernelFieldEstimator()
    D = centers.shape[1]
    index_old = KernelIndex(centers, weights, sigma)

    empty_index = KernelIndex(np.empty((0, D)), np.empty(0), sigma)
    old_centers_t = index_old.centers.unsqueeze(0).double()
    old_weights_t = index_old.weights.unsqueeze(0).double()
    E_old = float(kf.integrated_evidence_perturbed_batched_joint_torch(
        empty_index, old_centers_t, old_weights_t,
    )[0].item())

    xs_param = np.linspace(*x_bounds, resolution)
    ys_param = np.linspace(*y_bounds, resolution)
    xs_norm = (xs_param - x_bounds[0]) / (x_bounds[1] - x_bounds[0])
    ys_norm = (ys_param - y_bounds[0]) / (y_bounds[1] - y_bounds[0])

    gain_grid = np.zeros((resolution, resolution))
    for j in range(resolution):
        row_pts = np.stack([xs_norm, np.full(resolution, ys_norm[j])], axis=-1)
        row_pts_t = torch.from_numpy(row_pts).double().unsqueeze(1)
        weights_t = torch.ones(resolution, 1, dtype=torch.float64)
        e_new = kf.integrated_evidence_perturbed_batched_joint_torch(
            index_old, row_pts_t, weights_t,
        )
        gain_grid[j, :] = e_new.detach().cpu().numpy() - E_old

    return xs_param, ys_param, gain_grid


def compute_evidence_gain_grid(
    experiments: list,
    all_axes: list,
    x_key: str,
    y_key: str,
    sigma: float,
    resolution: int = 60,
    param_transform=None,
    estimator: EvidenceEstimator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 2D ΔE grid using the KernelField ANOVA pipeline.

    Handles ``expand_experiments`` → center extraction → normalization
    to [0,1]² → KernelField integration. Returns grid in parameter space.

    Parameters
    ----------
    experiments : list of ExperimentSpec / ExperimentData / dict
    all_axes : list of AxisSpec — all parameter axes (used for bounds)
    x_key, y_key : visible axis keys
    sigma : KDE bandwidth (in [0,1] normalized space)
    """
    from ..plotting.evidence import expand_experiments

    x_axis = next(a for a in all_axes if a.key == x_key)
    y_axis = next(a for a in all_axes if a.key == y_key)

    pts, _, _ = expand_experiments(experiments, param_transform)
    if not pts:
        xs = np.linspace(*x_axis.bounds, resolution)
        ys = np.linspace(*y_axis.bounds, resolution)
        return xs, ys, np.zeros((resolution, resolution))

    x_lo, x_hi = x_axis.bounds
    y_lo, y_hi = y_axis.bounds
    centers_2d = np.array([
        [
            np.clip((float(p.get(x_key, (x_lo + x_hi) / 2)) - x_lo) / (x_hi - x_lo), 0, 1),
            np.clip((float(p.get(y_key, (y_lo + y_hi) / 2)) - y_lo) / (y_hi - y_lo), 0, 1),
        ]
        for p in pts
    ])

    return _evidence_gain_grid_from_centers(
        centers_2d, np.ones(len(pts)), sigma,
        x_bounds=x_axis.bounds,
        y_bounds=y_axis.bounds,
        resolution=resolution,
        estimator=estimator,
    )
