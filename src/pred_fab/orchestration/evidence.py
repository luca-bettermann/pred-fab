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

def _choose_kde_regime(n_kernels: int, sigma: float, D: int) -> str:
    """Select KDE evaluation regime based on `(n_kernels, sigma, D)`.

    The right metric is **expected number of kernels contributing to a
    typical probe's density** — not n_kernels alone. Computed as
    ``n_kernels × V(5σ-ball in D dims) / V_unit_cube``. When ``n_active ≪
    n_kernels``, most kernels contribute negligibly and a sparse top-K
    gather wins. When ``n_active ≈ n_kernels``, dense is correct.

    Returns:
      ``"dense"``   — sum over all N kernels (regimes 1+2 from plan).
      ``"knn"``     — sparse top-K gather (regime 3, stubbed in commit 4).
      ``"cluster"`` — c-component summarisation (regime 4, stubbed in commit 4).

    See ``IMPLEMENTATION_PLAN.md`` for the full table of `(D, σ, n)` cases.
    """
    # 5σ ball volume in D dims, capped to unit cube domain.
    v_5sigma = (pi ** (D / 2)) * ((5.0 * sigma) ** D) / gamma(D / 2 + 1)
    v_5sigma = min(v_5sigma, 1.0)
    n_active = n_kernels * v_5sigma

    if n_kernels < 100:
        return "dense"  # too small for KNN overhead to pay
    if n_active > 10000 or n_kernels > 100000:
        return "cluster"
    if n_active < 50 or n_active < n_kernels * 0.5:
        return "knn"
    return "dense"


# ---------------------------------------------------------------------------
# Kernel index — O(M · log K) density evaluation via neighbour search
# ---------------------------------------------------------------------------

class KernelIndex:
    """Spatial index over Gaussian kernel centres for fast D(z) evaluation.

    replaced ``scipy.spatial.cKDTree`` neighbour
    lookup with batched ``torch.cdist`` distance computation. For
    n_kernels < ``truncation_threshold`` (default 10), full distance is
    computed directly (correct + cheap). Above the threshold a 5σ-radius
    mask is applied via ``torch.where`` — same semantics as the old
    cKDTree query_ball_point cutoff, but without scipy.
    """

    def __init__(
        self,
        centers: np.ndarray,
        weights: np.ndarray,
        sigma: float,
        cutoff_sigmas: float = 5.0,
        truncation_threshold: int = 10,
    ):
        self.centers = np.asarray(centers, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
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

    def density_at(self, points: np.ndarray, exclude_idx: int | None = None) -> np.ndarray:
        """D(z) = Σⱼ wⱼ · exp(−‖z−zⱼ‖²/2σ²) at each row of `points`.

        Peak-1 Gaussian per kernel — ρⱼ(zⱼ) = 1 — so D is bounded by the
        sum of weights when kernels overlap, and the saturation transform
        E = D/(1+D) keeps a usable gradient instead of pinning at 1 from a
        single kernel.

        If `exclude_idx` is given, kernel at that index is skipped — used by
        the KernelField estimator to substitute a precomputed self-density.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        M = points.shape[0]
        if self.is_empty or self._D == 0:
            return np.zeros(M)

        inv_2sig2 = 1.0 / (2.0 * self.sigma ** 2)

        if self._n < self.truncation_threshold:
            # Direct sum — small N, full Gaussian tail kept.
            diff = points[:, None, :] - self.centers[None, :, :]
            d2 = np.sum(diff * diff, axis=-1)
            exp_term = np.exp(-d2 * inv_2sig2)
            if exclude_idx is not None:
                mask = np.arange(self._n) != exclude_idx
                return (exp_term[:, mask] * self.weights[mask]).sum(axis=-1)
            return (exp_term * self.weights).sum(axis=-1)

        # Truncated sum via torch.cdist (replaces scipy.spatial.cKDTree).
        # Squared 5σ cutoff — kernels beyond contribute < exp(−12.5) ≈ 4e−6.
        cutoff_d2 = (self.cutoff_sigmas * self.sigma) ** 2
        pts_t = torch.from_numpy(points).double()
        cnt_t = torch.from_numpy(self.centers).double()
        # cdist returns euclidean distance; we want squared.
        d2_t = torch.cdist(pts_t, cnt_t, p=2.0).pow(2)  # (M, n_kernels)
        # Apply 5σ truncation: zero out distances above cutoff.
        in_range = d2_t <= cutoff_d2
        exp_term = torch.where(in_range, torch.exp(-d2_t * inv_2sig2), torch.zeros_like(d2_t))
        weights_t = torch.from_numpy(self.weights).double()
        if exclude_idx is not None:
            keep = torch.ones(self._n, dtype=torch.bool)
            keep[exclude_idx] = False
            return (exp_term[:, keep] * weights_t[keep]).sum(dim=-1).cpu().numpy()
        return (exp_term * weights_t).sum(dim=-1).cpu().numpy()


# ---------------------------------------------------------------------------
# Radial shell quadrature (Gaussian measure)
# ---------------------------------------------------------------------------

def _surface_area_unit_sphere(D: int) -> float:
    return 2.0 * pi ** (D / 2.0) / gamma(D / 2.0)


def _gaussian_radial_pdf(r: float, sigma: float, D: int) -> float:
    """Peak-1 Gaussian density at radius r — ρ(0) = 1.

    Used as the integration measure for radial shell quadrature, kept
    consistent with the peak-1 kernel definition in :class:`KernelIndex`.
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


def _in_unit_cube(points: np.ndarray) -> np.ndarray:
    return np.all((points >= 0.0) & (points <= 1.0), axis=-1)


def _in_unit_cube_torch(points: torch.Tensor) -> torch.Tensor:
    """Tensor mirror of ``_in_unit_cube`` — boolean mask over the last dim."""
    return ((points >= 0.0) & (points <= 1.0)).all(dim=-1)


# ---------------------------------------------------------------------------
# Estimator abstraction
# ---------------------------------------------------------------------------

class EvidenceEstimator(ABC):
    """Estimator for ∫_{[0,1]^D} D/(1+D) dz via per-kernel self-integrals."""

    @abstractmethod
    def self_integral(
        self, center: np.ndarray, index: KernelIndex, kernel_idx: int | None = None,
    ) -> float:
        """𝔼_{z~N(center, σ²I)} [1/(1+D(z)) · 1_{z ∈ [0,1]^D}].

        `kernel_idx` (when given) tells the estimator which kernel inside `index`
        owns this `center`, enabling self-density caching to skip a redundant
        kernel-tree query.
        """
        ...

    def integrated_evidence(self, index: KernelIndex) -> float:
        """I = Σⱼ wⱼ · self_integral(center_j)."""
        if index.is_empty:
            return 0.0
        total = 0.0
        for j in range(len(index.centers)):
            total += float(index.weights[j]) * self.self_integral(
                index.centers[j], index, kernel_idx=j,
            )
        return total

    def integrated_evidence_perturbed_batched(
        self,
        index_old: KernelIndex,
        new_centers_S: np.ndarray,
        new_weights_S: np.ndarray,
    ) -> np.ndarray:
        """E(old ∪ {(new_centers_S[s], new_weights_S[s])}) for each s. Returns ``(S,)``.

        Default implementation loops ``integrated_evidence`` per candidate,
        rebuilding ``index_new[s]`` each time. Subclasses override for
        vectorised perf — see :meth:`KernelFieldEstimator.integrated_evidence_perturbed_batched`.
        """
        S = int(new_centers_S.shape[0])
        if S == 0:
            return np.zeros(0, dtype=np.float64)
        out = np.zeros(S, dtype=np.float64)
        for s in range(S):
            if index_old.is_empty:
                all_centers = new_centers_S[s:s + 1]
                all_weights = new_weights_S[s:s + 1]
            else:
                all_centers = np.vstack([index_old.centers, new_centers_S[s:s + 1]])
                all_weights = np.concatenate([index_old.weights, new_weights_S[s:s + 1]])
            index_new = KernelIndex(
                all_centers, all_weights, index_old.sigma,
                cutoff_sigmas=index_old.cutoff_sigmas,
                truncation_threshold=index_old.truncation_threshold,
            )
            out[s] = self.integrated_evidence(index_new)
        return out


# ---------------------------------------------------------------------------
# KernelField — radial shell quadrature
# ---------------------------------------------------------------------------

@dataclass
class KernelFieldEstimator(EvidenceEstimator):
    """Deterministic shell quadrature: origin + shells × directions on S^{D−1}."""

    radii: tuple[float, ...] = DEFAULT_RADII
    angular_gap_deg: float = DEFAULT_ANGULAR_GAP_DEG

    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def _probes_and_weights(self, D: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
        """Probe offsets (relative to any centre) and quadrature weights."""
        offsets, weights, _ = self._probes_weights_self(D, sigma)
        return offsets, weights

    def _probes_weights_self(
        self, D: int, sigma: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Probe offsets, quadrature weights, and self-density at each probe.

        Self-density `ρ_self[k] = exp(−‖offsets[k]‖²/2σ²)` is peak-1 at the
        kernel centre, matching :meth:`KernelIndex.density_at`. Same constant
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

    def self_integral(
        self, center: np.ndarray, index: KernelIndex, kernel_idx: int | None = None,
    ) -> float:
        D = int(center.shape[-1])
        offsets, weights, self_density = self._probes_weights_self(D, index.sigma)
        probes = offsets + center
        in_domain = _in_unit_cube(probes).astype(float)
        if kernel_idx is not None:
            rho_other = index.density_at(probes, exclude_idx=kernel_idx)
            rho = self_density * float(index.weights[kernel_idx]) + rho_other
        else:
            rho = index.density_at(probes)
        integrand = 1.0 / (1.0 + rho)
        return float(np.sum(weights * integrand * in_domain))

    def integrated_evidence_perturbed_batched(
        self,
        index_old: KernelIndex,
        new_centers_S: np.ndarray,
        new_weights_S: np.ndarray,
    ) -> np.ndarray:
        """Vectorised E(old ∪ {(new[s], w[s])}) per s. Returns ``(S,)``.

        Replaces the S-times-rebuild-`index_new` + S-times-`integrated_evidence`
        loop with a single broadcast computation:

          - One ``(n_old, M, S)`` distance tensor for "new candidates → probes
            around each old kernel" — the perturbation each candidate s adds
            to each old self-integral.
          - One ``(S, M, n_old)`` distance tensor for "old kernels → probes
            around each new candidate" — needed for each new kernel's own
            self-integral.
          - All quadrature reductions vectorised across the candidate batch.

        Truncation: when ``index_old`` has ≥ ``truncation_threshold`` kernels,
        a 5σ mask is applied to both distance tensors; matches scalar
        ``KernelIndex.density_at`` semantics within numerical tolerance.

        Equivalence: matches the scalar
        ``EvidenceEstimator.integrated_evidence(index_new[s])`` to ~1e-5,
        with the loose tolerance reflecting different summation orders
        between the per-candidate scalar reduction and the broadcast tensor.
        """
        S = int(new_centers_S.shape[0])
        if S == 0:
            return np.zeros(0, dtype=np.float64)
        sigma = index_old.sigma
        D = int(new_centers_S.shape[1])
        inv_2sig2 = 1.0 / (2.0 * sigma ** 2)
        offsets, quad_weights, self_density = self._probes_weights_self(D, sigma)
        M = offsets.shape[0]

        # Truncation mask threshold (squared distance for cheap comparison).
        n_old = len(index_old.centers) if not index_old.is_empty else 0
        do_truncate = n_old >= index_old.truncation_threshold
        cutoff_d2 = (index_old.cutoff_sigmas * sigma) ** 2

        # === Probes around each new candidate: (S, M, D) ===
        probes_new = offsets[None, :, :] + new_centers_S[:, None, :]
        in_domain_new = _in_unit_cube(probes_new.reshape(-1, D)).reshape(S, M).astype(np.float64)

        # === Branch 1: no old kernels — only the new kernel contributes ===
        if n_old == 0:
            # Density at probes_new comes only from the new kernel itself.
            # rho_new[s, m] = self_density[m] * w_new[s]
            rho_new = self_density[None, :] * new_weights_S[:, None]
            integrand_new = 1.0 / (1.0 + rho_new)
            integral_new_per_s = (
                quad_weights[None, :] * integrand_new * in_domain_new
            ).sum(axis=-1)
            return new_weights_S * integral_new_per_s

        # === Branch 2: old kernels exist ===
        old_centers = index_old.centers
        old_weights = index_old.weights

        # ---- Per-old precompute (independent of s) ----
        # Probes around each old kernel: (n_old, M, D)
        probes_per_old = offsets[None, :, :] + old_centers[:, None, :]
        in_domain_per_old = (
            _in_unit_cube(probes_per_old.reshape(-1, D)).reshape(n_old, M).astype(np.float64)
        )

        # rho_other_per_old[j, m] = sum over k != j of w_k * exp(-||probes_per_old[j,m] - old_k||²/2σ²)
        # diff_old: (n_old, M, n_old, D) — broadcast probes_per_old vs old_centers
        diff_old = probes_per_old[:, :, None, :] - old_centers[None, None, :, :]
        d2_old = np.sum(diff_old * diff_old, axis=-1)  # (n_old, M, n_old)
        kernels_old = np.exp(-d2_old * inv_2sig2)  # (n_old, M, n_old)
        if do_truncate:
            kernels_old = np.where(d2_old <= cutoff_d2, kernels_old, 0.0)
        weighted_old = kernels_old * old_weights[None, None, :]  # (n_old, M, n_old)
        # Mask out k == j (last dim equals first): rho_other excludes j
        eye_jk = np.eye(n_old, dtype=np.float64)  # (n_old, n_old)
        keep_jk = (1.0 - eye_jk)  # (n_old, n_old)
        rho_other_per_old = (weighted_old * keep_jk[:, None, :]).sum(axis=-1)  # (n_old, M)
        self_contribution_per_old = self_density[None, :] * old_weights[:, None]  # (n_old, M)

        # ---- Per-(s, j) perturbation: density at probes_per_old[j, m] from new kernel s ----
        # diff_new_to_old_probes: (n_old, M, S, D)
        diff_new_to_old = probes_per_old[:, :, None, :] - new_centers_S[None, None, :, :]
        d2_new_to_old = np.sum(diff_new_to_old * diff_new_to_old, axis=-1)  # (n_old, M, S)
        kernels_new_to_old = np.exp(-d2_new_to_old * inv_2sig2)
        if do_truncate:
            kernels_new_to_old = np.where(d2_new_to_old <= cutoff_d2, kernels_new_to_old, 0.0)
        delta_density = kernels_new_to_old * new_weights_S[None, None, :]  # (n_old, M, S)

        # ---- Old kernels' self-integrals when new kernel is added ----
        # total[j, m, s] = self_contribution_per_old[j, m] + rho_other_per_old[j, m] + delta_density[j, m, s]
        total_density_old = (
            (self_contribution_per_old + rho_other_per_old)[:, :, None] + delta_density
        )  # (n_old, M, S)
        integrand_old = 1.0 / (1.0 + total_density_old)  # (n_old, M, S)
        # weighted by quadrature × in_domain, sum over m → (n_old, S)
        integral_old_per_s = (
            integrand_old * (quad_weights[None, :, None] * in_domain_per_old[:, :, None])
        ).sum(axis=1)

        # ---- New kernel's own self-integral ----
        # density at probes_new from all old kernels: (S, M, n_old) → sum over n_old → (S, M)
        diff_old_to_new = probes_new[:, :, None, :] - old_centers[None, None, :, :]
        d2_old_to_new = np.sum(diff_old_to_new * diff_old_to_new, axis=-1)  # (S, M, n_old)
        kernels_old_to_new = np.exp(-d2_old_to_new * inv_2sig2)
        if do_truncate:
            kernels_old_to_new = np.where(d2_old_to_new <= cutoff_d2, kernels_old_to_new, 0.0)
        rho_old_at_new_probes = (
            kernels_old_to_new * old_weights[None, None, :]
        ).sum(axis=-1)  # (S, M)
        self_contribution_new = self_density[None, :] * new_weights_S[:, None]  # (S, M)
        total_density_new = self_contribution_new + rho_old_at_new_probes  # (S, M)
        integrand_new = 1.0 / (1.0 + total_density_new)  # (S, M)
        integral_new_per_s = (quad_weights[None, :] * integrand_new * in_domain_new).sum(axis=-1)  # (S,)

        # ---- Combine ----
        # E_new[s] = sum_j w_j_old * integral_old_per_s[j, s] + w_new[s] * integral_new_per_s[s]
        e_new = (old_weights[:, None] * integral_old_per_s).sum(axis=0) + new_weights_S * integral_new_per_s
        return e_new

    def integrated_evidence_perturbed_batched_joint(
        self,
        index_old: KernelIndex,
        new_centers_SL: np.ndarray,
        new_weights_SL: np.ndarray,
    ) -> np.ndarray:
        """Joint-batched ``E(old ∪ {L kernels of candidate s})`` per s. Returns ``(S,)``.

        Like :meth:`integrated_evidence_perturbed_batched` but each candidate
        adds **L kernels jointly** (a schedule trajectory) instead of one. Used
        by schedule-mode acquisition where each DE candidate decodes to an
        L-step trajectory that's evaluated as a joint Δ∫E.

        Shapes:
          - ``new_centers_SL``: ``(S, L, D)`` — S candidates × L points each.
          - ``new_weights_SL``: ``(S, L)`` — per-point weights.

        Reduces to ``integrated_evidence_perturbed_batched`` when ``L == 1``
        (verified by equivalence test). For L > 1 each candidate's L points
        are added together; old kernels see all L additions; new kernels of
        the same candidate also see each other's contributions (correct
        joint-Δ∫E semantics).
        """
        S = int(new_centers_SL.shape[0])
        if S == 0:
            return np.zeros(0, dtype=np.float64)
        L = int(new_centers_SL.shape[1])
        D = int(new_centers_SL.shape[2])
        sigma = index_old.sigma
        inv_2sig2 = 1.0 / (2.0 * sigma ** 2)
        offsets, quad_weights, self_density = self._probes_weights_self(D, sigma)
        M = offsets.shape[0]

        n_old = len(index_old.centers) if not index_old.is_empty else 0
        do_truncate = n_old >= index_old.truncation_threshold
        cutoff_d2 = (index_old.cutoff_sigmas * sigma) ** 2

        # Probes around each new candidate's L kernels: (S, L, M, D)
        probes_new_SL = offsets[None, None, :, :] + new_centers_SL[:, :, None, :]
        in_domain_new_SL = (
            _in_unit_cube(probes_new_SL.reshape(-1, D)).reshape(S, L, M).astype(np.float64)
        )

        # Cross-influence between candidate-s's own L kernels at probes_s_l:
        # density at probes_s_l from the OTHER (L-1) kernels of the same candidate.
        # diff_self_l_to_lprime: (S, L, M, L, D) — probes around l vs centres at lprime
        diff_self = probes_new_SL[:, :, :, None, :] - new_centers_SL[:, None, None, :, :]  # (S, L, M, L, D)
        d2_self = np.sum(diff_self * diff_self, axis=-1)  # (S, L, M, L)
        kernels_self = np.exp(-d2_self * inv_2sig2)
        if do_truncate:
            kernels_self = np.where(d2_self <= cutoff_d2, kernels_self, 0.0)
        # Mask out l == lprime (that's the kernel's own self-density, handled separately)
        eye_LL = np.eye(L, dtype=np.float64)
        keep_LL = (1.0 - eye_LL)  # (L, L)
        weighted_self = kernels_self * new_weights_SL[:, None, None, :]  # broadcast over m
        rho_other_self_at_probes_SL = (
            weighted_self * keep_LL[None, :, None, :]
        ).sum(axis=-1)  # (S, L, M) — density from OTHER L-1 kernels of same candidate

        # === Branch 1: no old kernels ===
        if n_old == 0:
            # total density at probes_s_l: self_contribution + other-self-contribution
            self_contribution_new = self_density[None, None, :] * new_weights_SL[:, :, None]  # (S, L, M)
            total_density_new = self_contribution_new + rho_other_self_at_probes_SL
            integrand_new = 1.0 / (1.0 + total_density_new)
            integral_new_SL = (
                quad_weights[None, None, :] * integrand_new * in_domain_new_SL
            ).sum(axis=-1)  # (S, L)
            # E_new[s] = sum_l w_new_SL[s, l] * integral_new[s, l]
            return (new_weights_SL * integral_new_SL).sum(axis=-1)

        # === Branch 2: old kernels exist ===
        old_centers = index_old.centers
        old_weights = index_old.weights

        # ---- Per-old precompute (independent of s) ----
        probes_per_old = offsets[None, :, :] + old_centers[:, None, :]  # (n_old, M, D)
        in_domain_per_old = (
            _in_unit_cube(probes_per_old.reshape(-1, D)).reshape(n_old, M).astype(np.float64)
        )
        # rho_other_old_at_old_probes[j, m] = sum_{k!=j} w_k * exp(-||probes[j,m] - old_k||² / 2σ²)
        diff_old = probes_per_old[:, :, None, :] - old_centers[None, None, :, :]  # (n_old, M, n_old, D)
        d2_old = np.sum(diff_old * diff_old, axis=-1)
        kernels_old = np.exp(-d2_old * inv_2sig2)
        if do_truncate:
            kernels_old = np.where(d2_old <= cutoff_d2, kernels_old, 0.0)
        weighted_old = kernels_old * old_weights[None, None, :]
        eye_jk = np.eye(n_old, dtype=np.float64)
        keep_jk = (1.0 - eye_jk)
        rho_other_per_old = (weighted_old * keep_jk[:, None, :]).sum(axis=-1)  # (n_old, M)
        self_contribution_per_old = self_density[None, :] * old_weights[:, None]  # (n_old, M)

        # ---- Old kernels' self-integrals when L new kernels added ----
        # delta_density[j, m, s, l] = density at probes_per_old[j, m] from new kernel (s, l)
        # diff: (n_old, M, S, L, D)
        diff_new_to_old = probes_per_old[:, :, None, None, :] - new_centers_SL[None, None, :, :, :]
        d2_new_to_old = np.sum(diff_new_to_old * diff_new_to_old, axis=-1)  # (n_old, M, S, L)
        kernels_new_to_old = np.exp(-d2_new_to_old * inv_2sig2)
        if do_truncate:
            kernels_new_to_old = np.where(d2_new_to_old <= cutoff_d2, kernels_new_to_old, 0.0)
        delta_density = (
            kernels_new_to_old * new_weights_SL[None, None, :, :]
        ).sum(axis=-1)  # (n_old, M, S) — sum over L (joint addition)

        total_density_old = (
            (self_contribution_per_old + rho_other_per_old)[:, :, None] + delta_density
        )  # (n_old, M, S)
        integrand_old = 1.0 / (1.0 + total_density_old)
        integral_old_per_s = (
            integrand_old * (quad_weights[None, :, None] * in_domain_per_old[:, :, None])
        ).sum(axis=1)  # (n_old, S)

        # ---- New kernels' own self-integrals (each (s, l)) ----
        # density at probes_new_SL[s, l, m] from old kernels: (S, L, M, n_old) → sum n_old → (S, L, M)
        diff_old_to_new = probes_new_SL[:, :, :, None, :] - old_centers[None, None, None, :, :]
        d2_old_to_new = np.sum(diff_old_to_new * diff_old_to_new, axis=-1)  # (S, L, M, n_old)
        kernels_old_to_new = np.exp(-d2_old_to_new * inv_2sig2)
        if do_truncate:
            kernels_old_to_new = np.where(d2_old_to_new <= cutoff_d2, kernels_old_to_new, 0.0)
        rho_old_at_new_probes = (
            kernels_old_to_new * old_weights[None, None, None, :]
        ).sum(axis=-1)  # (S, L, M)

        # Self-contribution for kernel (s, l) at its own probes
        self_contribution_new = self_density[None, None, :] * new_weights_SL[:, :, None]  # (S, L, M)

        total_density_new = (
            self_contribution_new + rho_old_at_new_probes + rho_other_self_at_probes_SL
        )  # (S, L, M)
        integrand_new = 1.0 / (1.0 + total_density_new)
        integral_new_SL = (
            quad_weights[None, None, :] * integrand_new * in_domain_new_SL
        ).sum(axis=-1)  # (S, L)

        # ---- Combine ----
        e_new = (
            (old_weights[:, None] * integral_old_per_s).sum(axis=0)
            + (new_weights_SL * integral_new_SL).sum(axis=-1)
        )
        return e_new

    def integrated_evidence_perturbed_batched_joint_torch(
        self,
        index_old: KernelIndex,
        new_centers_SL: torch.Tensor,
        new_weights_SL: torch.Tensor,
    ) -> torch.Tensor:
        """Tensor-typed mirror of ``integrated_evidence_perturbed_batched_joint``.

        Returns ``(S,)`` torch tensor with autograd flowing from each
        candidate's ``E_new[s]`` back through ``new_centers_SL`` (and
        ``new_weights_SL`` if ``requires_grad=True``). Used by 's
        gradient-based acquisition optimiser (commit 5).

        ``index_old.centers`` / ``.weights`` arrive as numpy from the (still
        numpy-typed) ``KernelIndex``; converted to torch at call entry. Old
        kernels don't typically need gradients (they're frozen training-data
        encodings). New candidates may need grad — caller's choice.

        **Regime dispatch** (σ/D-aware): ``_choose_kde_regime(n_kernels, σ, D)``
        selects between dense / knn / cluster algorithms. Commit 4 ships the
        dense regime only; knn / cluster raise ``NotImplementedError`` with
        a pointer to the follow-up commits. Logged at INFO when a non-dense
        regime would be selected — gives empirical signal on when to ship
        commit 4b.

        Math is identical to the numpy variant
        ``integrated_evidence_perturbed_batched_joint``; the difference is
        op set (``torch.exp`` / ``torch.where`` / etc.) and the autograd
        graph it produces.
        """
        S = int(new_centers_SL.shape[0])
        if S == 0:
            return torch.zeros(0, dtype=new_centers_SL.dtype)
        L = int(new_centers_SL.shape[1])
        D = int(new_centers_SL.shape[2])
        sigma = index_old.sigma
        n_old = len(index_old.centers) if not index_old.is_empty else 0

        regime = _choose_kde_regime(n_old, sigma, D)
        if regime == "knn":
            _logger.info(
                "KDE regime 'knn' selected (n_kernels=%d, σ=%.4f, D=%d) — not yet "
                "implemented; ship commit 4b. Falling back to dense.",
                n_old, sigma, D,
            )
            regime = "dense"
        elif regime == "cluster":
            _logger.info(
                "KDE regime 'cluster' selected (n_kernels=%d, σ=%.4f, D=%d) — not yet "
                "implemented; ship commit 4c. Falling back to dense.",
                n_old, sigma, D,
            )
            regime = "dense"

        return self._integrated_evidence_joint_dense_torch(
            index_old, new_centers_SL, new_weights_SL,
        )

    def _integrated_evidence_joint_dense_torch(
        self,
        index_old: KernelIndex,
        new_centers_SL: torch.Tensor,
        new_weights_SL: torch.Tensor,
    ) -> torch.Tensor:
        """Dense-regime implementation: sum over all old kernels per probe.

        Mirrors the numpy ``integrated_evidence_perturbed_batched_joint``
        body in torch ops. All tensor moves target ``new_centers_SL.device``
        — when the caller has placed inputs on GPU, the entire dense KDE
        compute runs there end-to-end.
        """
        S = int(new_centers_SL.shape[0])
        L = int(new_centers_SL.shape[1])
        D = int(new_centers_SL.shape[2])
        dtype = new_centers_SL.dtype
        device = new_centers_SL.device
        sigma = index_old.sigma
        inv_2sig2 = 1.0 / (2.0 * sigma ** 2)

        # Probes / quad weights / per-probe self-density — convert to caller device.
        offsets_np, quad_weights_np, self_density_np = self._probes_weights_self(D, sigma)
        offsets = torch.from_numpy(offsets_np).to(device=device, dtype=dtype)
        quad_weights = torch.from_numpy(quad_weights_np).to(device=device, dtype=dtype)
        self_density = torch.from_numpy(self_density_np).to(device=device, dtype=dtype)
        M = offsets.shape[0]

        n_old = len(index_old.centers) if not index_old.is_empty else 0

        # Probes around each new candidate's L kernels: (S, L, M, D).
        probes_new_SL = offsets[None, None, :, :] + new_centers_SL[:, :, None, :]
        in_domain_new_SL = _in_unit_cube_torch(probes_new_SL.reshape(-1, D)).reshape(
            S, L, M
        ).to(dtype=dtype)

        # Cross-influence: density at probes_s_l from OTHER (L-1) kernels of same candidate.
        diff_self = probes_new_SL[:, :, :, None, :] - new_centers_SL[:, None, None, :, :]
        d2_self = (diff_self * diff_self).sum(dim=-1)  # (S, L, M, L)
        kernels_self = torch.exp(-d2_self * inv_2sig2)
        eye_LL = torch.eye(L, dtype=dtype)
        keep_LL = 1.0 - eye_LL
        weighted_self = kernels_self * new_weights_SL[:, None, None, :]
        rho_other_self_SL = (weighted_self * keep_LL[None, :, None, :]).sum(dim=-1)  # (S, L, M)

        # === Branch 1: no old kernels ===
        if n_old == 0:
            self_contribution_new = self_density[None, None, :] * new_weights_SL[:, :, None]
            total_density_new = self_contribution_new + rho_other_self_SL
            integrand_new = 1.0 / (1.0 + total_density_new)
            integral_new_SL = (
                quad_weights[None, None, :] * integrand_new * in_domain_new_SL
            ).sum(dim=-1)
            return (new_weights_SL * integral_new_SL).sum(dim=-1)

        # === Branch 2: old kernels exist ===
        old_centers = torch.from_numpy(index_old.centers).to(device=device, dtype=dtype)
        old_weights = torch.from_numpy(index_old.weights).to(device=device, dtype=dtype)

        # Per-old precompute (independent of S — could be cached across acquisition
        # generations in a future optimisation; for now recompute per call).
        probes_per_old = offsets[None, :, :] + old_centers[:, None, :]  # (n_old, M, D)
        in_domain_per_old = _in_unit_cube_torch(probes_per_old.reshape(-1, D)).reshape(
            n_old, M
        ).to(dtype=dtype)

        diff_old = probes_per_old[:, :, None, :] - old_centers[None, None, :, :]
        d2_old = (diff_old * diff_old).sum(dim=-1)  # (n_old, M, n_old)
        kernels_old = torch.exp(-d2_old * inv_2sig2)
        weighted_old = kernels_old * old_weights[None, None, :]
        eye_jk = torch.eye(n_old, dtype=dtype)
        keep_jk = 1.0 - eye_jk
        rho_other_per_old = (weighted_old * keep_jk[:, None, :]).sum(dim=-1)  # (n_old, M)
        self_contribution_per_old = self_density[None, :] * old_weights[:, None]

        # Old kernels' self-integrals when L new kernels added.
        diff_new_to_old = (
            probes_per_old[:, :, None, None, :] - new_centers_SL[None, None, :, :, :]
        )
        d2_new_to_old = (diff_new_to_old * diff_new_to_old).sum(dim=-1)  # (n_old, M, S, L)
        kernels_new_to_old = torch.exp(-d2_new_to_old * inv_2sig2)
        delta_density = (
            kernels_new_to_old * new_weights_SL[None, None, :, :]
        ).sum(dim=-1)  # (n_old, M, S)

        total_density_old = (
            (self_contribution_per_old + rho_other_per_old)[:, :, None] + delta_density
        )
        integrand_old = 1.0 / (1.0 + total_density_old)
        integral_old_per_s = (
            integrand_old * (quad_weights[None, :, None] * in_domain_per_old[:, :, None])
        ).sum(dim=1)  # (n_old, S)

        # New kernels' own self-integrals.
        diff_old_to_new = (
            probes_new_SL[:, :, :, None, :] - old_centers[None, None, None, :, :]
        )
        d2_old_to_new = (diff_old_to_new * diff_old_to_new).sum(dim=-1)  # (S, L, M, n_old)
        kernels_old_to_new = torch.exp(-d2_old_to_new * inv_2sig2)
        rho_old_at_new_probes = (
            kernels_old_to_new * old_weights[None, None, None, :]
        ).sum(dim=-1)  # (S, L, M)

        self_contribution_new = self_density[None, None, :] * new_weights_SL[:, :, None]
        total_density_new = (
            self_contribution_new + rho_old_at_new_probes + rho_other_self_SL
        )
        integrand_new = 1.0 / (1.0 + total_density_new)
        integral_new_SL = (
            quad_weights[None, None, :] * integrand_new * in_domain_new_SL
        ).sum(dim=-1)  # (S, L)

        e_new = (
            (old_weights[:, None] * integral_old_per_s).sum(dim=0)
            + (new_weights_SL * integral_new_SL).sum(dim=-1)
        )
        return e_new


# ---------------------------------------------------------------------------
# Sobol-local — QMC cube around each kernel
# ---------------------------------------------------------------------------

@dataclass
class SobolLocalEstimator(EvidenceEstimator):
    """Volume-weighted QMC in a [center ± box·σ]^D cube.

    `n_samples=None` (default) ties the Sobol probe count to the KernelField
    probe count at the same `D` — gives matched compute and matched extent
    (with `box=2.0`) for fair comparison. Set an explicit integer to override.
    """

    box: float = 2.0
    n_samples: int | None = None
    seed: int = 0

    def _resolve_n_samples(self, D: int) -> int:
        if self.n_samples is not None:
            return int(self.n_samples)
        return kernel_field_probe_count(D)

    def self_integral(
        self, center: np.ndarray, index: KernelIndex, kernel_idx: int | None = None,
    ) -> float:
        del kernel_idx  # Sobol regenerates per call; self-density caching not applicable.
        D = int(center.shape[-1])
        sigma = index.sigma
        box_side = 2.0 * self.box * sigma
        volume = box_side ** D
        n = self._resolve_n_samples(D)

        # Torch-native QMC. draw() returns (n, D)
        # in the unit cube; SobolEngine wants log2(n) for power-of-2 lengths
        # but accepts arbitrary n via .draw(n). Scrambled for sample diversity.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            engine = torch.quasirandom.SobolEngine(dimension=D, scramble=True, seed=self.seed)
            unit = engine.draw(n).numpy().astype(np.float64)
        samples = center + box_side * (unit - 0.5)

        # ρⱼ(z; center) — peak-1 Gaussian, no weight (caller scales by w_j)
        d2 = np.sum((samples - center) ** 2, axis=-1)
        rho_j = np.exp(-d2 / (2.0 * sigma * sigma))

        D_vals = index.density_at(samples)
        in_domain = _in_unit_cube(samples).astype(float)

        # ∫ρⱼ/(1+D) dz ≈ volume · mean[ρⱼ/(1+D) · 1_cube]
        integrand = rho_j / (1.0 + D_vals) * in_domain
        return volume * float(integrand.mean())


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
