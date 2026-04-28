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

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import gamma, pi
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[import-untyped]
from scipy.stats import qmc


# Module-level KernelField defaults — change here, propagates everywhere.
DEFAULT_RADII: tuple[float, ...] = (0.5, 1.0, 2.0)
DEFAULT_ANGULAR_GAP_DEG: float = 45.0


# ---------------------------------------------------------------------------
# Kernel index — O(M · log K) density evaluation via neighbour search
# ---------------------------------------------------------------------------

class KernelIndex:
    """Spatial index over Gaussian kernel centres for fast D(z) evaluation."""

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
        # cKDTree only when we have enough kernels to justify truncation —
        # below threshold we sum all kernels directly so density is never zero.
        self._tree = cKDTree(self.centers) if self._n >= self.truncation_threshold else None

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

        if self._tree is None:
            # Direct sum — small N, full Gaussian tail kept.
            diff = points[:, None, :] - self.centers[None, :, :]
            d2 = np.sum(diff * diff, axis=-1)
            exp_term = np.exp(-d2 * inv_2sig2)
            if exclude_idx is not None:
                mask = np.arange(self._n) != exclude_idx
                return (exp_term[:, mask] * self.weights[mask]).sum(axis=-1)
            return (exp_term * self.weights).sum(axis=-1)

        neighbor_lists = self._tree.query_ball_point(points, r=self.cutoff)
        out = np.zeros(M)
        for i, idxs in enumerate(neighbor_lists):
            if not idxs:
                continue
            idxs_arr = np.asarray(idxs, dtype=int)
            if exclude_idx is not None:
                idxs_arr = idxs_arr[idxs_arr != exclude_idx]
                if len(idxs_arr) == 0:
                    continue
            d2 = np.sum((points[i] - self.centers[idxs_arr]) ** 2, axis=-1)
            out[i] = np.sum(self.weights[idxs_arr] * np.exp(-d2 * inv_2sig2))
        return out


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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            unit = qmc.Sobol(d=D, scramble=True, rng=self.seed).random(n=n)
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
