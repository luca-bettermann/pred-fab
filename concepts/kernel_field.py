"""KernelField — a deterministic field of probes around each datapoint.

Each datapoint z_j carries a radial structure of evaluation points — one at
the center, the rest on concentric shells at fixed σ-multiples. We call the
set a *kernel field* (analogy to a gravitational field around a point mass)
and the individual points *probes*. Weights come from treating the shells as
a radial quadrature of the Gaussian density.

The field is computed once per (D, σ) and reused for every kernel — no RNG,
no variance, no σ dependence beyond the σ·radius scaling.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


# ---------- Angular lattices ----------

def unit_sphere_points(D: int, n: int, seed: int = 0) -> np.ndarray:
    """n points on the unit (D−1)-sphere in ℝ^D, quasi-uniformly spaced.

    D=1 → ±1 (n ignored).
    D=2 → n evenly-spaced directions.
    D=3 → Fibonacci sphere.
    D≥4 → deterministic random unit vectors (fixed seed).
    """
    if D == 1:
        return np.array([[-1.0], [1.0]])
    if D == 2:
        phi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.stack([np.cos(phi), np.sin(phi)], axis=-1)
    if D == 3:
        indices = np.arange(n) + 0.5
        phi = np.arccos(1.0 - 2.0 * indices / n)
        theta = np.pi * (1.0 + 5.0 ** 0.5) * indices
        return np.stack([
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ], axis=-1)
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, D))
    return g / np.linalg.norm(g, axis=-1, keepdims=True)


# ---------- Radial weights (Gaussian only) ----------

def _surface_area_unit_sphere(D: int) -> float:
    from math import gamma, pi
    return 2.0 * pi ** (D / 2.0) / gamma(D / 2.0)


def _gaussian_radial_pdf(r: float, sigma: float, D: int) -> float:
    """Radial marginal of isotropic Gaussian of total mass 1."""
    from math import pi, exp
    return (2.0 * pi * sigma * sigma) ** (-D / 2.0) * exp(-r * r / (2.0 * sigma * sigma))


def _gamma_half(D: int) -> float:
    from math import gamma
    return gamma(D / 2.0 + 1.0)


def radial_shell_weights(radii: np.ndarray, sigma: float, D: int) -> tuple[np.ndarray, float]:
    """Shell weights + center weight for the Gaussian density.

    radii: (K,) absolute shell radii (already σ-multiplied).
    Returns (weights_K, center_weight).
    """
    K = len(radii)
    weights = np.zeros(K)
    S = _surface_area_unit_sphere(D)
    r_ext = np.concatenate([[0.0], radii, [2.0 * radii[-1] - radii[-2]]])
    for k in range(K):
        r_k = radii[k]
        dr_k = 0.5 * (r_ext[k + 2] - r_ext[k])
        weights[k] = S * (r_k ** (D - 1)) * dr_k * _gaussian_radial_pdf(r_k, sigma, D)

    r0_half = 0.5 * r_ext[1]
    vol_ball = (np.pi ** (D / 2.0) / _gamma_half(D)) * (r0_half ** D)
    center_weight = vol_ball * _gaussian_radial_pdf(0.0, sigma, D)
    return weights, center_weight


# ---------- The class ----------

@dataclass(frozen=True)
class KernelField:
    """A deterministic field of probes around each kernel center.

    Two modes for shell placement:

        radii_mode = "fixed"           — shells at σ · radius_multipliers
        radii_mode = "chi2_quantile"   — shells at σ · √χ²_D.ppf(q) for q in
                                         radii_quantiles, so each shell sits at
                                         a fixed *mass percentile* of the
                                         radial Gaussian distribution. Outer
                                         radius scales with D automatically.

    Integration weights come from a radial-shell quadrature of the Gaussian
    density — no Monte Carlo, no sampling variance.
    """

    D: int
    sigma: float
    radius_multipliers: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0)
    n_directions: int = 16
    radii_mode: str = "fixed"
    radii_quantiles: tuple[float, ...] = (0.02, 0.2, 0.5, 0.8, 0.98)

    # ----- Derived geometry -----

    @cached_property
    def radii(self) -> np.ndarray:
        if self.radii_mode == "chi2_quantile":
            from scipy.stats import chi2
            qs = np.asarray(self.radii_quantiles, dtype=float)
            return np.sqrt(chi2.ppf(qs, df=self.D)) * self.sigma
        if self.radii_mode != "fixed":
            raise ValueError(f"unknown radii_mode: {self.radii_mode!r}")
        return np.asarray(self.radius_multipliers, dtype=float) * self.sigma

    @cached_property
    def directions(self) -> np.ndarray:
        return unit_sphere_points(self.D, self.n_directions)

    @cached_property
    def _built(self) -> tuple[np.ndarray, np.ndarray]:
        shell_w, center_w = radial_shell_weights(self.radii, self.sigma, self.D)
        offsets = [np.zeros(self.D)]
        weights = [center_w]
        n_dir = self.directions.shape[0]
        for k, r in enumerate(self.radii):
            w_per_point = shell_w[k] / n_dir
            for d in self.directions:
                offsets.append(r * d)
                weights.append(w_per_point)
        return np.stack(offsets), np.asarray(weights)

    @property
    def offsets(self) -> np.ndarray:
        """(n_probes, D) — probe offsets from any kernel center."""
        return self._built[0]

    @property
    def weights(self) -> np.ndarray:
        """(n_probes,) — quadrature weights; ∑ weights ≈ 1 by construction."""
        return self._built[1]

    @property
    def n_probes(self) -> int:
        return self.offsets.shape[0]

    # ----- Public API -----

    def probes_at(self, center: np.ndarray) -> np.ndarray:
        """(n_probes, D) — probes placed at a given kernel center."""
        return self.offsets + center

    def probes_for_batch(self, centers: np.ndarray) -> np.ndarray:
        """(N, n_probes, D) — probes per center, batch-friendly."""
        return centers[:, None, :] + self.offsets[None, :, :]

    def integrate(
        self,
        f_at_probes: np.ndarray,
        in_domain_mask: np.ndarray | None = None,
    ) -> float:
        """∫ f · ρ dz ≈ Σ_i w_i · f(probe_i) · [in_domain_i].

        f_at_probes:      (n_probes,) integrand values at each probe
        in_domain_mask:   (n_probes,) bool, or None to disable leakage masking
        """
        if in_domain_mask is None:
            return float(np.sum(self.weights * f_at_probes))
        return float(np.sum(self.weights * f_at_probes * in_domain_mask))

    def check_mass(self) -> float:
        """Quadrature estimate of ∫ρ dz (should be 1 analytically)."""
        return float(self.weights.sum())
