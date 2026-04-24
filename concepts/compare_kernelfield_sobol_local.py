"""compare_kernelfield_sobol_local.py — evidence integration restricted to
the kernel neighbourhood.

Both estimators target  ∫E(z) dz = ∫ ρ/(1+ρ) dz  for a single isolated
Gaussian kernel at (0.5, ..., 0.5):

    KernelField    Gaussian radial-shell quadrature. Shells at
                   [0.5, 1, 2, 3] · σ plus the centre. Per kernel:
                   `4·n_dirs + 1` probes.

    Sobol (local)  QMC inside a cube `[0.5 − 3σ, 0.5 + 3σ]^D` around
                   the datapoint. Estimate: `(6σ)^D · mean_i E(zᵢ)`.
                   `Outside the local cube, E ≈ 0` (kernel tail), so
                   the cube integral approximates the full ∫E.

Truth: importance sampling from ρ with n = 200k. Bias-free for an
isolated kernel because ∫ρ/(1+ρ) dz = 𝔼_{z~ρ}[1/(1+ρ(z))].

Two figures:
    probe_placement_local.png  3-D, σ = 0.10, nested cubes — unit cube
                                (outer) and sampling cube (inner). Probes
                                coloured by distance: centre → 2σ.
    error_vs_D_local.png       |relative error| vs D for σ ∈ {0.05, 0.10, 0.15}
                                at matched probe budget (n_dirs = 16).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import qmc

from kernel_field import KernelField
from _style import (
    apply_style, clean_spines, clean_3d_panes,
    evidence_cmap,
    ZINC_200, ZINC_300, ZINC_500, ZINC_600, ZINC_700,
    STEEL_500, EMERALD_500, RED,
)


KF_MULTIPLIERS: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
SOBOL_HALF_EXTENT: float = 3.0
TRUTH_N: int = 200_000

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ---------- Single-kernel density helpers ----------

def gaussian_density_iso(z: np.ndarray, center: np.ndarray, sigma: float) -> np.ndarray:
    """ρ(z; center, σ²I) — product-Gaussian pdf, ∫ρ dz = 1 in ℝ^D."""
    D = z.shape[-1]
    delta = z - center
    r2 = np.sum(delta ** 2, axis=-1)
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi)) ** D
    return norm * np.exp(-r2 / (2.0 * sigma ** 2))


# ---------- Truth + estimators ----------

def truth_single_kernel(sigma: float, D: int, n: int = TRUTH_N, seed: int = 42) -> float:
    """Bias-free ∫ρ/(1+ρ) dz via IS on ρ for a kernel at the cube centre."""
    rng = np.random.default_rng(seed)
    center = np.full(D, 0.5)
    eps = rng.standard_normal((n, D)) * sigma
    samples = center + eps
    rho = gaussian_density_iso(samples, center, sigma)
    return float((1.0 / (1.0 + rho)).mean())


def estimate_kf(sigma: float, D: int, n_dirs: int) -> tuple[float, int]:
    field = KernelField(
        D=D, sigma=sigma,
        radius_multipliers=KF_MULTIPLIERS,
        n_directions=n_dirs,
    )
    center = np.full(D, 0.5)
    probes = field.probes_at(center)
    rho = gaussian_density_iso(probes, center, sigma)
    u = 1.0 / (1.0 + rho)
    est = float(np.sum(field.weights * u))
    return est, probes.shape[0]


def estimate_sobol_local(
    sigma: float, D: int, budget: int, seed: int = 0,
) -> tuple[float, int]:
    """Cube QMC on `[0.5 − 3σ, 0.5 + 3σ]^D`."""
    n_exp = max(int(round(np.log2(max(budget, 2)))), 1)
    n = 2 ** n_exp
    sobol = qmc.Sobol(d=D, scramble=True, rng=seed).random_base2(m=n_exp)
    box_side = 2.0 * SOBOL_HALF_EXTENT * sigma
    center = np.full(D, 0.5)
    samples = center + box_side * (sobol - 0.5)
    rho = gaussian_density_iso(samples, center, sigma)
    E = rho / (1.0 + rho)
    volume = box_side ** D
    est = volume * float(E.mean())
    return est, n


# ---------- Cube wireframe ----------

def _draw_cube_wireframe(ax, lo: np.ndarray, hi: np.ndarray,
                         color: str, lw: float, alpha: float) -> None:
    corners = [
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for (i, j) in edges:
        p1, p2 = corners[i], corners[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=color, lw=lw, alpha=alpha, zorder=-1)


# ---------- Colour map: distance from centre, capped at 2σ ----------

def _color_by_distance(probes: np.ndarray, center: np.ndarray, sigma: float) -> np.ndarray:
    """Map each probe to a value in [0, 1]: 1 at centre, 0 at or beyond 2σ."""
    r_sigma = np.linalg.norm(probes - center, axis=-1) / sigma
    return np.clip(1.0 - r_sigma / 2.0, 0.0, 1.0)


# ---------- Figure 1: probe placement in 3-D ----------

def figure_probe_placement(sigma: float = 0.10, n_dirs: int = 12, seed: int = 0) -> Path:
    apply_style()
    D = 3
    center = np.full(D, 0.5)

    field = KernelField(
        D=D, sigma=sigma,
        radius_multipliers=KF_MULTIPLIERS, n_directions=n_dirs,
    )
    kf_probes = field.probes_at(center)
    kf_budget = kf_probes.shape[0]

    n_exp = max(int(round(np.log2(kf_budget))), 1)
    n_sobol = 2 ** n_exp
    sobol = qmc.Sobol(d=D, scramble=True, rng=seed).random_base2(m=n_exp)
    box_side = 2.0 * SOBOL_HALF_EXTENT * sigma
    sobol_probes = center + box_side * (sobol - 0.5)

    kf_cval = _color_by_distance(kf_probes, center, sigma)
    sobol_cval = _color_by_distance(sobol_probes, center, sigma)

    cmap = evidence_cmap()
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(12.5, 5.8))
    fig.subplots_adjust(left=0.01, right=0.88, top=0.98, bottom=0.02, wspace=0.02)

    panels = [
        (121, kf_probes, kf_cval, f"KernelField  ·  {kf_budget} probes"),
        (122, sobol_probes, sobol_cval, f"Sobol (local cube)  ·  {n_sobol} probes"),
    ]

    lo_inner = center - SOBOL_HALF_EXTENT * sigma
    hi_inner = center + SOBOL_HALF_EXTENT * sigma

    for spec, probes, cvals, label in panels:
        ax = fig.add_subplot(spec, projection="3d")
        _draw_cube_wireframe(ax, np.zeros(3), np.ones(3),
                             color=ZINC_300, lw=0.4, alpha=0.32)
        _draw_cube_wireframe(ax, lo_inner, hi_inner,
                             color=ZINC_500, lw=0.9, alpha=0.55)

        ax.scatter(
            probes[:, 0], probes[:, 1], probes[:, 2],
            c=cvals, cmap=cmap, norm=norm,
            s=10, alpha=0.95, edgecolors="none",
            depthshade=False, zorder=5,
        )
        ax.scatter(
            [center[0]], [center[1]], [center[2]],
            marker="o", c=RED, s=34, edgecolors="none",
            depthshade=False, zorder=10,
        )

        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_zticks([0, 0.5, 1])
        ax.set_xlabel("z₁"); ax.set_ylabel("z₂"); ax.set_zlabel("z₃")
        ax.view_init(elev=20, azim=35)
        clean_3d_panes(ax)

        ax.text2D(0.02, 0.96, label, transform=ax.transAxes,
                  fontsize=9.5, color=ZINC_600)

    cax = fig.add_axes([0.905, 0.22, 0.016, 0.56])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(colors=ZINC_500, labelsize=8)
    cbar.set_label("distance from centre", color=ZINC_600, fontsize=9)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["≥ 2σ", "1σ", "centre"])
    cbar.outline.set_edgecolor(ZINC_300)
    cbar.outline.set_linewidth(0.6)

    path = PLOTS_DIR / "probe_placement_local.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------- Figure 2: |err| vs D, one panel per σ ----------

def figure_error_vs_D(
    sigmas: tuple[float, ...] = (0.05, 0.10, 0.15),
    Ds: tuple[int, ...] = (2, 3, 4, 5, 6),
    n_dirs: int = 16,
) -> Path:
    apply_style()

    fig, axes = plt.subplots(1, len(sigmas), figsize=(12.5, 3.8),
                             constrained_layout=True, sharey=True)

    for ax, sigma in zip(axes, sigmas):
        kf_errs: list[float] = []
        sb_errs: list[float] = []
        for D in Ds:
            t = truth_single_kernel(sigma, D)
            kf_val, kf_n = estimate_kf(sigma, D, n_dirs)
            sb_val, _ = estimate_sobol_local(sigma, D, kf_n)
            denom = abs(t) + 1e-12
            kf_errs.append(max(100.0 * abs(kf_val - t) / denom, 1e-3))
            sb_errs.append(max(100.0 * abs(sb_val - t) / denom, 1e-3))

        x = list(Ds)
        ax.fill_between(x, kf_errs, 1e-3, color=STEEL_500, alpha=0.08, zorder=1)
        ax.fill_between(x, sb_errs, 1e-3, color=EMERALD_500, alpha=0.08, zorder=1)
        ax.plot(x, kf_errs, color=STEEL_500, lw=2.2, marker="o", ms=6,
                mec="none", label="KernelField", zorder=3)
        ax.plot(x, sb_errs, color=EMERALD_500, lw=2.2, marker="o", ms=6,
                mec="none", label="Sobol (local)", zorder=3)

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xlabel("D  (dimensions)")
        if sigma == sigmas[0]:
            ax.set_ylabel("|relative error|  [%]")
            ax.legend(loc="best", fontsize=8.5, frameon=False)

        probe_count = 4 * n_dirs + 1
        ax.text(0.03, 0.96, f"σ = {sigma}  ·  ~{probe_count} probes",
                transform=ax.transAxes, fontsize=9, color=ZINC_600)

        ax.grid(True, which="major", color=ZINC_200, alpha=0.5, lw=0.5)
        ax.grid(True, which="minor", color=ZINC_200, alpha=0.25, lw=0.4)
        clean_spines(ax)

    path = PLOTS_DIR / "error_vs_D_local.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("1/2  probe placement (3-D, nested cubes, σ = 0.10) ...")
    p1 = figure_probe_placement()
    print(f"      saved: {p1}")
    print("2/2  error vs D across σ ...")
    p2 = figure_error_vs_D()
    print(f"      saved: {p2}")
