"""compare_stroud5_sobol_local.py — Stroud-5 cubature vs Sobol-local.

Stroud's G_n^5 cubature rule: `2D² + 1` points, exact for polynomials of
total degree ≤ 5 against a standard Gaussian measure. A near-minimal
alternative to the angular-gap KernelField shell design, specifically
designed for Gaussian integration.

Points (standard-Gaussian coords, scaled by σ around the kernel centre):

    1        · origin                         · w = 2 / (D+2)
    2·D      · ±√(D+2) · eᵢ                    · w = (4−D) / (2(D+2)²)
    2·D(D−1) · √((D+2)/2) · (±eᵢ ± eⱼ)          · w = 1 / (D+2)²

Estimator for an isolated kernel at `centre`:
    ∫ ρ/(1+ρ) dz ≈ Σᵢ wᵢ · 1/(1 + ρ(centre + σ·xᵢ))

Note: for D ≥ 5 the axial weights turn negative. Mathematically fine
(the rule stays polynomial-exact), but for strongly-non-polynomial
integrands it can introduce extra bias in high D.
"""
from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import qmc

from _style import (
    apply_style, clean_spines, clean_3d_panes,
    evidence_cmap,
    ZINC_200, ZINC_300, ZINC_500, ZINC_600, ZINC_700,
    STEEL_500, EMERALD_500, RED,
)
from compare_kernelfield_sobol_local import (
    gaussian_density_iso,
    truth_single_kernel,
    estimate_sobol_local,
    _draw_cube_wireframe,
    _color_by_distance,
    SOBOL_HALF_EXTENT,
)


PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ---------- Stroud-5 construction ----------

def stroud5_points_and_weights(D: int) -> tuple[np.ndarray, np.ndarray]:
    """Stroud's G_n^5 cubature for the standard Gaussian measure on ℝ^D.

    Returns (points [N, D], weights [N]) where N = 2D² + 1 and
    ∫ f(x) · (2π)^(−D/2) · exp(−|x|²/2) dx  ≈  Σᵢ wᵢ · f(xᵢ)
    is exact for every polynomial of total degree ≤ 5.
    """
    N = 2 * D * D + 1
    points = np.zeros((N, D))
    weights = np.zeros(N)

    weights[0] = 2.0 / (D + 2)

    r1 = float(np.sqrt(D + 2))
    w1 = (4.0 - D) / (2.0 * (D + 2) ** 2)
    idx = 1
    for i in range(D):
        for s in (+1, -1):
            points[idx, i] = s * r1
            weights[idx] = w1
            idx += 1

    r2 = float(np.sqrt((D + 2) / 2.0))
    w2 = 1.0 / (D + 2) ** 2
    for i in range(D):
        for j in range(i + 1, D):
            for s1 in (+1, -1):
                for s2 in (+1, -1):
                    points[idx, i] = s1 * r2
                    points[idx, j] = s2 * r2
                    weights[idx] = w2
                    idx += 1

    return points, weights


def estimate_stroud5(sigma: float, D: int) -> tuple[float, int]:
    """Stroud-5 estimate of ∫ρ/(1+ρ) dz for an isolated Gaussian at centre."""
    points_std, weights = stroud5_points_and_weights(D)
    center = np.full(D, 0.5)
    probes = center + sigma * points_std
    rho = gaussian_density_iso(probes, center, sigma)
    u = 1.0 / (1.0 + rho)
    est = float(np.sum(weights * u))
    return est, probes.shape[0]


# ---------- Figure 1: probe placement (3-D) ----------

def figure_probe_placement(sigma: float = 0.10, seed: int = 0) -> Path:
    apply_style()
    D = 3
    center = np.full(D, 0.5)

    points_std, _ = stroud5_points_and_weights(D)
    stroud_probes = center + sigma * points_std
    stroud_budget = stroud_probes.shape[0]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        sobol = qmc.Sobol(d=D, scramble=True, rng=seed).random(n=stroud_budget)
    box_side = 2.0 * SOBOL_HALF_EXTENT * sigma
    sobol_probes = center + box_side * (sobol - 0.5)

    stroud_cval = _color_by_distance(stroud_probes, center, sigma)
    sobol_cval = _color_by_distance(sobol_probes, center, sigma)

    cmap = evidence_cmap()
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(12.5, 5.8))
    fig.subplots_adjust(left=0.01, right=0.88, top=0.98, bottom=0.02, wspace=0.02)

    panels = [
        (121, stroud_probes, stroud_cval, f"Stroud-5  ·  {stroud_budget} probes"),
        (122, sobol_probes, sobol_cval, f"Sobol (local cube)  ·  {stroud_budget} probes"),
    ]

    lo_inner = center - SOBOL_HALF_EXTENT * sigma
    hi_inner = center + SOBOL_HALF_EXTENT * sigma

    for spec, probes, cvals, label in panels:
        ax = fig.add_subplot(spec, projection="3d")
        _draw_cube_wireframe(ax, np.zeros(3), np.ones(3),
                             color=ZINC_300, lw=0.4, alpha=0.32)
        _draw_cube_wireframe(ax, lo_inner, hi_inner,
                             color=ZINC_500, lw=0.9, alpha=0.55)

        ax.scatter(probes[:, 0], probes[:, 1], probes[:, 2],
                   c=cvals, cmap=cmap, norm=norm,
                   s=10, alpha=0.95, edgecolors="none",
                   depthshade=False, zorder=5)
        ax.scatter([center[0]], [center[1]], [center[2]],
                   marker="o", c=RED, s=34, edgecolors="none",
                   depthshade=False, zorder=10)

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

    path = PLOTS_DIR / "probe_placement_stroud5.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------- Figure 2: |err| vs D at matched Stroud-5 budget ----------

def figure_error_vs_D(
    sigmas: tuple[float, ...] = (0.05, 0.10, 0.15),
    Ds: tuple[int, ...] = (2, 3, 4, 5, 6),
) -> Path:
    apply_style()

    budgets = [2 * D * D + 1 for D in Ds]

    fig, axes = plt.subplots(1, len(sigmas), figsize=(13.0, 4.2),
                             constrained_layout=True, sharey=True)

    for ax, sigma in zip(axes, sigmas):
        stroud_errs: list[float] = []
        sb_errs: list[float] = []
        for D, budget in zip(Ds, budgets):
            t = truth_single_kernel(sigma, D)
            st_val, _ = estimate_stroud5(sigma, D)
            sb_val = estimate_sobol_local(sigma, D, budget)
            denom = abs(t) + 1e-12
            stroud_errs.append(max(100.0 * abs(st_val - t) / denom, 1e-3))
            sb_errs.append(max(100.0 * abs(sb_val - t) / denom, 1e-3))

        x = list(Ds)
        ax.fill_between(x, stroud_errs, 1e-3, color=STEEL_500, alpha=0.08, zorder=1)
        ax.fill_between(x, sb_errs, 1e-3, color=EMERALD_500, alpha=0.08, zorder=1)
        ax.plot(x, stroud_errs, color=STEEL_500, lw=2.2, marker="o", ms=6,
                mec="none", label="Stroud-5", zorder=3)
        ax.plot(x, sb_errs, color=EMERALD_500, lw=2.2, marker="o", ms=6,
                mec="none", label="Sobol (local)", zorder=3)

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xlabel("D  (dimensions)")
        if sigma == sigmas[0]:
            ax.set_ylabel("|relative error|  [%]")
            ax.legend(loc="best", fontsize=8.5, frameon=False)

        ax.text(0.03, 0.96, f"σ = {sigma}",
                transform=ax.transAxes, fontsize=9.5, color=ZINC_600)

        ax.grid(True, which="major", color=ZINC_200, alpha=0.5, lw=0.5)
        ax.grid(True, which="minor", color=ZINC_200, alpha=0.25, lw=0.4)
        clean_spines(ax)

    for ax in axes:
        top = ax.secondary_xaxis("top")
        top.set_xticks(list(Ds))
        top.set_xticklabels([str(b) for b in budgets], fontsize=7, color=ZINC_500)
        top.tick_params(colors=ZINC_500, labelsize=7, length=0)
        top.set_xlabel("probes  (Stroud-5 = Sobol, matched)",
                       fontsize=8, color=ZINC_500, labelpad=4)

    path = PLOTS_DIR / "error_vs_D_stroud5.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("1/2  probe placement (Stroud-5 vs Sobol-local, D=3, σ=0.10) ...")
    p1 = figure_probe_placement()
    print(f"      saved: {p1}")
    print("2/2  error vs D across σ (Stroud-5 at 2D²+1 probes) ...")
    p2 = figure_error_vs_D()
    print(f"      saved: {p2}")
