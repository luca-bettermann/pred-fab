"""Concept: kernel shape choice — Gaussian vs Cauchy.

The prediction model represents each evaluated datapoint by a probability
density centred at its location (∫ρ dz = 1 in ℝ^D). Two candidates were on
the table:

    Gaussian (product):  ρ(z|zⱼ) = ∏_d  (1/(σ√2π)) · exp(−(z_d − zⱼ,d)² / 2σ²)
    Cauchy   (product):  ρ(z|zⱼ) = ∏_d  (σ/π) / ((z_d − zⱼ,d)² + σ²)

Both normalise to 1. They differ in **peak height** (evidence concentration
at the datapoint) and **tail decay** (how far evidence reaches). Gaussian
decays `exp(−r²/2σ²)`; Cauchy decays `1/(r² + σ²)` — polynomial, heavy.

Two figures:
    kernel_shape_single.png   single kernel at origin
    kernel_shape_mixture.png  5 kernels spread across the cube, as D(z)

Same σ in both panels of each figure; grey fill for density, Steel/Emerald
ring on the half-height contour marks the effective support.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from _style import (
    apply_style, clean_spines,
    ZINC_100, ZINC_300, ZINC_500, ZINC_600, ZINC_700, ZINC_900,
    STEEL_500, EMERALD_500,
)


# ---------- Kernel densities ----------

def gaussian_density(z: np.ndarray, z_j: np.ndarray, sigma: float) -> np.ndarray:
    D = z.shape[-1]
    d2 = np.sum((z - z_j) ** 2, axis=-1)
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi)) ** D
    return norm * np.exp(-d2 / (2.0 * sigma ** 2))


def cauchy_density(z: np.ndarray, z_j: np.ndarray, sigma: float) -> np.ndarray:
    diff = z - z_j
    per_dim = (sigma / np.pi) / (diff ** 2 + sigma ** 2)
    return np.prod(per_dim, axis=-1)


PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def _density_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "density", ["white", ZINC_100, "#D4D4D8", ZINC_500, ZINC_900], N=256,
    )


# ---------- Figure 1: single kernel ----------

def figure_single_kernel(sigma: float = 0.10, lim: float = 0.5, res: int = 301) -> Path:
    apply_style()
    x = np.linspace(-lim, lim, res)
    X, Y = np.meshgrid(x, x)
    Z = np.stack([X, Y], axis=-1)
    z_j = np.zeros(2)

    G = gaussian_density(Z, z_j, sigma)
    C = cauchy_density(Z, z_j, sigma)
    vmax = max(G.max(), C.max())

    fwhm_g = sigma * np.sqrt(2.0 * np.log(2.0))
    fwhm_c = sigma
    theta = np.linspace(0, 2 * np.pi, 240)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), constrained_layout=True)
    cmap = _density_cmap()

    for ax, field, label, color, fwhm in zip(
        axes, [G, C], ["Gaussian", "Cauchy"], [STEEL_500, EMERALD_500], [fwhm_g, fwhm_c]
    ):
        ax.contourf(x, x, field, levels=24, cmap=cmap, vmin=0, vmax=vmax)
        ax.contour(x, x, field, levels=6, colors=[ZINC_300], linewidths=0.4, alpha=0.55)
        ax.plot(fwhm * np.cos(theta), fwhm * np.sin(theta),
                color=color, lw=1.6, alpha=0.9,
                label=f"half-height  r = {fwhm:.3f}")
        ax.scatter([0], [0], c=color, s=28, edgecolors="none", zorder=5)

        ax.set_title(
            f"{label}  ·  σ = {sigma}   peak ρ = {field.max():.1f}",
            pad=6, color=ZINC_700,
        )
        ax.set_xlabel("z₁ − zⱼ,1"); ax.set_ylabel("z₂ − zⱼ,2")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        clean_spines(ax)

    fig.suptitle(
        "Kernel shape — single datapoint, same σ, different tails",
        fontsize=12, color=ZINC_700,
    )
    path = PLOTS_DIR / "kernel_shape_single.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------- Figure 2: five-kernel mixture D(z) ----------

def figure_mixture(sigma: float = 0.10, res: int = 301) -> Path:
    apply_style()
    centers = np.array([
        [0.22, 0.28], [0.74, 0.22], [0.50, 0.52],
        [0.26, 0.78], [0.78, 0.74],
    ])
    u = np.linspace(0.0, 1.0, res)
    U, V = np.meshgrid(u, u)
    Z = np.stack([U, V], axis=-1)

    G = np.zeros(Z.shape[:-1])
    C = np.zeros(Z.shape[:-1])
    for z_j in centers:
        G += gaussian_density(Z, z_j, sigma)
        C += cauchy_density(Z, z_j, sigma)

    vmax = max(G.max(), C.max())
    cmap = _density_cmap()

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.8), constrained_layout=True)
    for ax, field, label, color in zip(
        axes, [G, C], ["Gaussian mixture", "Cauchy mixture"], [STEEL_500, EMERALD_500]
    ):
        ax.contourf(u, u, field, levels=28, cmap=cmap, vmin=0, vmax=vmax)
        ax.contour(u, u, field, levels=8, colors=[ZINC_300], linewidths=0.4, alpha=0.5)
        ax.scatter(centers[:, 0], centers[:, 1],
                   c=color, s=22, edgecolors="none", zorder=5)
        ax.set_title(f"{label}  ·  σ = {sigma}", pad=6, color=ZINC_700)
        ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        clean_spines(ax)

    fig.suptitle(
        "Kernel mixture  D(z) = Σⱼ ρⱼ(z) — five datapoints, same σ",
        fontsize=12, color=ZINC_700,
    )
    path = PLOTS_DIR / "kernel_shape_mixture.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("1/2  single kernel ...")
    p1 = figure_single_kernel()
    print(f"      saved: {p1}")
    print("2/2  5-kernel mixture ...")
    p2 = figure_mixture()
    print(f"      saved: {p2}")
