"""From evidence to evidence-gain — what a candidate would add to the field.

Left:   E_old(z) — current evidence from existing data only. The candidate
        z_new is marked but not yet integrated; the σ-shells around it show
        the footprint it *would* contribute if added.
Right:  ΔE(z | z_new) = E(z | data ∪ {z_new}) − E_old(z). The gain field
        already encodes the kernel footprint, so radii would be redundant.
        z_new is marked as a bare red dot.

Reading the figure:
    The radial structure on the left becomes the gain bump on the right.
    Saturation flattens the gain wherever existing kernels already cover —
    i.e. ΔE is highest in *un-covered* regions and lowest in *covered* ones.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from _style import (
    apply_style, clean_spines, subplot_label, cmap, style_colorbar,
    add_kernel_radii_2d,
    ZINC_300, ZINC_700, RED,
)
from kernel_shapes import gaussian_density
from pred_fab.orchestration.evidence import DEFAULT_RADII


SIGMA: float = 0.10
Z_NEW: np.ndarray = np.array([0.50, 0.50])
EXISTING: np.ndarray = np.array([
    [0.22, 0.28],
    [0.78, 0.24],
    [0.26, 0.78],
    [0.78, 0.74],
])

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def _density_field(grid: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """Sum of Gaussian densities over a stack of centers, evaluated on `grid`."""
    field = np.zeros(grid.shape[:-1])
    for c in centers:
        field += gaussian_density(grid, c, sigma)
    return field


def figure(sigma: float = SIGMA, res: int = 301) -> Path:
    apply_style()
    u = np.linspace(0.0, 1.0, res)
    U, V = np.meshgrid(u, u)
    Z = np.stack([U, V], axis=-1)

    D_old = _density_field(Z, EXISTING, sigma)
    D_new = gaussian_density(Z, Z_NEW, sigma)
    D_after = D_old + D_new

    E_old = D_old / (1.0 + D_old)
    E_after = D_after / (1.0 + D_after)
    delta_E = E_after - E_old

    cmap_E = cmap("evidence")
    cmap_dE = cmap("evidence_gain")
    norm_E = Normalize(vmin=0.0, vmax=1.0)
    norm_dE = Normalize(vmin=0.0, vmax=float(delta_E.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0), constrained_layout=True)

    # ---------- Left: E_old(z) with z_new marker + σ-shells ----------
    ax = axes[0]
    ax.contourf(u, u, E_old, levels=28, cmap=cmap_E, norm=norm_E)
    ax.contour(u, u, E_old, levels=[0.25, 0.5, 0.75],
               colors=[ZINC_300], linewidths=0.5, alpha=0.55)
    ax.scatter(EXISTING[:, 0], EXISTING[:, 1], c=ZINC_700, s=18,
               edgecolors="none", zorder=5)
    add_kernel_radii_2d(ax, Z_NEW, sigma, DEFAULT_RADII, color_scale=False,
                        base_color=ZINC_700, alpha_max=0.45, lw=0.6)
    ax.scatter([Z_NEW[0]], [Z_NEW[1]], c=RED, s=32,
               edgecolors="none", zorder=10)
    ax.annotate("z_new", xy=(Z_NEW[0], Z_NEW[1]),
                xytext=(8, 6), textcoords="offset points",
                fontsize=8, color=RED)
    subplot_label(ax, "E_old(z)  ·  evidence before adding z_new")

    # ---------- Right: ΔE(z | z_new), no radii ----------
    ax = axes[1]
    ax.contourf(u, u, delta_E, levels=28, cmap=cmap_dE, norm=norm_dE)
    ax.contour(u, u, delta_E, levels=6, colors=[ZINC_300],
               linewidths=0.4, alpha=0.5)
    ax.scatter(EXISTING[:, 0], EXISTING[:, 1], c=ZINC_700, s=18,
               edgecolors="none", zorder=5)
    ax.scatter([Z_NEW[0]], [Z_NEW[1]], c=RED, s=32,
               edgecolors="none", zorder=10)
    ax.annotate("z_new", xy=(Z_NEW[0], Z_NEW[1]),
                xytext=(8, 6), textcoords="offset points",
                fontsize=8, color=RED)
    subplot_label(ax, f"ΔE(z | z_new)  ·  peak ≈ {delta_E.max():.2f}")

    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
        clean_spines(ax)

    cbar_E = fig.colorbar(ScalarMappable(norm=norm_E, cmap=cmap_E),
                          ax=axes[0], shrink=0.8, pad=0.02)
    style_colorbar(cbar_E)

    cbar_dE = fig.colorbar(ScalarMappable(norm=norm_dE, cmap=cmap_dE),
                           ax=axes[1], shrink=0.8, pad=0.02)
    style_colorbar(cbar_dE)

    path = PLOTS_DIR / "evidence_to_objective.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("evidence_to_objective ...")
    p = figure()
    print(f"      saved: {p}")
