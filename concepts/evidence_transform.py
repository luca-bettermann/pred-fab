"""Concept: the saturating evidence transform  E(z) = D(z) / (1 + D(z)).

Raw density `D(z) = Σⱼ wⱼ ρⱼ(z)` is unbounded — it grows without limit as
kernels stack, and its peak scales with `σ^−D` so the magnitude is
σ-dependent. We want an "evidence" field that:

  · is bounded in `[0, 1)` so it can be integrated and interpreted directly
  · saturates (diminishing returns) — a second kernel at the same point
    adds less than the first, a fourth less than a third, etc.
  · has a non-trivial gradient everywhere `D > 0`

The simplest map satisfying all three is the Möbius-style transform

    E(D) = D / (1 + D),     u(D) = 1 − E = 1 / (1 + D)  (uncertainty).

`E` approaches 1 asymptotically, hits 0.5 at `D = 1`, and stays differentiable
for every finite `D` — the optimiser never sees a flat patch as long as any
kernel has non-zero mass locally.

Two figures:
    evidence_scalar_curve.png  — E(D) and u(D) as scalar curves on [0, 10]
    evidence_field_comparison.png — D(z) and E(z) for a 5-kernel mixture
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

from _style import (
    apply_style, clean_spines,
    evidence_cmap,
    ZINC_100, ZINC_300, ZINC_500, ZINC_600, ZINC_700, ZINC_900,
    STEEL_500, EMERALD_500,
)
from kernel_shapes import gaussian_density


PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def _density_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "density", ["white", ZINC_100, "#D4D4D8", ZINC_500, ZINC_900], N=256,
    )


# ---------- Figure 1: scalar transform ----------

def figure_scalar_curve() -> Path:
    apply_style()
    D_vals = np.linspace(0.0, 10.0, 1000)
    E_vals = D_vals / (1.0 + D_vals)
    u_vals = 1.0 / (1.0 + D_vals)

    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)

    ax.plot(D_vals, E_vals, color=STEEL_500, lw=2.2, label="E = D / (1 + D)")
    ax.plot(D_vals, u_vals, color=EMERALD_500, lw=2.2, label="u = 1 − E")

    # Annotate characteristic points
    for D_pt in (1, 3, 9):
        ax.axvline(D_pt, color=ZINC_300, lw=0.6, ls="--", alpha=0.6, zorder=-1)
        ax.scatter([D_pt], [D_pt / (1 + D_pt)], c=STEEL_500, s=18, zorder=5, edgecolors="none")
        ax.annotate(
            f"D = {D_pt}\nE = {D_pt / (1 + D_pt):.2f}",
            xy=(D_pt, D_pt / (1 + D_pt)),
            xytext=(6, -22), textcoords="offset points",
            fontsize=7.5, color=ZINC_500,
        )

    ax.axhline(1.0, color=ZINC_300, lw=0.6, ls="--", alpha=0.6)

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("raw density  D")
    ax.set_ylabel("transformed value")
    ax.legend(loc="center right", fontsize=9, frameon=False)
    ax.grid(True, which="major", color="#E4E4E7", alpha=0.4, lw=0.5)
    clean_spines(ax)

    path = PLOTS_DIR / "evidence_scalar_curve.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------- Figure 2: field comparison ----------

def figure_field_comparison(sigma: float = 0.10, res: int = 301) -> Path:
    apply_style()
    centers = np.array([
        [0.22, 0.28], [0.74, 0.22], [0.50, 0.52],
        [0.26, 0.78], [0.78, 0.74],
    ])
    u = np.linspace(0.0, 1.0, res)
    U, V = np.meshgrid(u, u)
    Z = np.stack([U, V], axis=-1)

    D_field = np.zeros(Z.shape[:-1])
    for z_j in centers:
        D_field += gaussian_density(Z, z_j, sigma)
    E_field = D_field / (1.0 + D_field)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)

    # Left: raw density D(z) — unbounded
    cmap_D = _density_cmap()
    im_D = axes[0].contourf(u, u, D_field, levels=28, cmap=cmap_D, vmin=0, vmax=D_field.max())
    axes[0].contour(u, u, D_field, levels=8, colors=[ZINC_300], linewidths=0.4, alpha=0.5)
    axes[0].scatter(centers[:, 0], centers[:, 1], c=ZINC_700, s=18, edgecolors="none", zorder=5)
    axes[0].text(0.02, 0.96, f"D(z)  ·  peak ≈ {D_field.max():.0f}",
                 transform=axes[0].transAxes, fontsize=9.5, color=ZINC_600)

    # Right: evidence E(z) — bounded in [0, 1)
    cmap_E = evidence_cmap()
    im_E = axes[1].contourf(u, u, E_field, levels=28, cmap=cmap_E, vmin=0, vmax=1.0)
    axes[1].contour(u, u, E_field, levels=[0.25, 0.5, 0.75], colors=[ZINC_300],
                    linewidths=0.5, alpha=0.55)
    axes[1].scatter(centers[:, 0], centers[:, 1], c=ZINC_700, s=18, edgecolors="none", zorder=5)
    axes[1].text(0.02, 0.96, "E(z) = D / (1 + D)  ·  bounded in [0, 1)",
                 transform=axes[1].transAxes, fontsize=9.5, color=ZINC_600)

    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
        clean_spines(ax)

    # Two separate narrow colorbars, one per panel
    cbar_D = fig.colorbar(im_D, ax=axes[0], shrink=0.8, pad=0.02)
    cbar_D.ax.tick_params(colors=ZINC_500, labelsize=7)
    cbar_D.outline.set_edgecolor(ZINC_300)
    cbar_D.outline.set_linewidth(0.6)

    cbar_E = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap_E),
                          ax=axes[1], shrink=0.8, pad=0.02)
    cbar_E.ax.tick_params(colors=ZINC_500, labelsize=7)
    cbar_E.outline.set_edgecolor(ZINC_300)
    cbar_E.outline.set_linewidth(0.6)

    path = PLOTS_DIR / "evidence_field_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("1/2  scalar E(D) curve ...")
    p1 = figure_scalar_curve()
    print(f"      saved: {p1}")
    print("2/2  D(z) vs E(z) field ...")
    p2 = figure_field_comparison()
    print(f"      saved: {p2}")
