"""Probe placement: KernelField vs Sobol-local, in 2-D and 3-D.

Both estimators target the same self-integral
    𝔼_{z~N(c, σ²I)}[1/(1+D(z))]
using probes around an isolated kernel at the cube centre. KernelField
places probes deterministically on shells (atomic-orbital structure);
Sobol-local samples them quasi-randomly inside a cube around the kernel.
Probe colour encodes ρ(probe) / ρ(centre) — fraction of peak density.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import qmc

from pred_fab.orchestration.evidence import (
    DEFAULT_RADII,
    KernelFieldEstimator,
)
from _style import (
    apply_style, clean_spines, clean_3d_panes, subplot_label,
    cmap, cube_wireframe, square_wireframe, style_colorbar,
    add_kernel_radii_2d, add_kernel_radii_3d,
    ZINC_300, ZINC_500, ZINC_600, RED,
)


SOBOL_HALF_EXTENT: float = 3.0  # σ-units; matches SobolLocalEstimator default
SIGMA: float = 0.075
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def _density_fraction(probes: np.ndarray, center: np.ndarray, sigma: float) -> np.ndarray:
    d2 = np.sum((probes - center) ** 2, axis=-1)
    return np.exp(-d2 / (2.0 * sigma ** 2))


def _kf_offsets(D: int, sigma: float) -> np.ndarray:
    """Probe offsets (relative to centre) for the default KernelField estimator."""
    kf = KernelFieldEstimator()
    offsets, _w = kf._probes_and_weights(D, sigma)
    return offsets


def _sobol_offsets(D: int, n: int, sigma: float, seed: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        unit = qmc.Sobol(d=D, scramble=True, rng=seed).random(n=n)
    box = 2.0 * SOBOL_HALF_EXTENT * sigma
    return box * (unit - 0.5)


# ---------- 2-D figure ----------

def figure_2d(sigma: float = SIGMA, seed: int = 0) -> Path:
    apply_style()
    D = 2
    center = np.full(D, 0.5)

    kf_off = _kf_offsets(D, sigma)
    n_probes = kf_off.shape[0]
    sb_off = _sobol_offsets(D, n_probes, sigma, seed=seed)

    kf_pts = center + kf_off
    sb_pts = center + sb_off

    kf_density = _density_fraction(kf_pts, center, sigma)
    sb_density = _density_fraction(sb_pts, center, sigma)

    cm = cmap("evidence")
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0), constrained_layout=True)

    # KernelField
    ax = axes[0]
    add_kernel_radii_2d(ax, center, sigma, DEFAULT_RADII, color_scale=True)
    ax.scatter(kf_pts[:, 0], kf_pts[:, 1],
               c=kf_density, cmap=cm, norm=norm,
               s=14, alpha=0.95, edgecolors="none", zorder=5)
    ax.scatter([center[0]], [center[1]], marker="o", c=RED, s=34,
               edgecolors="none", zorder=10)
    subplot_label(ax, f"KernelField  ·  {n_probes} probes")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
    ax.set_aspect("equal")
    clean_spines(ax)

    # Sobol-local
    ax = axes[1]
    box_lo = center - SOBOL_HALF_EXTENT * sigma
    box_hi = center + SOBOL_HALF_EXTENT * sigma
    square_wireframe(ax, box_lo, box_hi, color=ZINC_500, lw=0.9, alpha=0.55)
    ax.scatter(sb_pts[:, 0], sb_pts[:, 1],
               c=sb_density, cmap=cm, norm=norm,
               s=14, alpha=0.95, edgecolors="none", zorder=5)
    ax.scatter([center[0]], [center[1]], marker="o", c=RED, s=34,
               edgecolors="none", zorder=10)
    subplot_label(ax, f"Sobol (local cube)  ·  {n_probes} probes")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
    ax.set_aspect("equal")
    clean_spines(ax)

    sm = ScalarMappable(norm=norm, cmap=cm)
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.7, pad=0.02)
    style_colorbar(cbar)
    cbar.set_label("ρ / ρ(centre)", color=ZINC_600, fontsize=9)

    path = PLOTS_DIR / "evidence_integration_2d.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------- 3-D figure ----------

def figure_3d(sigma: float = SIGMA, seed: int = 0) -> Path:
    apply_style()
    D = 3
    center = np.full(D, 0.5)

    kf_off = _kf_offsets(D, sigma)
    n_probes = kf_off.shape[0]
    sb_off = _sobol_offsets(D, n_probes, sigma, seed=seed)

    kf_pts = center + kf_off
    sb_pts = center + sb_off

    kf_density = _density_fraction(kf_pts, center, sigma)
    sb_density = _density_fraction(sb_pts, center, sigma)

    cm = cmap("evidence")
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(12.5, 5.8))
    fig.subplots_adjust(left=0.01, right=0.88, top=0.98, bottom=0.02, wspace=0.02)

    # KernelField
    ax = fig.add_subplot(121, projection="3d")
    cube_wireframe(ax, np.zeros(3), np.ones(3),
                   color=ZINC_300, lw=0.4, alpha=0.32)
    add_kernel_radii_3d(ax, center, sigma, DEFAULT_RADII, color_scale=True)
    ax.scatter(kf_pts[:, 0], kf_pts[:, 1], kf_pts[:, 2],  # type: ignore[arg-type]
               c=kf_density, cmap=cm, norm=norm,
               s=12, alpha=0.95, edgecolors="none",
               depthshade=False, zorder=5)
    ax.scatter([center[0]], [center[1]], [center[2]],  # type: ignore[arg-type]
               marker="o", c=RED, s=34, edgecolors="none",
               depthshade=False, zorder=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_zticks([0, 0.5, 1])  # type: ignore[operator]
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
    ax.set_zlabel("z₃")  # type: ignore[operator]
    ax.view_init(elev=20, azim=35)
    clean_3d_panes(ax)
    subplot_label(ax, f"KernelField  ·  {n_probes} probes")

    # Sobol-local
    ax = fig.add_subplot(122, projection="3d")
    cube_wireframe(ax, np.zeros(3), np.ones(3),
                   color=ZINC_300, lw=0.4, alpha=0.32)
    box_lo = center - SOBOL_HALF_EXTENT * sigma
    box_hi = center + SOBOL_HALF_EXTENT * sigma
    cube_wireframe(ax, box_lo, box_hi, color=ZINC_500, lw=0.9, alpha=0.55)
    ax.scatter(sb_pts[:, 0], sb_pts[:, 1], sb_pts[:, 2],  # type: ignore[arg-type]
               c=sb_density, cmap=cm, norm=norm,
               s=12, alpha=0.95, edgecolors="none",
               depthshade=False, zorder=5)
    ax.scatter([center[0]], [center[1]], [center[2]],  # type: ignore[arg-type]
               marker="o", c=RED, s=34, edgecolors="none",
               depthshade=False, zorder=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_zticks([0, 0.5, 1])  # type: ignore[operator]
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
    ax.set_zlabel("z₃")  # type: ignore[operator]
    ax.view_init(elev=20, azim=35)
    clean_3d_panes(ax)
    subplot_label(ax, f"Sobol (local cube)  ·  {n_probes} probes")

    cax = fig.add_axes((0.905, 0.22, 0.016, 0.56))
    sm = ScalarMappable(norm=norm, cmap=cm)
    cbar = fig.colorbar(sm, cax=cax)
    style_colorbar(cbar)
    cbar.set_label("ρ / ρ(centre)", color=ZINC_600, fontsize=9)

    path = PLOTS_DIR / "evidence_integration_3d.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("1/2  evidence_integration_2d ...")
    p1 = figure_2d()
    print(f"      saved: {p1}")
    print("2/2  evidence_integration_3d ...")
    p2 = figure_3d()
    print(f"      saved: {p2}")
