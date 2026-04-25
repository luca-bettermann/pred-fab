"""From evidence to evidence-gain — what a candidate would add to the field.

Left:   E_old(z) — current evidence from existing data only. The candidate
        z_new is overlaid as a KernelField atom (red centre + σ-shells +
        probes coloured by density) showing the footprint it *would* add.
        Existing kernels are shown as light-grey σ-shells + small dots —
        context, not focus.
Right:  ΔE(z | z_new) = E(z | data ∪ {z_new}) − E_old(z). Same overlays.
        The gain field already encodes z_new's footprint, so the visual
        story is "the σ-rings on the left become the gain bump on the right".

Both fields use the **peak-1 Gaussian** (same convention as
`evidence_transform.py`) so a single isolated kernel reaches `E = 0.5`
at its centre. Both colorbars are fixed at [0, 1] (bounded-spectrum rule).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from _style import (
    apply_style, clean_spines, subplot_label, cmap, style_colorbar,
    add_kernel_radii_2d,
    ZINC_300, ZINC_400, ZINC_500, RED,
)
from pred_fab.orchestration.evidence import DEFAULT_RADII, KernelFieldEstimator


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


def _gaussian_unit_peak(grid: np.ndarray, center: np.ndarray, sigma: float) -> np.ndarray:
    """Peak-1 Gaussian — single isolated kernel saturates to E=0.5 at its centre."""
    d2 = np.sum((grid - center) ** 2, axis=-1)
    return np.exp(-d2 / (2.0 * sigma ** 2))


def _kf_offsets(D: int, sigma: float) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        kf = KernelFieldEstimator()
        offsets, _w = kf._probes_and_weights(D, sigma)
    return offsets


def _draw_existing(ax, sigma: float) -> None:
    """Lightgrey σ-shells + small centre dots for the existing data points (context only)."""
    for c in EXISTING:
        add_kernel_radii_2d(ax, c, sigma, DEFAULT_RADII, color_scale=False,
                            base_color=ZINC_400, alpha_max=0.55, lw=0.5)
    ax.scatter(EXISTING[:, 0], EXISTING[:, 1], c=ZINC_500, s=12,
               edgecolors="none", zorder=4)


def _draw_z_new_atom(ax, sigma: float, cm, norm) -> None:
    """KernelField atom on z_new — red centre + density-coloured probes + σ-shells."""
    add_kernel_radii_2d(ax, Z_NEW, sigma, DEFAULT_RADII,
                        color_scale=True, alpha_max=0.85, lw=0.8)
    offsets = _kf_offsets(2, sigma)
    probes = Z_NEW + offsets
    probe_density_frac = np.exp(-np.sum(offsets ** 2, axis=-1) / (2.0 * sigma ** 2))
    ax.scatter(probes[:, 0], probes[:, 1],
               c=probe_density_frac, cmap=cm, norm=norm,
               s=18, alpha=0.95, edgecolors="none", zorder=6)
    ax.scatter([Z_NEW[0]], [Z_NEW[1]], c=RED, s=34,
               edgecolors="none", zorder=10)
    ax.annotate("z_new", xy=(Z_NEW[0], Z_NEW[1]),
                xytext=(8, 6), textcoords="offset points",
                fontsize=8, color=RED)


def figure(sigma: float = SIGMA, res: int = 301) -> Path:
    apply_style()
    u = np.linspace(0.0, 1.0, res)
    U, V = np.meshgrid(u, u)
    Z = np.stack([U, V], axis=-1)

    D_old = sum(_gaussian_unit_peak(Z, c, sigma) for c in EXISTING)
    D_new = _gaussian_unit_peak(Z, Z_NEW, sigma)
    D_after = D_old + D_new

    E_old = D_old / (1.0 + D_old)
    E_after = D_after / (1.0 + D_after)
    delta_E = E_after - E_old

    cmap_E = cmap("evidence")
    cmap_dE = cmap("evidence_gain")
    norm = Normalize(vmin=0.0, vmax=1.0)  # bounded spectrum: fixed [0, 1] on both panels

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0), constrained_layout=True)

    # ---------- Left: E_old(z) with z_new atom + existing context ----------
    ax = axes[0]
    ax.contourf(u, u, E_old, levels=28, cmap=cmap_E, norm=norm)
    ax.contour(u, u, E_old, levels=[0.25, 0.5, 0.75],
               colors=[ZINC_300], linewidths=0.5, alpha=0.55)
    _draw_existing(ax, sigma)
    _draw_z_new_atom(ax, sigma, cmap_E, norm)
    subplot_label(ax, f"E_old(z)  ·  evidence before adding z_new  ·  σ = {sigma:g}")

    # ---------- Right: ΔE(z | z_new) — same overlays ----------
    ax = axes[1]
    ax.contourf(u, u, delta_E, levels=28, cmap=cmap_dE, norm=norm)
    ax.contour(u, u, delta_E, levels=[0.1, 0.25, 0.4],
               colors=[ZINC_300], linewidths=0.5, alpha=0.55)
    _draw_existing(ax, sigma)
    _draw_z_new_atom(ax, sigma, cmap_dE, norm)
    subplot_label(ax, f"ΔE(z | z_new)  ·  peak ≈ {delta_E.max():.2f}  ·  σ = {sigma:g}")

    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
        clean_spines(ax)

    cbar_E = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_E),
                          ax=axes[0], shrink=0.8, pad=0.02)
    style_colorbar(cbar_E)

    cbar_dE = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_dE),
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
