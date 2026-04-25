"""From density to gain — the three views of the evidence pipeline.

A 3-panel figure showing the same kernel layout under three transforms:

    Panel 1 — D(z)              raw density (greys, unbounded → auto-scaled)
    Panel 2 — E_old(z)          saturation D/(1+D) (Blues, fixed [0, 1])
    Panel 3 — ΔE(z | z_new)     gain from adding z_new (Purples, fixed [0, 1])

z_new is overlaid on every panel as a KernelField "atom" — red centre,
density-coloured probes on σ-shells — to show what would be added.
Existing kernels appear as light-grey σ-shells with small centre dots —
context, not focus. Every concept figure in this folder uses the same
σ, the same EXISTING_POINTS layout, and the same Z_NEW (see `_config.py`).
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
    ZINC_300, ZINC_400, ZINC_500, ZINC_600, RED,
)
from _config import SIGMA, EXISTING_POINTS, Z_NEW, gaussian_unit_peak
from pred_fab.orchestration.evidence import DEFAULT_RADII, KernelFieldEstimator


PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def _kf_offsets(D: int, sigma: float) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        kf = KernelFieldEstimator()
        offsets, _w = kf._probes_and_weights(D, sigma)
    return offsets


def _draw_existing(ax, sigma: float) -> None:
    """Light-grey σ-shells + small centre dots for the existing data points (context)."""
    for c in EXISTING_POINTS:
        add_kernel_radii_2d(ax, c, sigma, DEFAULT_RADII, color_scale=False,
                            base_color=ZINC_400, alpha_max=0.5, lw=0.5)
    ax.scatter(EXISTING_POINTS[:, 0], EXISTING_POINTS[:, 1],
               c=ZINC_500, s=12, edgecolors="none", zorder=4)


def _draw_z_new_atom(ax, sigma: float) -> None:
    """Full KernelField atom on z_new: red centre + density-coloured probes + σ-shells."""
    add_kernel_radii_2d(ax, Z_NEW, sigma, DEFAULT_RADII,
                        color_scale=True, alpha_max=0.85, lw=0.8)
    offsets = _kf_offsets(2, sigma)
    probes = Z_NEW + offsets
    probe_density_frac = np.exp(-np.sum(offsets ** 2, axis=-1) / (2.0 * sigma ** 2))
    norm_atom = Normalize(vmin=0.0, vmax=1.0)
    ax.scatter(probes[:, 0], probes[:, 1],
               c=probe_density_frac, cmap=cmap("evidence"), norm=norm_atom,
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

    # Existing-only fields. z_new is shown as a marker but does not contribute.
    D_old = np.zeros(Z.shape[:-1])
    for c in EXISTING_POINTS:
        D_old += gaussian_unit_peak(Z, c, sigma)
    E_old = D_old / (1.0 + D_old)

    # Gain from hypothetically adding z_new.
    D_new = gaussian_unit_peak(Z, Z_NEW, sigma)
    E_after = (D_old + D_new) / (1.0 + D_old + D_new)
    delta_E = E_after - E_old

    cmap_D = cmap("density")
    cmap_E = cmap("evidence")
    cmap_dE = cmap("evidence_gain")
    norm_bounded = Normalize(vmin=0.0, vmax=1.0)
    norm_D = Normalize(vmin=0.0, vmax=float(D_old.max()))  # unbounded → auto-scale

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), constrained_layout=True)

    # ---------- Panel 1: D(z) — raw density (unbounded, greys) ----------
    ax = axes[0]
    im_D = ax.contourf(u, u, D_old, levels=28, cmap=cmap_D, norm=norm_D)
    ax.contour(u, u, D_old, levels=8, colors=[ZINC_300], linewidths=0.4, alpha=0.5)
    _draw_existing(ax, sigma)
    _draw_z_new_atom(ax, sigma)
    subplot_label(ax, f"D(z)  ·  raw density  ·  peak ≈ {D_old.max():.2f}")

    # ---------- Panel 2: E_old(z) — saturated evidence (bounded, Blues) ----------
    ax = axes[1]
    im_E = ax.contourf(u, u, E_old, levels=28, cmap=cmap_E, norm=norm_bounded)
    ax.contour(u, u, E_old, levels=[0.25, 0.5, 0.75],
               colors=[ZINC_300], linewidths=0.5, alpha=0.55)
    _draw_existing(ax, sigma)
    _draw_z_new_atom(ax, sigma)
    subplot_label(ax, "E_old(z) = D / (1 + D)  ·  evidence before z_new")

    # ---------- Panel 3: ΔE(z | z_new) — gain (bounded, Purples) ----------
    ax = axes[2]
    im_dE = ax.contourf(u, u, delta_E, levels=28, cmap=cmap_dE, norm=norm_bounded)
    ax.contour(u, u, delta_E, levels=[0.1, 0.25, 0.4],
               colors=[ZINC_300], linewidths=0.5, alpha=0.55)
    _draw_existing(ax, sigma)
    _draw_z_new_atom(ax, sigma)
    subplot_label(ax, f"ΔE(z | z_new)  ·  gain from z_new  ·  peak ≈ {delta_E.max():.2f}")

    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
        clean_spines(ax)

    cbar_D = fig.colorbar(im_D, ax=axes[0], shrink=0.8, pad=0.02)
    style_colorbar(cbar_D)
    cbar_E = fig.colorbar(ScalarMappable(norm=norm_bounded, cmap=cmap_E),
                          ax=axes[1], shrink=0.8, pad=0.02)
    style_colorbar(cbar_E)
    cbar_dE = fig.colorbar(ScalarMappable(norm=norm_bounded, cmap=cmap_dE),
                           ax=axes[2], shrink=0.8, pad=0.02)
    style_colorbar(cbar_dE)

    path = PLOTS_DIR / "evidence_to_objective.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("evidence_to_objective ...")
    p = figure()
    print(f"      saved: {p}")
