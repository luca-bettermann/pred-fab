"""From density to acquisition surface — the three views of the evidence pipeline.

A 3-panel figure showing the same kernel layout under three transforms:

    Panel 1 — D(z)         raw density (greys, unbounded → auto-scaled)
    Panel 2 — E_old(z)     saturation D/(1+D) (Blues, fixed [0, 1])
    Panel 3 — ΔI(z_new)    integrated evidence gain as a function of
                           candidate placement (Purples, auto-scaled)

Panels 1–2 show fields over z; the proposed z_new is overlaid as a
KernelField atom (red centre + density-coloured probes on σ-shells) to
illustrate what would be added. Panel 3 sweeps z_new across the whole
[0,1]^D and reports the integrated gain ΔI(z_new) = ∫(E_after − E_before) dz —
the actual acquisition surface the optimiser maximises. The argmax is
marked with a yellow star.

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
from _config import SIGMA, EXISTING_POINTS, Z_NEW, gaussian_density
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


def _delta_I_surface(
    sigma: float,
    inner_res: int,
    outer_res: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Acquisition surface ΔI(z_new) over the [0,1]² grid.

    For each candidate z_new, ΔI = ∫(E_after − E_before) dz integrated over
    the unit cube. Computed densely on an outer grid, with the integral
    approximated by a Riemann sum over a finer inner grid.
    """
    # Inner grid for the integral over z
    inner = np.linspace(0.0, 1.0, inner_res)
    UI, VI = np.meshgrid(inner, inner)
    Z_inner = np.stack([UI.ravel(), VI.ravel()], axis=-1)  # (M, 2)
    cell_area = (1.0 / (inner_res - 1)) ** 2

    D_old_inner = np.zeros(Z_inner.shape[0])
    for c in EXISTING_POINTS:
        D_old_inner += gaussian_density(Z_inner, c, sigma)
    E_old_inner = D_old_inner / (1.0 + D_old_inner)

    # Outer grid of candidate z_new placements
    outer = np.linspace(0.0, 1.0, outer_res)
    UO, VO = np.meshgrid(outer, outer)
    Z_outer = np.stack([UO.ravel(), VO.ravel()], axis=-1)  # (N, 2)
    inv_2sig2 = 1.0 / (2.0 * sigma ** 2)

    delta_I = np.zeros(Z_outer.shape[0])
    # Stream over candidates to keep peak memory below ~50 MB at typical
    # res. Each chunk computes the (chunk, M) ρ_new matrix densely.
    chunk = 256
    for start in range(0, Z_outer.shape[0], chunk):
        block = Z_outer[start:start + chunk]                    # (k, 2)
        diff = block[:, None, :] - Z_inner[None, :, :]          # (k, M, 2)
        d2 = (diff * diff).sum(-1)                              # (k, M)
        rho_new = np.exp(-d2 * inv_2sig2)                       # peak-1
        D_after = D_old_inner[None, :] + rho_new
        E_after = D_after / (1.0 + D_after)
        delta_I[start:start + chunk] = (E_after - E_old_inner[None, :]).sum(-1) * cell_area

    return outer, outer, delta_I.reshape(outer_res, outer_res)


def figure(sigma: float = SIGMA, res: int = 301) -> Path:
    apply_style()
    u = np.linspace(0.0, 1.0, res)
    U, V = np.meshgrid(u, u)
    Z = np.stack([U, V], axis=-1)

    # Existing-only fields. z_new is shown as a marker but does not contribute.
    D_old = np.zeros(Z.shape[:-1])
    for c in EXISTING_POINTS:
        D_old += gaussian_density(Z, c, sigma)
    E_old = D_old / (1.0 + D_old)

    # Acquisition surface ΔI(z_new) over the whole solution space —
    # what the optimiser actually maximises. Coarser grid since each
    # candidate triggers an inner integral.
    u_acq, v_acq, dI_grid = _delta_I_surface(sigma, inner_res=121, outer_res=81)
    arg_idx = int(np.argmax(dI_grid))
    arg_y, arg_x = np.unravel_index(arg_idx, dI_grid.shape)
    z_argmax = np.array([u_acq[arg_x], v_acq[arg_y]])

    cmap_D = cmap("density")
    cmap_E = cmap("evidence")
    cmap_dI = cmap("evidence_gain")
    norm_bounded = Normalize(vmin=0.0, vmax=1.0)
    norm_D = Normalize(vmin=0.0, vmax=float(D_old.max()))
    dI_max = float(dI_grid.max())
    norm_dI = Normalize(vmin=0.0, vmax=dI_max if dI_max > 0 else 1.0)

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

    # ---------- Panel 3: ΔI(z_new) — acquisition surface (auto-scaled, Purples) ----------
    ax = axes[2]
    im_dI = ax.contourf(u_acq, v_acq, dI_grid, levels=28, cmap=cmap_dI, norm=norm_dI)
    ax.contour(u_acq, v_acq, dI_grid,
               levels=np.linspace(0.0, dI_max, 6)[1:-1] if dI_max > 0 else [],
               colors=[ZINC_300], linewidths=0.5, alpha=0.55)
    _draw_existing(ax, sigma)
    # Mark argmax — same red dot + z_new label as the left panels.
    ax.scatter([z_argmax[0]], [z_argmax[1]], c=RED, s=34,
               edgecolors="none", zorder=10)
    ax.annotate("z_new", xy=(z_argmax[0], z_argmax[1]),
                xytext=(8, 6), textcoords="offset points",
                fontsize=8, color=RED)
    ax.text(0.98, 0.98, "z_new = argmax ΔI", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=ZINC_600, style="italic")
    subplot_label(ax,
                  f"ΔI(z_new)  ·  ∫(E_after − E_before) dz  ·  peak ≈ {dI_max:.3f}")

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
    cbar_dI = fig.colorbar(ScalarMappable(norm=norm_dI, cmap=cmap_dI),
                           ax=axes[2], shrink=0.8, pad=0.02)
    style_colorbar(cbar_dI)

    path = PLOTS_DIR / "density_to_evidence_to_objective.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("density_to_evidence_to_objective ...")
    p = figure()
    print(f"      saved: {p}")
