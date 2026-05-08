"""KernelField joint visualisation — 2D and 3D side by side.

Left:  2D KernelField probes around a single kernel centre.
Right: 3D KernelField probes (atom-orbital style) with z-axis spine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from _style import (
    apply_style, clean_spines, subplot_label, cmap, style_colorbar,
    add_kernel_radii_2d, add_kernel_radii_3d,
    ZINC_300, ZINC_500, ZINC_600, ZINC_700, RED,
)
from _config import SIGMA
from pred_fab.orchestration.evidence import DEFAULT_RADII, KernelFieldEstimator
from pred_fab.plotting._style import MARKERS, FONT

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def _kf_offsets(D, sigma):
    kf = KernelFieldEstimator()
    offsets, _, _ = kf._probes_weights_self(D, sigma)
    return offsets


def _density_fraction(probes, center, sigma):
    d2 = np.sum((probes - center) ** 2, axis=-1)
    return np.exp(-d2 / (2.0 * sigma ** 2))


def _gaussian_field(grid, center, sigma):
    d2 = np.sum((grid - center) ** 2, axis=-1)
    return np.exp(-d2 / (2.0 * sigma ** 2))


def _tinted_3d_panes(ax):
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("white")
        pane.set_alpha(0)
        pane.set_edgecolor(ZINC_300)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_tick_params(colors=ZINC_600, labelsize=7, pad=1)
        axis.label.set_color(ZINC_700)
        axis.line.set_color(ZINC_500)
    ax.grid(False)


def _angular_gap_marker(ax, center, sigma, ray_sigma=2.0):
    r = float(ray_sigma) * float(sigma)
    cx, cy = float(center[0]), float(center[1])
    ax.plot([cx, cx], [cy, cy + r], color=ZINC_300, lw=0.6, alpha=0.7, zorder=0)
    dx = r * np.cos(np.pi / 4.0)
    dy = r * np.sin(np.pi / 4.0)
    ax.plot([cx, cx + dx], [cy, cy + dy], color=ZINC_300, lw=0.6, alpha=0.7, zorder=0)
    label_r = 0.85 * r
    label_angle = 3.0 * np.pi / 8.0
    ax.text(cx + label_r * np.cos(label_angle),
            cy + label_r * np.sin(label_angle),
            "45°", fontsize=7, color=ZINC_500, ha="center", va="center", zorder=1)


def _radii_labels(ax, center, sigma, multipliers):
    for m in multipliers:
        r = float(m) * sigma
        x = float(center[0]) + r
        y = float(center[1]) + 0.04 * sigma
        ax.text(x, y, f"{m:g}σ", fontsize=7, color=ZINC_500,
                ha="left", va="bottom", zorder=6)


def main():
    apply_style()
    sigma = SIGMA
    cm = cmap("evidence")
    norm = Normalize(vmin=0.0, vmax=1.0)

    center_2d = np.full(2, 0.5)
    center_3d = np.full(3, 0.5)

    kf_off_2d = _kf_offsets(2, sigma)
    kf_off_3d = _kf_offsets(3, sigma)
    n_2d = kf_off_2d.shape[0]
    n_3d = kf_off_3d.shape[0]

    kf_pts_2d = center_2d + kf_off_2d
    kf_pts_3d = center_3d + kf_off_3d
    kf_dens_2d = _density_fraction(kf_pts_2d, center_2d, sigma)
    kf_dens_3d = _density_fraction(kf_pts_3d, center_3d, sigma)

    pad = 2.6 * sigma
    lo = 0.5 - pad
    hi = 0.5 + pad
    ticks = [round(0.5 - 2 * sigma, 2), 0.5, round(0.5 + 2 * sigma, 2)]

    # Background field for 2D
    grid_res = 240
    g_x = np.linspace(lo, hi, grid_res)
    G1, G2 = np.meshgrid(g_x, g_x)
    G = np.stack([G1, G2], axis=-1)
    bg = _gaussian_field(G, center_2d, sigma)

    fig = plt.figure(figsize=(12, 5.5))
    fig.subplots_adjust(left=0.04, right=0.90, top=0.92, bottom=0.08, wspace=0.15)

    # === 2D panel ===
    ax2 = fig.add_subplot(121)
    ax2.contourf(g_x, g_x, bg, levels=18, cmap=cm, norm=norm, alpha=0.18, zorder=0)
    _angular_gap_marker(ax2, center_2d, sigma)
    add_kernel_radii_2d(ax2, center_2d, sigma, DEFAULT_RADII, color_scale=True)
    _radii_labels(ax2, center_2d, sigma, DEFAULT_RADII)
    ax2.scatter(kf_pts_2d[1:, 0], kf_pts_2d[1:, 1],
                c=kf_dens_2d[1:], cmap=cm, norm=norm,
                s=20, alpha=0.95, edgecolors="none", zorder=5)
    ax2.scatter([center_2d[0]], [center_2d[1]], c=RED, s=38,
                edgecolors="none", zorder=10)
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    ax2.set_xlabel("z₁", fontsize=FONT["axis_label"], color=ZINC_600)
    ax2.set_ylabel("z₂", fontsize=FONT["axis_label"], color=ZINC_600)
    ax2.set_aspect("equal")
    subplot_label(ax2, f"KernelField 2D  ·  {n_2d} probes  ·  σ = {sigma:g}")
    clean_spines(ax2)

    # === 3D panel ===
    ax3 = fig.add_subplot(122, projection="3d")
    add_kernel_radii_3d(ax3, center_3d, sigma, DEFAULT_RADII,
                        color_scale=True, orbitals_per_shell=3, alpha_max=0.6, lw=0.8)
    ax3.scatter(kf_pts_3d[:, 0], kf_pts_3d[:, 1], kf_pts_3d[:, 2],  # type: ignore[arg-type]
                c=kf_dens_3d, cmap=cm, norm=norm,
                s=22, alpha=0.95, edgecolors="none",
                depthshade=False, zorder=5)
    ax3.scatter([center_3d[0]], [center_3d[1]], [center_3d[2]],  # type: ignore[arg-type]
                c=RED, s=42, edgecolors="none",
                depthshade=False, zorder=10)

    pad_3d = 1.0 * sigma
    lo3 = 0.5 - pad_3d
    hi3 = 0.5 + pad_3d
    ax3.set_xlim(lo3, hi3)
    ax3.set_ylim(lo3, hi3)
    ax3.set_zlim(lo3, hi3)  # type: ignore[attr-defined]

    # No coordinate axes — just the orbital structure
    ax3.set_axis_off()
    ax3.set_xticks(ticks)
    ax3.set_yticks(ticks)
    ax3.set_zticks(ticks)  # type: ignore[attr-defined]
    ax3.set_xlabel("z₁", fontsize=FONT["axis_label"])
    ax3.set_ylabel("z₂", fontsize=FONT["axis_label"])
    ax3.set_zlabel("z₃", fontsize=FONT["axis_label"])  # type: ignore[attr-defined]
    ax3.view_init(elev=20.0, azim=35.0)
    ax3.dist = 7  # default is ~10, lower = closer
    _tinted_3d_panes(ax3)
    subplot_label(ax3, f"KernelField 3D  ·  {n_3d} probes  ·  σ = {sigma:g}")

    # Shared colorbar
    sm = ScalarMappable(norm=norm, cmap=cm)
    cbar = fig.colorbar(sm, ax=[ax2, ax3], location="right", shrink=0.7, pad=0.02)
    style_colorbar(cbar)
    cbar.set_label("ρ / ρ(centre)", color=ZINC_600, fontsize=9)

    path = PLOTS_DIR / "kernelfield_joint_visual.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
