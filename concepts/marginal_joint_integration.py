"""Marginal-joint integration — three concept figures.

Figure 1 (density):     Joint density ρ + marginal densities ρ_x, ρ_y
Figure 2 (evidence):    Joint evidence E=D/(1+D) + marginal evidence
Figure 3 (evidence gain): Joint ΔE surface + marginal ΔE curves

All use three kernels where A and B share near-identical x-values —
the marginal integral detects this overlap while the joint integral doesn't.

Layout per figure:
    Left (wide):   2D field with KernelField probes + projection lines
    Right (stack): Two 1D marginal plots (x on top, y on bottom)
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
    ZINC_300, ZINC_400, ZINC_500, ZINC_600, ZINC_700,
)
from _config import SIGMA
from pred_fab.orchestration.evidence import DEFAULT_RADII, KernelFieldEstimator
from pred_fab.plotting._style import (
    SURFACES, MARKERS, LINES, FILL_ALPHA, FONT, RED,
    surface as get_surface,
)

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# A and B near-identical x (offset slightly for visibility), C separate.
CENTERS = np.array([
    [0.28, 0.20],
    [0.32, 0.75],
    [0.75, 0.50],
])
LABELS = ["A", "B", "C"]
SIGMA_VIS = 0.08


def _density_1d(xs, centers_1d, sigma):
    rho = np.zeros_like(xs)
    for c in centers_1d:
        rho += np.exp(-(xs - c) ** 2 / (2 * sigma ** 2))
    return rho


def _density_2d(xx, yy, centers, sigma):
    rho = np.zeros_like(xx)
    for c in centers:
        d2 = (xx - c[0]) ** 2 + (yy - c[1]) ** 2
        rho += np.exp(-d2 / (2 * sigma ** 2))
    return rho


def _draw_2d_panel(ax, centers, sigma, field_2d, xs, surface_name):
    """Draw the 2D panel with KernelField probes and projection lines."""
    surf = get_surface(surface_name)
    cm = cmap(surface_name)
    norm = Normalize(vmin=0, vmax=field_2d.max())

    ax.contourf(xs, xs, field_2d, levels=18, cmap=cm, norm=norm)
    ax.contour(xs, xs, field_2d, levels=8, colors=[ZINC_300], linewidths=0.4)

    # KernelField probes
    kf = KernelFieldEstimator()
    offsets, _, _ = kf._probes_weights_self(2, sigma)

    probe_d2 = np.sum(offsets ** 2, axis=-1)
    probe_density = np.exp(-probe_d2 / (2 * sigma ** 2))
    probe_vis = np.clip(probe_density, 0.4, 1.0)  # minimum 40% visibility

    for ci, (c, lab) in enumerate(zip(centers, LABELS)):
        # Shell radii in current surface colormap
        add_kernel_radii_2d(ax, c, sigma, DEFAULT_RADII, color_scale=False,
                            base_color=ZINC_400)
        # Probes colored by inverted density (outer=dark, inner=light)
        pts = c + offsets
        probe_inv = 1.0 - probe_vis[1:]
        probe_norm = Normalize(vmin=0, vmax=1)
        ax.scatter(pts[1:, 0], pts[1:, 1],
                   c=probe_inv, cmap=cm, norm=probe_norm,
                   s=MARKERS["probe"].size + 2, alpha=0.9, edgecolors="none", zorder=4)
        # Centre (red)
        m = MARKERS["sample"]
        ax.scatter([c[0]], [c[1]], c=m.color, s=m.size, edgecolors=m.edgecolor,
                   linewidth=m.linewidth, zorder=10)
        # Label
        dx = 0.05 if ci != 1 else -0.07
        ax.text(c[0] + dx, c[1] + 0.04, lab, fontsize=FONT["annotation"],
                color=ZINC_600, zorder=11)

    # Projection lines to axes with labelled intersection points
    proj = LINES["projection"]
    proj_color_x = ZINC_500
    proj_color_y = ZINC_400
    for ci, (c, lab) in enumerate(zip(centers, LABELS)):
        # Vertical projection to x-axis
        ax.plot([c[0], c[0]], [0, c[1]],
                color=proj_color_x, lw=0.8, linestyle=proj.linestyle,
                alpha=0.5, zorder=1)
        ax.scatter([c[0]], [0], c=RED, s=18, edgecolors="white",
                   linewidth=0.5, zorder=8, clip_on=False)
        _bbox = dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="none", alpha=1.0)
        ax.text(c[0], -0.04, f"{lab}ₓ", fontsize=7, color=proj_color_x,
                ha="center", va="top", zorder=9, clip_on=False, bbox=_bbox)
        # Horizontal projection to y-axis
        ax.plot([0, c[0]], [c[1], c[1]],
                color=proj_color_y, lw=0.8, linestyle=proj.linestyle,
                alpha=0.5, zorder=1)
        ax.scatter([0], [c[1]], c=RED, s=18, edgecolors="white",
                   linewidth=0.5, zorder=8, clip_on=False)
        ax.text(-0.04, c[1], f"{lab}ᵧ", fontsize=7, color=proj_color_y,
                ha="right", va="center", zorder=9, clip_on=False, bbox=_bbox)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=FONT["axis_label"], color=ZINC_600)
    ax.set_ylabel("y", fontsize=FONT["axis_label"], color=ZINC_600)
    clean_spines(ax)
    return cm, norm


def _draw_marginal_panels(ax_x, ax_y, centers, sigma, curve_x, curve_y, surface_name, res=200):
    """Draw the two stacked 1D marginal panels."""
    res = 200
    xs = np.linspace(0, 1, res)
    surf = get_surface(surface_name)
    cm = cmap(surface_name)
    line_color = cm(0.7)
    fill_color = cm(0.5)

    proj_color_x = ZINC_500
    proj_color_y = ZINC_400
    bbox = dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="none", alpha=1.0)

    # Y-axis limits: bounded [0,1]; density: shared max so scale differences are visible
    if surf.bounded:
        y_max = surf.vmax
    else:
        y_max = max(curve_x.max(), curve_y.max()) * 1.15

    def _gradient_fill(ax, xs, curve, y_max, cm_obj, norm_fill):
        """Fill under curve with a vertical gradient matching the surface cmap."""
        res_y = 100
        extent = [0, 1, 0, y_max]
        gradient = np.linspace(0, 1, res_y).reshape(-1, 1) * np.ones((1, len(xs)))
        # Scale gradient by the curve value at each x
        curve_norm = curve / y_max if y_max > 0 else curve
        gradient = gradient * curve_norm[None, :]
        ax.imshow(gradient, aspect="auto", origin="lower", extent=extent,
                  cmap=cm_obj, norm=norm_fill, alpha=0.7, zorder=0)
        # Mask above the curve
        ax.fill_between(xs, curve, y_max, color="white", zorder=1)

    fill_norm = Normalize(vmin=0, vmax=1)

    # X marginal (top)
    _gradient_fill(ax_x, xs, curve_x, y_max, cm, fill_norm)
    ax_x.plot(xs, curve_x, color=line_color, linewidth=1.5)
    for ci, (c, lab) in enumerate(zip(centers, LABELS)):
        val_at_x = np.interp(c[0], xs, curve_x)
        ax_x.plot([c[0], c[0]], [0, val_at_x],
                  color=proj_color_x, lw=0.8, linestyle=":", alpha=0.5, zorder=3)
        ax_x.scatter([c[0]], [0], c=RED, s=18, edgecolors="white",
                     linewidth=0.5, zorder=10, clip_on=False)
        ax_x.text(c[0], -0.08 * y_max, f"{lab}ₓ", fontsize=7,
                  color=proj_color_x, ha="center", va="top", zorder=11,
                  clip_on=False, bbox=bbox)
    ax_x.set_xlim(0, 1)
    ax_x.set_ylim(0, y_max)
    ax_x.set_xlabel("x", fontsize=FONT["axis_label"], color=ZINC_600)
    clean_spines(ax_x)

    # Y marginal (bottom)
    _gradient_fill(ax_y, xs, curve_y, y_max, cm, fill_norm)
    ax_y.plot(xs, curve_y, color=line_color, linewidth=1.5)
    for ci, (c, lab) in enumerate(zip(centers, LABELS)):
        val_at_y = np.interp(c[1], xs, curve_y)
        ax_y.plot([c[1], c[1]], [0, val_at_y],
                  color=proj_color_y, lw=0.8, linestyle=":", alpha=0.5, zorder=3)
        ax_y.scatter([c[1]], [0], c=RED, s=18, edgecolors="white",
                     linewidth=0.5, zorder=10, clip_on=False)
        ax_y.text(c[1], -0.08 * y_max, f"{lab}ᵧ", fontsize=7,
                  color=proj_color_y, ha="center", va="top", zorder=11,
                  clip_on=False, bbox=bbox)
    ax_y.set_xlim(0, 1)
    ax_y.set_ylim(0, y_max)
    ax_y.set_xlabel("y", fontsize=FONT["axis_label"], color=ZINC_600)
    clean_spines(ax_y)


def _make_figure(surface_name, title_2d, title_x, title_y, field_2d, curve_x, curve_y, filename):
    apply_style()
    sigma = SIGMA_VIS
    res = 200
    xs = np.linspace(0, 1, res)

    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.5, wspace=0.25,
                          left=0.06, right=0.95, top=0.92, bottom=0.12)
    ax_joint = fig.add_subplot(gs[:, 0])
    ax_mx = fig.add_subplot(gs[0, 1])
    ax_my = fig.add_subplot(gs[1, 1])

    cm, norm = _draw_2d_panel(ax_joint, CENTERS, sigma, field_2d, xs, surface_name)
    subplot_label(ax_joint, title_2d)

    # Colorbar snug against the joint plot
    sm = ScalarMappable(norm=norm, cmap=cm)
    cbar = fig.colorbar(sm, ax=ax_joint, location="right", shrink=0.85, pad=0.05)

    _draw_marginal_panels(ax_mx, ax_my, CENTERS, sigma, curve_x, curve_y, surface_name)
    subplot_label(ax_mx, title_x)
    subplot_label(ax_my, title_y)

    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def main():
    sigma = SIGMA_VIS
    res = 200
    xs = np.linspace(0, 1, res)
    xx, yy = np.meshgrid(xs, xs)

    rho_2d = _density_2d(xx, yy, CENTERS, sigma)
    rho_x = _density_1d(xs, CENTERS[:, 0], sigma)
    rho_y = _density_1d(xs, CENTERS[:, 1], sigma)

    evidence_2d = rho_2d / (1.0 + rho_2d)
    evidence_x = rho_x / (1.0 + rho_x)
    evidence_y = rho_y / (1.0 + rho_y)

    gain_2d = 1.0 / (1.0 + evidence_2d)
    gain_x = 1.0 / (1.0 + evidence_x)
    gain_y = 1.0 / (1.0 + evidence_y)

    _make_figure("density",
                 "Joint density  ρ(x, y)", "Marginal density  ρ(x)", "Marginal density  ρ(y)",
                 rho_2d, rho_x, rho_y,
                 "marginal_joint_density.png")

    _make_figure("evidence",
                 "Joint evidence  E(x, y)", "Marginal evidence  E(x)", "Marginal evidence  E(y)",
                 evidence_2d, evidence_x, evidence_y,
                 "marginal_joint_evidence.png")

    # Figure 3: Evidence gain topology — acquisition surface with Z_new
    _figure_acquisition(xs, xx, yy, gain_2d, sigma)


def _figure_acquisition(xs, xx, yy, gain_2d, sigma):
    """Evidence gain topology computed via real KernelField ΔE."""
    apply_style()
    import torch
    from pred_fab.orchestration.evidence import KernelIndex, KernelFieldEstimator

    kf = KernelFieldEstimator()
    index_old = KernelIndex(CENTERS, np.ones(len(CENTERS)), sigma)

    # Compute real ΔE at each grid point
    res = len(xs)
    gain_real = np.zeros((res, res))
    for j in range(res):
        # Batch all x values for this y row
        row_pts = np.stack([xs, np.full(res, xs[j])], axis=-1)  # (res, 2)
        row_pts_t = torch.from_numpy(row_pts).double().unsqueeze(1)  # (res, 1, 2)
        weights_t = torch.ones(res, 1, dtype=torch.float64)
        de = kf.integrated_evidence_perturbed_batched_joint_torch(
            index_old, row_pts_t, weights_t,
        )
        gain_real[j, :] = de.detach().cpu().numpy()

    cm = cmap("evidence_gain")
    norm = Normalize(vmin=0, vmax=gain_real.max())

    # Find optimal Z_new from real ΔE
    idx_flat = np.argmax(gain_real)
    iy, ix = np.unravel_index(idx_flat, gain_real.shape)
    z_new = np.array([xs[ix], xs[iy]])

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.contourf(xs, xs, gain_real, levels=18, cmap=cm, norm=norm)
    ax.contour(xs, xs, gain_real, levels=8, colors=["white"], linewidths=0.4, alpha=0.5)

    # Existing points
    m = MARKERS["sample"]
    for c, lab in zip(CENTERS, LABELS):
        ax.scatter([c[0]], [c[1]], c=m.color, s=m.size, edgecolors=m.edgecolor,
                   linewidth=m.linewidth, zorder=10)
        ax.text(c[0] + 0.03, c[1] + 0.03, lab, fontsize=FONT["annotation"],
                color="white", fontweight="bold", zorder=11)

    # Proposed Z_new
    from pred_fab.plotting._style import ACCENT_YELLOW
    ax.scatter([z_new[0]], [z_new[1]], c=ACCENT_YELLOW, s=80,
               marker="X", edgecolors="white", linewidth=1.2, zorder=12)
    ax.text(z_new[0] + 0.03, z_new[1] + 0.03, "Z_new", fontsize=FONT["annotation"],
            color=ACCENT_YELLOW, fontweight="bold", zorder=13)

    # Colorbar
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(norm=norm, cmap=cm)
    cbar = fig.colorbar(sm, ax=ax, location="right", shrink=0.85, pad=0.03)
    style_colorbar(cbar)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=FONT["axis_label"], color=ZINC_600)
    ax.set_ylabel("y", fontsize=FONT["axis_label"], color=ZINC_600)
    subplot_label(ax, "Evidence gain  ΔE")
    clean_spines(ax)

    path = PLOTS_DIR / "marginal_joint_evidence_gain.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
