"""Three-panel concept sequence: density → evidence integration → evidence gain.

Panel 1 (density):       Joint density ρ(x,y) from existing points. No probes,
                          no candidate — pure "what we know."
Panel 2 (evidence):      Joint evidence E(x,y) with marginal evidence E(x), E(y).
                          KernelField probes shown ONLY around the candidate z
                          to illustrate the integration mechanism. Existing points
                          projected onto axes.
Panel 3 (evidence gain): ΔE topology — sweeps the candidate across the space.
                          The candidate from panel 2 is at the optimum (Z_new).

All three use the same kernel layout (CENTERS) and candidate position.
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
from _config import SIGMA, CONCEPT_POINTS, CONCEPT_LABELS, SIGMA_VIS, make_topology_marginal_layout
from panels import draw_proposal
from pred_fab.orchestration.evidence import DEFAULT_RADII, KernelFieldEstimator
from pred_fab.plotting._style import (
    SURFACES, MARKERS, LINES, FILL_ALPHA, FONT, RED, ACCENT_YELLOW,
    surface as get_surface,
)

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

CENTERS = CONCEPT_POINTS
LABELS = CONCEPT_LABELS


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


def _draw_existing_points(ax, centers, labels, label_color="white"):
    """Scatter existing points with labels."""
    m = MARKERS["sample"]
    for c, lab in zip(centers, labels):
        ax.scatter([c[0]], [c[1]], c=m.color, s=m.size, edgecolors=m.edgecolor,
                   linewidth=m.linewidth, zorder=10)
        dx = 0.05 if lab != "B" else -0.07
        ax.text(c[0] + dx, c[1] + 0.04, lab, fontsize=8,
                color=label_color, zorder=11)


def _draw_projections(ax, centers, labels):
    """Project existing points onto x and y axes with labelled dots."""
    proj = LINES["projection"]
    bbox = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="none", alpha=1.0)
    for c, lab in zip(centers, labels):
        ax.plot([c[0], c[0]], [0, c[1]],
                color=ZINC_500, lw=0.8, linestyle=proj.linestyle, alpha=0.5, zorder=1)
        ax.scatter([c[0]], [0], c=RED, s=18, edgecolors="white",
                   linewidth=0.5, zorder=8, clip_on=False)
        ax.text(c[0], -0.05, f"{lab}ₓ", fontsize=7, color=ZINC_500,
                ha="center", va="top", zorder=9, clip_on=False, bbox=bbox)
        ax.plot([0, c[0]], [c[1], c[1]],
                color=ZINC_400, lw=0.8, linestyle=proj.linestyle, alpha=0.5, zorder=1)
        ax.scatter([0], [c[1]], c=RED, s=18, edgecolors="white",
                   linewidth=0.5, zorder=8, clip_on=False)
        ax.text(-0.05, c[1], f"{lab}ᵧ", fontsize=7, color=ZINC_400,
                ha="right", va="center", zorder=9, clip_on=False, bbox=bbox)


def _draw_candidate_kernelfield(ax, z_candidate, sigma, label="z"):
    """Draw KernelField probes ONLY around the candidate — the integration mechanism."""
    kf = KernelFieldEstimator()
    offsets, _, _ = kf._probes_weights_self(2, sigma)
    probes = z_candidate + offsets

    probe_d2 = np.sum(offsets ** 2, axis=-1)
    probe_density = np.exp(-probe_d2 / (2 * sigma ** 2))

    add_kernel_radii_2d(ax, z_candidate, sigma, DEFAULT_RADII,
                        color_scale=True, alpha_max=0.85, lw=0.8)
    cm = cmap("evidence")
    probe_vis = np.clip(probe_density, 0.45, 1.0)
    norm = Normalize(vmin=0.0, vmax=1.0)
    ax.scatter(probes[:, 0], probes[:, 1],
               c=probe_vis, cmap=cm, norm=norm,
               s=18, alpha=0.95, edgecolors="none", zorder=6)
    ax.scatter([z_candidate[0]], [z_candidate[1]], c=RED, s=34,
               edgecolors="none", zorder=10)
    ax.annotate(label, xy=(z_candidate[0], z_candidate[1]),
                xytext=(10, 8), textcoords="offset points",
                fontsize=8, color=RED)


def _draw_marginal_panels(ax_x, ax_y, centers, labels, sigma, curve_x, curve_y, surface_name):
    """1D marginal panels with gradient fill and projection markers."""
    res = 200
    xs = np.linspace(0, 1, res)
    cm = cmap(surface_name)
    line_color = cm(0.7)
    surf = get_surface(surface_name)
    bbox = dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="none", alpha=1.0)

    y_max = surf.vmax if surf.bounded else max(curve_x.max(), curve_y.max()) * 1.15

    def _gradient_fill(ax, xs_arr, curve, y_max_val, cm_obj):
        res_y = 100
        extent = [0, 1, 0, y_max_val]
        gradient = np.linspace(0, 1, res_y).reshape(-1, 1) * np.ones((1, len(xs_arr)))
        curve_norm = curve / y_max_val if y_max_val > 0 else curve
        gradient = gradient * curve_norm[None, :]
        norm_fill = Normalize(vmin=0, vmax=1)
        ax.imshow(gradient, aspect="auto", origin="lower", extent=extent,
                  cmap=cm_obj, norm=norm_fill, alpha=0.7, zorder=0)
        ax.fill_between(xs_arr, curve, y_max_val, color="white", zorder=1)

    _gradient_fill(ax_x, xs, curve_x, y_max, cm)
    ax_x.plot(xs, curve_x, color=line_color, linewidth=1.5)
    for c, lab in zip(centers, labels):
        val = np.interp(c[0], xs, curve_x)
        ax_x.plot([c[0], c[0]], [0, val], color=ZINC_500, lw=0.8, linestyle=":", alpha=0.5, zorder=3)
        ax_x.scatter([c[0]], [0], c=RED, s=18, edgecolors="white", linewidth=0.5, zorder=10, clip_on=False)
        ax_x.text(c[0], -0.08 * y_max, f"{lab}ₓ", fontsize=7, color=ZINC_500,
                  ha="center", va="top", zorder=11, clip_on=False, bbox=bbox)
    ax_x.set_xlim(0, 1)
    ax_x.set_ylim(0, y_max)
    ax_x.set_xlabel("x", fontsize=FONT["axis_label"], color=ZINC_600)
    clean_spines(ax_x)

    _gradient_fill(ax_y, xs, curve_y, y_max, cm)
    ax_y.plot(xs, curve_y, color=line_color, linewidth=1.5)
    for c, lab in zip(centers, labels):
        val = np.interp(c[1], xs, curve_y)
        ax_y.plot([c[1], c[1]], [0, val], color=ZINC_400, lw=0.8, linestyle=":", alpha=0.5, zorder=3)
        ax_y.scatter([c[1]], [0], c=RED, s=18, edgecolors="white", linewidth=0.5, zorder=10, clip_on=False)
        ax_y.text(c[1], -0.08 * y_max, f"{lab}ᵧ", fontsize=7, color=ZINC_400,
                  ha="center", va="top", zorder=11, clip_on=False, bbox=bbox)
    ax_y.set_xlim(0, 1)
    ax_y.set_ylim(0, y_max)
    ax_y.set_xlabel("y", fontsize=FONT["axis_label"], color=ZINC_600)
    clean_spines(ax_y)


def _setup_2d_axes(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=FONT["axis_label"], color=ZINC_600)
    ax.set_ylabel("y", fontsize=FONT["axis_label"], color=ZINC_600)
    clean_spines(ax)


def main():
    sigma = SIGMA_VIS
    res = 200
    xs = np.linspace(0, 1, res)
    xx, yy = np.meshgrid(xs, xs)

    rho_2d = _density_2d(xx, yy, CENTERS, sigma)
    evidence_2d = rho_2d / (1.0 + rho_2d)

    rho_x = _density_1d(xs, CENTERS[:, 0], sigma)
    rho_y = _density_1d(xs, CENTERS[:, 1], sigma)
    evidence_x = rho_x / (1.0 + rho_x)
    evidence_y = rho_y / (1.0 + rho_y)

    # Compute Z_new via real ANOVA evidence gain
    from pred_fab.orchestration.evidence import _evidence_gain_grid_from_centers

    xs_gain, ys_gain, gain_grid = _evidence_gain_grid_from_centers(
        CENTERS, np.ones(len(CENTERS)), sigma, resolution=80,
    )

    idx_flat = np.argmax(gain_grid)
    iy, ix = np.unravel_index(idx_flat, gain_grid.shape)
    z_new = np.array([xs_gain[ix], ys_gain[iy]])

    # ================================================================
    # Figure 1: Joint density — what we know
    # ================================================================
    apply_style()
    fig1, ax1 = plt.subplots(figsize=(6, 5.5))
    cm_d = cmap("density")
    norm_d = Normalize(vmin=0.0, vmax=float(rho_2d.max()))
    ax1.set_facecolor("white")
    levels_d = np.linspace(0.02, float(rho_2d.max()), 18)
    ax1.contourf(xs, xs, rho_2d, levels=levels_d, cmap=cm_d, norm=norm_d, extend="neither")
    ax1.contour(xs, xs, rho_2d, levels=8, colors=[ZINC_300], linewidths=0.4)
    _draw_existing_points(ax1, CENTERS, LABELS)
    subplot_label(ax1, r"Joint density  $\rho(x, y)$")
    _setup_2d_axes(ax1)
    sm = ScalarMappable(norm=norm_d, cmap=cm_d)
    cbar = fig1.colorbar(sm, ax=ax1, shrink=0.85, pad=0.06)
    style_colorbar(cbar)
    path1 = PLOTS_DIR / "01_density.png"
    fig1.savefig(path1, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {path1}")

    # ================================================================
    # Figure 2: Evidence integration — how we measure
    # ================================================================
    apply_style()
    fig2, ax_joint, ax_mx, ax_my = make_topology_marginal_layout()

    cm_e = cmap("evidence")
    norm_e = Normalize(vmin=0, vmax=evidence_2d.max())
    ax_joint.set_facecolor("white")
    levels_e = np.linspace(0.02, float(evidence_2d.max()), 18)
    ax_joint.contourf(xs, xs, evidence_2d, levels=levels_e, cmap=cm_e, norm=norm_e, extend="neither")
    ax_joint.contour(xs, xs, evidence_2d, levels=8, colors=[ZINC_300], linewidths=0.4)
    _draw_existing_points(ax_joint, CENTERS, LABELS)
    _draw_projections(ax_joint, CENTERS, LABELS)
    _draw_candidate_kernelfield(ax_joint, z_new, sigma, label="z")
    subplot_label(ax_joint, r"Joint evidence  $E(x, y)$")
    _setup_2d_axes(ax_joint)
    sm_e = ScalarMappable(norm=norm_e, cmap=cm_e)
    cbar_e = fig2.colorbar(sm_e, ax=ax_joint, location="right", shrink=0.85, pad=0.06)
    style_colorbar(cbar_e)

    _draw_marginal_panels(ax_mx, ax_my, CENTERS, LABELS, sigma, evidence_x, evidence_y, "evidence")
    subplot_label(ax_mx, r"Marginal evidence  $E(x)$")
    subplot_label(ax_my, r"Marginal evidence  $E(y)$")

    path2 = PLOTS_DIR / "02_evidence_integration.png"
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {path2}")

    # ================================================================
    # Figure 3: Evidence gain — where to place
    # ================================================================
    apply_style()
    fig3, ax3 = plt.subplots(figsize=(6, 5.5))
    cm_g = cmap("evidence_gain")
    norm_g = Normalize(vmin=gain_grid.min(), vmax=gain_grid.max())
    ax3.contourf(xs_gain, xs_gain, gain_grid, levels=24, cmap=cm_g, norm=norm_g)
    ax3.contour(xs_gain, xs_gain, gain_grid, levels=12, colors=["white"], linewidths=0.3, alpha=0.4)

    gain_dark = cmap("evidence_gain")(0.95)
    _draw_existing_points(ax3, CENTERS, LABELS, label_color=gain_dark)

    draw_proposal(ax3, z_new[0], z_new[1])

    subplot_label(ax3, r"Evidence gain  $\Delta E$")
    _setup_2d_axes(ax3)
    sm_g = ScalarMappable(norm=norm_g, cmap=cm_g)
    cbar_g = fig3.colorbar(sm_g, ax=ax3, shrink=0.85, pad=0.06)
    style_colorbar(cbar_g)

    path3 = PLOTS_DIR / "03_evidence_gain.png"
    fig3.savefig(path3, dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {path3}")


if __name__ == "__main__":
    main()
