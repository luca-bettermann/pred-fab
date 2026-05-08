"""Marginal-joint integration — two concept figures.

Figure 1 (density): What the evidence field looks like.
    Left:  2D joint density with KernelField probes (red centres, probe dots)
           + subtle projection lines to axes
    Right: Two stacked 1D marginal density plots (x on top, y on bottom)

Figure 2 (evidence): What the optimizer sees.
    Left:  2D evidence gain surface 1/(1+ρ)
    Right: Two stacked 1D marginal evidence curves 1/(1+ρ_d)

Both use three kernels where A and B share the same x-value — the
marginal integral detects this overlap while the joint integral doesn't.
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
    STEEL_500, EMERALD_500, ZINC_200, ZINC_300, ZINC_400, ZINC_500, ZINC_600, RED,
)
from _config import SIGMA
from pred_fab.orchestration.evidence import DEFAULT_RADII, KernelFieldEstimator

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

CENTERS = np.array([
    [0.30, 0.20],   # A
    [0.30, 0.75],   # B — same x as A
    [0.75, 0.50],   # C
])
LABELS = ["A", "B", "C"]
SIGMA_VIS = 0.08


def _density_1d(xs: np.ndarray, centers_1d: np.ndarray, sigma: float) -> np.ndarray:
    rho = np.zeros_like(xs)
    for c in centers_1d:
        rho += np.exp(-(xs - c) ** 2 / (2 * sigma ** 2))
    return rho


def _density_2d(xx: np.ndarray, yy: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    rho = np.zeros_like(xx)
    for c in centers:
        d2 = (xx - c[0]) ** 2 + (yy - c[1]) ** 2
        rho += np.exp(-d2 / (2 * sigma ** 2))
    return rho


def _draw_joint_panel(ax, centers, sigma, field, xs, cm, norm, show_evidence: bool = False):
    """Draw the 2D panel — density or evidence."""
    label_text = "Evidence gain  1/(1+ρ)" if show_evidence else "Joint density  ρ(x, y)"
    cmap_name = cm

    levels = 18
    ax.contourf(xs, xs, field, levels=levels, cmap=cmap_name, norm=norm, alpha=0.5)
    ax.contour(xs, xs, field, levels=8, colors=[ZINC_300], linewidths=0.4)

    # KernelField probes around each centre
    kf = KernelFieldEstimator()
    offsets, _, _ = kf._probes_weights_self(2, sigma)
    probe_density = np.exp(-np.sum(offsets ** 2, axis=-1) / (2 * sigma ** 2))
    probe_cm = cmap("evidence")
    probe_norm = Normalize(vmin=0, vmax=1)

    for ci, (c, lab) in enumerate(zip(centers, LABELS)):
        # Shell probes
        pts = c + offsets
        ax.scatter(pts[1:, 0], pts[1:, 1],
                   c=probe_density[1:], cmap=probe_cm, norm=probe_norm,
                   s=8, alpha=0.5, edgecolors="none", zorder=4)
        # Centre
        ax.scatter([c[0]], [c[1]], c=RED, s=40, edgecolors="white",
                   linewidth=0.8, zorder=10)
        # Label
        offset_x = 0.04 if ci != 1 else -0.06
        ax.text(c[0] + offset_x, c[1] + 0.04, lab, fontsize=9,
                color=ZINC_600, fontweight="bold", zorder=11)

    # Projection lines to axes
    for c in centers:
        ax.plot([c[0], c[0]], [0, c[1]], color=ZINC_300, lw=0.6,
                linestyle=":", alpha=0.6, zorder=1)
        ax.plot([0, c[0]], [c[1], c[1]], color=ZINC_400, lw=0.6,
                linestyle=":", alpha=0.6, zorder=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=9, color=ZINC_600)
    ax.set_ylabel("y", fontsize=9, color=ZINC_600)
    subplot_label(ax, label_text)
    clean_spines(ax)


def _draw_marginal_panels(ax_x, ax_y, centers, sigma, show_evidence: bool = False):
    """Draw the two stacked 1D marginal panels."""
    res = 200
    xs = np.linspace(0, 1, res)

    rho_x = _density_1d(xs, centers[:, 0], sigma)
    rho_y = _density_1d(xs, centers[:, 1], sigma)

    if show_evidence:
        curve_x = 1.0 / (1.0 + rho_x)
        curve_y = 1.0 / (1.0 + rho_y)
        fill_color = EMERALD_500
        line_color = EMERALD_500
        label_x = "Marginal evidence  1/(1+ρ_x)"
        label_y = "Marginal evidence  1/(1+ρ_y)"
    else:
        curve_x = rho_x
        curve_y = rho_y
        fill_color = STEEL_500
        line_color = STEEL_500
        label_x = "Marginal density  ρ_x"
        label_y = "Marginal density  ρ_y"

    # X marginal (top)
    ax_x.fill_between(xs, 0, curve_x, alpha=0.15, color=fill_color)
    ax_x.plot(xs, curve_x, color=line_color, linewidth=1.5)
    for c, lab in zip(centers, LABELS):
        ax_x.scatter([c[0]], [0], c=RED, s=30, edgecolors="white",
                     linewidth=0.6, zorder=10, clip_on=False)
    ax_x.set_xlim(0, 1)
    ax_x.set_ylim(0, None)
    ax_x.set_xlabel("x", fontsize=9, color=ZINC_600)
    subplot_label(ax_x, label_x)
    clean_spines(ax_x)

    # Y marginal (bottom)
    ax_y.fill_between(xs, 0, curve_y, alpha=0.15, color=fill_color)
    ax_y.plot(xs, curve_y, color=line_color, linewidth=1.5)
    for c, lab in zip(centers, LABELS):
        ax_y.scatter([c[1]], [0], c=RED, s=30, edgecolors="white",
                     linewidth=0.6, zorder=10, clip_on=False)
    ax_y.set_xlim(0, 1)
    ax_y.set_ylim(0, None)
    ax_y.set_xlabel("y", fontsize=9, color=ZINC_600)
    subplot_label(ax_y, label_y)
    clean_spines(ax_y)


def figure_density() -> Path:
    """Figure 1: density field — what the evidence field looks like."""
    apply_style()
    sigma = SIGMA_VIS
    res = 200
    xs = np.linspace(0, 1, res)
    xx, yy = np.meshgrid(xs, xs)
    rho_2d = _density_2d(xx, yy, CENTERS, sigma)

    cm = cmap("evidence")
    norm = Normalize(vmin=0, vmax=rho_2d.max())

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.35, wspace=0.3)
    ax_joint = fig.add_subplot(gs[:, 0])
    ax_mx = fig.add_subplot(gs[0, 1])
    ax_my = fig.add_subplot(gs[1, 1])

    _draw_joint_panel(ax_joint, CENTERS, sigma, rho_2d, xs, cm, norm, show_evidence=False)
    _draw_marginal_panels(ax_mx, ax_my, CENTERS, sigma, show_evidence=False)

    path = PLOTS_DIR / "marginal_joint_density.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def figure_evidence() -> Path:
    """Figure 2: evidence gain — what the optimizer sees."""
    apply_style()
    sigma = SIGMA_VIS
    res = 200
    xs = np.linspace(0, 1, res)
    xx, yy = np.meshgrid(xs, xs)
    rho_2d = _density_2d(xx, yy, CENTERS, sigma)
    evidence_2d = 1.0 / (1.0 + rho_2d)

    cm = "YlGn"
    norm = Normalize(vmin=evidence_2d.min(), vmax=1.0)

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.35, wspace=0.3)
    ax_joint = fig.add_subplot(gs[:, 0])
    ax_mx = fig.add_subplot(gs[0, 1])
    ax_my = fig.add_subplot(gs[1, 1])

    _draw_joint_panel(ax_joint, CENTERS, sigma, evidence_2d, xs, cm, norm, show_evidence=True)
    _draw_marginal_panels(ax_mx, ax_my, CENTERS, sigma, show_evidence=True)

    path = PLOTS_DIR / "marginal_joint_evidence.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


if __name__ == "__main__":
    figure_density()
    figure_evidence()
