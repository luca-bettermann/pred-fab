"""Marginal-joint integration decomposition — the ANOVA KernelField.

Shows how the evidence integral decomposes into independent 1D marginal
integrals (one per dimension) and one D-dimensional joint integral:

    E = (1/2D) Σ_d ∫₀¹ 1/(1+ρ_d) dx_d  +  (1/2) ∫ 1/(1+ρ) dx

Three panels for a 2D example with 3 kernels:
    Left   — Marginal integral on x: 1D density + shaded evidence area
    Centre — Marginal integral on y: 1D density + shaded evidence area
    Right  — Joint integral: 2D contour with shell probes

Annotated with computed values showing how marginals detect per-dimension
gaps that the joint integral misses.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from _style import (
    apply_style, clean_spines, subplot_label,
    STEEL_500, EMERALD_500, ZINC_200, ZINC_300, ZINC_400, ZINC_600, ZINC_700,
)
from _config import SIGMA

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Three kernels placed to illustrate the key insight:
# Kernels A and B share the same x-value (0.3) but differ in y.
# Kernel C is at a different x. The marginal x-integral sees the
# A/B overlap; the joint integral doesn't (they're far in 2D).
CENTERS = np.array([
    [0.30, 0.25],   # A
    [0.30, 0.75],   # B — same x as A!
    [0.75, 0.50],   # C
])
LABELS = ["A", "B", "C"]
SIGMA_VIS = 0.08  # slightly larger than production for visibility


def _density_1d(xs: np.ndarray, centers_1d: np.ndarray, sigma: float) -> np.ndarray:
    """1D Gaussian density from multiple kernels."""
    rho = np.zeros_like(xs)
    for c in centers_1d:
        rho += np.exp(-(xs - c) ** 2 / (2 * sigma ** 2))
    return rho


def _density_2d(xx: np.ndarray, yy: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """2D isotropic Gaussian density from multiple kernels."""
    rho = np.zeros_like(xx)
    for c in centers:
        d2 = (xx - c[0]) ** 2 + (yy - c[1]) ** 2
        rho += np.exp(-d2 / (2 * sigma ** 2))
    return rho


def main() -> None:
    apply_style()
    sigma = SIGMA_VIS
    res = 200
    xs = np.linspace(0, 1, res)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # === Panel 1: Marginal integral on x ===
    ax = axes[0]
    rho_x = _density_1d(xs, CENTERS[:, 0], sigma)
    evidence_x = 1.0 / (1.0 + rho_x)

    ax.fill_between(xs, 0, rho_x, alpha=0.15, color=STEEL_500, label="ρ_x (density)")
    ax.plot(xs, rho_x, color=STEEL_500, linewidth=1.5)
    ax.fill_between(xs, 0, evidence_x, alpha=0.2, color=EMERALD_500, label="1/(1+ρ_x)")
    ax.plot(xs, evidence_x, color=EMERALD_500, linewidth=1.5)

    for c, lab in zip(CENTERS, LABELS):
        ax.axvline(c[0], color=ZINC_400, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.text(c[0], max(rho_x) * 1.05, lab, ha="center", fontsize=9, color=ZINC_600)

    # Highlight the overlap: A and B at x=0.3
    ax.annotate("A & B overlap\n(same x)", xy=(0.30, rho_x[int(0.30 * res)]),
                xytext=(0.48, max(rho_x) * 0.85),
                arrowprops=dict(arrowstyle="->", color=ZINC_600, lw=1),
                fontsize=8, color=ZINC_600, ha="center")

    E_x = float(np.trapezoid(evidence_x, xs))
    ax.set_xlabel("x (dimension 1)", fontsize=9, color=ZINC_700)
    ax.set_ylabel("Value", fontsize=9, color=ZINC_700)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    subplot_label(ax, f"Marginal x:  E_x = {E_x:.3f}")
    clean_spines(ax)

    # === Panel 2: Marginal integral on y ===
    ax = axes[1]
    rho_y = _density_1d(xs, CENTERS[:, 1], sigma)
    evidence_y = 1.0 / (1.0 + rho_y)

    ax.fill_between(xs, 0, rho_y, alpha=0.15, color=STEEL_500, label="ρ_y (density)")
    ax.plot(xs, rho_y, color=STEEL_500, linewidth=1.5)
    ax.fill_between(xs, 0, evidence_y, alpha=0.2, color=EMERALD_500, label="1/(1+ρ_y)")
    ax.plot(xs, evidence_y, color=EMERALD_500, linewidth=1.5)

    for c, lab in zip(CENTERS, LABELS):
        ax.axvline(c[1], color=ZINC_400, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.text(c[1], max(rho_y) * 1.05, lab, ha="center", fontsize=9, color=ZINC_600)

    # Highlight: A, B, C are all at different y → good marginal coverage
    ax.annotate("All separated\n(good y-coverage)", xy=(0.50, evidence_y[int(0.50 * res)]),
                xytext=(0.50, max(rho_y) * 0.85),
                fontsize=8, color=ZINC_600, ha="center")

    E_y = float(np.trapezoid(evidence_y, xs))
    ax.set_xlabel("y (dimension 2)", fontsize=9, color=ZINC_700)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    subplot_label(ax, f"Marginal y:  E_y = {E_y:.3f}")
    clean_spines(ax)

    # === Panel 3: Joint integral (2D) ===
    ax = axes[2]
    xx, yy = np.meshgrid(xs, xs)
    rho_2d = _density_2d(xx, yy, CENTERS, sigma)
    evidence_2d = 1.0 / (1.0 + rho_2d)

    levels = np.linspace(0, rho_2d.max(), 12)
    ax.contourf(xx, yy, rho_2d, levels=levels, cmap="Blues", alpha=0.6)
    ax.contour(xx, yy, rho_2d, levels=levels[1::2], colors=[ZINC_300], linewidths=0.5)

    for c, lab in zip(CENTERS, LABELS):
        ax.scatter(c[0], c[1], s=60, c="white", edgecolors=STEEL_500, linewidth=1.5, zorder=5)
        ax.text(c[0] + 0.04, c[1] + 0.04, lab, fontsize=9, color=ZINC_600, zorder=6)

    # Draw shell probes around kernel A
    from pred_fab.orchestration.evidence import KernelFieldEstimator
    kf = KernelFieldEstimator()
    offsets_np, _, _ = kf._probes_weights_self(2, sigma)
    for off in offsets_np[1:]:  # skip centre
        ax.plot(CENTERS[0, 0] + off[0], CENTERS[0, 1] + off[1], ".",
                color=ZINC_400, markersize=3, alpha=0.5, zorder=4)

    E_joint = float(np.trapezoid(np.trapezoid(evidence_2d, xs, axis=1), xs))
    ax.set_xlabel("x", fontsize=9, color=ZINC_700)
    ax.set_ylabel("y", fontsize=9, color=ZINC_700)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    subplot_label(ax, f"Joint (2D):  E_joint = {E_joint:.3f}")
    clean_spines(ax)

    # === Bottom annotation: the formula ===
    E_marginal = (E_x + E_y) / 2
    E_total = (E_marginal + E_joint) / 2
    fig.text(0.5, -0.02,
             f"E = (E_marginal + E_joint) / 2 = ({E_marginal:.3f} + {E_joint:.3f}) / 2 = {E_total:.3f}\n"
             f"Marginal detects A/B x-overlap (E_x = {E_x:.3f} < E_y = {E_y:.3f})  ·  "
             f"Joint sees them as well-separated in 2D",
             ha="center", fontsize=9, color=ZINC_600)

    fig.tight_layout()
    path = PLOTS_DIR / "marginal_joint_integration.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
