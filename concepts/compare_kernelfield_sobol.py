"""Compare KernelField (χ² quantile shells) vs Sobol as estimators of ∫E dz.

Two figures:

    1. visual_comparison.png — single kernel at (0.5, 0.5) in 2-D. Shows where
       each method places its probes for the same budget. Low n_dirs so the
       geometry is readable.

    2. error_vs_D.png — relative error vs dimensionality at fixed budget.
       Ground truth via per-kernel importance sampling with 200k samples per
       kernel. Test topology: 3 clustered kernels + 3 scattered, σ = 0.10.

Context: the production integrator in pred_fab uses Sobol (see
`PredictionSystem._integrated_evidence`). This script preserves the rationale
for that choice — the shell quadrature was the principal alternative, and at
matched budget Sobol is cheaper per call (O(N·n) vs O(N²·M)) and more
accurate across D at moderate σ. The small-σ regime where KernelField would
degrade more gracefully is handled by the σ ≥ SIGMA_MIN floor in production.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import qmc

from kernel_field import KernelField


# ---------- Config ----------

N_DIRS = 16
QUANTILES = (0.02, 0.2, 0.5, 0.8, 0.98)
N_SOBOL = 512
SIGMA = 0.10
N_KERNELS = 6

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

STEEL = "#0369a1"
EMERALD = "#16a34a"
RED = "#dc2626"
ZINC_300 = "#d4d4d8"
ZINC_500 = "#71717a"
ZINC_700 = "#3f3f46"
ZINC_900 = "#18181b"


# ---------- Shared density ----------

def raw_density(z: np.ndarray, centers: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    """D(z) = Σⱼ wⱼ · ρ(z; zⱼ, σ²I) with ∫ρ dz = 1 in ℝ^D."""
    D = z.shape[-1]
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi)) ** D
    d2 = np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    K = norm * np.exp(-d2 / (2.0 * sigma ** 2))
    return (K * weights[None, :]).sum(axis=-1)


def _clean_axes(ax: Axes) -> None:
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(ZINC_300)
    ax.spines["bottom"].set_color(ZINC_300)
    ax.tick_params(colors=ZINC_500, labelsize=8)


# ---------- Figure 1: visual comparison ----------

def figure_visual(n_dirs: int = 8, sigma: float = 0.15) -> Path:
    """Single kernel at centre — show probe placement for both estimators."""
    center = np.array([0.5, 0.5])

    field = KernelField(
        D=2, sigma=sigma, n_directions=n_dirs,
        radii_mode="chi2_quantile", radii_quantiles=QUANTILES,
    )
    kf_probes = field.probes_at(center)
    kf_budget = kf_probes.shape[0]

    n_sobol = 2 ** int(np.log2(max(kf_budget, 2)))
    sobol = qmc.Sobol(d=2, scramble=True, rng=0).random(n=n_sobol)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.4), constrained_layout=True)

    axes[0].scatter(kf_probes[:, 0], kf_probes[:, 1],
                    c=STEEL, s=8, alpha=0.9, edgecolors="none")
    axes[0].scatter([center[0]], [center[1]],
                    marker="x", c=RED, s=70, linewidths=1.8, zorder=5)
    axes[0].set_title(
        f"KernelField — χ² shells\n{kf_budget} probes (n_dirs={n_dirs})",
        fontsize=10, color=ZINC_700,
    )

    axes[1].scatter(sobol[:, 0], sobol[:, 1],
                    c=EMERALD, s=8, alpha=0.9, edgecolors="none")
    axes[1].scatter([center[0]], [center[1]],
                    marker="x", c=RED, s=70, linewidths=1.8, zorder=5)
    axes[1].set_title(
        f"Sobol — cube uniform\n{n_sobol} probes",
        fontsize=10, color=ZINC_700,
    )

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        _clean_axes(ax)

    fig.suptitle(
        f"Probe placement — single kernel at (0.5, 0.5), σ = {sigma}",
        fontsize=11, color=ZINC_700,
    )
    path = PLOTS_DIR / "visual_comparison.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------- Figure 2: error vs D at fixed budget ----------

def _test_centers(D: int, N: int = N_KERNELS, seed: int = 0) -> np.ndarray:
    """3 clustered near centre + 3 scattered — realistic overlap topology."""
    rng = np.random.default_rng(seed)
    cluster = np.full(D, 0.5) + rng.normal(0.0, 0.08, size=(3, D))
    scattered = rng.uniform(0.12, 0.88, size=(3, D))
    return np.clip(np.vstack([cluster, scattered]), 0.02, 0.98)


def reference_per_kernel_is(centers: np.ndarray, sigma: float, D: int,
                             n_per_kernel: int = 200_000, seed: int = 42) -> float:
    """Ground truth via per-kernel IS — robust at any σ."""
    rng = np.random.default_rng(seed)
    N = len(centers)
    total = 0.0
    for j in range(N):
        eps = rng.standard_normal((n_per_kernel, D)) * sigma
        samples = centers[j] + eps
        in_domain = np.all((samples >= 0.0) & (samples <= 1.0), axis=1)
        D_vals = raw_density(samples, centers, np.ones(N), sigma)
        total += float(np.where(in_domain, 1.0 / (1.0 + D_vals), 0.0).mean())
    return total


def estimate_kf(centers: np.ndarray, sigma: float, n_dirs: int, D: int) -> tuple[float, int]:
    field = KernelField(
        D=D, sigma=sigma, n_directions=n_dirs,
        radii_mode="chi2_quantile", radii_quantiles=QUANTILES,
    )
    probes = field.probes_for_batch(centers)
    flat = probes.reshape(-1, D)
    in_domain = np.all((flat >= 0.0) & (flat <= 1.0), axis=1)
    D_vals = raw_density(flat, centers, np.ones(len(centers)), sigma)
    integrand = 1.0 / (1.0 + D_vals) * in_domain.astype(float)
    M = probes.shape[1]
    per_k = (integrand.reshape(len(centers), M) * field.weights[None, :]).sum(axis=1)
    return float(per_k.sum()), len(centers) * M


def estimate_sobol(centers: np.ndarray, sigma: float, n: int, D: int,
                    seed: int = 0) -> float:
    sobol = qmc.Sobol(d=D, scramble=True, rng=seed).random(n=n)
    D_vals = raw_density(sobol, centers, np.ones(len(centers)), sigma)
    return float((D_vals / (1.0 + D_vals)).mean())


def figure_error_vs_D(
    Ds: tuple[int, ...] = (2, 3, 4, 5, 6, 7),
    n_dirs: int = N_DIRS,
    n_sobol: int = N_SOBOL,
    sigma: float = SIGMA,
) -> Path:
    kf_errs, sb_errs, kf_budgets = [], [], []
    for D in Ds:
        centers = _test_centers(D)
        truth = reference_per_kernel_is(centers, sigma, D)
        kf_val, budget = estimate_kf(centers, sigma, n_dirs, D)
        sb_val = estimate_sobol(centers, sigma, n_sobol, D)
        denom = abs(truth) + 1e-12
        kf_errs.append(100 * (kf_val - truth) / denom)
        sb_errs.append(100 * (sb_val - truth) / denom)
        kf_budgets.append(budget)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    kf_budget_lbl = (f"{kf_budgets[0]}" if len(set(kf_budgets)) == 1
                     else f"{kf_budgets[0]}–{kf_budgets[-1]}")
    ax.plot(Ds, kf_errs, color=STEEL, lw=1.8, marker="o", ms=6,
            label=f"KernelField  (n_dirs={n_dirs}, {kf_budget_lbl} probes)")
    ax.plot(Ds, sb_errs, color=EMERALD, lw=1.8, marker="s", ms=6,
            label=f"Sobol  (n={n_sobol} probes, fixed)")
    ax.axhline(0, color=ZINC_300, lw=0.8, ls="--")

    ax.set_xlabel("D (dimensions)")
    ax.set_ylabel("relative error [%]")
    ax.set_xticks(list(Ds))
    ax.legend(loc="best", fontsize=8, frameon=False)
    _clean_axes(ax)

    fig.suptitle(
        f"Relative error vs D — matched budget   (σ={sigma}, N={N_KERNELS} kernels)",
        fontsize=11, color=ZINC_700,
    )
    path = PLOTS_DIR / "error_vs_D.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------- Main ----------

if __name__ == "__main__":
    print("1/2  visual comparison (single kernel, low n_dirs) ...")
    p1 = figure_visual()
    print(f"      saved: {p1}")
    print("2/2  error vs D at fixed budget ...")
    p2 = figure_error_vs_D()
    print(f"      saved: {p2}")
