"""Compare KernelField (χ² quantile shells) vs Sobol as estimators of ∫E dz.

Two publication figures:

    1. probe_placement.png — 3-D unit cube with a single kernel at the
       centre. Each probe coloured by its evidence value E = D/(1+D).
       KernelField's probes cluster near the centre and read saturated
       orange; Sobol's scatter fills the cube and mostly reads grey,
       with only the handful of points close to the kernel lighting up.
       Same σ, matched probe budget, identical camera angle.

    2. error_convergence.png — |relative error| vs budget at fixed σ,
       one panel per D ∈ {2..5}. Ground truth via dense Sobol reference
       (n = 100k). Shows Sobol converging while KernelField plateaus once
       its shells saturate.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import qmc

from kernel_field import KernelField
from _style import (
    apply_style, clean_spines, clean_3d_panes,
    evidence_cmap,
    ZINC_200, ZINC_300, ZINC_500, ZINC_600, ZINC_700,
    STEEL_500, EMERALD_500, RED,
)


QUANTILES = (0.02, 0.2, 0.5, 0.8, 0.98)
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ---------- Shared density ----------

def raw_density(z: np.ndarray, centers: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    """D(z) = Σⱼ wⱼ · ρ(z; zⱼ, σ²I), with ∫ρ dz = 1 in ℝ^D."""
    D = z.shape[-1]
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi)) ** D
    d2 = np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    K = norm * np.exp(-d2 / (2.0 * sigma ** 2))
    return (K * weights[None, :]).sum(axis=-1)


def evidence_at(probes: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    D_vals = raw_density(probes, centers, np.ones(len(centers)), sigma)
    return D_vals / (1.0 + D_vals)


# ---------- Cube wireframe for 3-D panels ----------

def _cube_wireframe(ax, color: str = ZINC_300, lw: float = 0.6, alpha: float = 0.35) -> None:
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for (i, j) in edges:
        p1, p2 = corners[i], corners[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=color, lw=lw, alpha=alpha, zorder=-1)


# ---------- Figure 1: probe placement (3-D) ----------

def figure_probe_placement(
    n_dirs: int = 12,
    sigma: float = 0.08,
    seed: int = 0,
) -> Path:
    apply_style()
    center = np.array([0.5, 0.5, 0.5])
    centers_arr = center[None, :]

    field = KernelField(
        D=3, sigma=sigma, n_directions=n_dirs,
        radii_mode="chi2_quantile", radii_quantiles=QUANTILES,
    )
    kf_probes = field.probes_at(center)
    kf_budget = kf_probes.shape[0]

    n_sobol = 2 ** int(round(np.log2(kf_budget)))
    sobol = qmc.Sobol(d=3, scramble=True, rng=seed).random(n=n_sobol)

    E_kf = evidence_at(kf_probes, centers_arr, sigma)
    E_sb = evidence_at(sobol, centers_arr, sigma)

    cmap = evidence_cmap()
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(12.5, 6.2))
    fig.subplots_adjust(left=0.02, right=0.88, top=0.92, bottom=0.04, wspace=0.02)

    panels = [
        (121, kf_probes, E_kf,
         f"KernelField — χ² shells\n{kf_budget} probes · n_dirs = {n_dirs}"),
        (122, sobol, E_sb,
         f"Sobol — cube uniform\n{n_sobol} probes"),
    ]

    for spec, probes, evals, title in panels:
        ax = fig.add_subplot(spec, projection="3d")
        _cube_wireframe(ax)

        ax.scatter(
            probes[:, 0], probes[:, 1], probes[:, 2],
            c=evals, cmap=cmap, norm=norm,
            s=16, alpha=0.95, edgecolors="none",
            depthshade=False, zorder=5,
        )
        ax.scatter(
            [center[0]], [center[1]], [center[2]],
            marker="o", c=RED, s=52, edgecolors="none",
            depthshade=False, zorder=10,
        )

        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_zticks([0, 0.5, 1])
        ax.set_xlabel("z₁"); ax.set_ylabel("z₂"); ax.set_zlabel("z₃")
        ax.view_init(elev=20, azim=35)
        clean_3d_panes(ax)
        ax.set_title(title, fontsize=10.5, color=ZINC_700, pad=-8)

    cax = fig.add_axes([0.905, 0.22, 0.016, 0.56])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(colors=ZINC_500, labelsize=8)
    cbar.set_label("evidence  E(z) = D(z) / (1 + D(z))", color=ZINC_600, fontsize=9)
    cbar.outline.set_edgecolor(ZINC_300)
    cbar.outline.set_linewidth(0.6)

    fig.suptitle(
        f"Probe placement — single kernel at (0.5, 0.5, 0.5), σ = {sigma}",
        fontsize=12.5, color=ZINC_700, y=0.985,
    )

    path = PLOTS_DIR / "probe_placement.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------- Figure 2: error convergence ----------

def _test_centers(D: int, N: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cluster = np.full(D, 0.5) + rng.normal(0.0, 0.08, size=(3, D))
    scattered = rng.uniform(0.12, 0.88, size=(3, D))
    return np.clip(np.vstack([cluster, scattered]), 0.02, 0.98)


def reference_dense_sobol(
    centers: np.ndarray, sigma: float, D: int,
    n: int = 100_000, seed: int = 42,
) -> float:
    """Ground truth via dense scrambled Sobol. Reliable for σ ≥ ~0.05."""
    sobol = qmc.Sobol(d=D, scramble=True, rng=seed).random(n=n)
    D_vals = raw_density(sobol, centers, np.ones(len(centers)), sigma)
    return float((D_vals / (1.0 + D_vals)).mean())


def estimate_kf(
    centers: np.ndarray, sigma: float, n_dirs: int, D: int,
) -> tuple[float, int]:
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


def estimate_sobol_at_budget(
    centers: np.ndarray, sigma: float, n: int, D: int, seed: int = 0,
) -> float:
    sobol = qmc.Sobol(d=D, scramble=True, rng=seed).random(n=n)
    D_vals = raw_density(sobol, centers, np.ones(len(centers)), sigma)
    return float((D_vals / (1.0 + D_vals)).mean())


def figure_error_convergence(
    sigma: float = 0.10,
    Ds: tuple[int, ...] = (2, 3, 4, 5),
    n_dirs_values: tuple[int, ...] = (4, 8, 16, 32, 64),
) -> Path:
    apply_style()

    results = []
    for D in Ds:
        centers = _test_centers(D)
        truth = reference_dense_sobol(centers, sigma, D)
        denom = abs(truth) + 1e-12
        kf_series: dict[str, list[float]] = {"budget": [], "err": []}
        sb_series: dict[str, list[float]] = {"budget": [], "err": []}
        for nd in n_dirs_values:
            kf_val, budget = estimate_kf(centers, sigma, nd, D)
            n_sob = 2 ** int(np.ceil(np.log2(max(budget, 2))))
            sb_val = estimate_sobol_at_budget(centers, sigma, n_sob, D)
            kf_series["budget"].append(float(budget))
            kf_series["err"].append(max(100.0 * abs(kf_val - truth) / denom, 1e-2))
            sb_series["budget"].append(float(n_sob))
            sb_series["err"].append(max(100.0 * abs(sb_val - truth) / denom, 1e-2))
        results.append((D, kf_series, sb_series))

    fig, axes = plt.subplots(
        1, len(Ds), figsize=(13.5, 3.6),
        constrained_layout=True, sharey=True,
    )

    for ax, (D, kf, sb) in zip(axes, results):
        ax.fill_between(kf["budget"], kf["err"], 1e-2,
                        color=STEEL_500, alpha=0.08, zorder=1)
        ax.fill_between(sb["budget"], sb["err"], 1e-2,
                        color=EMERALD_500, alpha=0.08, zorder=1)

        ax.plot(kf["budget"], kf["err"], color=STEEL_500,
                lw=2.2, marker="o", ms=6, mec="none",
                label="KernelField", zorder=3)
        ax.plot(sb["budget"], sb["err"], color=EMERALD_500,
                lw=2.2, marker="o", ms=6, mec="none",
                label="Sobol", zorder=3)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("budget (probes)")
        if D == Ds[0]:
            ax.set_ylabel("|relative error|   [%]")
            ax.legend(loc="upper right", fontsize=8.5, frameon=False)
        ax.set_title(f"D = {D}", color=ZINC_700, pad=6)

        ax.grid(True, which="major", color=ZINC_200, alpha=0.5, lw=0.5)
        ax.grid(True, which="minor", color=ZINC_200, alpha=0.25, lw=0.4)
        clean_spines(ax)

    fig.suptitle(
        f"Convergence to ground truth — dense Sobol reference (n = 100k)   "
        f"σ = {sigma}, N = 6 kernels",
        fontsize=12, color=ZINC_700,
    )

    path = PLOTS_DIR / "error_convergence.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("1/2  probe placement (3-D) ...")
    p1 = figure_probe_placement()
    print(f"      saved: {p1}")
    print("2/2  error convergence ...")
    p2 = figure_error_convergence()
    print(f"      saved: {p2}")
