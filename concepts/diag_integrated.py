"""Diagnostic: integrated-objective Phase 1 baseline.

Math:
    ρ_j(z)   = w_j · N(z; z_j, σ²I)     proper Gaussian density, mass w_j in ℝ^D
    D(z)     = Σ_j ρ_j(z)                raw evidence density, unbounded
    E(z)     = D(z) / (1 + D(z))         actual evidence, [0, 1)
    objective= ∫_[0,1]^D E(z) dz         (maximize)

    ΔE_total from adding new points factors out the new ρ, but for joint
    placement from empty prior the integral objective above is the goal.

MC estimate:
    Fixed Sobol quasi-random samples over [0,1]^D, deterministic per DE run.
    ∫ E dz ≈ (1/M) Σ_m E(z_m),  z_m ~ Sobol([0,1]^D)

No α, no boundary term, no +1 hack. Leakage enters via integration bounds.

Goal of this script:
    With N=5, D=3, σ=0.20·√D, compare against today's pointwise baseline
    (which puts 9/15 coords at the boundary). Target: 0 boundary hits.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.stats import qmc


# ---------- Kernel and evidence ----------

def gaussian_density(z: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """Peak-1 Gaussian density — matches production (ρ(c) = 1 per centre).

    z:       (M, D)
    centers: (N, D)
    Returns: (M, N) density values; each column is ρ(· ; centers[n], σ²I)
             with peak ρ = 1 at the centre.
    """
    d2 = np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    return np.exp(-d2 / (2.0 * sigma ** 2))


def raw_density(z: np.ndarray, centers: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    """D(z) = Σ w_j · ρ_j(z).  z: (M, D),  centers: (N, D),  weights: (N,)  -> (M,)."""
    if len(centers) == 0:
        return np.zeros(z.shape[0])
    K = gaussian_density(z, centers, sigma)           # (M, N)
    return (K * weights[None, :]).sum(axis=-1)         # (M,)


def integrated_evidence(
    centers: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    sobol_samples: np.ndarray,
) -> float:
    """∫_[0,1]^D E(z) dz via fixed Sobol MC.  Returns scalar in [0, 1)."""
    D_vals = raw_density(sobol_samples, centers, weights, sigma)
    E_vals = D_vals / (1.0 + D_vals)
    return float(E_vals.mean())  # unit cube volume = 1


# ---------- Phase 1 joint optimizer ----------

def optimize_joint_phase1(
    N: int, D: int, sigma: float, M_samples: int, seed: int = 0, maxiter: int = 100,
) -> tuple[np.ndarray, float, int]:
    """Maximize ∫E dz over N batch positions in [0,1]^D."""
    # Fixed Sobol samples: deterministic objective surface
    sampler = qmc.Sobol(d=D, scramble=True, seed=seed)
    sobol_samples = sampler.random(n=M_samples)

    def objective(x_flat: np.ndarray) -> float:
        centers = x_flat.reshape(N, D)
        weights = np.ones(N)
        return -integrated_evidence(centers, weights, sigma, sobol_samples)

    bounds = [(0.0, 1.0)] * (N * D)
    res = differential_evolution(
        objective, bounds, seed=seed, maxiter=maxiter, tol=1e-6,
        popsize=15, mutation=(0.5, 1.0), recombination=0.7, workers=1,
    )
    centers = res.x.reshape(N, D)
    return centers, -res.fun, res.nfev


# ---------- Boundary hit diagnostics ----------

def boundary_stats(X: np.ndarray, tol: float = 0.02) -> tuple[int, int]:
    """Return (n_hits, n_total_coords) where a hit is a coord within `tol` of {0,1}."""
    at_bound = np.logical_or(X < tol, X > 1.0 - tol)
    return int(at_bound.sum()), int(X.size)


# ---------- Reference: pointwise baseline (today's behavior) ----------

def cauchy_kernel(d2: np.ndarray, sigma: float) -> np.ndarray:
    """Unnormalized Cauchy, K(0)=1.  For parity with today's production."""
    return 1.0 / (1.0 + d2 / sigma ** 2)


def boundary_evidence_pointwise(z: np.ndarray, sigma: float) -> np.ndarray:
    """Current production's per-dim boundary term."""
    lo = cauchy_kernel(z ** 2, sigma)
    hi = cauchy_kernel((1.0 - z) ** 2, sigma)
    return 0.5 * (lo + hi).sum(axis=-1)


def pointwise_batch_u(X: np.ndarray, sigma: float, use_boundary: bool = True) -> float:
    """Today's batch-aware UCB-κ=1 objective: mean of u_k = 1/(1+E_k).

    Each sibling contributes K(|z_k - z_j|²) at every other batch point.
    """
    N = X.shape[0]
    u = np.empty(N)
    for k in range(N):
        others = np.delete(X, k, axis=0)
        d2 = np.sum((others - X[k]) ** 2, axis=-1)
        e = cauchy_kernel(d2, sigma).sum()
        if use_boundary:
            e += boundary_evidence_pointwise(X[k], sigma)
        u[k] = 1.0 / (1.0 + e)
    return float(u.mean())


def optimize_pointwise_phase1(
    N: int, D: int, sigma: float, seed: int = 0, maxiter: int = 60, use_boundary: bool = True,
) -> np.ndarray:
    """Replicate today's Baseline Process objective with Cauchy + boundary."""

    def obj(x_flat: np.ndarray) -> float:
        return -pointwise_batch_u(x_flat.reshape(N, D), sigma, use_boundary=use_boundary)

    bounds = [(0.0, 1.0)] * (N * D)
    res = differential_evolution(
        obj, bounds, seed=seed, maxiter=maxiter, tol=1e-5,
        popsize=10, mutation=(0.5, 1.0), recombination=0.7, workers=1,
    )
    return res.x.reshape(N, D)


# ---------- Reporting ----------

def summarize(name: str, X: np.ndarray, extra: dict | None = None) -> None:
    hits, total = boundary_stats(X)
    print(f"\n── {name} ──")
    print(f"  boundary hits: {hits} / {total}  ({100*hits/total:.0f}%)")
    mean_pair = np.mean([
        np.linalg.norm(X[i] - X[j])
        for i in range(len(X)) for j in range(i + 1, len(X))
    ])
    print(f"  mean pairwise dist: {mean_pair:.3f}")
    print(f"  placement (normalized):")
    for k, row in enumerate(X):
        marks = ["*" if (v < 0.02 or v > 0.98) else " " for v in row]
        print(f"    pt{k}: " + "  ".join(f"{v:.3f}{m}" for v, m in zip(row, marks)))
    if extra:
        for k, v in extra.items():
            print(f"  {k}: {v}")


# ---------- Main ----------

def main() -> None:
    out_dir = Path(__file__).parent / "plots"
    out_dir.mkdir(exist_ok=True)

    N, D = 5, 3
    exploration_radius = 0.20
    sigma = exploration_radius * np.sqrt(float(D))
    M_samples = 2 ** (D + 4)  # 128 for D=3

    print("=" * 70)
    print(f"Phase 1 validation: N={N}, D={D}, σ={sigma:.3f}")
    print(f"MC samples (Sobol): M = 2^(D+4) = {M_samples}")
    print("=" * 70)

    # 1) Today's pointwise baseline (reproduces the corner-winning bug)
    X_today = optimize_pointwise_phase1(N, D, sigma, seed=0, maxiter=60, use_boundary=True)
    summarize("A) Today's pointwise baseline (Cauchy + boundary term)", X_today)

    # 2) Pointwise without boundary term (should be even worse — no boundary gradient)
    X_no_b = optimize_pointwise_phase1(N, D, sigma, seed=0, maxiter=60, use_boundary=False)
    summarize("B) Pointwise, boundary term removed (stripped baseline)", X_no_b)

    # 3) Integrated objective — σ sweep to find spreading regime
    print("\n" + "=" * 70)
    print("σ sweep for integrated objective (N=5, D=3):")
    print("=" * 70)
    sigma_candidates = [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.28, 0.346]
    sweep_results = []
    for s in sigma_candidates:
        X, obj, nfev = optimize_joint_phase1(
            N, D, s, M_samples=M_samples, seed=0, maxiter=100,
        )
        hits, total = boundary_stats(X)
        mean_pair = np.mean([
            np.linalg.norm(X[i] - X[j])
            for i in range(len(X)) for j in range(i + 1, len(X))
        ])
        sweep_results.append((s, X, obj, hits, mean_pair))
        coll = "COLLAPSE" if mean_pair < 0.05 else ("boundary" if hits > 0 else "spread")
        print(f"  σ={s:.3f}  hits={hits:2d}/{total}  mean_pair={mean_pair:.3f}  "
              f"∫E={obj:.4f}  → {coll}")

    # Pick the best σ (0 hits and max mean_pair) and show placement
    good = [r for r in sweep_results if r[3] == 0 and r[4] > 0.1]
    if good:
        best = max(good, key=lambda r: r[4])
        s_best, X_best, obj_best, _, _ = best
        summarize(f"C) Integrated, best σ={s_best:.2f}", X_best,
                  extra={"∫E dz": f"{obj_best:.4f}"})

    X_integ = sweep_results[3][1]  # σ=0.15 for plot

    # Plot the three placements in the first 2 dims
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (label, X) in zip(axes, [
        ("A) today (pointwise)", X_today),
        ("B) pointwise, no boundary", X_no_b),
        ("C) integrated (proposal)", X_integ),
    ]):
        ax.scatter(X[:, 0], X[:, 1], c="C0", s=80, edgecolors="black")
        for k, row in enumerate(X):
            ax.annotate(f"{k}", (row[0], row[1]), xytext=(5, 5),
                        textcoords="offset points", fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="red", alpha=0.3)
        ax.axhline(1, color="red", alpha=0.3)
        ax.axvline(0, color="red", alpha=0.3)
        ax.axvline(1, color="red", alpha=0.3)
    out_path = out_dir / "diag_integrated.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    main()
