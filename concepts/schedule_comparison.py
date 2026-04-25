"""Concept: why per-step schedules beat single-shot parameters.

A common pattern in fabrication: the optimum of a process parameter drifts
across a process dimension. Print speed that is ideal for the first layer is
not ideal for the tenth — bottom layers tolerate higher speed (more slip,
less sag), top layers need slower speed to settle. A *fixed* parameter
choice picks one value and lives with the loss elsewhere; a *scheduled*
choice tracks the drifting optimum step by step.

Toy problem (1-D parameter x, step index k = 0..N−1):

    optimum drift     x*(k) = x₀ + Δ · k / (N−1)
    per-step score    f(x, k) = exp(−(x − x*(k))² / 2σ²)
    fixed strategy    x_fixed = (x*(0) + x*(N−1)) / 2     (best single x)
    schedule          x(k) = x*(k)                         (tracks the drift)

Two panels:
    1.  topology f(x, k) with both strategies overlaid as paths
    2.  per-step score for each strategy + cumulative-mean lines

The headline reads off panel 2: the fixed strategy is optimal at exactly
one step (the midpoint) and bleeds at both ends; the schedule stays on the
ridge throughout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from _style import (
    apply_style, clean_spines, subplot_label, cmap, style_colorbar,
    ACCENT_YELLOW, EMERALD_500, STEEL_500,
    ZINC_300, ZINC_500, ZINC_600,
)


PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def _problem(n_steps: int, x0: float, delta: float, sigma: float, res: int):
    steps = np.arange(n_steps)
    optimum = x0 + delta * steps / max(n_steps - 1, 1)

    x_grid = np.linspace(0.0, 1.0, res)
    K, X = np.meshgrid(steps, x_grid, indexing="ij")
    OPT = x0 + delta * K / max(n_steps - 1, 1)
    field = np.exp(-((X - OPT) ** 2) / (2.0 * sigma ** 2))
    return steps, optimum, x_grid, field


def figure_schedule_comparison(
    n_steps: int = 10,
    x0: float = 0.30,
    delta: float = 0.40,
    sigma: float = 0.06,
    res: int = 240,
) -> Path:
    apply_style()
    steps, optimum, x_grid, field = _problem(n_steps, x0, delta, sigma, res)
    x_fixed = float((optimum[0] + optimum[-1]) / 2.0)

    score_fixed = np.exp(-((x_fixed - optimum) ** 2) / (2.0 * sigma ** 2))
    score_sched = np.ones(n_steps)  # x(k) = x*(k) → score = 1 at every step

    cum_fixed = np.cumsum(score_fixed) / np.arange(1, n_steps + 1)
    cum_sched = np.cumsum(score_sched) / np.arange(1, n_steps + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)

    # ── Panel 1: topology with both strategies overlaid ──
    ax = axes[0]
    perf_cm = cmap("performance")
    norm = Normalize(vmin=0.0, vmax=1.0)
    ax.contourf(steps, x_grid, field.T, levels=24, cmap=perf_cm, norm=norm)
    ax.contour(steps, x_grid, field.T, levels=8, colors="white",
               linewidths=0.3, alpha=0.45)
    ax.plot(steps, optimum, color="white", lw=1.0, alpha=0.7,
            label="optimum  x*(k)")
    ax.plot(steps, [x_fixed] * n_steps, color=ACCENT_YELLOW, lw=2.0,
            label=f"fixed  x = {x_fixed:.2f}")
    ax.plot(steps, optimum, color=EMERALD_500, lw=2.0, ls="--", alpha=0.85,
            label="schedule  x(k)")
    ax.scatter(steps, optimum, c=EMERALD_500, s=20, zorder=5,
               edgecolors="white", linewidths=0.5)
    ax.set_xlabel("step  k")
    ax.set_ylabel("parameter  x")
    ax.set_xlim(0, n_steps - 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=ZINC_600)
    subplot_label(ax, f"f(x, k) = exp(−(x − x*(k))² / 2σ²)  ·  σ = {sigma}")
    clean_spines(ax)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=perf_cm),
                        ax=ax, shrink=0.85, pad=0.02)
    style_colorbar(cbar)
    cbar.set_label("score", color=ZINC_600, fontsize=9)

    # ── Panel 2: per-step performance ──
    ax = axes[1]
    ax.plot(steps, score_fixed, "o-", color=ACCENT_YELLOW, lw=1.8, ms=6,
            markeredgecolor="white", markeredgewidth=0.6,
            label=f"fixed  (mean = {score_fixed.mean():.3f})")
    ax.plot(steps, score_sched, "o-", color=EMERALD_500, lw=1.8, ms=6,
            markeredgecolor="white", markeredgewidth=0.6,
            label=f"schedule  (mean = {score_sched.mean():.3f})")
    ax.plot(steps, cum_fixed, color=ACCENT_YELLOW, lw=0.8, alpha=0.5, ls=":")
    ax.plot(steps, cum_sched, color=EMERALD_500, lw=0.8, alpha=0.5, ls=":")
    ax.fill_between(steps, score_fixed, score_sched,
                    color=STEEL_500, alpha=0.10, label="loss avoided")
    ax.set_xlabel("step  k")
    ax.set_ylabel("per-step score  f")
    ax.set_xlim(0, n_steps - 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=ZINC_600)
    subplot_label(ax, "fixed loses at both ends; schedule stays on the ridge")
    clean_spines(ax)
    ax.grid(True, alpha=0.2, color=ZINC_300)

    path = PLOTS_DIR / "schedule_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("1/1  schedule comparison ...")
    p = figure_schedule_comparison()
    print(f"      saved: {p}")
