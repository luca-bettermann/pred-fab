"""Concept: importance weighting via score-conditional sigmoid.

The prediction model is trained on every experiment in the dataset, but not
all experiments are equally informative. A high-performing point near the
optimum tells us more about the manufacturing-relevant region than a poor
exploratory probe at the boundary. Importance weighting amplifies the
training signal from high-score experiments without discarding the rest.

Weight function (one knob — `floor` — plus a data-driven steepness):

    w(s) = floor + (1 − floor) · σ(k · (s − μ_s))
    k    = steepness / σ_s          (so the curve scales to score variance)

where s is the experiment's combined score, μ_s and σ_s the dataset's mean
and standard deviation. Bottom plateau is `floor` (no point is ignored);
top plateau is 1 (the best points get full weight). The midpoint sits at
score = μ_s, so the dataset is centred on the inflection by construction.

This figure shows three panels of the same toy distribution to make the
intuition concrete:

    1.  raw scores (lollipops, height = 1)
    2.  the weighting curve overlaid with where each experiment lands
    3.  effective weights (lollipops, height = w(sᵢ)) — the optimum-region
        cluster carries the loss; tail points still contribute via `floor`
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _style import (
    apply_style, clean_spines, subplot_label,
    ACCENT_RED, EMERALD_500, STEEL_500,
    ZINC_300, ZINC_400, ZINC_600,
)


PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def _toy_scores(n: int = 28, seed: int = 0) -> np.ndarray:
    """Synthetic combined-score distribution: long tail of poor probes plus
    a tight cluster near the optimum, like a real exploration run."""
    rng = np.random.default_rng(seed)
    poor = rng.uniform(0.10, 0.55, size=int(0.7 * n))
    good = rng.normal(0.78, 0.05, size=n - poor.size).clip(0.55, 0.97)
    return np.sort(np.concatenate([poor, good]))


def _weights(scores: np.ndarray, floor: float, steepness: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    s_mean = float(scores.mean())
    s_std = float(scores.std())
    k = steepness / s_std if s_std > 1e-10 else 0.0
    sig = 1.0 / (1.0 + np.exp(-k * (scores - s_mean)))
    return floor + (1.0 - floor) * sig, np.array([s_mean, s_std]), k, s_mean


def figure_importance_weights(
    floor: float = 0.10,
    steepness: float = 0.8,
    seed: int = 0,
) -> Path:
    """3-panel figure: raw scores, sigmoid weighting curve, weighted scores."""
    apply_style()
    scores = _toy_scores(seed=seed)
    weights, stats, k, s_mean = _weights(scores, floor, steepness)
    s_std = float(stats[1])

    perf_range = np.linspace(0.0, 1.0, 400)
    sig_curve = 1.0 / (1.0 + np.exp(-k * (perf_range - s_mean)))
    w_curve = floor + (1.0 - floor) * sig_curve

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)

    # ── Panel 1: raw scores ──
    ax = axes[0]
    ax.vlines(scores, 0, 1, color=STEEL_500, lw=1.2, alpha=0.7)
    ax.scatter(scores, np.ones_like(scores), c=STEEL_500, s=22, zorder=5,
               edgecolors="white", linewidths=0.5)
    ax.axvline(s_mean, color=ACCENT_RED, ls="--", lw=1, alpha=0.7,
               label=f"μ = {s_mean:.2f}")
    ax.axvspan(s_mean - s_std, s_mean + s_std, alpha=0.06, color=ACCENT_RED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("combined score  s")
    ax.set_ylabel("uniform contribution")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_yticks([])
    subplot_label(ax, f"unweighted  ·  n = {len(scores)}")
    clean_spines(ax)
    ax.grid(True, axis="x", alpha=0.2, color=ZINC_300)

    # ── Panel 2: sigmoid weighting curve ──
    ax = axes[1]
    ax.fill_between(perf_range, floor, w_curve, alpha=0.10, color=STEEL_500)
    ax.plot(perf_range, w_curve, color=STEEL_500, lw=2)
    ax.scatter(scores, weights, c=ACCENT_RED, s=28, zorder=5,
               edgecolors="darkred", linewidths=0.4,
               label="experiment weights")
    ax.axhline(floor, color=ZINC_400, ls=":", lw=0.8,
               label=f"floor = {floor}")
    ax.axhline(1.0, color=ZINC_400, ls=":", lw=0.8)
    midpoint = (1.0 + floor) / 2.0
    ax.axhline(midpoint, color=ZINC_400, ls="--", lw=0.8,
               label=f"midpoint = {midpoint:.2f}")
    ax.axvline(s_mean, color=ACCENT_RED, ls="--", lw=1, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("combined score  s")
    ax.set_ylabel("weight  w(s)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    subplot_label(ax,
                  f"w(s) = {floor} + (1−{floor})·σ(k·(s−μ))  ·  k = {steepness}/σ_s = {k:.1f}")
    clean_spines(ax)
    ax.grid(True, alpha=0.2, color=ZINC_300)

    # ── Panel 3: weighted contribution ──
    ax = axes[2]
    ax.vlines(scores, 0, weights, color=EMERALD_500, lw=1.2, alpha=0.75)
    ax.scatter(scores, weights, c=EMERALD_500, s=24, zorder=5,
               edgecolors="white", linewidths=0.5)
    ax.axhline(floor, color=ZINC_400, ls=":", lw=0.8, alpha=0.7)
    ax.axhline(1.0, color=ZINC_400, ls=":", lw=0.8, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("combined score  s")
    ax.set_ylabel("effective contribution  w(s)")
    total = float(weights.sum())
    top_share = float(weights[scores >= s_mean].sum() / total)
    subplot_label(ax,
                  f"weighted  ·  top half carries {top_share*100:.0f}% of the loss")
    clean_spines(ax)
    ax.grid(True, alpha=0.2, color=ZINC_300)

    path = PLOTS_DIR / "importance_weights.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("1/1  importance weights ...")
    p = figure_importance_weights()
    print(f"      saved: {p}")
