"""Concept: importance weighting via score-conditional sigmoid.

The prediction model is trained on every experiment in the dataset, but not
all experiments are equally informative. A high-performing point near the
optimum tells us more about the manufacturing-relevant region than a poor
exploratory probe at the boundary. Importance weighting amplifies the
training signal from high-score experiments without discarding the rest.

Weight function — one knob (`floor`) plus a data-driven steepness:

    w(s) = floor + (1 − floor) · σ(k · (s − μ_s))
    k    = steepness / σ_s          (so the curve scales to score variance)

where s is the experiment's combined score, μ_s and σ_s the dataset's mean
and standard deviation. Only the floor is an explicit bound — the upper
limit is just where the sigmoid asymptotes. By construction the dataset is
centred on the inflection (midpoint at score = μ_s).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _style import (
    apply_style, clean_spines, subplot_label,
    ACCENT_RED, STEEL_500,
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


def figure_importance_weights(
    floor: float = 0.10,
    steepness: float = 0.8,
    seed: int = 0,
) -> Path:
    """Single panel: sigmoid weighting curve with experiment dots."""
    apply_style()
    scores = _toy_scores(seed=seed)
    s_mean = float(scores.mean())
    s_std = float(scores.std())
    k = steepness / s_std if s_std > 1e-10 else 0.0

    perf_range = np.linspace(0.0, 1.0, 400)
    sig_curve = 1.0 / (1.0 + np.exp(-k * (perf_range - s_mean)))
    w_curve = floor + (1.0 - floor) * sig_curve

    exp_sig = 1.0 / (1.0 + np.exp(-k * (scores - s_mean)))
    weights = floor + (1.0 - floor) * exp_sig

    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)

    # Curve + experiment dots
    ax.fill_between(perf_range, floor, w_curve, alpha=0.10, color=STEEL_500)
    ax.plot(perf_range, w_curve, color=STEEL_500, lw=2, label="w(s)")
    ax.scatter(scores, weights, c=ACCENT_RED, s=30, zorder=5,
               edgecolors="darkred", linewidths=0.4,
               label=f"experiments (n = {len(scores)})")

    # Bounds: only floor is an explicit cap — top is the sigmoid asymptote.
    ax.axhline(floor, color=ZINC_400, ls=":", lw=0.9,
               label=f"floor = {floor}")
    midpoint = (1.0 + floor) / 2.0
    ax.axhline(midpoint, color=ZINC_400, ls="--", lw=0.8,
               label=f"midpoint = {midpoint:.2f}")

    # Score statistics
    ax.axvline(s_mean, color=ACCENT_RED, ls="--", lw=1, alpha=0.6,
               label=f"μ_s = {s_mean:.2f}")
    ax.axvspan(s_mean - s_std, s_mean + s_std, alpha=0.05, color=ACCENT_RED)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("combined score  s")
    ax.set_ylabel("weight  w(s)")
    ax.legend(loc="lower right", fontsize=9, frameon=False, labelcolor=ZINC_600)
    subplot_label(
        ax,
        f"w(s) = floor + (1 − floor) · σ(k·(s − μ_s))   ·   k = {steepness} / σ_s = {k:.1f}",
    )
    ax.grid(True, alpha=0.2, color=ZINC_300)
    clean_spines(ax)

    path = PLOTS_DIR / "importance_weights.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("1/1  importance weights ...")
    p = figure_importance_weights()
    print(f"      saved: {p}")
