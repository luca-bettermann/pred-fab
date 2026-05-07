"""Optimization convergence plots."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._style import (
    save_fig, apply_style, clean_spines,
    STEEL_500, EMERALD_500, ZINC_300, ZINC_700,
)


def plot_convergence(
    save_path: str,
    histories: dict[str, list[float]],
) -> None:
    """Convergence plot: x-axis normalised to [0, 1] (fraction of maxiter),
    y-axis shows absolute objective value. Multiple lines overlay.
    """
    if not histories or all(len(h) == 0 for h in histories.values()):
        return

    apply_style()
    colors = [STEEL_500, EMERALD_500, "#DD8452", "#D65F5F"]
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, (label, history) in enumerate(histories.items()):
        if not history or len(history) < 2:
            continue
        color = colors[i % len(colors)]
        h = np.array(history)
        # Normalise: improvement from start (1.0) toward best (0.0).
        h_best = h.min()
        h_range = h[0] - h_best
        if abs(h_range) > 1e-15:
            h_rel = (h - h_best) / h_range
        else:
            h_rel = np.ones_like(h)
        x = np.linspace(0, 1, len(h))
        ax.plot(x, h_rel, color=color, linewidth=1.5, label=label, alpha=0.8)

    ax.set_xlabel("Progress (fraction of max iterations)", fontsize=9, color=ZINC_700)
    ax.set_ylabel("Relative Objective (1 = start, 0 = best)", fontsize=9, color=ZINC_700)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.2, color=ZINC_300)
    clean_spines(ax)

    save_fig(save_path)
