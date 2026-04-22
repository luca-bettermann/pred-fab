"""Optimization convergence plots."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._style import save_fig, STEEL_500, EMERALD_500, ZINC_400, ZINC_700


def plot_convergence(
    save_path: str,
    histories: dict[str, list[float]],
) -> None:
    """Normalized convergence plot: each line shows relative improvement from its start.

    histories: mapping of label → list of best-so-far energy per DE iteration.
    """
    if not histories or all(len(h) == 0 for h in histories.values()):
        return

    colors = [STEEL_500, EMERALD_500, "#DD8452", "#D65F5F", ZINC_400]
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, (label, history) in enumerate(histories.items()):
        if not history or history[0] == 0:
            continue
        color = colors[i % len(colors)]
        # Normalize: value / initial_value → starts at 1.0, decreases
        h = np.array(history)
        normalized = h / h[0]
        ax.plot(range(1, len(normalized) + 1), normalized, color=color, linewidth=1.5,
                label=label, alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Iteration", fontsize=9, color=ZINC_700)
    ax.set_ylabel("Relative Energy", fontsize=9, color=ZINC_700)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.2)

    save_fig(save_path)
