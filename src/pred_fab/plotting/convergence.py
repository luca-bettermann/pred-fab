"""Optimization convergence plots."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._style import save_fig, STEEL_500, EMERALD_500, ZINC_400, ZINC_700


def plot_convergence(
    save_path: str,
    histories: dict[str, list[float]],
    *,
    title: str = "Optimization Convergence",
) -> None:
    """Log-scale convergence plot showing energy/score improvement per iteration.

    histories: mapping of label → list of convergence values per DE iteration.
    """
    if not histories or all(len(h) == 0 for h in histories.values()):
        return

    colors = [STEEL_500, EMERALD_500, "#DD8452", "#D65F5F", ZINC_400]
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, (label, history) in enumerate(histories.items()):
        if not history:
            continue
        color = colors[i % len(colors)]
        ax.plot(range(1, len(history) + 1), history, color=color, linewidth=1.5,
                label=label, alpha=0.8)

    ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontsize=9, color=ZINC_700)
    ax.set_ylabel("Convergence", fontsize=9, color=ZINC_700)
    ax.set_title(title, fontsize=12, fontweight="bold", color=ZINC_700)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.2)

    save_fig(save_path)
