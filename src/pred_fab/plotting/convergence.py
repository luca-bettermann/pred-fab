"""Optimization convergence plots."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from ._style import (
    save_fig, apply_style, clean_spines,
    STEEL_500, EMERALD_500, ZINC_300, ZINC_500, ZINC_700,
    style_colorbar,
)


def plot_convergence(
    save_path: str,
    histories: dict[str, list[list[float]]],
) -> None:
    """Per-start convergence plot coloured by Sobol rank.

    Each phase (Baseline, Schedule, …) gets its own x-axis reset.
    Within a phase, every LBFGS start is a separate line coloured from
    dark (best Sobol rank) to light (worst).
    """
    if not histories or all(len(h) == 0 for h in histories.values()):
        return

    apply_style()
    phase_cmaps = ["Blues", "Greens", "Oranges", "Reds", "Purples"]

    fig, ax = plt.subplots(figsize=(10, 4))
    all_vals: list[float] = []

    for pi, (label, starts) in enumerate(histories.items()):
        if not starts:
            continue

        n_starts = len(starts)
        cmap = plt.get_cmap(phase_cmaps[pi % len(phase_cmaps)])

        for si, start_h in enumerate(starts):
            if not start_h:
                continue
            h = np.abs(np.array(start_h))
            h = np.maximum(h, 1e-10)
            x = np.arange(len(h))
            all_vals.extend(h.tolist())

            t = si / max(n_starts - 1, 1)
            color = cmap(0.35 + 0.55 * (1.0 - t))
            lw = 1.8 if si == 0 else 0.9
            alpha = 0.95 if si == 0 else 0.6
            ax.plot(x, h, color=color, linewidth=lw, alpha=alpha)

        # Legend entry for this phase (single representative line)
        ax.plot([], [], color=cmap(0.7), linewidth=1.5, label=label)

    if all_vals:
        ax.set_yscale("log")
        v_min = min(all_vals) * 0.8
        v_max = max(all_vals) * 1.2
        ax.set_ylim(v_min, v_max)

    ax.set_xlabel("Iteration (per start)", fontsize=9, color=ZINC_700)
    ax.set_ylabel("|Objective|", fontsize=9, color=ZINC_700)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(True, alpha=0.2, color=ZINC_300, which="both")
    clean_spines(ax)

    save_fig(save_path)
