"""Optimization convergence plots."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from ._style import (
    save_fig, apply_style, clean_spines,
    ZINC_300, ZINC_700, style_colorbar,
)


def plot_convergence(
    save_path: str,
    histories: dict[str, list[list[float]]],
) -> None:
    """Per-start convergence plot coloured by Sobol rank.

    Each LBFGS start is a separate line. Colour runs from dark (best
    Sobol candidate) to light (worst) using a continuous colormap with
    a colorbar showing the rank mapping.

    Y-axis shows raw (negative) objective on a symlog scale so that
    the converged region — where starts differ — gets more visual space.
    """
    if not histories or all(len(h) == 0 for h in histories.values()):
        return

    apply_style()
    phase_cmaps = ["Blues", "Greens", "Oranges", "Reds", "Purples"]

    fig, ax = plt.subplots(figsize=(10, 4))
    all_vals: list[float] = []
    last_cmap_name: str | None = None
    last_n_starts: int = 0

    for pi, (label, starts) in enumerate(histories.items()):
        if not starts:
            continue

        n_starts = len(starts)
        cmap_name = phase_cmaps[pi % len(phase_cmaps)]
        cmap = plt.get_cmap(cmap_name)
        last_cmap_name = cmap_name
        last_n_starts = n_starts

        for si, start_h in enumerate(starts):
            if not start_h:
                continue
            h = np.array(start_h)
            x = np.arange(len(h))
            all_vals.extend(h.tolist())

            t = si / max(n_starts - 1, 1)
            color = cmap(0.35 + 0.55 * (1.0 - t))
            lw = 1.8 if si == 0 else 0.9
            alpha = 0.95 if si == 0 else 0.6
            ax.plot(x, h, color=color, linewidth=lw, alpha=alpha)

        ax.plot([], [], color=cmap(0.7), linewidth=1.5, label=label)

    if all_vals:
        v_min = min(all_vals)
        v_max = max(all_vals)
        margin = abs(v_max - v_min) * 0.05
        ax.set_ylim(v_min - margin, v_max + margin)

    ax.set_xlabel("Iteration (per start)", fontsize=9, color=ZINC_700)
    ax.set_ylabel("Objective", fontsize=9, color=ZINC_700)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(True, alpha=0.2, color=ZINC_300, which="both")
    clean_spines(ax)

    if last_cmap_name and last_n_starts > 1:
        sm = ScalarMappable(
            cmap=plt.get_cmap(last_cmap_name),
            norm=Normalize(vmin=0, vmax=last_n_starts - 1),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Sobol rank", fontsize=8, color=ZINC_700)
        style_colorbar(cbar)

    save_fig(save_path)
