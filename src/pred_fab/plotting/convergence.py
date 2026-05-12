"""Optimization convergence plots."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from ._style import (
    save_fig, apply_style, clean_spines,
    ACCENT_YELLOW, ZINC_300, ZINC_700, style_colorbar,
)


def plot_convergence(
    save_path: str,
    histories: dict[str, list[list[float]]],
) -> None:
    """Per-start convergence plot coloured by Sobol rank.

    Best Sobol start (rank 0) is drawn dark. The overall best start
    (lowest final objective) is highlighted in yellow with a marker
    on the colorbar showing its Sobol rank.
    """
    if not histories or all(len(h) == 0 for h in histories.values()):
        return

    apply_style()
    phase_cmaps = ["Blues", "Greens", "Oranges", "Reds", "Purples"]

    fig, ax = plt.subplots(figsize=(10, 4))
    all_vals: list[float] = []
    last_cmap_name: str | None = None
    last_n_starts: int = 0
    best_rank: int = 0
    best_final: float = float("inf")

    for pi, (label, starts) in enumerate(histories.items()):
        if not starts:
            continue

        n_starts = len(starts)
        cmap_name = phase_cmaps[pi % len(phase_cmaps)]
        cmap = plt.get_cmap(cmap_name)
        last_cmap_name = cmap_name
        last_n_starts = n_starts

        # First pass: find the best start (lowest final value)
        for si, start_h in enumerate(starts):
            if start_h:
                final = min(start_h)
                if final < best_final:
                    best_final = final
                    best_rank = si

        # Second pass: draw all lines, best line last (on top)
        for si, start_h in enumerate(starts):
            if not start_h or si == best_rank:
                continue
            h = np.array(start_h)
            x = np.arange(len(h))
            all_vals.extend(h.tolist())

            t = si / max(n_starts - 1, 1)
            color = cmap(0.35 + 0.55 * (1.0 - t))
            lw = 1.8 if si == 0 else 0.9
            alpha = 0.95 if si == 0 else 0.6
            ax.plot(x, h, color=color, linewidth=lw, alpha=alpha)

        # Draw best line on top in yellow
        if starts[best_rank]:
            h = np.array(starts[best_rank])
            x = np.arange(len(h))
            all_vals.extend(h.tolist())
            ax.plot(x, h, color=ACCENT_YELLOW, linewidth=2.2, alpha=1.0,
                    label=f"best (rank {best_rank})", zorder=10)

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
        cbar.ax.axhline(best_rank, color=ACCENT_YELLOW, linewidth=2.5)

    save_fig(save_path)
