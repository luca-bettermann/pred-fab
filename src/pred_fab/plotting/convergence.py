"""Optimization convergence plots."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._style import (
    save_fig, apply_style, clean_spines,
    STEEL_500, EMERALD_500, ZINC_300, ZINC_500, ZINC_700,
)


def plot_convergence(
    save_path: str,
    histories: dict[str, list[float]],
) -> None:
    """Convergence plot: chained phases on a shared iteration axis, log scale.

    All phases (Global, Trajectory rounds) are concatenated on the x-axis.
    Background shading separates phases. Y-axis shows true (unscaled)
    negated objective on log scale.
    """
    if not histories or all(len(h) == 0 for h in histories.values()):
        return

    apply_style()
    phase_colors = [STEEL_500, EMERALD_500, "#DD8452", "#D65F5F", "#8C564B", "#9467BD"]
    bg_alphas = [0.06, 0.03]

    fig, ax = plt.subplots(figsize=(10, 4))

    offset = 0
    all_vals: list[float] = []
    phase_spans: list[tuple[int, int, str]] = []

    for i, (label, history) in enumerate(histories.items()):
        if not history or len(history) < 1:
            continue
        h = np.array(history)
        vals = np.abs(h)
        vals = np.maximum(vals, 1e-10)
        x = np.arange(len(h)) + offset
        all_vals.extend(vals.tolist())

        color = phase_colors[i % len(phase_colors)]
        ax.plot(x, vals, color=color, linewidth=1.5, label=label, alpha=0.85)

        phase_spans.append((offset, offset + len(h) - 1, label))
        offset += len(h)

    # Background shading per phase
    for pi, (x_start, x_end, _label) in enumerate(phase_spans):
        bg = bg_alphas[pi % len(bg_alphas)]
        ax.axvspan(x_start - 0.5, x_end + 0.5, alpha=bg, color=ZINC_500, zorder=0)

    if all_vals:
        ax.set_yscale("log")
        v_min = min(all_vals) * 0.8
        v_max = max(all_vals) * 1.2
        ax.set_ylim(v_min, v_max)

    ax.set_xlim(-0.5, offset - 0.5)
    ax.set_xlabel("Iteration", fontsize=9, color=ZINC_700)
    ax.set_ylabel("|Objective|", fontsize=9, color=ZINC_700)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(True, alpha=0.2, color=ZINC_300, which="both")
    clean_spines(ax)

    save_fig(save_path)
