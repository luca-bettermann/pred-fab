"""Radar performance plot — per-attribute visualization with dataset comparison."""

from collections.abc import Mapping

import numpy as np
import matplotlib.pyplot as plt

from ._style import (
    save_fig, EMERALD_300, EMERALD_500,
    ZINC_200, ZINC_300, ZINC_400, ZINC_500, ZINC_600, ZINC_700,
)


def plot_performance_radar(
    save_path: str,
    performance: dict[str, float],
    dataset_performances: list[dict[str, float]],
    weights: Mapping[str, float | int],
    *,
    combined_score: float | None = None,
    dataset_combined: float | None = None,
    exp_code: str = "",
    title: str = "Performance Profile",
) -> None:
    """Radar/spider plot of per-attribute performance with dataset average overlay."""
    attributes = sorted(performance.keys())
    n = len(attributes)
    if n < 3:
        return

    values = [performance[a] for a in attributes]

    # Dataset mean
    mean_vals = []
    for a in attributes:
        attr_vals = [p[a] for p in dataset_performances if a in p]
        mean_vals.append(float(np.mean(attr_vals)) if attr_vals else 0.0)

    # Angles for radar
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_closed = values + [values[0]]
    mean_closed = mean_vals + [mean_vals[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    # Shrink the radar to ~50% of figure, centered with room for labels + footer
    ax.set_position([0.20, 0.22, 0.55, 0.55])  # type: ignore[arg-type]

    # Style the polar plot
    ax.set_theta_offset(np.pi / 2)  # type: ignore[attr-defined]
    ax.set_theta_direction(-1)  # type: ignore[attr-defined]
    ax.set_rlabel_position(45)  # type: ignore[attr-defined]

    # Grid
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0.5", "", "1.0"],
                       fontsize=10, color=ZINC_400)
    ax.spines["polar"].set_color(ZINC_200)
    ax.grid(color=ZINC_200, alpha=0.4, linewidth=0.6)

    # Axis labels
    ax.set_xticks(angles)
    labels = [a.replace("_", " ").title() for a in attributes]
    ax.set_xticklabels(labels, fontsize=11, color=ZINC_600)
    ax.tick_params(axis="x", pad=20)

    # Dataset mean — background polygon
    ax.plot(angles_closed, mean_closed, "o-",
            color=ZINC_400, linewidth=1.2, markersize=4, alpha=0.5)
    ax.fill(angles_closed, mean_closed, alpha=0.10, color=ZINC_300)

    # Current experiment — foreground polygon (Emerald = performance)
    ax.plot(angles_closed, values_closed, "o-",
            color=EMERALD_500, linewidth=2.0, markersize=6,
            markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    ax.fill(angles_closed, values_closed, alpha=0.15, color=EMERALD_300)

    # Title — at the very top
    title_text = title
    if exp_code:
        title_text += f"  ·  {exp_code}"
    fig.text(0.5, 0.96, title_text, fontsize=13, fontweight="bold",
             color=ZINC_700, ha="center", va="top")

    # System performance — percentage display (experiment=green, dataset=grey)
    if combined_score is not None:
        pct = combined_score * 100
        fig.text(0.78, 0.12, f"{pct:.0f}%", fontsize=28, fontweight="bold",
                 color=EMERALD_500, ha="center", va="bottom")
        if dataset_combined is not None:
            avg_pct = dataset_combined * 100
            fig.text(0.78, 0.06, f"{avg_pct:.0f}%", fontsize=17,
                     color=ZINC_400, ha="center", va="bottom")

    # Legend (bottom-left)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=EMERALD_500, linewidth=2, marker="o",
               markersize=5, markeredgecolor="white", label="Experiment"),
        Line2D([0], [0], color=ZINC_400, linewidth=1.2, marker="o",
               markersize=4, alpha=0.5, label="Dataset avg"),
    ]
    fig.legend(handles=legend_elements, loc="lower left",
               bbox_to_anchor=(0.06, 0.04), fontsize=10,
               frameon=False, labelcolor=ZINC_500)

    save_fig(save_path)
