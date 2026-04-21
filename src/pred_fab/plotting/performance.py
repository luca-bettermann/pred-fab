"""Radar performance plot — per-attribute visualization with dataset comparison."""

from collections.abc import Mapping

import numpy as np
import matplotlib.pyplot as plt

from ._style import (
    save_fig, STEEL_100, STEEL_300, STEEL_500,
    ZINC_200, ZINC_400, ZINC_600, ZINC_700,
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
    """Radar/spider plot of per-attribute performance with dataset average overlay.

    Axes radiate from center, one per attribute. The current experiment forms
    a filled polygon; the dataset mean forms a background polygon. Markers
    are scaled by attribute weight.
    """
    attributes = sorted(performance.keys())
    n = len(attributes)
    if n < 3:
        return

    # Compute values
    values = [performance[a] for a in attributes]
    weight_vals = [weights.get(a, 1.0) for a in attributes]

    # Dataset mean
    mean_vals = []
    for a in attributes:
        attr_vals = [p[a] for p in dataset_performances if a in p]
        mean_vals.append(float(np.mean(attr_vals)) if attr_vals else 0.0)

    # Angles for radar
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    # Close the polygon
    values_closed = values + [values[0]]
    mean_closed = mean_vals + [mean_vals[0]]
    angles_closed = angles + [angles[0]]

    # Normalize weights for marker sizing (min 4pt, max 12pt)
    w_arr = np.array(weight_vals, dtype=float)
    if w_arr.max() > w_arr.min():
        w_norm = (w_arr - w_arr.min()) / (w_arr.max() - w_arr.min())
    else:
        w_norm = np.ones_like(w_arr) * 0.5
    marker_sizes = 4.0 + w_norm * 8.0

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})

    # Style the polar plot
    ax.set_theta_offset(np.pi / 2)  # type: ignore[attr-defined]
    ax.set_theta_direction(-1)  # type: ignore[attr-defined]
    ax.set_rlabel_position(0)  # type: ignore[attr-defined]

    # Grid styling
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                       fontsize=7, color=ZINC_400)
    ax.spines["polar"].set_color(ZINC_200)
    ax.grid(color=ZINC_200, alpha=0.4, linewidth=0.6)

    # Axis labels
    ax.set_xticks(angles)
    labels = [a.replace("_", " ").title() for a in attributes]
    ax.set_xticklabels(labels, fontsize=8, color=ZINC_600)

    # Dataset mean — background polygon
    ax.plot(angles_closed, mean_closed, "o-",
            color=ZINC_400, linewidth=1.0, markersize=3, alpha=0.6)
    ax.fill(angles_closed, mean_closed, alpha=0.08, color=ZINC_400)

    # Current experiment — foreground polygon
    ax.plot(angles_closed, values_closed, "-",
            color=STEEL_500, linewidth=2.0)
    ax.fill(angles_closed, values_closed, alpha=0.18, color=STEEL_300)

    # Weighted markers on current experiment
    for i, (angle, val, ms) in enumerate(zip(angles, values, marker_sizes)):
        ax.plot(angle, val, "o", color=STEEL_500, markersize=ms,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)

    # Title
    title_text = title
    if exp_code:
        title_text += f"  ·  {exp_code}"
    ax.set_title(title_text, fontsize=12, fontweight="bold",
                 color=ZINC_700, pad=20)

    # Combined score annotation (bottom-right area)
    if combined_score is not None:
        score_text = f"Combined: {combined_score:.3f}"
        if dataset_combined is not None:
            diff = combined_score - dataset_combined
            score_text += f"\nDataset avg: {dataset_combined:.3f}  ({diff:+.3f})"

        fig.text(0.82, 0.08, score_text, fontsize=9, color=ZINC_700,
                 ha="center", va="bottom", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=STEEL_100,
                           edgecolor=STEEL_300, linewidth=0.8))

    # Weight legend (bottom-left)
    fig.text(0.18, 0.08, "Marker size = weight", fontsize=7,
             color=ZINC_400, ha="center", va="bottom", style="italic")

    save_fig(save_path)
