"""Radar performance plot — per-attribute visualization with dataset comparison."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ._style import (
    FONT, PUBLICATION_DPI, save_fig, apply_style, EMERALD_300, EMERALD_500,
    STEEL_100, STEEL_500,
    ZINC_200, ZINC_300, ZINC_400, ZINC_500, ZINC_600, ZINC_700,
)


def radar_chart(
    ax,
    attribute_names: list[str],
    values: list[float] | None = None,
    stds: list[float] | None = None,
    *,
    score: float | None = None,
    score_std: float | None = None,
    ref_values: list[float] | None = None,
    ref_stds: list[float] | None = None,
    color: str = EMERALD_500,
    fill_color: str = EMERALD_300,
    ref_color: str = ZINC_400,
    label: str | None = None,
    ref_label: str | None = None,
) -> None:
    """General-purpose radar/spider chart on a polar axes.

    Modes:
      - values only → single primary polygon
      - values + stds → primary polygon with ±1σ band
      - values + ref_values (± stds) → two polygons with optional bands
      - ref_values only (values=None) → single polygon in ref_color

    The primary polygon uses ``color``/``fill_color`` (emerald by default).
    The reference polygon is subtle: dashed outline, no fill, soft zinc.
    When only ref_values is provided, it renders as the primary polygon
    using ref_color — for side-by-side comparison layouts.
    """
    if values is None and ref_values is not None:
        values = ref_values
        stds = ref_stds
        color = ref_color
        fill_color = ref_color
        label = ref_label
        ref_values = None
        ref_stds = None
        ref_label = None

    if values is None:
        return

    n = len(attribute_names)
    if n < 3:
        return

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                       fontsize=FONT["tick"] - 3, color=ZINC_400)
    ax.spines["polar"].set_color(ZINC_300)
    ax.grid(color=ZINC_300, linewidth=0.4, alpha=0.5)

    wrapped = [n.replace(" ", "\n") if "\n" not in n else n for n in attribute_names]
    ax.set_xticks(angles)
    ax.set_xticklabels(wrapped, fontsize=FONT["tick"], color=ZINC_600)
    ax.tick_params(axis="x", pad=14)

    def _close(v):
        return list(v) + [v[0]]

    CAP = 0.06

    def _draw_errorbar(ax_obj, angle, vlo, vhi, c, alpha, zorder):
        ax_obj.plot([angle, angle], [vlo, vhi], color=c,
                    linewidth=1.0, alpha=alpha, zorder=zorder)
        for end in (vlo, vhi):
            ax_obj.plot([angle - CAP, angle + CAP], [end, end],
                        color=c, linewidth=1.0, alpha=alpha, zorder=zorder)

    # Reference polygon (behind primary)
    if ref_values is not None:
        rc = _close(ref_values)
        ax.plot(angles_closed, rc, color=ref_color, linewidth=1.2,
                linestyle="--", alpha=0.6, zorder=3)
        if ref_stds is not None:
            lo = [max(0, v - s) for v, s in zip(ref_values, ref_stds)]
            hi = [min(1, v + s) for v, s in zip(ref_values, ref_stds)]
            ax.fill_between(angles_closed, _close(lo), _close(hi),
                            color=ref_color, alpha=0.08, zorder=2)
            for a, vlo, vhi in zip(angles, lo, hi):
                _draw_errorbar(ax, a, vlo, vhi, ref_color, 0.4, 3)
        for a, s in zip(angles, ref_values):
            ax.scatter([a], [s], c=ref_color, s=35, zorder=4,
                       edgecolors="white", linewidth=0.5, alpha=0.6)

    # Primary polygon
    vc = _close(values)
    ax.plot(angles_closed, vc, color=color, linewidth=1.8, zorder=5)
    ax.fill(angles_closed, vc, color=fill_color, alpha=0.15, zorder=4)

    if stds is not None:
        lo = [max(0, v - s) for v, s in zip(values, stds)]
        hi = [min(1, v + s) for v, s in zip(values, stds)]
        ax.fill_between(angles_closed, _close(lo), _close(hi),
                        color=color, alpha=0.10, zorder=4)
        for a, vlo, vhi in zip(angles, lo, hi):
            _draw_errorbar(ax, a, vlo, vhi, color, 0.4, 5)

    for a, s in zip(angles, values):
        ax.scatter([a], [s], c=color, s=40, zorder=6,
                   edgecolors="white", linewidth=0.6)

    # Legend: mean dot + ±σ error bar (symbols only, no labels)
    from matplotlib.lines import Line2D
    handles = []
    handles.append(Line2D([0], [0], marker="o", color="none",
                          markerfacecolor=color, markeredgecolor="white",
                          markersize=7, label="mean"))
    if stds is not None:
        handles.append(Line2D([0], [0], color=color, linewidth=1.0,
                              marker="|", markersize=6, markeredgewidth=1.0,
                              label="±σ"))
    ax.legend(handles=handles, loc="upper left", fontsize=FONT["legend"],
              frameon=False, markerscale=1.3,
              bbox_to_anchor=(-0.25, 1.15))

    if score is not None:
        s_text = f"$S$ = {score:.2f}"
        if score_std is not None:
            s_text += f" ± {score_std:.2f}"
        ax.text(1.15, 1.12, s_text, transform=ax.transAxes,
                ha="right", va="top", fontsize=FONT["title"], color=color)


@dataclass
class RadarPanel:
    """Configuration for one panel in a radar figure."""
    attribute_names: list[str]
    values: list[float] | None = None
    stds: list[float] | None = None
    ref_values: list[float] | None = None
    ref_stds: list[float] | None = None
    score: float | None = None
    score_std: float | None = None
    title: str | None = None
    label: str | None = None
    ref_label: str | None = None
    color: str = field(default=EMERALD_500)
    fill_color: str = field(default=EMERALD_300)
    ref_color: str = field(default=ZINC_400)


def plot_radar_panels(
    panels: list[RadarPanel],
    *,
    save_path: str | Path,
    figsize_per_panel: tuple[float, float] = (5.5, 5.5),
    suptitle: str | None = None,
) -> None:
    """Render N polar panels in a single row and save.

    Each panel is delegated to ``radar_chart``. Figure layout, save,
    and optional suptitle handled here.
    """
    n = len(panels)
    if n == 0:
        return

    apply_style()
    pw, ph = figsize_per_panel
    fig, axes = plt.subplots(1, n, figsize=(pw * n, ph),
                             subplot_kw=dict(projection="polar"), squeeze=False)

    for ax, panel in zip(axes[0], panels):
        radar_chart(
            ax, panel.attribute_names,
            values=panel.values, stds=panel.stds,
            ref_values=panel.ref_values, ref_stds=panel.ref_stds,
            score=panel.score, score_std=panel.score_std,
            color=panel.color, fill_color=panel.fill_color,
            ref_color=panel.ref_color,
            label=panel.label, ref_label=panel.ref_label,
        )
        if panel.title:
            ax.set_title(panel.title, fontsize=FONT["title"],
                         color=ZINC_700, pad=20)

    if suptitle:
        fig.suptitle(suptitle, fontsize=FONT["title"], color=ZINC_700)

    path = str(save_path)
    save_fig(path)
    print(f"Saved: {path}")


def plot_performance_radar(
    save_path: str,
    performance: dict[str, float],
    dataset_performances: list[dict[str, float]],
    weights: Mapping[str, float | int],
    *,
    combined_score: float | None = None,
    dataset_combined: float | None = None,
    exp_code: str = "",
) -> None:
    """Radar/spider plot of per-attribute performance with dataset average overlay."""
    apply_style()
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

    # ── Figure layout ──
    # Use gridspec: row 0 = title, row 1 = radar, row 2 = footer (legend + score)
    fig = plt.figure(figsize=(7, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.06, 0.78, 0.16], hspace=0.0)

    # Title area (reserved by gridspec but left empty)

    # ── Radar axes ──
    ax = fig.add_subplot(gs[1], projection="polar")
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
    ax.tick_params(axis="x", pad=18)

    # Dataset mean — background polygon
    ax.plot(angles_closed, mean_closed, "o-",
            color=ZINC_400, linewidth=1.2, markersize=4, alpha=0.5)
    ax.fill(angles_closed, mean_closed, alpha=0.10, color=ZINC_300)

    # Current experiment — foreground polygon (Emerald = performance)
    ax.plot(angles_closed, values_closed, "o-",
            color=EMERALD_500, linewidth=2.0, markersize=6,
            markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    ax.fill(angles_closed, values_closed, alpha=0.15, color=EMERALD_300)

    # ── Footer: legend (left) + system performance (right) ──
    legend_elements = [
        Line2D([0], [0], color=EMERALD_500, linewidth=2, marker="o",
               markersize=5, markeredgecolor="white", label="Experiment"),
        Line2D([0], [0], color=ZINC_400, linewidth=1.2, marker="o",
               markersize=4, alpha=0.5, label="Dataset avg"),
    ]
    fig.legend(handles=legend_elements, loc="lower left",
               bbox_to_anchor=(0.08, 0.02), fontsize=10,
               frameon=False, labelcolor=ZINC_500)

    if combined_score is not None:
        # Header
        fig.text(0.80, 0.13, "System Performance", fontsize=FONT["axis_label"],
                 color=ZINC_500, ha="center", va="bottom")
        # Experiment score (green)
        pct = combined_score * 100
        fig.text(0.80, 0.06, f"{pct:.0f}%", fontsize=28, fontweight="bold",
                 color=EMERALD_500, ha="center", va="bottom")
        # Dataset avg (grey)
        if dataset_combined is not None:
            avg_pct = dataset_combined * 100
            fig.text(0.80, 0.01, f"{avg_pct:.0f}%", fontsize=16,
                     color=ZINC_400, ha="center", va="bottom")

    save_fig(save_path)
