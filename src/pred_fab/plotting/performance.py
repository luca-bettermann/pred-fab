"""Radar performance plot — unified radar_chart + plot_radar_panels."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ._style import (
    FONT, save_fig, apply_style, EMERALD_300, EMERALD_500,
    ZINC_300, ZINC_400, ZINC_500, ZINC_600, ZINC_700,
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

    save_fig(str(save_path))
