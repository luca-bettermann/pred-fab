"""Parallel coordinates plot for experiment proposals.

Shows all parameters at once — one vertical axis per parameter, one
polyline per experiment. Existing experiments in Steel, proposals in Yellow.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._style import (
    apply_style, clean_spines, save_fig, AxisSpec,
    STEEL_500, STEEL_300, STEEL_100,
    ZINC_300, ZINC_400, ZINC_500, ZINC_600, ZINC_700,
    ACCENT_YELLOW,
    FONT,
)


def plot_parallel_coordinates(
    path: str,
    axes: list[AxisSpec],
    experiments: list[Any],
    *,
    highlight: set[str] | list[str] | None = None,
    param_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> None:
    """Parallel coordinates: one vertical axis per parameter, one line per experiment.

    Args:
        path: output file path
        axes: parameter axes (label, key, bounds, unit)
        experiments: ExperimentData / ExperimentSpec / dict
        highlight: experiment codes (or indices) to draw in Yellow;
            rest drawn in Steel. If None, all in Steel.
        param_transform: optional param dict transform
    """
    apply_style()
    transform = param_transform or (lambda d: d)
    n_axes = len(axes)
    if n_axes < 2:
        return

    highlight_set = set(highlight) if highlight else set()

    existing_lines: list[list[float]] = []
    proposed_lines: list[list[float]] = []

    for i, exp in enumerate(experiments):
        if hasattr(exp, "parameters"):
            p = transform(exp.parameters.get_values_dict())
            code = getattr(exp, "code", str(i))
        elif hasattr(exp, "initial_params"):
            p = transform(dict(exp.initial_params.to_dict()))
            code = str(i)
        else:
            p = transform(dict(exp))
            code = str(i)

        normed = []
        for ax in axes:
            val = p.get(ax.key)
            if val is None or ax.bounds is None:
                normed.append(0.5)
                continue
            lo, hi = ax.bounds
            span = hi - lo
            normed.append((float(val) - lo) / span if span > 0 else 0.5)

        if code in highlight_set or str(i) in highlight_set:
            proposed_lines.append(normed)
        else:
            existing_lines.append(normed)

    fig_w = max(4.0, 1.8 * n_axes)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))

    x_positions = np.arange(n_axes)

    for line in existing_lines:
        ax.plot(x_positions, line, color=STEEL_500, alpha=0.4, linewidth=1.5, zorder=2)
        for j, v in enumerate(line):
            ax.scatter([j], [v], c=STEEL_500, s=18, alpha=0.5, zorder=3,
                       edgecolors="white", linewidth=0.4)

    for line in proposed_lines:
        ax.plot(x_positions, line, color=ACCENT_YELLOW, alpha=0.9, linewidth=2.0, zorder=5)
        for j, v in enumerate(line):
            ax.scatter([j], [v], marker="x", c=ACCENT_YELLOW, s=30, zorder=6,
                       linewidths=1.0)

    ax.set_xlim(-0.3, n_axes - 0.7)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [a.display_label for a in axes],
        fontsize=FONT["axis_label"], color=ZINC_600,
    )

    for j, a in enumerate(axes):
        if a.bounds:
            lo, hi = a.bounds
            ax.text(j, -0.08, f"{lo}", ha="center", va="top",
                    fontsize=FONT["tick"], color=ZINC_500)
            ax.text(j, 1.08, f"{hi}", ha="center", va="bottom",
                    fontsize=FONT["tick"], color=ZINC_500)
        ax.plot([j, j], [0, 1], color=ZINC_300, linewidth=0.8, zorder=0)

    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if proposed_lines and existing_lines:
        ax.plot([], [], color=STEEL_500, linewidth=1.2, alpha=0.5, label="existing")
        ax.plot([], [], color=ACCENT_YELLOW, linewidth=2.0, label="proposed")
        ax.legend(fontsize=FONT["legend"], loc="upper right", frameon=False)

    save_fig(path)
