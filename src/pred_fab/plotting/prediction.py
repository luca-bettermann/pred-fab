"""Prediction model quality plots."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from ..utils.metrics import importance_weight, IMPORTANCE_FLOOR, IMPORTANCE_STEEPNESS
from ._style import (
    AxisSpec, FONT, fig_size, save_fig, _add_fixed_subtitle,
    subplot_topology, apply_style, clean_spines, row_colorbar,
    ACCENT_RED, EMERALD_500,
    STEEL_300, STEEL_500, ZINC_300, ZINC_400, ZINC_600, ZINC_700,
)


def plot_topology_comparison(
    save_path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    x_values: np.ndarray,
    y_values: np.ndarray,
    grids: dict[str, np.ndarray],
    *,
    fixed_params: dict[str, Any] | None = None,
    fit_to_data: bool = False,
    evidence_grids: dict[str, np.ndarray] | None = None,
) -> None:
    """Side-by-side contour plots for comparing topologies on shared color scale.

    Shared scale is the bounded [0,1] default, or — with ``fit_to_data`` —
    bounds computed across *all* grids. ``evidence_grids`` (keyed like
    ``grids``) fades the matching panels where evidence is low — pass it for
    model-derived panels only.
    """
    apply_style()
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=fig_size(n, panel_w=4.6, panel_h=5.0),
                             layout="constrained", squeeze=False)
    axes = axes[0]
    _add_fixed_subtitle(fig, fixed_params)

    vmin = vmax = None
    if fit_to_data:
        all_vals = np.concatenate([g.ravel() for g in grids.values()])
        vmin, vmax = float(all_vals.min()), float(all_vals.max())

    im = None
    for ax, (label, data) in zip(axes, grids.items()):
        im = subplot_topology(ax, x_axis, y_axis, x_values, y_values, data,
                              cmap_name="performance", label=label,
                              vmin=vmin, vmax=vmax, fit_to_data=fit_to_data,
                              evidence_grid=(evidence_grids or {}).get(label),
                              show_colorbar=False)

    row_colorbar(fig, axes, im)
    save_fig(save_path)


def overlay_diagnosed_points(
    ax,
    diagnostic: Any,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    *,
    point_size: float = 32,
) -> None:
    """Overlay CV error-vs-coverage diagnosed points, quadrant-coded.

    ``diagnostic`` is an ``ErrorCoverageDiagnostic`` (or any iterable of
    ``DiagnosedPoint``-likes with ``params``/``label``). Color carries the
    quadrant: red = model problem, hollow = under-explored,
    emerald = trustworthy, zinc = sparse-but-ok. Opt-in — meant for
    prediction topologies, where it shows where the model was *checked*.
    """
    # Lazy import: the label constants' SSOT lives with the CV instrument,
    # but plotting must not pull orchestration (and torch) in at import time.
    from ..orchestration.cross_validation import (
        MODEL_PROBLEM, SPARSE_OK, TRUSTWORTHY, UNDER_EXPLORED,
    )
    styles: dict[str, tuple[str, str]] = {
        MODEL_PROBLEM:  (ACCENT_RED, "white"),
        UNDER_EXPLORED: ("white", ZINC_600),
        TRUSTWORTHY:    (EMERALD_500, "white"),
        SPARSE_OK:      (ZINC_400, "white"),
    }
    points = getattr(diagnostic, "points", diagnostic)
    for p in points:
        face, edge = styles.get(p.label, (ZINC_400, "white"))
        ax.scatter([float(p.params[x_axis.key])], [float(p.params[y_axis.key])],
                   c=face, edgecolors=edge, linewidths=0.7, s=point_size,
                   zorder=7)


def plot_importance_weights(
    save_path: str,
    experiment_scores: np.ndarray,
    floor: float = IMPORTANCE_FLOOR,
    steepness: float = IMPORTANCE_STEEPNESS,
    validation_gaps: dict[str, float] | None = None,
) -> None:
    """1x2: sigmoid importance curve with experiment dots + per-feature R²_inf gaps."""
    apply_style()
    scores = np.asarray(experiment_scores)

    # Same weighting as the prediction system (single source) — curve anchored
    # to the experiment scores via ref_scores.
    perf_range = np.linspace(0.0, 1.0, 200)
    weights_curve = importance_weight(
        perf_range, floor=floor, steepness=steepness, ref_scores=scores)
    exp_weights = importance_weight(
        scores, floor=floor, steepness=steepness)
    midpoint = (1.0 + floor) / 2.0
    # Display-only stats for the annotations (the weighting itself is computed
    # by importance_weight above).
    s_mean = float(scores.mean())
    s_std = float(scores.std())
    k = steepness / s_std if s_std > 1e-10 else 0.0

    n_panels = 2 if validation_gaps else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    ax1 = axes[0]
    ax1.plot(perf_range, weights_curve, color=STEEL_500, lw=2, zorder=1)
    ax1.scatter(scores, exp_weights, c=ACCENT_RED, s=40, zorder=3,
                edgecolors="darkred", linewidths=0.5,
                label=f"experiments (n={len(scores)})")
    ax1.axhline(midpoint, color=ZINC_400, ls="--", lw=0.8, alpha=0.5,
                label=f"midpoint = {midpoint:.2f}")
    ax1.axhline(floor, color=ZINC_400, ls=":", lw=0.8, alpha=0.5)
    ax1.axhline(1.0, color=ZINC_400, ls=":", lw=0.8, alpha=0.5)
    ax1.axvline(s_mean, color=ACCENT_RED, ls="--", lw=1, alpha=0.7,
                label=f"mean = {s_mean:.3f}")
    ax1.axvspan(s_mean - s_std, s_mean + s_std, alpha=0.06, color=ACCENT_RED)
    ax1.fill_between(perf_range, floor, weights_curve, alpha=0.08,
                     color=STEEL_300)
    ax1.set_xlabel("Combined Performance Score")
    ax1.set_ylabel("Importance Weight")
    ax1.set_title("Importance Weighting")
    ax1.text(0.02, 0.98, f"k = {steepness}/σ = {k:.1f}",
             transform=ax1.transAxes, ha="left", va="top",
             fontsize=FONT["annotation"], color=ZINC_400)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.08)
    ax1.legend(fontsize=FONT["legend"])
    ax1.grid(True, alpha=0.2, color=ZINC_300)
    clean_spines(ax1)

    if validation_gaps and n_panels == 2:
        ax2 = axes[1]
        features = list(validation_gaps.keys())
        gaps = [validation_gaps[f] for f in features]
        y_pos = np.arange(len(features))
        colors = [EMERALD_500 if g >= 0 else ACCENT_RED for g in gaps]
        ax2.barh(y_pos, gaps, height=0.6, color=colors, alpha=0.7)
        for i, (feat, g) in enumerate(zip(features, gaps)):
            sign = 1 if g >= 0 else -1
            ax2.text(g + 0.002 * sign, i, f"{g:+.4f}", va="center",
                     ha="left" if g >= 0 else "right", fontsize=8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(features, fontsize=8)
        ax2.invert_yaxis()
        ax2.axvline(0, color=ZINC_700, lw=1)
        ax2.set_xlabel("Gap (R²_inf − R²)")
        ax2.set_title("Actual Validation Gaps")
        max_gap = max(abs(g) for g in gaps) if gaps else 0.05
        margin = max(max_gap * 1.5, 0.02)
        ax2.set_xlim(-margin, margin)
        ax2.grid(True, alpha=0.2, axis="x", color=ZINC_300)
        clean_spines(ax2)

    save_fig(save_path)
