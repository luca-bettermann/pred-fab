"""Performance concept figure — predicted feature + scoring + performance topology.

Accepts callables for prediction and evaluation so consumers wire in
their actual models. The three panels show:
  1. Predicted feature surface over a 2D parameter slice
  2. Scoring function (feature → performance)
  3. Performance topology (the composition)
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

from ._style import (
    apply_style, clean_spines, save_fig, style_colorbar,
    AxisSpec, subplot_label,
    STEEL_500,
    EMERALD_500,
    ZINC_300, ZINC_400, ZINC_500, ZINC_700,
    ACCENT_YELLOW,
    FILL_ALPHA,
)


def plot_performance_concept(
    path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    predict_fn: Callable[[dict[str, Any]], dict[str, float]],
    score_fn: Callable[[dict[str, float], dict[str, Any]], float],
    feature_code: str,
    target_value: float,
    scaling: float,
    fixed_params: dict[str, Any],
    experiments: list[Any] | None = None,
    param_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    score_curve_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    score_curve_label: str | None = None,
    resolution: int = 40,
) -> None:
    """Generate the performance concept figure.

    Args:
        path: output file path
        x_axis / y_axis: parameter axes for the 2D slice
        predict_fn: ``(params_dict) -> {feat_code: predicted_value, ...}``
        score_fn: ``(feature_values_dict, params_dict) -> float in [0, 1]``
        feature_code: which predicted feature to show in panel 1
        target_value: target for the scoring function curve (panel 2)
        scaling: denominator for the scoring function curve
        fixed_params: values for non-plotted parameters
        experiments: optional list of ExperimentData / ExperimentSpec / dict
        param_transform: optional param dict transform
        resolution: grid resolution per axis
    """
    apply_style()
    transform = param_transform or (lambda d: d)

    x_lo, x_hi = x_axis.bounds  # type: ignore[misc]
    y_lo, y_hi = y_axis.bounds  # type: ignore[misc]
    xs = np.linspace(x_lo, x_hi, resolution)
    ys = np.linspace(y_lo, y_hi, resolution)

    feat_grid = np.zeros((resolution, resolution))
    perf_grid = np.zeros((resolution, resolution))

    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            params = dict(fixed_params)
            params[x_axis.key] = float(xv)
            params[y_axis.key] = float(yv)
            params = transform(params)

            features = predict_fn(params)
            feat_grid[j, i] = features.get(feature_code, 0.0)
            perf_grid[j, i] = score_fn(features, params)

    # Extract experiment locations
    exp_x, exp_y, exp_feat = [], [], []
    if experiments:
        for exp in experiments:
            if hasattr(exp, "parameters"):
                p = exp.parameters.get_values_dict()
            elif hasattr(exp, "initial_params"):
                p = dict(exp.initial_params.to_dict())
            else:
                p = dict(exp)
            p = transform(p)
            xv = p.get(x_axis.key)
            yv = p.get(y_axis.key)
            if xv is not None and yv is not None:
                exp_x.append(float(xv))
                exp_y.append(float(yv))
                exp_feat.append(predict_fn(p).get(feature_code, 0.0))

    opt_idx = np.unravel_index(np.argmax(perf_grid), perf_grid.shape)
    opt_xv, opt_yv = xs[opt_idx[1]], ys[opt_idx[0]]

    fig = plt.figure(figsize=(10.5, 4.2))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 0.5, 1.0],
                           wspace=0.35, left=0.06, right=0.95, bottom=0.12, top=0.88)

    # ── Panel 1: Feature prediction topology ──
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(xs, ys, feat_grid, levels=20, cmap="Blues")
    ax1.contour(xs, ys, feat_grid, levels=8, colors="white", linewidths=0.3, alpha=0.5)
    if exp_x:
        ax1.scatter(exp_x, exp_y, s=30, c="white", edgecolors=ZINC_700,
                    linewidth=0.5, zorder=5)
    tc = ax1.contour(xs, ys, feat_grid, levels=[target_value],
                     colors=[ZINC_300], linewidths=1.2, linestyles="--")
    ax1.clabel(tc, fmt=f"t={target_value:.2f}", fontsize=7, colors=ZINC_400)
    ax1.set_xlabel(x_axis.display_label)
    ax1.set_ylabel(y_axis.display_label)
    subplot_label(ax1, f"1. Predicted feature ({feature_code})")
    clean_spines(ax1)
    cb1 = plt.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
    style_colorbar(cb1)

    # ── Panel 2: Scoring function ──
    ax2 = fig.add_subplot(gs[0, 1])
    f_lo, f_hi = float(feat_grid.min()), float(feat_grid.max())
    feat_range = np.linspace(f_lo, f_hi, 200)
    if score_curve_fn is not None:
        perf_curve = np.clip(score_curve_fn(feat_range), 0, 1)
    else:
        perf_curve = np.clip(1.0 - np.abs(feat_range - target_value) / scaling, 0, 1)
    ax2.fill_between(feat_range, perf_curve, alpha=FILL_ALPHA["area"], color=EMERALD_500)
    ax2.plot(feat_range, perf_curve, color=EMERALD_500, linewidth=1.8)
    ax2.axvline(target_value, color=ZINC_300, linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.text(target_value, 1.02, f"t={target_value:.2f}", fontsize=7,
             color=ZINC_400, ha="center", va="bottom")
    if exp_feat:
        for fx in exp_feat[:6]:
            py = float(np.clip(
                score_curve_fn(np.array([fx]))[0] if score_curve_fn
                else max(0.0, 1.0 - abs(fx - target_value) / scaling),
                0, 1,
            ))
            ax2.plot(fx, py, "o", color=STEEL_500, markersize=3.5, zorder=5,
                     markeredgecolor="white", markeredgewidth=0.4)
            ax2.plot([fx, fx], [0, py], color=ZINC_300, linewidth=0.4, linestyle=":")
    ax2.set_xlabel("$\\hat{f}$")
    ax2.set_ylabel("$p$")
    subplot_label(ax2, "2. Scoring")
    ax2.set_ylim(0, 1.08)
    clean_spines(ax2)
    if score_curve_label:
        ax2.text(0.5, -0.18, score_curve_label,
                 fontsize=9, color=ZINC_500, ha="center", va="top", transform=ax2.transAxes)
    elif score_curve_fn is None:
        ax2.text(0.5, -0.18, "$p = 1 - \\frac{|\\hat{f} - t|}{s}$",
                 fontsize=9, color=ZINC_500, ha="center", va="top", transform=ax2.transAxes)

    # ── Panel 3: Performance topology ──
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.contourf(xs, ys, perf_grid, levels=20, cmap="RdYlGn", vmin=0, vmax=1)
    ax3.contour(xs, ys, perf_grid, levels=8, colors="white", linewidths=0.3, alpha=0.5)
    if exp_x:
        ax3.scatter(exp_x, exp_y, s=30, c="white", edgecolors=ZINC_700,
                    linewidth=0.5, zorder=5)
    ax3.scatter([opt_xv], [opt_yv], s=55, marker="x", c=ACCENT_YELLOW,
                linewidths=1.0, zorder=10)
    ax3.annotate("$x^*$", (opt_xv, opt_yv), xytext=(8, 8), textcoords="offset points",
                 fontsize=8, color=ZINC_700,
                 arrowprops=dict(arrowstyle="->", color=ZINC_400, lw=0.8))
    ax3.set_xlabel(x_axis.display_label)
    ax3.set_ylabel(y_axis.display_label)
    subplot_label(ax3, "3. Performance topology")
    clean_spines(ax3)
    cb3 = plt.colorbar(im3, ax=ax3, shrink=0.8, pad=0.02)
    style_colorbar(cb3)

    # ── Section labels ──
    fig.text(0.02, 0.96, "(a) Prediction and evaluation", fontsize=11,
             color=ZINC_700, fontweight="bold", va="top")
    fig.text(0.68, 0.96, "(b) Performance topology", fontsize=11,
             color=ZINC_700, fontweight="bold", va="top")

    arrow = FancyArrowPatch(
        (0.585, 0.5), (0.64, 0.5),
        transform=fig.transFigure, arrowstyle="->",
        color=ZINC_400, linewidth=1.2, mutation_scale=12,
    )
    fig.patches.append(arrow)

    save_fig(path, dpi=200)
