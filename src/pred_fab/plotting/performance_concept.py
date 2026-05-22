"""Performance concept figure — predicted feature + scoring + performance topology.

Accepts model methods directly — no lambdas or wrappers needed:
  - ``predict_fn``: same signature as prediction model's predict
  - ``score_fn``: same signature as ``IEvaluationModel._score_row``
  - ``score_curve_fn``: optional, same signature as ``_score_row`` for panel 2 curve
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    fixed_params: dict[str, Any],
    *,
    score_curve_fn: Callable[[dict[str, float], dict[str, Any]], float] | None = None,
    score_curve_label: str | None = None,
    target_value: float | None = None,
    experiments: list[Any] | None = None,
    param_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    resolution: int = 40,
) -> None:
    """Generate the performance concept figure.

    Args:
        predict_fn: ``(params) -> {feat_code: value}``. Pass the model's
            predict method directly — the plot injects axis values into params.
        score_fn: ``(feature_values, params) -> float``. Same signature as
            ``IEvaluationModel._score_row``. Used for the performance topology.
        feature_code: which predicted feature to show in panel 1.
        fixed_params: values for non-plotted parameters (including dimension
            params like N_layers, N_nodes).
        score_curve_fn: optional scoring function for panel 2's 1D curve.
            Same ``_score_row`` signature. When None, uses ``score_fn``.
        score_curve_label: optional formula annotation below panel 2.
        target_value: optional target line on panels 1 and 2.
        experiments: optional list of ExperimentData / ExperimentSpec / dict.
        param_transform: optional param dict transform.
        resolution: grid resolution per axis.
    """
    apply_style()
    transform = param_transform or (lambda d: d)
    curve_fn = score_curve_fn or score_fn

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
    if target_value is not None:
        tc = ax1.contour(xs, ys, feat_grid, levels=[target_value],
                         colors=[ZINC_300], linewidths=1.2, linestyles="--")
        ax1.clabel(tc, fmt=f"t={target_value:.2f}", fontsize=7, colors=ZINC_400)
    ax1.set_xlabel(x_axis.display_label)
    ax1.set_ylabel(y_axis.display_label)
    subplot_label(ax1, f"1. Predicted feature ({feature_code})")
    clean_spines(ax1)
    cb1 = plt.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
    style_colorbar(cb1)

    # ── Panel 2: Scoring function (1D curve) ──
    ax2 = fig.add_subplot(gs[0, 1])
    f_lo, f_hi = float(feat_grid.min()), float(feat_grid.max())
    feat_range = np.linspace(f_lo, f_hi, 200)
    ref_params = dict(fixed_params)
    ref_params = transform(ref_params)
    perf_curve = np.array([
        float(np.clip(curve_fn({feature_code: float(fv)}, ref_params), 0, 1))
        for fv in feat_range
    ])
    ax2.fill_between(feat_range, perf_curve, alpha=FILL_ALPHA["area"], color=EMERALD_500)
    ax2.plot(feat_range, perf_curve, color=EMERALD_500, linewidth=1.8)
    if target_value is not None:
        ax2.axvline(target_value, color=ZINC_300, linewidth=0.8, linestyle="--", alpha=0.5)
        ax2.text(target_value, 1.02, f"t={target_value:.2f}", fontsize=7,
                 color=ZINC_400, ha="center", va="bottom")
    if exp_feat:
        for fx in exp_feat[:6]:
            py = float(np.clip(curve_fn({feature_code: fx}, ref_params), 0, 1))
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

    save_fig(path, dpi=200)
