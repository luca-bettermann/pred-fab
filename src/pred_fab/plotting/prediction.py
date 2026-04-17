"""Prediction model quality plots."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from ._style import AxisSpec, save_fig, _apply_axes, _add_fixed_subtitle, ACCENT_RED


def plot_prediction_scatter(
    save_path: str,
    model_results: dict[str, dict[str, Any]],
    *,
    title: str = "Prediction Accuracy",
) -> None:
    """Scatter of predicted vs actual for each feature.

    model_results: {feat: {"r2": float, "y_true": ndarray, "y_pred": ndarray}}
    """
    features = list(model_results.keys())
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, feat in zip(axes, features):
        r = model_results[feat]
        ax.scatter(r["y_true"], r["y_pred"], s=20, alpha=0.6,
                   c="#DD8452", edgecolors="white", linewidth=0.3)
        lims = [min(r["y_true"].min(), r["y_pred"].min()),
                max(r["y_true"].max(), r["y_pred"].max())]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{feat}\nR\u00b2 = {r['r2']:.3f}", fontsize=10)
        ax.grid(True, alpha=0.2)

    save_fig(save_path)


def plot_topology_comparison(
    save_path: str,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    x_values: np.ndarray,
    y_values: np.ndarray,
    grids: dict[str, np.ndarray],
    *,
    title: str = "Topology Comparison",
    fixed_params: dict[str, Any] | None = None,
) -> None:
    """Side-by-side contour plots for comparing topologies on shared color scale."""
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")
    _add_fixed_subtitle(fig, fixed_params)

    all_vals = np.concatenate([g.ravel() for g in grids.values()])
    vmin, vmax = all_vals.min(), all_vals.max()

    for ax, (label, data) in zip(axes, grids.items()):
        im = ax.contourf(x_values, y_values, data, levels=20, cmap="RdYlGn",
                          vmin=vmin, vmax=vmax)
        ax.contour(x_values, y_values, data, levels=10, colors="white",
                    linewidths=0.3, alpha=0.5)
        _apply_axes(ax, x_axis, y_axis)
        ax.set_title(label, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    save_fig(save_path)


def plot_importance_weights(
    save_path: str,
    experiment_scores: np.ndarray,
    floor: float = 0.1,
    steepness: float = 0.8,
    validation_gaps: dict[str, float] | None = None,
    *,
    title: str = "R\u00b2_adj Importance Weighting",
) -> None:
    """1x2: sigmoid importance curve with experiment dots + per-feature R\u00b2_adj gaps."""
    scores = np.asarray(experiment_scores)
    s_mean = float(scores.mean())
    s_std = float(scores.std())
    k = steepness / s_std if s_std > 1e-10 else 0.0

    perf_range = np.linspace(0.0, 1.0, 200)
    sigmoid_curve = 1.0 / (1.0 + np.exp(-k * (perf_range - s_mean)))
    weights_curve = floor + (1.0 - floor) * sigmoid_curve
    midpoint = (1.0 + floor) / 2.0

    exp_sigmoid = 1.0 / (1.0 + np.exp(-k * (scores - s_mean)))
    exp_weights = floor + (1.0 - floor) * exp_sigmoid

    n_panels = 2 if validation_gaps else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(perf_range, weights_curve, "b-", lw=2, zorder=1)
    ax1.scatter(scores, exp_weights, c=ACCENT_RED, s=40, zorder=3,
                edgecolors="darkred", linewidths=0.5,
                label=f"experiments (n={len(scores)})")
    ax1.axhline(midpoint, color="gray", ls="--", lw=0.8, alpha=0.5,
                label=f"midpoint = {midpoint:.2f}")
    ax1.axhline(floor, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax1.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax1.axvline(s_mean, color=ACCENT_RED, ls="--", lw=1, alpha=0.7,
                label=f"mean = {s_mean:.3f}")
    ax1.axvspan(s_mean - s_std, s_mean + s_std, alpha=0.06, color=ACCENT_RED)
    ax1.fill_between(perf_range, floor, weights_curve, alpha=0.08, color="blue")
    ax1.set_xlabel("Combined Performance Score")
    ax1.set_ylabel("Importance Weight")
    ax1.set_title(f"sigmoid(k\u00b7(perf \u2212 mean)),  k = {steepness}/\u03c3 = {k:.1f}")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.08)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    if validation_gaps and n_panels == 2:
        ax2 = axes[1]
        features = list(validation_gaps.keys())
        gaps = [validation_gaps[f] for f in features]
        y_pos = np.arange(len(features))
        colors = ["#2ca02c" if g >= 0 else "#d62728" for g in gaps]
        ax2.barh(y_pos, gaps, height=0.6, color=colors, alpha=0.7)
        for i, (feat, g) in enumerate(zip(features, gaps)):
            sign = 1 if g >= 0 else -1
            ax2.text(g + 0.002 * sign, i, f"{g:+.4f}", va="center",
                     ha="left" if g >= 0 else "right", fontsize=8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(features, fontsize=8)
        ax2.invert_yaxis()
        ax2.axvline(0, color="black", lw=1)
        ax2.set_xlabel("Gap (R\u00b2_adj \u2212 R\u00b2)")
        ax2.set_title("Actual Validation Gaps")
        max_gap = max(abs(g) for g in gaps) if gaps else 0.05
        margin = max(max_gap * 1.5, 0.02)
        ax2.set_xlim(-margin, margin)
        ax2.grid(True, alpha=0.2, axis="x")

    save_fig(save_path)
