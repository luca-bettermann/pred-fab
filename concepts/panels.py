"""Reusable topology panels for concept figures.

Each function renders one panel on a provided axes. Concept scripts
compose these into multi-panel figures without duplicating rendering code.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from _style import (
    clean_spines, subplot_label, style_colorbar, cmap,
    ZINC_300, ZINC_400, ZINC_500, ZINC_600, ZINC_700,
    ACCENT_YELLOW,
)
from pred_fab.plotting._style import FONT, FILL_ALPHA, surface as get_surface


def draw_experiments(ax, exp_x: list[float], exp_y: list[float]) -> None:
    """White dots with Zinc-700 edge — baseline experiment style."""
    for ex, ey in zip(exp_x, exp_y):
        ax.scatter([ex], [ey], c="white", s=30, edgecolors=ZINC_700,
                   linewidth=0.5, zorder=10)


def setup_axes(ax, x_label: str, y_label: str,
               x_bounds: tuple[float, float], y_bounds: tuple[float, float]) -> None:
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xlabel(x_label, fontsize=FONT["axis_label"], color=ZINC_600)
    ax.set_ylabel(y_label, fontsize=FONT["axis_label"], color=ZINC_600)
    clean_spines(ax)


def feature_topology(
    fig, ax,
    xs: np.ndarray, ys: np.ndarray, grid: np.ndarray,
    x_label: str, y_label: str,
    x_bounds: tuple[float, float], y_bounds: tuple[float, float],
    target_value: float | None = None,
    label: str = "$\\hat{f}(x, y)$",
) -> None:
    """Predicted feature surface — Greys (density, raw data)."""
    cm_f = cmap("density")
    norm_f = Normalize(vmin=0.0, vmax=float(grid.max()))
    levels = np.linspace(0.02, float(grid.max()), 18)
    ax.contourf(xs, ys, grid, levels=levels, cmap=cm_f, norm=norm_f, alpha=0.8)
    ax.contour(xs, ys, grid, levels=8, colors=[ZINC_300], linewidths=0.4)
    if target_value is not None:
        tc = ax.contour(xs, ys, grid, levels=[target_value],
                        colors=[ZINC_300], linewidths=1.2, linestyles="--")
        ax.clabel(tc, fmt=f"t={target_value:.2f}", fontsize=7, colors=ZINC_400)
    subplot_label(ax, label)
    setup_axes(ax, x_label, y_label, x_bounds, y_bounds)
    sm = ScalarMappable(norm=norm_f, cmap=cm_f)
    cb = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.06)
    style_colorbar(cb)


def performance_topology(
    fig, ax,
    xs: np.ndarray, ys: np.ndarray, grid: np.ndarray,
    x_label: str, y_label: str,
    x_bounds: tuple[float, float], y_bounds: tuple[float, float],
    show_optimum: bool = True,
    label: str = "$P(x, y)$",
    fit_colorbar: bool = True,
) -> None:
    """Performance surface — RdYlGn (quality judgment)."""
    cm_p = cmap("performance")
    if fit_colorbar:
        norm_p = Normalize(vmin=float(grid.min()), vmax=float(grid.max()))
    else:
        norm_p = Normalize(vmin=0, vmax=1)
    ax.contourf(xs, ys, grid, levels=18, cmap=cm_p, norm=norm_p, alpha=0.8)
    ax.contour(xs, ys, grid, levels=8, colors="white", linewidths=0.3, alpha=0.4)
    if show_optimum:
        opt_idx = np.unravel_index(np.argmax(grid), grid.shape)
        opt_xv, opt_yv = xs[opt_idx[1]], ys[opt_idx[0]]
        ax.scatter([opt_xv], [opt_yv], marker="x", c=ACCENT_YELLOW, s=55,
                   linewidths=1.0, zorder=10)
        ax.annotate("$x^*$", (opt_xv, opt_yv), xytext=(8, 8),
                    textcoords="offset points", fontsize=8, color=ZINC_700,
                    arrowprops=dict(arrowstyle="->", color=ZINC_400, lw=0.8))
    subplot_label(ax, label)
    setup_axes(ax, x_label, y_label, x_bounds, y_bounds)
    sm = ScalarMappable(norm=norm_p, cmap=cm_p)
    cb = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.06)
    style_colorbar(cb)


def evidence_gain_topology(
    fig, ax,
    xs: np.ndarray, ys: np.ndarray, grid: np.ndarray,
    x_label: str, y_label: str,
    x_bounds: tuple[float, float], y_bounds: tuple[float, float],
    label: str = "$\\Delta E(x, y)$",
    fit_colorbar: bool = True,
) -> None:
    """Evidence gain surface — YlGn (exploration signal)."""
    cm_e = cmap("evidence_gain")
    if fit_colorbar:
        norm_e = Normalize(vmin=float(grid.min()), vmax=float(grid.max()))
    else:
        norm_e = Normalize(vmin=0, vmax=1)
    ax.contourf(xs, ys, grid, levels=18, cmap=cm_e, norm=norm_e, alpha=0.8)
    ax.contour(xs, ys, grid, levels=8, colors="white", linewidths=0.3, alpha=0.4)
    subplot_label(ax, label)
    setup_axes(ax, x_label, y_label, x_bounds, y_bounds)
    sm = ScalarMappable(norm=norm_e, cmap=cm_e)
    cb = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.06)
    style_colorbar(cb)


def acquisition_topology(
    fig, ax,
    xs: np.ndarray, ys: np.ndarray, grid: np.ndarray,
    x_label: str, y_label: str,
    x_bounds: tuple[float, float], y_bounds: tuple[float, float],
    kappa: float = 0.5,
    label: str | None = None,
) -> None:
    """Acquisition surface — magma (combined objective)."""
    cm_a = cmap("acquisition")
    norm_a = Normalize(vmin=float(grid.min()), vmax=float(grid.max()))
    ax.contourf(xs, ys, grid, levels=18, cmap=cm_a, norm=norm_a, alpha=0.8)
    ax.contour(xs, ys, grid, levels=8, colors="white", linewidths=0.3, alpha=0.4)

    subplot_label(ax, label or f"$A(x, y)$   $\\kappa={kappa}$")
    setup_axes(ax, x_label, y_label, x_bounds, y_bounds)
    sm = ScalarMappable(norm=norm_a, cmap=cm_a)
    cb = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.06)
    style_colorbar(cb)


def marginal_performance(
    ax, vals: np.ndarray, curve: np.ndarray,
    axis_label: str, panel_label: str,
    fit_colorbar: bool = True,
) -> None:
    """Marginal performance curve with gradient fill — RdYlGn."""
    cm_p = cmap("performance")
    line_color = cm_p(0.7)
    if fit_colorbar:
        y_min, y_max = float(curve.min()), float(curve.max())
        pad = (y_max - y_min) * 0.05 or 0.05
        y_lo, y_hi = y_min - pad, y_max + pad
    else:
        y_lo, y_hi = 0.0, 1.0
    res_y = 100
    extent = [float(vals[0]), float(vals[-1]), y_lo, y_hi]
    gradient = np.linspace(0, 1, res_y).reshape(-1, 1) * np.ones((1, len(vals)))
    curve_norm = (curve - y_lo) / (y_hi - y_lo)
    gradient = gradient * curve_norm[None, :]
    norm_fill = Normalize(vmin=0, vmax=1)
    ax.imshow(gradient, aspect="auto", origin="lower", extent=extent,
              cmap=cm_p, norm=norm_fill, alpha=0.7, zorder=0)
    ax.fill_between(vals, curve, y_hi, color="white", zorder=1)
    ax.plot(vals, curve, color=line_color, linewidth=1.5)
    ax.set_xlim(float(vals[0]), float(vals[-1]))
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(axis_label, fontsize=FONT["axis_label"], color=ZINC_600)
    subplot_label(ax, panel_label)
    clean_spines(ax)
