"""Exploration concept: evidence gain + performance + acquisition topology.

Three side-by-side topologies showing the κ-blend:
  1. Evidence gain ΔE(x, y)  — where information is missing
  2. Performance P(x, y)     — where quality is high
  3. Acquisition A(x, y)     — what the optimizer sees

Usage with real model::

    cal = agent.calibration_system
    main(
        evidence_fn=...,  # params → evidence gain float (from KDE)
        score_fn=cal._compute_normalised_perf_for_params,  # params → combined [0,1]
        kappa=0.5,
        ...
    )

All callables accept a raw params dict. No wrapping needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _style import apply_style
from pred_fab.plotting._style import ZINC_500, save_fig
from panels import (
    draw_experiments, evidence_gain_topology,
    performance_topology, acquisition_topology,
)

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def main(
    save_path: str | None = None,
    evidence_fn: Callable[[dict[str, Any]], float] | None = None,
    score_fn: Callable[[dict[str, Any]], float] | None = None,
    kappa: float = 0.5,
    x_key: str = "V_fab",
    y_key: str = "calibrationFactor",
    x_bounds: tuple[float, float] = (0.05, 0.1),
    y_bounds: tuple[float, float] = (1.8, 2.2),
    x_label: str = "Print Speed [m/s]",
    y_label: str = "Calibration Factor",
    fixed_params: dict[str, Any] | None = None,
    datapoints: list[dict[str, float]] | None = None,
    resolution: int = 60,
):
    fixed = fixed_params or {}
    x_lo, x_hi = x_bounds
    y_lo, y_hi = y_bounds

    if evidence_fn is None:
        def evidence_fn(params):
            xn = (params.get(x_key, 0.075) - x_lo) / (x_hi - x_lo)
            yn = (params.get(y_key, 2.0) - y_lo) / (y_hi - y_lo)
            e = 0.6 * np.exp(-((xn-0.2)**2 + (yn-0.8)**2) / 0.06)
            e += 0.45 * np.exp(-((xn-0.85)**2 + (yn-0.15)**2) / 0.08)
            e += 0.3 * np.exp(-((xn-0.5)**2 + (yn-0.5)**2) / 0.15)
            return float(np.clip(e, 0, 1))

    if score_fn is None:
        def score_fn(params):
            xn = (params.get(x_key, 0.075) - x_lo) / (x_hi - x_lo)
            yn = (params.get(y_key, 2.0) - y_lo) / (y_hi - y_lo)
            f = 0.7 * np.exp(-((xn-0.4)**2 + (yn-0.55)**2) / 0.15)
            f += 0.35 * np.exp(-((xn-0.8)**2 + (yn-0.25)**2) / 0.2)
            return float(np.clip(1.0 - abs(f - 0.65) / 0.7, 0, 1))

    xs = np.linspace(x_lo, x_hi, resolution)
    ys = np.linspace(y_lo, y_hi, resolution)

    evidence_grid = np.zeros((resolution, resolution))
    perf_grid = np.zeros((resolution, resolution))
    acq_grid = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            params = dict(fixed)
            params[x_key] = float(xs[i])
            params[y_key] = float(ys[j])
            ev = evidence_fn(params)
            perf = score_fn(params)
            evidence_grid[j, i] = ev
            perf_grid[j, i] = perf
            acq_grid[j, i] = (1 - kappa) * perf + kappa * ev

    exp_x = [d[x_key] for d in datapoints] if datapoints else []
    exp_y = [d[y_key] for d in datapoints] if datapoints else []

    apply_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5))
    fig.subplots_adjust(wspace=0.30, left=0.04, right=0.97, bottom=0.12, top=0.90)

    evidence_gain_topology(fig, ax1, xs, ys, evidence_grid,
                           x_label, y_label, x_bounds, y_bounds)
    if exp_x:
        draw_experiments(ax1, exp_x, exp_y)

    performance_topology(fig, ax2, xs, ys, perf_grid,
                         x_label, y_label, x_bounds, y_bounds, show_optimum=False,
                         label="$P_{\\mathrm{sys}}(x, y)$")
    if exp_x:
        draw_experiments(ax2, exp_x, exp_y)

    acquisition_topology(fig, ax3, xs, ys, acq_grid,
                         x_label, y_label, x_bounds, y_bounds, kappa=kappa)
    if exp_x:
        draw_experiments(ax3, exp_x, exp_y)

    fig.text(0.5, 0.02,
             "$A = (1 - \\kappa) \\cdot P + \\kappa \\cdot \\Delta E$",
             ha="center", fontsize=11, color=ZINC_500)

    path = save_path or str(PLOTS_DIR / "exploration_acquisition.png")
    save_fig(path, dpi=200)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
