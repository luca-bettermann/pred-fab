"""Exploration concept: evidence gain + performance + acquisition topology.

Three side-by-side topologies showing the κ-blend:
  1. Evidence gain ΔE(x, y)  — sliced through the full-D KDE
  2. Performance P_sys(x, y) — sliced through the full-D model
  3. Acquisition A(x, y)     — (1-κ)·P_sys + κ·ΔE

Usage with real model::

    xs, ys, ev_grid, perf_grid, acq_grid = cal.compute_acquisition_grids(
        x_key, y_key, x_bounds, y_bounds,
        fixed_params=proposed_params,  # slice at proposed hidden-dim values
        kappa=0.5, resolution=60,
    )
    main(xs, ys, ev_grid, perf_grid, acq_grid, kappa=0.5, ...)

All grids computed by slicing through the real pipeline — same functions
the optimizer uses. No separate evidence/performance computation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _style import apply_style
from pred_fab.plotting._style import ACCENT_YELLOW, save_fig
from panels import (
    draw_experiments, evidence_gain_topology,
    performance_topology, acquisition_topology,
)

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def main(
    xs: np.ndarray,
    ys: np.ndarray,
    evidence_grid: np.ndarray,
    perf_grid: np.ndarray,
    acq_grid: np.ndarray,
    kappa: float,
    save_path: str | None = None,
    x_label: str = "Print Speed [m/s]",
    y_label: str = "Calibration Factor",
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
    proposed_params: dict[str, Any] | None = None,
    x_key: str = "V_fab",
    y_key: str = "calibrationFactor",
    datapoints: list[dict[str, float]] | None = None,
    fit_colorbar: bool = True,
):
    xb = x_bounds or (float(xs[0]), float(xs[-1]))
    yb = y_bounds or (float(ys[0]), float(ys[-1]))

    exp_x = [d[x_key] for d in datapoints] if datapoints else []
    exp_y = [d[y_key] for d in datapoints] if datapoints else []

    apply_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5))
    fig.subplots_adjust(wspace=0.30, left=0.04, right=0.97, bottom=0.12, top=0.90)

    evidence_gain_topology(fig, ax1, xs, ys, evidence_grid,
                           x_label, y_label, xb, yb)
    if exp_x:
        draw_experiments(ax1, exp_x, exp_y)

    performance_topology(fig, ax2, xs, ys, perf_grid,
                         x_label, y_label, xb, yb, show_optimum=False,
                         label="$P_{\\mathrm{sys}}(x, y)$",
                         fit_colorbar=fit_colorbar)
    if exp_x:
        draw_experiments(ax2, exp_x, exp_y)

    acquisition_topology(fig, ax3, xs, ys, acq_grid,
                         x_label, y_label, xb, yb, kappa=kappa)
    if proposed_params is not None:
        ax3.scatter([float(proposed_params[x_key])], [float(proposed_params[y_key])],
                    marker="x", c=ACCENT_YELLOW, s=100, linewidths=1.8, zorder=12)
    if exp_x:
        draw_experiments(ax3, exp_x, exp_y)

    path = save_path or str(PLOTS_DIR / "exploration_acquisition.png")
    save_fig(path, dpi=200)
    print(f"Saved: {path}")


if __name__ == "__main__":
    raise RuntimeError("Requires pre-computed grids from cal.compute_acquisition_grids()")
