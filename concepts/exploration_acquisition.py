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
    panels: list[dict[str, Any]],
    kappa: float,
    save_path: str | None = None,
    proposed_params: dict[str, Any] | None = None,
    fit_colorbar: bool = True,
    sigma: float | None = None,
):
    """Render acquisition panels for one or more axis pairs.

    Each entry in *panels* is a dict with keys:
      xs, ys, evidence_grid, perf_grid, acq_grid,
      x_label, y_label, x_bounds, y_bounds, x_key, y_key,
      datapoints (list[dict])
    When multiple panels are provided they are stacked as rows with
    shared colorbar scales for direct comparability.
    """
    n_rows = len(panels)
    if n_rows == 0:
        return

    apply_style()
    fig, axes = plt.subplots(n_rows, 3, figsize=(16.5, 5 * n_rows), squeeze=False)
    fig.subplots_adjust(wspace=0.30, left=0.04, right=0.97, bottom=0.08, top=0.92)

    for r, p in enumerate(panels):
        xs, ys = p["xs"], p["ys"]
        xb = p.get("x_bounds") or (float(xs[0]), float(xs[-1]))
        yb = p.get("y_bounds") or (float(ys[0]), float(ys[-1]))
        xl, yl = p["x_label"], p["y_label"]
        xk, yk = p["x_key"], p["y_key"]
        dp = p.get("datapoints") or []
        exp_x = [d[xk] for d in dp]
        exp_y = [d[yk] for d in dp]

        ax1, ax2, ax3 = axes[r]

        # Scale sigma from [0,1] to physical x-axis units for radius circles
        sigma_phys = sigma * (xb[1] - xb[0]) if sigma else None

        evidence_gain_topology(fig, ax1, xs, ys, p["evidence_grid"],
                               xl, yl, xb, yb, fit_colorbar=True)
        if exp_x:
            draw_experiments(ax1, exp_x, exp_y, sigma=sigma_phys)

        performance_topology(fig, ax2, xs, ys, p["perf_grid"],
                             xl, yl, xb, yb, show_optimum=False,
                             label="Predicted $P_{\\mathrm{sys}}$",
                             fit_colorbar=True)
        if exp_x:
            draw_experiments(ax2, exp_x, exp_y, sigma=sigma_phys)

        acquisition_topology(fig, ax3, xs, ys, p["acq_grid"],
                             xl, yl, xb, yb, kappa=kappa)
        if proposed_params is not None:
            from matplotlib.patheffects import withStroke
            px, py = float(proposed_params[xk]), float(proposed_params[yk])
            ax3.scatter([px], [py], marker="x", c=ACCENT_YELLOW, s=80,
                        linewidths=1.8, zorder=12,
                        path_effects=[withStroke(linewidth=3, foreground="white")])
        if exp_x:
            draw_experiments(ax3, exp_x, exp_y, sigma=sigma_phys)

    path = save_path or str(PLOTS_DIR / "exploration_acquisition.png")
    save_fig(path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    raise RuntimeError("Requires pre-computed grids from cal.compute_acquisition_grids()")
