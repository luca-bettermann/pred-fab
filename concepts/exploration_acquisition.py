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
    draw_experiments, draw_proposal, evidence_gain_topology,
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
        pt_styles = p.get("styles")

        ax1, ax2, ax3 = axes[r]

        sigma_xy = (sigma * (xb[1] - xb[0]), sigma * (yb[1] - yb[0])) if sigma else None

        evidence_gain_topology(fig, ax1, xs, ys, p["evidence_grid"],
                               xl, yl, xb, yb, fit_colorbar=fit_colorbar)
        if exp_x:
            draw_experiments(ax1, exp_x, exp_y, sigma_xy=sigma_xy, styles=pt_styles)

        if r == 0:
            from pred_fab.plotting._style import ZINC_700, STEEL_500
            legend_items = [
                plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                           markeredgecolor=ZINC_700, markersize=6, label="Discovery"),
                plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=STEEL_500,
                           markeredgecolor=STEEL_500, markersize=6, label="Exploration"),
            ]
            if proposed_params is not None:
                from matplotlib.legend_handler import HandlerBase
                class _CrossHandler(HandlerBase):
                    def create_artists(self, legend, orig, xdescent, ydescent, w, h, fontsize, trans):
                        cx, cy = w / 2 - xdescent, h / 2 - ydescent
                        bg = plt.Line2D([cx], [cy], marker="x", color="none",
                                        markeredgecolor="black", markersize=8,
                                        markeredgewidth=2.8, transform=trans)
                        fg = plt.Line2D([cx], [cy], marker="x", color="none",
                                        markeredgecolor="white", markersize=7,
                                        markeredgewidth=1.5, transform=trans)
                        return [bg, fg]
                proposed_handle = plt.Line2D([0], [0], label="Proposed")
                legend_items.append(proposed_handle)
                handler_map = {proposed_handle: _CrossHandler()}
            hmap = handler_map if proposed_params is not None else {}
            ax1.legend(handles=legend_items, handler_map=hmap,
                       loc="upper left", fontsize=9,
                       frameon=True, framealpha=0.85, facecolor="white",
                       edgecolor="#D4D4D8", borderpad=0.8, markerscale=1.3)

        performance_topology(fig, ax2, xs, ys, p["perf_grid"],
                             xl, yl, xb, yb, show_optimum=False,
                             label="Predicted $P_{\\mathrm{sys}}$",
                             fit_colorbar=fit_colorbar)
        if exp_x:
            draw_experiments(ax2, exp_x, exp_y, sigma_xy=sigma_xy, styles=pt_styles)

        acquisition_topology(fig, ax3, xs, ys, p["acq_grid"],
                             xl, yl, xb, yb, kappa=kappa)
        if proposed_params is not None:
            draw_proposal(ax3, float(proposed_params[xk]), float(proposed_params[yk]))
        if exp_x:
            draw_experiments(ax3, exp_x, exp_y, sigma_xy=sigma_xy, styles=pt_styles)

    path = save_path or str(PLOTS_DIR / "exploration_acquisition.png")
    save_fig(path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    raise RuntimeError("Requires pre-computed grids from cal.compute_acquisition_grids()")
