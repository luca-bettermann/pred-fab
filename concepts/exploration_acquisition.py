"""Exploration concept: evidence gain + performance + acquisition topology.

Three side-by-side topologies showing the κ-blend:
  1. Evidence gain ΔE(x, y)  — KernelField ANOVA integrated evidence
  2. Performance P_sys(x, y) — where quality is high
  3. Acquisition A(x, y)     — (1-κ)·P_sys + κ·ΔE

Usage with real model::

    from pred_fab.orchestration.evidence import compute_evidence_gain_grid

    xs, ys, ev_grid = compute_evidence_gain_grid(
        experiments, all_axes, x_key, y_key, sigma, resolution=60,
    )

    main(
        evidence_grid=ev_grid,
        score_fn=cal.system_performance,
        kappa=0.5,
        ...
    )

Evidence grid uses compute_evidence_gain_grid (KernelField ANOVA).
Performance is swept via score_fn. Acquisition: A = (1-κ)·P_sys + κ·ΔE.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _style import apply_style
from pred_fab.plotting._style import ZINC_500, ACCENT_YELLOW, save_fig
from panels import (
    draw_experiments, evidence_gain_topology,
    performance_topology, acquisition_topology,
)

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def main(
    evidence_grid: np.ndarray,
    score_fn: Callable[[dict[str, Any]], float],
    kappa: float,
    save_path: str | None = None,
    x_key: str = "V_fab",
    y_key: str = "calibrationFactor",
    x_bounds: tuple[float, float] = (0.05, 0.1),
    y_bounds: tuple[float, float] = (1.8, 2.2),
    x_label: str = "Print Speed [m/s]",
    y_label: str = "Calibration Factor",
    fixed_params: dict[str, Any] | None = None,
    proposed_params: dict[str, Any] | None = None,
    datapoints: list[dict[str, float]] | None = None,
    fit_colorbar: bool = True,
    resolution: int = 60,
):
    fixed = fixed_params or {}
    x_lo, x_hi = x_bounds
    y_lo, y_hi = y_bounds

    xs = np.linspace(x_lo, x_hi, resolution)
    ys = np.linspace(y_lo, y_hi, resolution)

    perf_grid = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            params = dict(fixed)
            params[x_key] = float(xs[i])
            params[y_key] = float(ys[j])
            perf_grid[j, i] = score_fn(params)

    ev_norm = evidence_grid / evidence_grid.max() if evidence_grid.max() > 0 else evidence_grid
    acq_grid = (1 - kappa) * perf_grid + kappa * ev_norm

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
                         label="$P_{\\mathrm{sys}}(x, y)$",
                         fit_colorbar=fit_colorbar)
    if exp_x:
        draw_experiments(ax2, exp_x, exp_y)

    acquisition_topology(fig, ax3, xs, ys, acq_grid,
                         x_label, y_label, x_bounds, y_bounds, kappa=kappa)
    if proposed_params is not None:
        ax3.scatter([float(proposed_params[x_key])], [float(proposed_params[y_key])],
                    marker="x", c=ACCENT_YELLOW, s=100, linewidths=1.8, zorder=12)
    if exp_x:
        draw_experiments(ax3, exp_x, exp_y)

    fig.text(0.5, 0.02,
             "$A = (1 - \\kappa) \\cdot P_{\\mathrm{sys}} + \\kappa \\cdot \\Delta E$",
             ha="center", fontsize=11, color=ZINC_500)

    path = save_path or str(PLOTS_DIR / "exploration_acquisition.png")
    save_fig(path, dpi=200)
    print(f"Saved: {path}")


if __name__ == "__main__":
    raise RuntimeError("Requires evidence_grid and score_fn — call main() directly.")
