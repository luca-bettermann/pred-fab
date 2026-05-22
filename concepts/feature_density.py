"""Feature density concept: two topology panels with different axis pairs.

Two side-by-side feature topologies for the same predicted feature,
each swept over a different pair of parameters (e.g. geometry vs extrusion).

Usage with real model::

    cal = agent.calibration_system
    main(
        predict_fn=cal._compute_perf_dict_for_params,
        feature_code="extrusion_consistency",
        left_axes=("layer_height", "path_offset", (0.003, 0.008), (0.03, 0.05),
                   "Layer Height [m]", "Path Offset [m]"),
        right_axes=("V_fab", "calibrationFactor", (0.05, 0.1), (1.8, 2.2),
                    "Print Speed [m/s]", "Calibration Factor"),
        fixed_params={"layer_height": 0.005, "path_offset": 0.04,
                      "V_fab": 0.075, "calibrationFactor": 2.0},
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
from pred_fab.plotting._style import save_fig
from panels import draw_experiments, feature_topology

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

AxisSpec = tuple[str, str, tuple[float, float], tuple[float, float], str, str]


def _build_grid(
    predict_fn: Callable[[dict[str, Any]], dict[str, float]],
    feature_code: str,
    x_key: str,
    y_key: str,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    fixed: dict[str, Any],
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(*x_bounds, resolution)
    ys = np.linspace(*y_bounds, resolution)
    grid = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            params = dict(fixed)
            params[x_key] = float(xs[i])
            params[y_key] = float(ys[j])
            features = predict_fn(params)
            grid[j, i] = features.get(feature_code, 0.0)
    return xs, ys, grid


def main(
    save_path: str | None = None,
    predict_fn: Callable[[dict[str, Any]], dict[str, float]] | None = None,
    feature_code: str = "extrusion_consistency",
    target_value: float | None = None,
    left_axes: AxisSpec = (
        "layer_height", "path_offset",
        (0.003, 0.008), (0.03, 0.05),
        "Layer Height [m]", "Path Offset [m]",
    ),
    right_axes: AxisSpec = (
        "V_fab", "calibrationFactor",
        (0.05, 0.1), (1.8, 2.2),
        "Print Speed [m/s]", "Calibration Factor",
    ),
    fixed_params: dict[str, Any] | None = None,
    experiments: list[dict[str, float]] | None = None,
    resolution: int = 80,
):
    fixed = fixed_params or {}

    if predict_fn is None:
        def predict_fn(params):
            vals = [float(v) for v in params.values() if isinstance(v, (int, float))]
            s = sum(vals) if vals else 1.0
            f = 0.5 + 0.3 * np.sin(s * 10.0)
            return {feature_code: float(np.clip(f, 0, 1))}

    lx_key, ly_key, lx_bounds, ly_bounds, lx_label, ly_label = left_axes
    rx_key, ry_key, rx_bounds, ry_bounds, rx_label, ry_label = right_axes

    lxs, lys, l_grid = _build_grid(
        predict_fn, feature_code, lx_key, ly_key, lx_bounds, ly_bounds,
        fixed, resolution,
    )
    rxs, rys, r_grid = _build_grid(
        predict_fn, feature_code, rx_key, ry_key, rx_bounds, ry_bounds,
        fixed, resolution,
    )

    apply_style()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(wspace=0.30, left=0.06, right=0.97, bottom=0.12, top=0.90)

    feature_topology(fig, ax_l, lxs, lys, l_grid,
                     lx_label, ly_label, lx_bounds, ly_bounds,
                     target_value=target_value,
                     label=f"$\\hat{{f}}$({lx_key}, {ly_key})")

    feature_topology(fig, ax_r, rxs, rys, r_grid,
                     rx_label, ry_label, rx_bounds, ry_bounds,
                     target_value=target_value,
                     label=f"$\\hat{{f}}$({rx_key}, {ry_key})")

    if experiments is not None:
        l_exp_x = [e[lx_key] for e in experiments if lx_key in e]
        l_exp_y = [e[ly_key] for e in experiments if ly_key in e]
        r_exp_x = [e[rx_key] for e in experiments if rx_key in e]
        r_exp_y = [e[ry_key] for e in experiments if ry_key in e]
        if l_exp_x:
            draw_experiments(ax_l, l_exp_x, l_exp_y)
        if r_exp_x:
            draw_experiments(ax_r, r_exp_x, r_exp_y)

    path = save_path or str(PLOTS_DIR / "feature_density.png")
    save_fig(path, dpi=200)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
