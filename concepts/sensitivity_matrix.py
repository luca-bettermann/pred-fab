"""Sensitivity analysis concept: Sobol total-order heatmap.

Usage with real model::

    from pred_fab.diagnostics import sobol_sensitivity
    from pred_fab.plotting import plot_sensitivity_matrix

    result = sobol_sensitivity(
        predict_fn=cal.predict_features,
        inputs=["V_fab", "calibrationFactor", "pathOffset", "H_layer"],
        bounds=param_bounds,
        n=1024,
        dimension_derivations=cal.dimension_derivations,
        fixed_params=fixed,
    )
    plot_sensitivity_matrix(
        "sensitivity.png", result.S_T, result.inputs, result.outputs,
        S_T_conf=result.S_T_conf,
    )
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def main(save_path: str | None = None):
    from pred_fab.diagnostics import sobol_sensitivity
    from pred_fab.plotting import plot_sensitivity_matrix

    def mock_predict(params):
        s = params.get("speed", 0.075)
        c = params.get("calib", 2.0)
        o = params.get("offset", 2.0)
        h = params.get("height", 2.5)
        return {
            "extrusion_consistency": 0.5 + 0.3 * np.sin(s * 40) + 0.1 * c,
            "robot_energy": 200 + 800 * s + 50 * h + 10 * o,
            "structural_integrity": 0.7 - 0.2 * abs(c - 2.0) + 0.1 * o,
        }

    result = sobol_sensitivity(
        predict_fn=mock_predict,
        inputs=["speed", "calib", "offset", "height"],
        bounds={
            "speed": (0.05, 0.1),
            "calib": (1.8, 2.2),
            "offset": (1.0, 3.0),
            "height": (2.0, 3.0),
        },
        n=1024,
    )

    for w in result.warnings:
        print(f"  ! {w}")

    path = save_path or str(PLOTS_DIR / "sensitivity_matrix.png")
    plot_sensitivity_matrix(
        path, result.S_T, result.inputs, result.outputs,
        S_T_conf=result.S_T_conf,
    )
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
