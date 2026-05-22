"""Sobol global sensitivity analysis for prediction models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class SobolResult:
    """Result of a Sobol sensitivity analysis."""
    inputs: list[str]
    outputs: list[str]
    S_T: np.ndarray             # (n_outputs, n_inputs) total-order indices
    S_T_conf: np.ndarray        # (n_outputs, n_inputs) confidence intervals
    S1: np.ndarray              # (n_outputs, n_inputs) first-order indices
    S1_conf: np.ndarray         # (n_outputs, n_inputs) confidence intervals
    n_samples: int
    warnings: list[str] = field(default_factory=list)


def sobol_sensitivity(
    predict_fn: Callable[[dict[str, Any]], dict[str, float]],
    inputs: list[str],
    bounds: dict[str, tuple[float, float]],
    outputs: list[str] | None = None,
    n: int = 1024,
    seed: int = 0,
    dimension_derivations: dict[str, Callable[[dict[str, Any]], int]] | None = None,
    fixed_params: dict[str, Any] | None = None,
) -> SobolResult:
    """Run Sobol global sensitivity analysis on a prediction function.

    Parameters
    ----------
    predict_fn : params dict → {output_code: float}
        Model to analyse. Accepts a parameter dict, returns output dict.
    inputs : list of parameter codes to vary
    bounds : {param_code: (lo, hi)} for each input
    outputs : output codes to analyse (None = use all from first prediction)
    n : base sample size. Total evaluations = N * (D + 2).
    seed : random seed for reproducibility
    dimension_derivations : {axis_code: fn} to inject derived dimensions
    fixed_params : fixed values for non-varied parameters
    """
    try:
        from SALib.sample import saltelli
        from SALib.analyze import sobol
    except ImportError:
        raise ImportError("SALib is required: pip install SALib>=1.5")

    D = len(inputs)
    fixed = dict(fixed_params or {})
    derivations = dimension_derivations or {}
    warnings: list[str] = []

    if n < 64 * D:
        warnings.append(
            f"Sample size N={n} may be too small for D={D} inputs "
            f"(recommended: N >= {64 * D})"
        )

    problem = {
        "num_vars": D,
        "names": inputs,
        "bounds": [list(bounds[k]) for k in inputs],
    }

    X = saltelli.sample(problem, n, calc_second_order=False, seed=seed)
    n_total = X.shape[0]

    # Evaluate model at all sample points
    first_result = None
    Y_dict: dict[str, list[float]] = {}

    for i in range(n_total):
        params = dict(fixed)
        for j, key in enumerate(inputs):
            params[key] = float(X[i, j])
        for axis_code, derive_fn in derivations.items():
            if axis_code not in params:
                params[axis_code] = derive_fn(params)

        result = predict_fn(params)

        if first_result is None:
            first_result = result
            if outputs is None:
                outputs = list(result.keys())
            for out_key in outputs:
                Y_dict[out_key] = []

        for out_key in outputs:
            val = result.get(out_key)
            Y_dict[out_key].append(float(val) if val is not None else 0.0)

    if outputs is None:
        outputs = []

    n_outputs = len(outputs)
    S_T = np.zeros((n_outputs, D))
    S_T_conf = np.zeros((n_outputs, D))
    S1 = np.zeros((n_outputs, D))
    S1_conf = np.zeros((n_outputs, D))

    for i, out_key in enumerate(outputs):
        Y = np.array(Y_dict[out_key])
        if np.std(Y) < 1e-12:
            warnings.append(f"Output '{out_key}' has near-zero variance — sensitivity undefined")
            continue
        Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
        S_T[i, :] = Si["ST"]
        S_T_conf[i, :] = Si["ST_conf"]
        S1[i, :] = Si["S1"]
        S1_conf[i, :] = Si["S1_conf"]

    return SobolResult(
        inputs=inputs,
        outputs=outputs,
        S_T=S_T,
        S_T_conf=S_T_conf,
        S1=S1,
        S1_conf=S1_conf,
        n_samples=n_total,
        warnings=warnings,
    )
