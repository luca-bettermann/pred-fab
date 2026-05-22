"""Global sensitivity analysis for prediction models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np


@dataclass
class SensitivityResult:
    """Result of a global sensitivity analysis (Sobol or Morris)."""
    method: str
    inputs: list[str]
    outputs: list[str]
    S_T: np.ndarray             # (n_outputs, n_inputs) total-order / mu_star indices
    S_T_conf: np.ndarray        # (n_outputs, n_inputs) confidence intervals / sigma
    S1: np.ndarray | None       # (n_outputs, n_inputs) first-order (Sobol only)
    S1_conf: np.ndarray | None  # (n_outputs, n_inputs) confidence (Sobol only)
    n_samples: int
    warnings: list[str] = field(default_factory=list)


def _evaluate_samples(
    X: np.ndarray,
    inputs: list[str],
    predict_fn: Callable[[dict[str, Any]], dict[str, float]],
    fixed_params: dict[str, Any],
    dimension_derivations: dict[str, Callable[[dict[str, Any]], int]],
    outputs: list[str] | None,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Evaluate predict_fn at all sample points. Returns (output_codes, {code: Y_array})."""
    n_total = X.shape[0]
    Y_dict: dict[str, list[float]] = {}
    resolved_outputs: list[str] | None = outputs

    for i in range(n_total):
        params = dict(fixed_params)
        for j, key in enumerate(inputs):
            params[key] = float(X[i, j])
        for axis_code, derive_fn in dimension_derivations.items():
            if axis_code not in params:
                params[axis_code] = derive_fn(params)

        result = predict_fn(params)

        if resolved_outputs is None:
            resolved_outputs = list(result.keys())
            for k in resolved_outputs:
                Y_dict[k] = []

        for k in resolved_outputs:
            val = result.get(k)
            Y_dict[k].append(float(val) if val is not None else 0.0)

    out_codes = resolved_outputs or []
    return out_codes, {k: np.array(v) for k, v in Y_dict.items()}


def _sobol(
    predict_fn: Callable[[dict[str, Any]], dict[str, float]],
    inputs: list[str],
    bounds: dict[str, tuple[float, float]],
    outputs: list[str] | None,
    n: int,
    seed: int,
    fixed_params: dict[str, Any],
    derivations: dict[str, Callable[[dict[str, Any]], int]],
) -> SensitivityResult:
    from SALib.sample import saltelli
    from SALib.analyze import sobol

    D = len(inputs)
    warnings: list[str] = []
    if n < 64 * D:
        warnings.append(f"Sample size N={n} may be too small for D={D} (recommended: N >= {64 * D})")

    problem = {"num_vars": D, "names": inputs, "bounds": [list(bounds[k]) for k in inputs]}
    X = saltelli.sample(problem, n, calc_second_order=False, seed=seed)
    out_codes, Y_dict = _evaluate_samples(X, inputs, predict_fn, fixed_params, derivations, outputs)

    n_out = len(out_codes)
    S_T = np.zeros((n_out, D))
    S_T_conf = np.zeros((n_out, D))
    S1 = np.zeros((n_out, D))
    S1_conf = np.zeros((n_out, D))

    for i, key in enumerate(out_codes):
        Y = Y_dict[key]
        if np.std(Y) < 1e-12:
            warnings.append(f"Output '{key}' has near-zero variance — sensitivity undefined")
            continue
        Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
        S_T[i, :] = Si["ST"]
        S_T_conf[i, :] = Si["ST_conf"]
        S1[i, :] = Si["S1"]
        S1_conf[i, :] = Si["S1_conf"]

    return SensitivityResult(
        method="sobol", inputs=inputs, outputs=out_codes,
        S_T=S_T, S_T_conf=S_T_conf, S1=S1, S1_conf=S1_conf,
        n_samples=X.shape[0], warnings=warnings,
    )


def _morris(
    predict_fn: Callable[[dict[str, Any]], dict[str, float]],
    inputs: list[str],
    bounds: dict[str, tuple[float, float]],
    outputs: list[str] | None,
    n: int,
    seed: int,
    fixed_params: dict[str, Any],
    derivations: dict[str, Callable[[dict[str, Any]], int]],
) -> SensitivityResult:
    from SALib.sample import morris as morris_sample
    from SALib.analyze import morris as morris_analyze

    D = len(inputs)
    warnings: list[str] = []
    problem = {"num_vars": D, "names": inputs, "bounds": [list(bounds[k]) for k in inputs]}
    X = morris_sample.sample(problem, n, seed=seed)
    out_codes, Y_dict = _evaluate_samples(X, inputs, predict_fn, fixed_params, derivations, outputs)

    n_out = len(out_codes)
    S_T = np.zeros((n_out, D))
    S_T_conf = np.zeros((n_out, D))

    for i, key in enumerate(out_codes):
        Y = Y_dict[key]
        if np.std(Y) < 1e-12:
            warnings.append(f"Output '{key}' has near-zero variance — sensitivity undefined")
            continue
        Si = morris_analyze.analyze(problem, X, Y, print_to_console=False)
        # mu_star = mean absolute elementary effect (analogous to S_T)
        S_T[i, :] = Si["mu_star"]
        S_T_conf[i, :] = Si["sigma"]

    return SensitivityResult(
        method="morris", inputs=inputs, outputs=out_codes,
        S_T=S_T, S_T_conf=S_T_conf, S1=None, S1_conf=None,
        n_samples=X.shape[0], warnings=warnings,
    )


def sobol_sensitivity(
    predict_fn: Callable[[dict[str, Any]], dict[str, float]],
    inputs: list[str],
    bounds: dict[str, tuple[float, float]],
    outputs: list[str] | None = None,
    n: int = 1024,
    seed: int = 0,
    method: Literal["sobol", "morris", "auto"] = "auto",
    dimension_derivations: dict[str, Callable[[dict[str, Any]], int]] | None = None,
    fixed_params: dict[str, Any] | None = None,
) -> SensitivityResult:
    """Run global sensitivity analysis on a prediction function.

    Parameters
    ----------
    predict_fn : params dict → {output_code: float}
    inputs : parameter codes to vary
    bounds : {param_code: (lo, hi)} for each input
    outputs : output codes to analyse (None = all from first prediction)
    n : base sample size
    seed : random seed
    method : "sobol" (rigorous, cost N*(D+2)), "morris" (cheap screening),
        or "auto" (sobol if D <= 30, morris otherwise)
    dimension_derivations : {axis_code: fn} to inject derived dimensions
    fixed_params : fixed values for non-varied parameters
    """
    try:
        import SALib  # noqa: F401
    except ImportError:
        raise ImportError("SALib is required: pip install 'pred-fab[diagnostics]'")

    D = len(inputs)
    fixed = dict(fixed_params or {})
    derivations = dimension_derivations or {}

    if method == "auto":
        method = "morris" if D > 30 else "sobol"

    if method == "morris":
        return _morris(predict_fn, inputs, bounds, outputs, n, seed, fixed, derivations)
    return _sobol(predict_fn, inputs, bounds, outputs, n, seed, fixed, derivations)


def filter_top_k(
    result: SensitivityResult,
    k_inputs: int | None = None,
    k_outputs: int | None = None,
) -> SensitivityResult:
    """Keep only the top-K most sensitive inputs and/or outputs.

    Ranks by max S_T across the other axis.
    """
    S_T = result.S_T
    inputs = list(result.inputs)
    outputs = list(result.outputs)

    if k_inputs is not None and k_inputs < len(inputs):
        col_max = S_T.max(axis=0)
        top_cols = np.argsort(col_max)[-k_inputs:][::-1]
        S_T = S_T[:, top_cols]
        inputs = [inputs[j] for j in top_cols]
        if result.S_T_conf is not None:
            result = SensitivityResult(
                method=result.method, inputs=inputs, outputs=outputs,
                S_T=S_T, S_T_conf=result.S_T_conf[:, top_cols],
                S1=result.S1[:, top_cols] if result.S1 is not None else None,
                S1_conf=result.S1_conf[:, top_cols] if result.S1_conf is not None else None,
                n_samples=result.n_samples, warnings=result.warnings,
            )

    if k_outputs is not None and k_outputs < len(outputs):
        row_max = S_T.max(axis=1)
        top_rows = np.argsort(row_max)[-k_outputs:][::-1]
        return SensitivityResult(
            method=result.method,
            inputs=inputs,
            outputs=[outputs[i] for i in top_rows],
            S_T=result.S_T[top_rows, :],
            S_T_conf=result.S_T_conf[top_rows, :],
            S1=result.S1[top_rows, :] if result.S1 is not None else None,
            S1_conf=result.S1_conf[top_rows, :] if result.S1_conf is not None else None,
            n_samples=result.n_samples,
            warnings=result.warnings,
        )

    return result
