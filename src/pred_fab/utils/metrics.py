"""Metrics utility for calculating performance metrics."""

from typing import Any

import numpy as np


# Importance-weighting defaults (R²_inf) — single source for the formula and
# its constants, shared by the prediction system and its plot.
IMPORTANCE_FLOOR = 0.1
IMPORTANCE_STEEPNESS = 0.8


def importance_weight(
    scores: Any,
    *,
    floor: float = IMPORTANCE_FLOOR,
    steepness: float = IMPORTANCE_STEEPNESS,
    ref_scores: Any = None,
) -> np.ndarray:
    """Performance-importance weight ``floor + (1−floor)·sigmoid(k·(s−mean))``,
    ``k = steepness/std``.

    ``mean``/``std``/``k`` are taken from ``ref_scores`` (default: ``scores``
    itself), so a plot can evaluate the same curve over an arbitrary range while
    anchoring the sigmoid to the experiment scores. The single source for the
    R²_inf importance weights and their plot.
    """
    s = np.asarray(scores, dtype=float)
    ref = s if ref_scores is None else np.asarray(ref_scores, dtype=float)
    mean = float(ref.mean()) if ref.size else 0.0
    std = float(ref.std()) if ref.size else 0.0
    k = steepness / std if std > 1e-10 else 0.0
    return floor + (1.0 - floor) / (1.0 + np.exp(-k * (s - mean)))


def combined_score(
    performance: dict[str, Any],
    weights: dict[str, float],
) -> Any:
    """Weighted combined performance score.

    Computes sum(w_i * perf_i) / sum(w_i) over exactly the keys that have a
    non-None performance value *and* a corresponding weight. The denominator
    is summed over the same contributing keys as the numerator, so a missing
    or NaN performance renormalises over what is present rather than deflating
    the score by the absent term's weight (callers such as the acquisition
    objective deliberately drop NaN performances before calling).
    Works with both Python floats and torch Tensors (preserves gradient).
    """
    contributing = [
        (weights[k], v)
        for k, v in performance.items()
        if v is not None and k in weights
    ]
    total_w = sum(w for w, _ in contributing)
    if total_w == 0:
        return 0.0
    score = sum(w * v for w, v in contributing)
    return score / total_w


class Metrics:
    """Static class for calculating regression metrics."""

    @staticmethod
    def _r2(y_true: np.ndarray, y_pred: np.ndarray,
            weights: np.ndarray | None = None) -> float:
        """Weighted R² score.  When *weights* is None, computes standard R²."""
        if weights is None:
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        else:
            w = weights / weights.sum()
            mean_w = float(np.sum(w * y_true))
            ss_res = float(np.sum(w * (y_true - y_pred) ** 2))
            ss_tot = float(np.sum(w * (y_true - mean_w) ** 2))

        if ss_tot < 1e-8:
            return 0.0 if ss_res > 1e-8 else 1.0
        return float(1.0 - ss_res / ss_tot)

    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Compute MAE and R²; returns dict with 'mae', 'r2', 'n_samples'."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) != len(y_pred):
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

        if len(y_true) == 0:
            return {'r2': 0.0, 'mae': 0.0, 'n_samples': 0}

        mae = float(np.mean(np.abs(y_true - y_pred)))
        r2 = Metrics._r2(y_true, y_pred)

        return {'r2': r2, 'mae': mae, 'n_samples': len(y_true)}

    @staticmethod
    def calculate_informed_r2(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        importance: np.ndarray,
    ) -> dict[str, float]:
        """Performance-weighted R² (a.k.a. predictive relevance / informed R²).

        Returns dict with 'r2', 'r2_inf', and 'n_samples'.

        The *importance* array contains pre-computed per-sample weights
        (typically in [floor, 1.0], see _build_importance_weights).
        Samples with higher weight contribute more to R²_inf.

        Interpretation of the gap (r2_inf - r2):
          gap > 0 -> high-importance samples predicted better
          gap < 0 -> high-importance samples predicted worse
          gap ~ 0 -> prediction quality is uniform across the space
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        importance = np.asarray(importance, dtype=float)

        if len(y_true) != len(y_pred) or len(y_true) != len(importance):
            raise ValueError(
                f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, "
                f"importance={len(importance)}"
            )

        n = len(y_true)
        if n == 0:
            return {'r2': 0.0, 'r2_inf': 0.0, 'n_samples': 0}

        r2 = Metrics._r2(y_true, y_pred)
        r2_inf = Metrics._r2(y_true, y_pred, weights=importance)

        return {'r2': r2, 'r2_inf': r2_inf, 'n_samples': n}
