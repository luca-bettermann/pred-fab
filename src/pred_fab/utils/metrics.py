"""Metrics utility for calculating performance metrics."""

from typing import Any

import numpy as np


def combined_score(
    performance: dict[str, Any],
    weights: dict[str, float],
) -> float:
    """Weighted combined performance score.

    Computes sum(w_i * perf_i) / sum(w_i) over all keys in performance
    that have a non-None value and a corresponding weight.
    """
    total_w = sum(weights.values())
    if total_w == 0:
        return 0.0
    score = sum(
        weights.get(k, 0.0) * float(v)
        for k, v in performance.items() if v is not None
    )
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
        """Compute MAE, RMSE, and R²; returns dict with 'mae', 'rmse', 'r2', 'n_samples'."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) != len(y_pred):
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

        if len(y_true) == 0:
            return {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0, 'n_samples': 0}

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        r2 = Metrics._r2(y_true, y_pred)

        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'n_samples': len(y_true)}

    @staticmethod
    def calculate_adjusted_r2(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        importance: np.ndarray,
    ) -> dict[str, float]:
        """Performance-weighted R² that emphasizes high-importance samples.

        Returns dict with 'r2', 'r2_adj', and 'n_samples'.

        The *importance* array contains pre-computed per-sample weights
        (typically in [floor, 1.0], see _build_importance_weights).
        Samples with higher weight contribute more to R²_adj.

        Interpretation of the gap (r2_adj - r2):
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
            return {'r2': 0.0, 'r2_adj': 0.0, 'n_samples': 0}

        r2 = Metrics._r2(y_true, y_pred)
        r2_adj = Metrics._r2(y_true, y_pred, weights=importance)

        return {'r2': r2, 'r2_adj': r2_adj, 'n_samples': n}
