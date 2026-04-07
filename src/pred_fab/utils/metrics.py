"""Metrics utility for calculating performance metrics."""

import numpy as np
from typing import Dict, Optional


class Metrics:
    """Static class for calculating regression metrics."""

    @staticmethod
    def _r2(y_true: np.ndarray, y_pred: np.ndarray,
            weights: Optional[np.ndarray] = None) -> float:
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
    ) -> Dict[str, float]:
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
        alpha: float = 0.0,
        symmetric: bool = True,
    ) -> Dict[str, float]:
        """Performance-adjusted R² that up-weights high-importance samples.

        Returns dict with 'r2', 'r2_adj', and 'n_samples'.

        The *importance* array (one value per sample, in [0, 1]) expresses how
        relevant each sample is for the downstream optimization objective.
        When *symmetric* is True (default), the caller should compute
        ``importance_i = max(perf_true_i, perf_pred_i)`` so that both
        actually-good and predicted-good samples count.  When False, only
        ground-truth performance is used — better for cross-method comparison
        since weights become model-independent.

        *alpha* ∈ [0, 1] sets the minimum weight floor:
          ``weight_i = alpha + (1 − alpha) · importance_i``
        At alpha=0 (default): weights = importance (pure importance weighting).
        At alpha=1: weights = 1 for all samples (r2_adj = r2).

        Interpretation of the gap (r2_adj − r2):
          gap > 0  →  high-importance samples predicted better (exploration working)
          gap < 0  →  high-importance samples predicted worse
          gap ≈ 0  →  prediction quality is uniform across the space
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

        alpha = float(np.clip(alpha, 0.0, 1.0))

        # Up-weight important samples: floor at alpha, scale to 1 at importance=1
        weights = alpha + (1.0 - alpha) * importance

        r2 = Metrics._r2(y_true, y_pred)
        r2_adj = Metrics._r2(y_true, y_pred, weights=weights)

        return {'r2': r2, 'r2_adj': r2_adj, 'n_samples': n}
