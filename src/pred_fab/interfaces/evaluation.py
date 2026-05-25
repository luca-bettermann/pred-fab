"""Abstract interface for evaluation models that score features into performance."""

import numpy as np
from typing import Any, final
from abc import abstractmethod
from numpy.typing import NDArray

from .base_interface import BaseInterface
from ..core import Parameters
from ..core.data_objects import DataArray
from ..utils import PfabLogger


class IEvaluationModel(BaseInterface):
    """Abstract base for evaluation models that score features into performance.

    The user owns the scoring formula via ``_score_row`` (per-row numpy)
    and ``_score_tensor`` (batched torch for gradient acquisition).
    The framework orchestrates row iteration, batching, and NaN handling.

    Subclasses declare:
      - ``input_features: list[str]`` — one or more feature codes to score
      - ``output_performance: str`` — the performance attribute produced
      - ``_score_row(feature_values, params, **dims) -> float`` — scalar scoring
      - ``_score_tensor(feature_tensors, params_list) -> Tensor`` — gradient path
    """

    def __init__(self, logger: PfabLogger):
        super().__init__(logger)

    # === USER CONTRACT ===

    @property
    @abstractmethod
    def input_features(self) -> list[str]:
        """Feature codes consumed at measurement time — iterated by _score_row."""
        ...

    @property
    def acquisition_features(self) -> list[str]:
        """Feature codes consumed at acquisition time — looked up by _score_tensor.

        Defaults to input_features. Override when prediction-time scoring
        operates on a different feature granularity than measurement-time
        scoring (e.g., MLP predicts per-layer mean, measurement uses per-node).
        """
        return self.input_features

    @property
    @abstractmethod
    def output_performance(self) -> str:
        """Performance attribute code produced by this model."""
        ...

    @abstractmethod
    def _score_row(
        self,
        feature_values: dict[str, float],
        params: dict[str, Any],
        **dimensions: int,
    ) -> float:
        """Score one row. Returns a value in [0, 1].

        ``feature_values`` maps each ``input_features`` code to its scalar
        value at the current dimension context.
        """
        ...

    @abstractmethod
    def _score_tensor(
        self,
        feature_tensors: dict[str, "torch.Tensor"],
        parameters_list: list[Parameters],
    ) -> "torch.Tensor":
        """Batched tensor scoring for gradient acquisition. Returns ``(S,)``.

        ``feature_tensors`` maps each ``input_features`` code to an
        ``(S, n_rows)`` tensor. Must return ``(S,)`` mean scores,
        gradient-traversable.
        """
        ...

    # === FRAMEWORK ORCHESTRATION ===

    @final
    def compute_performance(
        self,
        feature_arrays: dict[str, NDArray],
        parameters: Parameters,
    ) -> tuple[float | None, list[float | None]]:
        """Score each row by calling ``_score_row``; returns (avg, per-row list).

        ``feature_arrays`` maps each ``input_features`` code to a 2-D table
        ``(n_rows, n_dims + 1)`` where the last column is the value.
        All arrays must share the same dimension columns and row count.
        """
        params = parameters.get_values_dict()

        first_code = self.input_features[0]
        first_array = feature_arrays[first_code]
        n_rows = first_array.shape[0]

        first_feat_obj = self._ref_features.get(first_code)
        if first_feat_obj is not None and isinstance(first_feat_obj, DataArray):
            dim_iterator_codes = list(first_feat_obj.columns[:-1])
        else:
            dim_iterator_codes = []

        performance_list: list[float | None] = []

        for i in range(n_rows):
            current_dim = first_array[i, :-1]
            current_dim_dict = dict(zip(dim_iterator_codes, current_dim))

            fv: dict[str, float] = {}
            has_nan = False
            for code in self.input_features:
                val = float(feature_arrays[code][i, -1])
                if np.isnan(val):
                    has_nan = True
                fv[code] = val

            if has_nan:
                performance_list.append(None)
                continue

            score = self._score_row(fv, params, **current_dim_dict)
            performance_list.append(float(np.clip(score, 0.0, 1.0)))

        perf_arr = np.array([v if v is not None else np.nan for v in performance_list])
        avg = float(np.nanmean(perf_arr)) if len(perf_arr) > 0 else None
        return avg, performance_list

    @final
    def compute_performance_tensor(
        self,
        feature_tensors: dict[str, "torch.Tensor"],
        parameters_list: list[Parameters],
    ) -> "torch.Tensor":
        """Call ``_score_tensor`` and handle NaN. Returns ``(S,)``.

        ``feature_tensors`` maps each ``input_features`` code to an
        ``(S, n_rows)`` tensor.
        """
        import torch
        first = next(iter(feature_tensors.values()))
        S = int(first.shape[0])
        if S == 0:
            return torch.zeros(0, dtype=first.dtype)

        avgs = self._score_tensor(feature_tensors, parameters_list)

        nan_mask = torch.isnan(avgs)
        if nan_mask.any():
            avgs = torch.where(nan_mask, torch.zeros_like(avgs), avgs)
        return avgs

    # === WRAPPERS (BaseInterface compatibility) ===

    @final
    @property
    def outputs(self) -> list[str]:
        """Wrap output_performance as a single-element list."""
        perf_code = self.output_performance
        if not isinstance(perf_code, str):
            raise TypeError(f"output_performance must return str, got {type(perf_code).__name__}")
        return [perf_code]
