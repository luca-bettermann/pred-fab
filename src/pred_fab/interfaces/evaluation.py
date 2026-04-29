import numpy as np
from typing import final
from abc import ABC, abstractmethod
from numpy.typing import NDArray

from .base_interface import BaseInterface
from ..core import Parameters, Dataset
from ..core.data_objects import DataArray
from ..utils import PfabLogger


class IEvaluationModel(BaseInterface):
    """Abstract base for evaluation models that score features against target values."""

    # Class flag: set True if ``_compute_target_value`` and ``_compute_scaling_factor``
    # do not depend on the per-row dimension iterator values (i.e. target / scaling
    # are functions of ``params`` only, constant across rows of a single
    # ``compute_performance`` call). When True, ``compute_performance_batched``
    # takes a vectorised fast path: target/scaling are evaluated once per
    # candidate and the per-row arithmetic runs as numpy broadcasts across
    # ``(S, n_rows)``. Default False for backwards compatibility (per-row
    # variation supported via the scalar loop).
    TARGETS_CONSTANT: bool = False

    def __init__(self, logger: PfabLogger):
        super().__init__(logger)

    # === ABSTRACT METHODS ===

    # abstract methods from BaseInterface:
    # - input_parameters

    @property
    @abstractmethod
    def input_feature(self) -> str:
        """Code of the single input feature required by this evaluation model."""
        ...

    @property
    @abstractmethod
    def output_performance(self) -> str:
        """Code of the performance attribute produced by this evaluation model."""
        ...

    @abstractmethod
    def _compute_target_value(self, params: dict, **dimensions) -> float:
        """Compute the target (ideal) value for scoring at the given parameter context."""
        ...

    def _compute_scaling_factor(self, params: dict, **dimensions) -> float | None:
        """Optionally return a scaling factor for performance normalization; None uses target_value as denominator."""
        return None
    
    # === PUBLIC API ===

    @final
    def compute_performance(
        self,
        feature_array: NDArray,
        parameters: Parameters,
        feature_std: NDArray | None = None,
    ) -> tuple[float | None, list[float | None], list[float | None] | None]:
        """Score each row of feature_array against its target; returns (avg, per-row list, per-row std or None)."""
        # Unpack DataBlocks
        params = parameters.get_values_dict()
        # Derive iterator codes from the DataArray columns registered for the input feature.
        # Columns are [iterator_code, ..., feature_code]; all except the last are iterators.
        input_feat_obj = self._ref_features.get(self.input_feature)
        if input_feat_obj is not None and isinstance(input_feat_obj, DataArray):
            dim_iterator_codes = list(input_feat_obj.columns[:-1])
        else:
            dim_iterator_codes = []

        performance_list: list[float | None] = []
        std_list: list[float | None] = []

        for i, row in enumerate(feature_array):
            # Extract current dimension values
            current_dim = row[:-1]
            feature_value = row[-1]

            # merge dims and params into single dict
            current_dim_dict = dict(zip(dim_iterator_codes, current_dim))

            # Compute target value, scaling factor
            target_value = self._compute_target_value(params, **current_dim_dict)
            scaling_factor = self._compute_scaling_factor(params, **current_dim_dict)

            # Validate outputs from user implementation
            if not isinstance(target_value, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"_compute_target_value() must return numeric. "
                    f"Expected int/float, got {type(target_value).__name__}"
                )
            if scaling_factor is not None and not isinstance(scaling_factor, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"_compute_scaling_factor() must return numeric or None. "
                    f"Expected int/float/None, got {type(scaling_factor).__name__}"
                )

            # Compute performance value
            performance_value = self._compute_performance_value(feature_value, target_value, scaling_factor)
            performance_list.append(performance_value)

            # Propagate feature uncertainty to performance uncertainty if provided
            if feature_std is not None:
                sigma_feat = float(feature_std[i]) if feature_std[i] is not None else None
                if sigma_feat is not None and sigma_feat >= 0.0:
                    # Determine the same denominator used in _compute_performance_value
                    if scaling_factor is not None and scaling_factor > 0:
                        denom = float(scaling_factor)
                    elif target_value > 0:
                        denom = float(target_value)
                    else:
                        denom = None
                    std_perf = float(sigma_feat / denom) if denom else None
                else:
                    std_perf = None
                std_list.append(std_perf)

        performance_array = np.array(performance_list)
        avg_performance = float(np.nanmean(performance_array)) if len(performance_array) > 0 else None

        return avg_performance, performance_list, std_list if feature_std is not None else None

    @final
    def compute_performance_batched(
        self,
        feature_arrays_S: list[NDArray],
        parameters_list: list[Parameters],
    ) -> list[float | None]:
        """Vectorised ``compute_performance`` over S candidates.

        Default path loops the scalar ``compute_performance`` per candidate.
        Fast path (when ``TARGETS_CONSTANT=True``): target/scaling are computed
        once per candidate (no dim_dict), then perf arithmetic vectorises
        across all ``(S, n_rows)`` cells. Saves ``~S × n_rows`` Python calls
        per acquisition objective evaluation.

        Returns a list of ``avg_performance`` values, one per candidate. The
        per-row + std outputs of the scalar API are not exposed here — they
        aren't consumed by the calibration hot path.
        """
        S = len(feature_arrays_S)
        if S == 0:
            return []
        if not self.TARGETS_CONSTANT:
            return [
                self.compute_performance(arr, params)[0]
                for arr, params in zip(feature_arrays_S, parameters_list)
            ]

        # Fast path: vectorised. Stack feature arrays into (S, n_rows, n_cols).
        # All arrays share shape — same experiment topology per acquisition call.
        try:
            F_S = np.stack(feature_arrays_S, axis=0)
        except ValueError:
            # Heterogeneous shapes: fall back to scalar loop.
            return [
                self.compute_performance(arr, params)[0]
                for arr, params in zip(feature_arrays_S, parameters_list)
            ]
        feature_values = F_S[..., -1]  # (S, n_rows)

        # Compute target / scaling once per candidate (TARGETS_CONSTANT contract).
        S_actual = F_S.shape[0]
        targets = np.empty(S_actual, dtype=np.float64)
        denoms = np.empty(S_actual, dtype=np.float64)
        for s, params_obj in enumerate(parameters_list):
            params = params_obj.get_values_dict()
            t = float(self._compute_target_value(params))
            sc = self._compute_scaling_factor(params)
            targets[s] = t
            if sc is not None and sc > 0:
                denoms[s] = float(sc)
            elif t > 0:
                denoms[s] = t
            else:
                denoms[s] = np.nan  # neither valid → unscaled, returns NaN below

        # Broadcast (S, 1) against (S, n_rows). NaN feature values propagate;
        # nanmean ignores them. Clamp to [0, 1] matches the scalar path.
        diffs = np.abs(feature_values - targets[:, None])
        perfs = 1.0 - diffs / denoms[:, None]
        perfs = np.clip(perfs, 0.0, 1.0)
        nan_feat = np.isnan(feature_values)
        perfs = np.where(nan_feat, np.nan, perfs)

        with np.errstate(invalid='ignore'):
            avgs = np.nanmean(perfs, axis=1)

        return [None if np.isnan(a) else float(a) for a in avgs]

    @final
    def compute_performance_tensor(
        self,
        feature_values_S: "torch.Tensor",
        parameters_list: list[Parameters],
    ) -> "torch.Tensor":
        """Tensor-typed mirror of ``compute_performance_batched``. Returns ``(S,)``.

        ``feature_values_S`` is a 2-D tensor of shape ``(S, n_rows)`` — the
        value column only (no iterator indices), one row per cell, one set
        per candidate. Output is a ``(S,)`` tensor of mean performances,
        gradient-traversable when ``feature_values_S`` has ``requires_grad=True``.

        Math matches ``compute_performance_batched`` (the TARGETS_CONSTANT
        formulation):
            perf[s, i] = clamp(1 − |feat[s, i] − target[s]| / denom[s], 0, 1)
            avg[s]    = mean_i (perf[s, i])  ignoring NaN feature values
        Targets and scalings are computed in numpy (using user-defined
        ``_compute_target_value`` / ``_compute_scaling_factor`` which take
        Python dicts) — they're per-candidate constants, no gradient flows
        through them. Gradient flows from ``avg[s]`` back through
        ``feature_values_S[s, i]`` via the affine ``(feat - target) /
        denom`` and the ``torch.clamp`` (zero gradient outside [0, 1] —
        correct behaviour for saturated scores).

        Used by the gradient-based acquisition where the
        prediction tensor flows from a leaf params tensor and we need
        ``∂avg/∂feat`` to backprop further.
        """
        import torch  # local import — keep numpy-only consumers torch-free
        S = int(feature_values_S.shape[0])
        if S == 0:
            return torch.zeros(0, dtype=feature_values_S.dtype)

        # Per-candidate target + denom (numpy/Python — non-differentiable).
        targets = torch.empty(S, dtype=feature_values_S.dtype)
        denoms = torch.empty(S, dtype=feature_values_S.dtype)
        for s, params_obj in enumerate(parameters_list):
            params = params_obj.get_values_dict()
            t = float(self._compute_target_value(params))
            sc = self._compute_scaling_factor(params)
            targets[s] = t
            if sc is not None and sc > 0:
                denoms[s] = float(sc)
            elif t > 0:
                denoms[s] = t
            else:
                denoms[s] = float('nan')  # neither valid → NaN propagates to perf

        # Broadcast (S, 1) against (S, n_rows). NaN feature values propagate;
        # nanmean ignores them. Clamp to [0, 1] matches scalar path.
        diffs = (feature_values_S - targets[:, None]).abs()
        perfs = 1.0 - diffs / denoms[:, None]
        perfs = torch.clamp(perfs, 0.0, 1.0)

        # NaN-aware mean: zero-out NaN entries, count valid, divide.
        nan_mask = torch.isnan(feature_values_S)
        perfs_safe = torch.where(nan_mask, torch.zeros_like(perfs), perfs)
        valid_count = (~nan_mask).sum(dim=1).to(perfs.dtype)
        sum_perfs = perfs_safe.sum(dim=1)
        # Avoid divide-by-zero: candidates with all-NaN features yield NaN avg.
        safe_count = torch.where(valid_count > 0, valid_count, torch.ones_like(valid_count))
        avgs = sum_perfs / safe_count
        avgs = torch.where(valid_count > 0, avgs, torch.full_like(avgs, float('nan')))
        return avgs

    @final
    def _compute_performance_value(
        self, feature_value: float, target_value: float, scaling_factor: float | None
    ) -> float | None:
        """Return 1 − |feature − target| / denominator, clamped to [0, 1].

        denominator is scaling_factor when provided, else target_value.
        Returns None for NaN/None inputs.
        """
        # Handle missing values
        if feature_value is None or np.isnan(feature_value) or target_value is None:
            self.logger.warning("Feature or target is None/NaN, returning None")
            return None
        
        # Compute difference and normalize
        diff = feature_value - target_value
        if scaling_factor is not None and scaling_factor > 0:
            performance_value = 1.0 - np.abs(diff) / scaling_factor
        elif target_value > 0:
            performance_value = 1.0 - np.abs(diff) / target_value
        else:
            performance_value = np.abs(diff)
            self.logger.warning("Performance not scaled (target_value <= 0)")
        
        # Clamp to valid range
        if not 0 <= performance_value <= 1:
            self.logger.warning(f"Performance {performance_value:.3f} out of bounds, clamping")
            performance_value = np.clip(performance_value, 0, 1)
        
        self.logger.debug(
            f"Performance: feature={feature_value:.3f}, target={target_value:.3f}, "
            f"diff={diff:.3f}, scaling={scaling_factor}, perf={performance_value:.3f}"
        )
        return float(performance_value)

    # === WRAPPERS ===

    @final
    @property
    def input_features(self) -> list[str]:
        """Wrap input_feature scalar as a single-element list."""
        input_feat = self.input_feature
        if not isinstance(input_feat, str):
            raise TypeError(f"input_feature() must return str, got {type(input_feat).__name__}")
        return [input_feat]

    @final
    @property
    def outputs(self) -> list[str]:
        """Wrap output_performance scalar as a single-element list."""
        perf_code = self.output_performance
        if not isinstance(perf_code, str):
            raise TypeError(f"performance_code() must return str, got {type(perf_code).__name__}")
        return [perf_code]
