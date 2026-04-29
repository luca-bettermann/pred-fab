"""nn.Module-based normalisers.

Replaces the legacy dict-of-stats representation
(``{"method": NormMethod.STANDARD, "mean": 0.5, "std": 0.2}``) with proper
``nn.Module`` instances. Stats live in ``state_dict()`` so models serialise
via ``torch.save`` for free, and the affine transform composes cleanly with
autograd when applied to tensor inputs.

Each module supports both ``module(x)`` (forward) and ``module.reverse(x)``
APIs, plus dict-like ``module["mean"]`` / ``module.get("min")`` access for
backwards-compat readers that previously consumed the stat dict.

Mock-scale users get the same dict-like ergonomics; production-scale users
get gradient-traversable normalisation, GPU support via ``module.to('cuda')``,
and free serialisation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..utils import NormMethod


_EPSILON = 1e-8


class NormaliserModule(nn.Module):
    """Base class for normaliser modules with dict-like accessor.

    Subclasses set ``method`` (a NormMethod) and register stat buffers via
    ``register_buffer``. ``forward`` applies normalisation;
    ``reverse`` inverts it. Inputs may be numpy arrays or torch tensors —
    output type matches input.
    """
    method: NormMethod

    def reverse(self, x):  # pragma: no cover (overridden)
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        """Dict-like access for backwards compat with the legacy stat-dict callers."""
        if key == "method":
            return self.method
        # buffers stored as private (e.g. _mean) → expose mean / std / etc. via property names.
        attr_name = f"_{key}"
        if hasattr(self, attr_name):
            buf = getattr(self, attr_name)
            if isinstance(buf, torch.Tensor):
                return float(buf.item()) if buf.numel() == 1 else buf.detach().cpu().numpy()
        # also try direct attribute (for properties that compute from buffers)
        if hasattr(self, key):
            attr = getattr(self, key)
            if isinstance(attr, torch.Tensor):
                return float(attr.item()) if attr.numel() == 1 else attr.detach().cpu().numpy()
            return attr
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        if key == "method":
            return True
        if hasattr(self, f"_{key}") or hasattr(self, key):
            return True
        return False

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    @staticmethod
    def _to_tensor_like(x: Any) -> tuple[torch.Tensor, bool]:
        """Convert numpy array → torch.Tensor; return (tensor, was_numpy)."""
        if isinstance(x, torch.Tensor):
            return x, False
        return torch.as_tensor(x, dtype=torch.float64), True

    @staticmethod
    def _back(x: torch.Tensor, was_numpy: bool):
        if was_numpy:
            return x.detach().cpu().numpy()
        return x


class IdentityNormaliser(NormaliserModule):
    """No-op normaliser (NormMethod.NONE)."""
    method = NormMethod.NONE

    def forward(self, x):
        return x

    def reverse(self, x):
        return x


class StandardScalerModule(NormaliserModule):
    """``(x − mean) / (std + eps)``. NormMethod.STANDARD."""
    method = NormMethod.STANDARD

    def __init__(self, mean: float, std: float):
        super().__init__()
        self.register_buffer("_mean", torch.tensor(float(mean), dtype=torch.float64))
        self.register_buffer("_std", torch.tensor(float(std), dtype=torch.float64))

    @property
    def mean(self) -> float:
        return float(self._mean.item())  # type: ignore[union-attr]

    @property
    def std(self) -> float:
        return float(self._std.item())  # type: ignore[union-attr]

    def forward(self, x):
        x_t, was_np = self._to_tensor_like(x)
        mean_buf: torch.Tensor = self._mean  # type: ignore[assignment]
        std_buf: torch.Tensor = self._std    # type: ignore[assignment]
        out = (x_t - mean_buf.to(x_t.dtype)) / (std_buf.to(x_t.dtype) + _EPSILON)
        return self._back(out, was_np)

    def reverse(self, x):
        x_t, was_np = self._to_tensor_like(x)
        mean_buf: torch.Tensor = self._mean  # type: ignore[assignment]
        std_buf: torch.Tensor = self._std    # type: ignore[assignment]
        out = x_t * std_buf.to(x_t.dtype) + mean_buf.to(x_t.dtype)
        return self._back(out, was_np)


class MinMaxScalerModule(NormaliserModule):
    """``(x − min) / (max − min + eps)``, clamped to ``[0, 1]`` only when max==min. NormMethod.MIN_MAX."""
    method = NormMethod.MIN_MAX

    def __init__(self, min_val: float, max_val: float):
        super().__init__()
        self.register_buffer("_min", torch.tensor(float(min_val), dtype=torch.float64))
        self.register_buffer("_max", torch.tensor(float(max_val), dtype=torch.float64))

    @property
    def min(self) -> float:
        min_buf: torch.Tensor = self._min  # type: ignore[assignment]
        return float(min_buf.item())

    @property
    def max(self) -> float:
        max_buf: torch.Tensor = self._max  # type: ignore[assignment]
        return float(max_buf.item())

    def forward(self, x):
        x_t, was_np = self._to_tensor_like(x)
        min_buf: torch.Tensor = self._min  # type: ignore[assignment]
        max_buf: torch.Tensor = self._max  # type: ignore[assignment]
        denom = float(max_buf.item()) - float(min_buf.item())
        if abs(denom) < 1e-12:
            return self._back(torch.zeros_like(x_t), was_np)
        out = (x_t - min_buf.to(x_t.dtype)) / (
            (max_buf - min_buf).to(x_t.dtype) + _EPSILON
        )
        return self._back(out, was_np)

    def reverse(self, x):
        x_t, was_np = self._to_tensor_like(x)
        min_buf: torch.Tensor = self._min  # type: ignore[assignment]
        max_buf: torch.Tensor = self._max  # type: ignore[assignment]
        denom = float(max_buf.item()) - float(min_buf.item())
        if abs(denom) < 1e-12:
            return self._back(torch.full_like(x_t, float(min_buf.item())), was_np)
        out = x_t * (max_buf - min_buf).to(x_t.dtype) + min_buf.to(x_t.dtype)
        return self._back(out, was_np)


class RobustScalerModule(NormaliserModule):
    """``(x − median) / (IQR + eps)`` where IQR = q3 − q1. NormMethod.ROBUST."""
    method = NormMethod.ROBUST

    def __init__(self, median: float, q1: float, q3: float):
        super().__init__()
        self.register_buffer("_median", torch.tensor(float(median), dtype=torch.float64))
        self.register_buffer("_q1", torch.tensor(float(q1), dtype=torch.float64))
        self.register_buffer("_q3", torch.tensor(float(q3), dtype=torch.float64))

    @property
    def median(self) -> float:
        median_buf: torch.Tensor = self._median  # type: ignore[assignment]
        return float(median_buf.item())

    @property
    def q1(self) -> float:
        q1_buf: torch.Tensor = self._q1  # type: ignore[assignment]
        return float(q1_buf.item())

    @property
    def q3(self) -> float:
        q3_buf: torch.Tensor = self._q3  # type: ignore[assignment]
        return float(q3_buf.item())

    def forward(self, x):
        x_t, was_np = self._to_tensor_like(x)
        median_buf: torch.Tensor = self._median  # type: ignore[assignment]
        q1_buf: torch.Tensor = self._q1          # type: ignore[assignment]
        q3_buf: torch.Tensor = self._q3          # type: ignore[assignment]
        iqr = q3_buf - q1_buf
        out = (x_t - median_buf.to(x_t.dtype)) / (iqr.to(x_t.dtype) + _EPSILON)
        return self._back(out, was_np)

    def reverse(self, x):
        x_t, was_np = self._to_tensor_like(x)
        median_buf: torch.Tensor = self._median  # type: ignore[assignment]
        q1_buf: torch.Tensor = self._q1          # type: ignore[assignment]
        q3_buf: torch.Tensor = self._q3          # type: ignore[assignment]
        iqr = q3_buf - q1_buf
        out = x_t * iqr.to(x_t.dtype) + median_buf.to(x_t.dtype)
        return self._back(out, was_np)


def make_normaliser(method: NormMethod, data) -> NormaliserModule:
    """Fit a normaliser to ``data`` (numpy array or tensor) and return the module.

    Replaces the legacy ``DataModule._compute_normalization_stats``.
    """
    if method == NormMethod.NONE:
        return IdentityNormaliser()

    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        import numpy as np
        data_np = np.asarray(data)

    if method == NormMethod.STANDARD:
        import numpy as np
        return StandardScalerModule(mean=float(np.mean(data_np)), std=float(np.std(data_np)))
    if method == NormMethod.MIN_MAX:
        import numpy as np
        return MinMaxScalerModule(min_val=float(np.min(data_np)), max_val=float(np.max(data_np)))
    if method == NormMethod.ROBUST:
        import numpy as np
        return RobustScalerModule(
            median=float(np.median(data_np)),
            q1=float(np.percentile(data_np, 25)),
            q3=float(np.percentile(data_np, 75)),
        )
    raise ValueError(f"Unknown normalisation method: {method}")


def normaliser_from_dict(stats: dict[str, Any]) -> NormaliserModule:
    """Build a NormaliserModule from a legacy stats dict.

    Supports the on-disk JSON state format from prior PFAB versions —
    ``get_normalization_state`` writes legacy dicts; this rebuilds modules
    from them on load.
    """
    method = stats.get("method")
    if method == NormMethod.NONE or method is None:
        return IdentityNormaliser()
    if method == NormMethod.STANDARD:
        return StandardScalerModule(mean=stats["mean"], std=stats["std"])
    if method == NormMethod.MIN_MAX:
        return MinMaxScalerModule(min_val=stats["min"], max_val=stats["max"])
    if method == NormMethod.ROBUST:
        return RobustScalerModule(median=stats["median"], q1=stats["q1"], q3=stats["q3"])
    raise ValueError(f"Unknown normalisation method: {method}")
