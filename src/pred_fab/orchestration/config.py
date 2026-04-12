"""Configuration dataclasses for agent.configure().

Group related parameters into typed, documented config objects.
All fields default to None — only non-None fields are applied.
"""

from dataclasses import dataclass, field
from typing import Any

from .calibration import Optimizer


@dataclass
class OptimizerConfig:
    """Optimizer backend and tuning parameters."""
    backend: Optimizer | None = None          # offline optimizer (DE or LBFGSB)
    online_backend: Optimizer | None = None   # online/adaptation optimizer
    de_maxiter: int | None = None             # DE: max generations (default 100)
    de_popsize: int | None = None             # DE: population size (default 10)
    lbfgsb_maxfun: int | None = None          # L-BFGS-B: max evals per start
    lbfgsb_eps: float | None = None           # L-BFGS-B: finite-diff step size


@dataclass
class ExplorationConfig:
    """Exploration and uncertainty parameters."""
    radius: float | None = None               # KDE bandwidth: h = c*sqrt(d)/sqrt(N)
    boundary_buffer: tuple[float, float, float] | None = None  # (extent, strength, exponent)


@dataclass
class TrajectoryConfig:
    """Trajectory and MPC parameters."""
    step_parameters: dict[str, str] | None = None  # {param_code: dim_code}
    adaptation_delta: dict[str, float] | None = None  # trust-region half-widths
    ofat_strategy: list[str] | None = None     # OFAT cycling order
    mpc_lookahead: int | None = None           # N-step lookahead (0 = greedy)
    mpc_discount: float | None = None          # discount factor (default 0.9)
    smoothing: float | None = None             # speed-change penalty (0 = off)
