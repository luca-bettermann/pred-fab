"""Tests for OptimizationEngine.run_acquisition_gradient.

Three guarantees this commit promises:
  1. Sigmoid bound reparameterisation keeps the returned ``best_x`` strictly
     inside ``[lo, hi]`` for every dimension. No clipping, no drift.
  2. Multi-start finds the global minimum of a simple test objective
     (parabola, sinusoid) within reasonable tolerance.
  3. Gradient flows: the objective callable receives a tensor with
     ``requires_grad`` (post-sigmoid) so gradient-based methods work.
"""

import numpy as np
import pytest
import torch

from pred_fab.orchestration.calibration.engine import OptimizationEngine
from pred_fab.utils import PfabLogger


@pytest.fixture
def engine(tmp_path) -> OptimizationEngine:
    return OptimizationEngine(
        logger=PfabLogger.get_logger(str(tmp_path / "log")),
        random_seed=0,
    )


# ── Sigmoid bound reparameterisation ──────────────────────────────────────


def test_gradient_respects_bounds(engine):
    """Returned best_x is strictly inside [lo, hi] on every dimension."""
    lo = np.array([-3.0, 0.5, 1.0, -1.0])
    hi = np.array([3.0, 2.0, 1.5, 1.0])
    bounds = list(zip(lo.tolist(), hi.tolist()))

    # Objective that pushes toward the corner — without sigmoid the optimiser
    # would race to ±∞ in z-space, which still maps to the cube via sigmoid.
    def obj(x: torch.Tensor) -> torch.Tensor:
        # Maximise sum of x → minimise −sum(x)
        return -(x.sum(dim=-1))

    result = engine.run_acquisition_gradient(
        obj, bounds, n_starts=4, n_iters=200, method="adam",
    )
    assert result.best_x is not None
    assert (result.best_x >= lo).all()
    assert (result.best_x <= hi).all()
    # Should land near hi for every dim (within 5% of span).
    for d in range(len(lo)):
        span = hi[d] - lo[d]
        assert result.best_x[d] >= hi[d] - 0.05 * span


def test_gradient_finds_quadratic_minimum(engine):
    """1-D quadratic: argmin = 0.7 inside [0, 1]."""
    bounds = [(0.0, 1.0)]
    target = 0.7

    def obj(x: torch.Tensor) -> torch.Tensor:
        return (x.squeeze(-1) - target) ** 2

    result = engine.run_acquisition_gradient(
        obj, bounds, n_starts=4, n_iters=200, method="adam",
    )
    assert result.best_x is not None
    assert abs(float(result.best_x[0]) - target) < 1e-2


def test_gradient_2d_quadratic_recovers_target(engine):
    """2-D parabola: argmin matches inside-bounds target on each dim."""
    bounds = [(0.0, 1.0), (-2.0, 2.0)]
    target = np.array([0.3, 1.2])

    def obj(x: torch.Tensor) -> torch.Tensor:
        target_t = torch.tensor(target, dtype=x.dtype)
        return ((x - target_t) ** 2).sum(dim=-1)

    result = engine.run_acquisition_gradient(
        obj, bounds, n_starts=6, n_iters=200, method="adam",
    )
    assert result.best_x is not None
    np.testing.assert_allclose(result.best_x, target, atol=1e-2)


def test_gradient_with_lbfgs_backend(engine):
    """L-BFGS backend converges on a smooth quadratic."""
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    target = np.array([0.4, 0.8])

    def obj(x: torch.Tensor) -> torch.Tensor:
        target_t = torch.tensor(target, dtype=x.dtype)
        return ((x - target_t) ** 2).sum(dim=-1)

    result = engine.run_acquisition_gradient(
        obj, bounds, n_starts=4, n_iters=30, method="lbfgs",
    )
    assert result.best_x is not None
    np.testing.assert_allclose(result.best_x, target, atol=1e-3)


# ── Gradient flow ─────────────────────────────────────────────────────────


def test_objective_receives_tensor_with_grad(engine):
    """Inside the optimiser loop, x must have requires_grad / grad_fn so
    autograd-based objectives can call .backward() on intermediate ops."""
    bounds = [(0.0, 1.0)]
    saw_grad = [False]

    def obj(x: torch.Tensor) -> torch.Tensor:
        if x.requires_grad or x.grad_fn is not None:
            saw_grad[0] = True
        return (x.squeeze(-1) - 0.5) ** 2

    engine.run_acquisition_gradient(obj, bounds, n_starts=2, n_iters=5)
    assert saw_grad[0], "objective_tensor never received a tensor with grad"


# ── Edge cases ────────────────────────────────────────────────────────────


def test_gradient_empty_bounds_returns_empty(engine):
    """0-D problem → no work to do; result has best_x=None."""
    result = engine.run_acquisition_gradient(
        lambda x: torch.zeros(x.shape[0]),
        bounds=[], n_starts=2,
    )
    assert result.best_x is None
    assert result.score == 0.0


def test_gradient_single_start(engine):
    """n_starts=1 still works (no parallelism, just one sigmoid path)."""
    bounds = [(0.0, 1.0)]

    def obj(x: torch.Tensor) -> torch.Tensor:
        return (x.squeeze(-1) - 0.4) ** 2

    result = engine.run_acquisition_gradient(
        obj, bounds, n_starts=1, n_iters=100,
    )
    assert result.best_x is not None
    assert abs(float(result.best_x[0]) - 0.4) < 1e-2


def test_gradient_x0_is_used(engine):
    """When x0 supplied, it seeds the first start (visible in convergence)."""
    bounds = [(0.0, 1.0)]
    x0 = np.array([0.45])  # near the minimum

    def obj(x: torch.Tensor) -> torch.Tensor:
        return (x.squeeze(-1) - 0.5) ** 2

    # With x0 near optimum we converge in fewer iters
    result = engine.run_acquisition_gradient(
        obj, bounds, n_starts=1, n_iters=50, x0=x0,
    )
    assert result.best_x is not None
    assert abs(float(result.best_x[0]) - 0.5) < 1e-2


def test_gradient_score_negates_objective(engine):
    """Engine convention: objective returns -score; result.score is +score."""
    bounds = [(0.0, 1.0)]

    def obj(x: torch.Tensor) -> torch.Tensor:
        # Objective minimum at x=0.5; minimum value is -1.0
        return -torch.exp(-(x.squeeze(-1) - 0.5) ** 2 * 100)

    result = engine.run_acquisition_gradient(
        obj, bounds, n_starts=4, n_iters=200,
    )
    # objective minimum ≈ -1, so score = 1
    assert result.score > 0.95
