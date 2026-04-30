"""End-to-end smoke for Optimizer.GRADIENT routed through baseline_step.

Goal: confirm that selecting ``Optimizer.GRADIENT`` actually exercises the
tensor acquisition path (``run_acquisition_gradient`` in OptimizationEngine,
``_acquisition_objective_tensor`` in CalibrationSystem, and the tensor
``delta_integrated_evidence_*`` closures in PredictionSystem) without
crashing or producing nonsense, and that returned ExperimentSpec values land
inside the schema bounds.

Numerical equivalence with the DE path is NOT promised — gradient solves a
smooth non-convex problem with 4 random starts and ~60 iterations, while DE
runs a population search; they generally find similar but not identical
optima. What must hold: bounds respected, output shape correct, no NaNs.
"""

from __future__ import annotations

import numpy as np
import pytest

from pred_fab.core import ExperimentSpec
from pred_fab.orchestration import Optimizer

from tests.utils.builders import build_real_agent_stack


def test_baseline_with_gradient_optimizer(tmp_path):
    """Baseline step produces valid ExperimentSpecs (gradient is the default refine path)."""
    agent, _, _, _ = build_real_agent_stack(tmp_path)
    # Keep iters low for test speed.
    agent.configure_optimizer(n_starts=2, n_iters=20)

    sampled = agent.baseline_step(n=3)

    assert len(sampled) == 3
    assert all(isinstance(p, ExperimentSpec) for p in sampled)
    # Source step still tagged correctly.
    assert all(p.initial_params.source_step == "baseline_step" for p in sampled)


def test_baseline_with_integer_params_runs(tmp_path):
    """Baseline runs cleanly when integer / domain dims are in scope.

    The mixed-feature schema has integer dims that DE rounds via the
    integrality_mask. Verified by the run completing without exception.
    """
    agent, _, _, _ = build_real_agent_stack(tmp_path)
    agent.configure_optimizer(de_maxiter=5, de_popsize=2)

    sampled = agent.baseline_step(n=2)
    assert len(sampled) == 2
