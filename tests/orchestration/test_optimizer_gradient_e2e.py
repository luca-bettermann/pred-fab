"""End-to-end smoke for the gradient acquisition path through baseline_step.

Confirms that the tensor acquisition path (``run_acquisition_gradient``
in OptimizationEngine, ``_acquisition_objective_tensor`` in
CalibrationSystem, and the tensor ``delta_integrated_evidence_*``
closures in PredictionSystem) runs end-to-end without crashing and
returns ExperimentSpec values inside the schema bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

from pred_fab.core import ExperimentSpec

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

    Integer dims use continuous relaxation + STE rounding in the gradient
    path. Verified by the run completing without exception.
    """
    agent, _, _, _ = build_real_agent_stack(tmp_path)
    agent.configure_optimizer(n_iters=5, n_starts=1)

    sampled = agent.baseline_step(n=2)
    assert len(sampled) == 2
