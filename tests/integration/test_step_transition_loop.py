"""End-to-end agent step-transition loop on a single agent.

The per-step contract tests (``test_agent_step_contracts``) rebuild a fresh
stack for each step and the workflow test stops at one exploration step.
Neither drives the full sequence on one carried-forward agent. This does:
discovery → (evaluate + train) → exploration → inference, asserting the
``SourceStep`` each phase stamps and that the loop keeps producing valid
proposals across iterations. ``adaptation_step`` (not yet implemented) is
confirmed excluded.
"""

import pytest

from pred_fab.core import ExperimentSpec
from pred_fab.utils import SourceStep
from tests.utils.builders import build_real_agent_stack


def _prime(agent, exp, datamodule):
    """Evaluate the seed experiment and train, so calibration has a model + KDE."""
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)


def test_discovery_to_exploration_to_inference_transition(tmp_path):
    """One agent advances DISCOVERY → EXPLORATION → INFERENCE, each stamping the
    matching source_step."""
    agent, _dataset, exp, datamodule = build_real_agent_stack(tmp_path)

    discovery = agent.discovery_step(n=3)
    assert len(discovery) == 3
    assert all(isinstance(s, ExperimentSpec) for s in discovery)

    _prime(agent, exp, datamodule)

    exploration = agent.exploration_step(datamodule=datamodule, kappa=0.8)
    inference = agent.inference_step(exp_data=exp, datamodule=datamodule, recompute=True)

    observed_steps = [
        discovery[0].initial_params.source_step,
        exploration.initial_params.source_step,
        inference.initial_params.source_step,
    ]
    assert observed_steps == [SourceStep.DISCOVERY, SourceStep.EXPLORATION, SourceStep.INFERENCE]


def test_loop_carries_forward_across_iterations(tmp_path):
    """Repeated acquisition on the same agent keeps yielding valid proposals —
    the agent state advances rather than requiring a fresh stack per step."""
    agent, _dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    _prime(agent, exp, datamodule)

    for _ in range(3):
        explore = agent.exploration_step(datamodule=datamodule, kappa=0.6)
        assert isinstance(explore, ExperimentSpec)
        assert explore.initial_params.source_step == SourceStep.EXPLORATION

    infer = agent.inference_step(exp_data=exp, datamodule=datamodule, recompute=True)
    assert infer.initial_params.source_step == SourceStep.INFERENCE


def test_default_kappa_routes_to_exploration(tmp_path):
    """kappa=None resolves to the default (0.5 > 0) → EXPLORATION, not INFERENCE."""
    agent, _dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    _prime(agent, exp, datamodule)

    assert agent.calibration_system.kappa_default > 0.0
    result = agent.acquisition_step(datamodule=datamodule)
    assert result.initial_params.source_step == SourceStep.EXPLORATION


def test_adaptation_step_excluded_from_loop(tmp_path):
    """adaptation_step is not yet implemented and must not silently no-op."""
    agent, _dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    _prime(agent, exp, datamodule)
    with pytest.raises(NotImplementedError):
        agent.adaptation_step()
