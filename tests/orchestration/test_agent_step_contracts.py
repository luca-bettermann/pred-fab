from pred_fab.core import ExperimentSpec
from pred_fab.utils import SourceStep
from tests.utils.builders import build_real_agent_stack


def test_discovery_step_returns_experiment_specs(tmp_path):
    """discovery_step returns ExperimentSpec instances."""
    agent, _, _, _ = build_real_agent_stack(tmp_path)
    sampled = agent.discovery_step(n=3)

    assert len(sampled) == 3
    assert all(isinstance(p, ExperimentSpec) for p in sampled)
    assert all(p.initial_params.source_step == SourceStep.DISCOVERY for p in sampled)


def test_exploration_step_returns_experiment_spec(tmp_path):
    """exploration_step runs through run_acquisition with kappa > 0."""
    agent, _, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    result = agent.exploration_step(datamodule=datamodule, kappa=0.8)

    assert isinstance(result, ExperimentSpec)
    assert result.initial_params.source_step == SourceStep.EXPLORATION


def test_inference_step_returns_experiment_spec(tmp_path):
    """inference_step runs through run_acquisition with kappa=0."""
    agent, _, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    result = agent.inference_step(exp_data=exp, datamodule=datamodule, recompute=True)

    assert isinstance(result, ExperimentSpec)
    assert result.initial_params.source_step == SourceStep.INFERENCE


def test_acquisition_step_resolves_default_kappa(tmp_path):
    """kappa=None falls back to calibration_system.kappa_default."""
    agent, _, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    agent.calibration_system.kappa_default = 0.3
    result = agent.acquisition_step(datamodule=datamodule)

    assert isinstance(result, ExperimentSpec)
    assert result.initial_params.source_step == SourceStep.EXPLORATION


def test_evaluate_executes_real_evaluation_method(tmp_path):
    agent, _, exp, _ = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    assert exp.performance.has_value("performance_1")


def test_train_executes_real_training_method(tmp_path):
    agent, _, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    result = agent.train(datamodule=datamodule, validate=False, test=False)
    assert result is None
    assert agent.pred_system.datamodule is datamodule
