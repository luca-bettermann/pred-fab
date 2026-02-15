from pred_fab.core import ParameterProposal
from pred_fab.utils import Mode
from tests.utils.builders import build_real_agent_stack


def test_exploration_step_returns_parameter_proposal(tmp_path):
    agent, _, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)

    captured = {}
    original = agent.calibration_system.run_calibration

    def _spy_run_calibration(datamodule, mode, w_explore, n_optimization_rounds):
        captured["mode"] = mode
        return {"param_1": 3.0, "param_2": 2, "dim_1": 1, "dim_2": 0}

    agent.calibration_system.run_calibration = _spy_run_calibration  # type: ignore[assignment]
    try:
        proposal = agent.exploration_step(datamodule=datamodule, w_explore=0.8, n_optimization_rounds=1)
    finally:
        agent.calibration_system.run_calibration = original  # type: ignore[assignment]

    assert isinstance(proposal, ParameterProposal)
    assert proposal.source_step == "exploration_step"
    assert proposal["param_1"] == 3.0
    assert captured["mode"] == Mode.EXPLORATION


def test_inference_step_runs_real_eval_then_returns_parameter_proposal(tmp_path):
    agent, _, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)

    captured = {}
    original = agent.calibration_system.run_calibration

    def _spy_run_calibration(datamodule, mode, w_explore, n_optimization_rounds):
        captured["mode"] = mode
        return {"param_1": 4.0, "param_2": 2, "dim_1": 1, "dim_2": 0}

    agent.calibration_system.run_calibration = _spy_run_calibration  # type: ignore[assignment]
    try:
        proposal = agent.inference_step(exp_data=exp, datamodule=datamodule, recompute=True, visualize=False)
    finally:
        agent.calibration_system.run_calibration = original  # type: ignore[assignment]

    assert isinstance(proposal, ParameterProposal)
    assert proposal.source_step == "inference_step"
    assert proposal["param_1"] == 4.0
    assert captured["mode"] == Mode.INFERENCE
    assert exp.performance.has_value("performance_1")


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


def test_baseline_sampling_returns_parameter_proposals(tmp_path):
    agent, _, _, _ = build_real_agent_stack(tmp_path)
    sampled = agent.sample_baseline_experiments(n_samples=3)

    assert len(sampled) == 3
    assert all(isinstance(p, ParameterProposal) for p in sampled)
    assert all(p.source_step == "baseline_sampling" for p in sampled)
