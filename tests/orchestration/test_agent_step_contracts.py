from pred_fab.core import ParameterProposal, ExperimentSpec
from pred_fab.utils import Mode
from tests.utils.builders import build_real_agent_stack


def test_exploration_step_returns_experiment_spec(tmp_path):
    agent, _, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)

    captured = {}
    original = agent.calibration_system.run_calibration

    def _spy_run_calibration(datamodule, mode, **kwargs):
        captured["mode"] = mode
        return ExperimentSpec(
            initial_params=ParameterProposal.from_dict(
                {"param_1": 3.0, "param_2": 2, "dim_1": 1, "dim_2": 0},
                source_step="exploration_step",
            ),
        )

    agent.calibration_system.run_calibration = _spy_run_calibration  # type: ignore[assignment]
    try:
        result = agent.exploration_step(datamodule=datamodule, kappa=0.8, n_optimization_rounds=1)
    finally:
        agent.calibration_system.run_calibration = original  # type: ignore[assignment]

    assert isinstance(result, ExperimentSpec)
    assert result.initial_params.source_step == "exploration_step"
    assert result["param_1"] == 3.0
    assert captured["mode"] == Mode.EXPLORATION


def test_inference_step_returns_experiment_spec(tmp_path):
    agent, _, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)

    captured = {}
    original = agent.calibration_system.run_calibration

    def _spy_run_calibration(datamodule, mode, **kwargs):
        captured["mode"] = mode
        return ExperimentSpec(
            initial_params=ParameterProposal.from_dict(
                {"param_1": 4.0, "param_2": 2, "dim_1": 1, "dim_2": 0},
                source_step="inference_step",
            ),
        )

    agent.calibration_system.run_calibration = _spy_run_calibration  # type: ignore[assignment]
    try:
        result = agent.inference_step(exp_data=exp, datamodule=datamodule, recompute=True, visualize=False)
    finally:
        agent.calibration_system.run_calibration = original  # type: ignore[assignment]

    assert isinstance(result, ExperimentSpec)
    assert result.initial_params.source_step == "inference_step"
    assert result["param_1"] == 4.0
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


def test_baseline_step_returns_experiment_specs(tmp_path):
    """baseline_step returns ExperimentSpec instances with LHS spacing."""
    agent, _, _, _ = build_real_agent_stack(tmp_path)
    sampled = agent.baseline_step(n=3)

    assert len(sampled) == 3
    assert all(isinstance(p, ExperimentSpec) for p in sampled)
    assert all(p.initial_params.source_step == "baseline_step" for p in sampled)
