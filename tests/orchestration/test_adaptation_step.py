from pred_fab.core import ParameterProposal, ExperimentSpec
from tests.utils.builders import build_real_agent_stack


def _build_real_agent_and_data(tmp_path):
    agent, _, exp, datamodule = build_real_agent_stack(tmp_path)
    # Populate tensors/performance through the real evaluation flow.
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    # Prepare/train so adaptation has normalization state + prediction model context.
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)
    return agent, exp


def test_adaptation_step_runs_real_tuning_flow_and_returns_experiment_spec(tmp_path):
    agent, exp = _build_real_agent_and_data(tmp_path)

    result = agent.adaptation_step(
        dimension="dim_1",
        step_index=1,
        exp_data=exp,
        record=False,
    )

    assert isinstance(result, ExperimentSpec)
    assert result.initial_params.source_step == "adaptation_step"
    assert "param_1" in result and "dim_1" in result and "dim_2" in result
    assert result["param_1"] == 2.5
    assert len(exp.parameter_updates) == 0


def test_adaptation_step_record_uses_effective_params_after_prior_updates(tmp_path):
    agent, exp = _build_real_agent_and_data(tmp_path)
    exp.record_parameter_update(
        ParameterProposal.from_dict({"param_1": 5.0}, source_step="adaptation_step"),
        dimension="dim_1",
        step_index=0,
    )

    result = agent.adaptation_step(
        dimension="dim_1",
        step_index=1,
        exp_data=exp,
        record=True,
    )

    assert result["param_1"] == 5.0
    # No-op proposals are intentionally not logged as additional update events.
    assert len(exp.parameter_updates) == 1
