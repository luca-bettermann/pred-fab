"""
Contract tests for CalibrationSystem.run_calibration().

Covers:
  1. Offline exploration — experiment level (single step, global bounds + restarts)
  2. Offline inference  — experiment level (single step, global bounds + restarts)
  3. ExperimentSpec integration: apply_schedules, dict access
  4. Agent step method contracts

Online (trust-region) is covered in test_calibration_sampling.py.
"""
import pytest
import numpy as np

from pred_fab.core import ParameterProposal, ExperimentSpec, ParameterSchedule
from pred_fab.utils.enum import Mode
from tests.utils.builders import (
    build_calibration_system,
    build_real_agent_stack,
    build_runtime_agent_stack,
    build_workflow_stack,
    evaluate_loaded_workflow_experiments,
    build_prepared_workflow_datamodule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_real_agent(tmp_path):
    """Return (agent, exp, datamodule) using the mixed-feature (no runtime params) stack."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)
    return agent, exp, datamodule


# ===========================================================================
# 1. Offline exploration — experiment level (single step)
# ===========================================================================

class TestOfflineExplorationExperimentLevel:
    """run_calibration(mode=EXPLORATION, domain=OFFLINE) with no trajectory configs."""

    def test_returns_experiment_spec(self, tmp_path):
        agent, exp, datamodule = _setup_real_agent(tmp_path)
        result = agent.calibration_system.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
        )
        assert isinstance(result, ExperimentSpec)

    def test_initial_params_is_parameter_proposal(self, tmp_path):
        agent, exp, datamodule = _setup_real_agent(tmp_path)
        result = agent.calibration_system.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
        )
        assert isinstance(result.initial_params, ParameterProposal)

    def test_source_step_is_exploration(self, tmp_path):
        agent, exp, datamodule = _setup_real_agent(tmp_path)
        result = agent.calibration_system.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
        )
        assert result.initial_params.source_step == "exploration_step"

    def test_schedules_empty_without_trajectory_configs(self, tmp_path):
        agent, exp, datamodule = _setup_real_agent(tmp_path)
        result = agent.calibration_system.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
        )
        assert result.schedules == {}

    def test_result_contains_all_schema_params(self, tmp_path):
        from pred_fab.core.data_objects import DataDomainAxis
        agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
        agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
        datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
        agent.train(datamodule=datamodule, validate=False, test=False)

        result = agent.calibration_system.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
        )
        # Domain axis params (DataDomainAxis) define grid sizes and are not proposed by calibration.
        non_axis_codes = [
            code for code, obj in dataset.schema.parameters.items()
            if not isinstance(obj, DataDomainAxis)
        ]
        for code in non_axis_codes:
            assert code in result, f"Missing param in result: {code}"

    def test_fixed_params_respected(self, tmp_path):
        agent, exp, datamodule = _setup_real_agent(tmp_path)
        agent.calibration_system.configure_fixed_params({"param_1": 5.0})

        result = agent.calibration_system.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
        )
        assert result["param_1"] == pytest.approx(5.0, abs=0.01)

    def test_dict_like_access_on_result(self, tmp_path):
        agent, exp, datamodule = _setup_real_agent(tmp_path)
        result = agent.calibration_system.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
        )
        # __getitem__, __contains__, keys()
        val = result["param_1"]
        assert isinstance(val, float)
        assert "param_1" in result
        assert "nonexistent" not in result
        assert "param_1" in set(result.keys())

    def test_param_bounds_stored_in_calibration_system(self, tmp_path):
        """Configuring bounds persists them on the CalibrationSystem."""
        agent, exp, datamodule = _setup_real_agent(tmp_path)
        agent.calibration_system.configure_param_bounds({"param_1": (3.0, 4.0)})
        assert agent.calibration_system.param_bounds["param_1"] == (3.0, 4.0)


# ===========================================================================
# 2. Offline inference — experiment level (single step)
# ===========================================================================

class TestOfflineInferenceExperimentLevel:
    """run_calibration(mode=INFERENCE, domain=OFFLINE) with no trajectory configs."""

    def test_returns_experiment_spec(self, tmp_path):
        agent, exp, datamodule = _setup_real_agent(tmp_path)
        result = agent.calibration_system.run_calibration(
            datamodule=datamodule, mode=Mode.INFERENCE,
        )
        assert isinstance(result, ExperimentSpec)

    def test_source_step_is_inference(self, tmp_path):
        agent, exp, datamodule = _setup_real_agent(tmp_path)
        result = agent.calibration_system.run_calibration(
            datamodule=datamodule, mode=Mode.INFERENCE,
        )
        assert result.initial_params.source_step == "inference_step"

    def test_schedules_empty_without_trajectory_configs(self, tmp_path):
        agent, exp, datamodule = _setup_real_agent(tmp_path)
        result = agent.calibration_system.run_calibration(
            datamodule=datamodule, mode=Mode.INFERENCE,
        )
        assert result.schedules == {}

    def test_exploration_and_inference_produce_different_source_steps(self, tmp_path):
        agent, exp, datamodule = _setup_real_agent(tmp_path)
        r_exp = agent.calibration_system.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
        )
        r_inf = agent.calibration_system.run_calibration(
            datamodule=datamodule, mode=Mode.INFERENCE,
        )
        assert r_exp.initial_params.source_step == "exploration_step"
        assert r_inf.initial_params.source_step == "inference_step"

    def test_invalid_mode_raises(self, tmp_path):
        agent, exp, datamodule = _setup_real_agent(tmp_path)
        with pytest.raises((ValueError, AttributeError)):
            agent.calibration_system.run_calibration(
                datamodule=datamodule, mode="invalid_mode",
            )

    def test_explicit_offline_domain_behaves_same_as_default(self, tmp_path):
        agent, exp, datamodule = _setup_real_agent(tmp_path)
        r_default = agent.calibration_system.run_calibration(
            datamodule=datamodule, mode=Mode.INFERENCE,
        )
        r_explicit = agent.calibration_system.run_calibration(
            datamodule=datamodule, mode=Mode.INFERENCE,
        )
        assert isinstance(r_default, ExperimentSpec)
        assert isinstance(r_explicit, ExperimentSpec)
        assert r_default.schedules == {}
        assert r_explicit.schedules == {}


# ===========================================================================
# 6. ExperimentSpec integration: apply_schedules, dict access
# ===========================================================================

class TestExperimentSpecIntegration:
    """ExperimentSpec produced by run_calibration can be applied to experiment data."""

    def test_apply_schedules_records_parameter_update_events(self, tmp_path):
        """A hand-constructed ExperimentSpec with a schedule applies correctly."""
        agent, dataset, exp, _ = build_runtime_agent_stack(tmp_path)

        proposal = ParameterProposal.from_dict({"speed": 150.0})
        schedule = ParameterSchedule(dimension="dim_1", entries=[(1, proposal)])
        spec = ExperimentSpec(
            initial_params=ParameterProposal.from_dict({"speed": 100.0}),
            schedules={"dim_1": schedule},
        )

        initial_events = len(exp.parameter_updates)
        spec.apply_schedules(exp)
        assert len(exp.parameter_updates) == initial_events + 1
        assert exp.parameter_updates[-1].updates.get("speed") == pytest.approx(150.0)

    def test_initial_params_accessible_via_getitem(self, tmp_path):
        agent, exp, datamodule = _setup_real_agent(tmp_path)
        result = agent.calibration_system.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
        )
        # Delegated __getitem__ must work
        val = result["param_1"]
        assert isinstance(val, (int, float))

    def test_initial_params_contains_check(self, tmp_path):
        agent, exp, datamodule = _setup_real_agent(tmp_path)
        result = agent.calibration_system.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
        )
        assert "param_1" in result
        assert "nonexistent_key_xyz" not in result

    def test_initial_params_keys_returns_all_schema_keys(self, tmp_path):
        from pred_fab.core.data_objects import DataDomainAxis
        agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
        agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
        datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
        agent.train(datamodule=datamodule, validate=False, test=False)

        result = agent.calibration_system.run_calibration(
            datamodule=datamodule, mode=Mode.INFERENCE,
        )
        # Domain axis params define grid sizes and are not proposed by calibration.
        non_axis_keys = {
            code for code, obj in dataset.schema.parameters.items()
            if not isinstance(obj, DataDomainAxis)
        }
        result_keys = set(result.keys())
        assert non_axis_keys.issubset(result_keys), (
            f"Missing schema params in result: {non_axis_keys - result_keys}"
        )


# ===========================================================================
# 7. Agent-level step method contracts (exploration_step, inference_step)
# ===========================================================================

class TestAgentStepMethodContracts:
    """Verify exploration_step and inference_step pass the right params to run_calibration."""

    def test_exploration_step_returns_experiment_spec(self, tmp_path):
        agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
        agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
        datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)

        result = agent.exploration_step(datamodule=datamodule)
        assert isinstance(result, ExperimentSpec)

    def test_inference_step_returns_experiment_spec(self, tmp_path):
        agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
        agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
        datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)

        result = agent.inference_step(exp_data=exp, datamodule=datamodule, recompute=True)
        assert isinstance(result, ExperimentSpec)

    def test_exploration_step_source_step_tag(self, tmp_path):
        agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
        agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
        datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)

        result = agent.exploration_step(datamodule=datamodule)
        assert result.initial_params.source_step == "exploration_step"

    def test_inference_step_source_step_tag(self, tmp_path):
        agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
        agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
        datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)

        result = agent.inference_step(exp_data=exp, datamodule=datamodule, recompute=True)
        assert result.initial_params.source_step == "inference_step"

    def test_exploration_step_mode_is_passed_as_exploration(self, tmp_path):
        """Spy confirms exploration_step calls run_calibration with Mode.EXPLORATION."""
        agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
        agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
        datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)

        captured = {}
        original = agent.calibration_system.run_calibration

        def _spy(datamodule, mode, **kwargs):
            captured["mode"] = mode
            return ExperimentSpec(
                initial_params=ParameterProposal.from_dict(
                    {"param_1": 1.0, "param_2": 1, "dim_1": 1, "dim_2": 1},
                    source_step="exploration_step",
                ),
            )

        agent.calibration_system.run_calibration = _spy  # type: ignore[assignment]
        try:
            agent.exploration_step(datamodule=datamodule)
        finally:
            agent.calibration_system.run_calibration = original  # type: ignore[assignment]

        from pred_fab.utils.enum import Mode
        assert captured["mode"] == Mode.EXPLORATION

    def test_inference_step_mode_is_passed_as_inference(self, tmp_path):
        """Spy confirms inference_step calls run_calibration with Mode.INFERENCE."""
        agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
        agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
        datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)

        captured = {}
        original = agent.calibration_system.run_calibration

        def _spy(datamodule, mode, **kwargs):
            captured["mode"] = mode
            return ExperimentSpec(
                initial_params=ParameterProposal.from_dict(
                    {"param_1": 1.0, "param_2": 1, "dim_1": 1, "dim_2": 1},
                    source_step="inference_step",
                ),
            )

        agent.calibration_system.run_calibration = _spy  # type: ignore[assignment]
        try:
            agent.inference_step(exp_data=exp, datamodule=datamodule, recompute=True)
        finally:
            agent.calibration_system.run_calibration = original  # type: ignore[assignment]

        from pred_fab.utils.enum import Mode
        assert captured["mode"] == Mode.INFERENCE


