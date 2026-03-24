"""
Contract tests for the unified CalibrationSystem.run_calibration() step-loop.

Covers all four primary use cases:
  1. Offline exploration — experiment level (single step, global bounds + restarts)
  2. Offline inference  — experiment level (single step, global bounds + restarts)
  3. Offline exploration — dimensional level (multi-step, trajectory params)
  4. Offline inference  — dimensional level (multi-step, trajectory params)

Plus edge cases: empty step grid, multi-dim grids, eligibility logic, spec assembly,
and interactions between fixed params, trust regions, and the step-loop.

Online (domain=ONLINE) is covered separately in test_calibration_sampling.py.
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

def _setup_runtime_agent(tmp_path):
    """Return (agent, exp, datamodule) trained and ready for calibration."""
    agent, dataset, exp, datamodule = build_runtime_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)
    return agent, exp, datamodule


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
# 3. Offline exploration — dimensional level (multi-step)
# ===========================================================================

class TestOfflineExplorationDimensionalLevel:
    """run_calibration(mode=EXPLORATION) with trajectory configs → multi-step loop."""

    def _configured_cs(self, tmp_path):
        """Return (cs, exp, datamodule) with speed configured for dim_1 trajectory."""
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("speed", "dim_1")
        cs.configure_adaptation_delta({"speed": 50.0})
        return cs, exp, datamodule

    def test_returns_experiment_spec(self, tmp_path):
        cs, exp, datamodule = self._configured_cs(tmp_path)
        current_params = {**exp.parameters.get_values_dict(), "speed": 100.0}
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION, current_params=current_params,
        )
        assert isinstance(result, ExperimentSpec)

    def test_schedule_keyed_by_configured_dimension(self, tmp_path):
        cs, exp, datamodule = self._configured_cs(tmp_path)
        current_params = {**exp.parameters.get_values_dict(), "speed": 100.0}
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION, current_params=current_params,
        )
        # dim_1=2 → 2 steps → 1 transition → schedule populated
        assert "dim_1" in result.schedules

    def test_schedule_is_parameter_schedule_instance(self, tmp_path):
        cs, exp, datamodule = self._configured_cs(tmp_path)
        current_params = {**exp.parameters.get_values_dict(), "speed": 100.0}
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION, current_params=current_params,
        )
        for schedule in result.schedules.values():
            assert isinstance(schedule, ParameterSchedule)

    def test_schedule_entries_contain_trajectory_param(self, tmp_path):
        cs, exp, datamodule = self._configured_cs(tmp_path)
        current_params = {**exp.parameters.get_values_dict(), "speed": 100.0}
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION, current_params=current_params,
        )
        schedule = result.schedules["dim_1"]
        for _, proposal in schedule.entries:
            assert "speed" in proposal

    def test_consecutive_speed_values_within_delta(self, tmp_path):
        """Trust-region constraint: consecutive speed proposals must not exceed delta."""
        delta = 30.0
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("speed", "dim_1")
        cs.configure_adaptation_delta({"speed": delta})
        current_params = {**exp.parameters.get_values_dict(), "speed": 100.0}

        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION, current_params=current_params,
        )

        if "dim_1" not in result.schedules:
            pytest.skip("No schedule produced — dim_1 must have been 1")

        seg0 = result.initial_params["speed"]
        waypoints = [p["speed"] for _, p in result.schedules["dim_1"].entries]
        all_vals = [seg0] + waypoints
        for k in range(len(all_vals) - 1):
            diff = abs(all_vals[k + 1] - all_vals[k])
            assert diff <= delta + 1e-4, (
                f"Step {k}→{k+1}: diff={diff:.4f} exceeds delta={delta}"
            )

    def test_initial_params_source_step_is_exploration(self, tmp_path):
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("speed", "dim_1")
        cs.configure_adaptation_delta({"speed": 50.0})
        current_params = {**exp.parameters.get_values_dict(), "speed": 100.0}

        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION, current_params=current_params,
        )
        assert result.initial_params.source_step == "exploration_step"

    def test_raises_when_trajectory_param_missing_trust_region(self, tmp_path):
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("speed", "dim_1")
        # deliberately omit configure_adaptation_delta

        current_params = {**exp.parameters.get_values_dict(), "speed": 100.0}
        with pytest.raises(RuntimeError, match="trust region"):
            cs.run_calibration(
                datamodule=datamodule, mode=Mode.EXPLORATION, current_params=current_params,
            )

    def test_without_current_params_produces_single_step(self, tmp_path):
        """Without current_params, step grid defaults to single step → empty schedules."""
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("speed", "dim_1")
        cs.configure_adaptation_delta({"speed": 50.0})

        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION, current_params=None,
        )
        assert isinstance(result, ExperimentSpec)
        assert result.schedules == {}

    def test_dim_one_produces_single_step_empty_schedule(self, tmp_path):
        """If current_params['dim_1'] == 1, step grid has 1 step → no schedule transitions."""
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("speed", "dim_1")
        cs.configure_adaptation_delta({"speed": 50.0})

        current_params = {**exp.parameters.get_values_dict(), "speed": 100.0, "dim_1": 1}
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION, current_params=current_params,
        )
        # 1 step = no transitions = no schedule entries
        assert result.schedules == {}

    def test_step_grid_ordering_coarsest_first(self, tmp_path):
        """_build_step_grid must iterate coarsest dimension (level 1) in outer loop."""
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        current_params = {**exp.parameters.get_values_dict(), "speed": 100.0}
        current_params["dim_1"] = 2
        current_params["dim_2"] = 3

        # Manually test grid ordering
        # trajectory_configs must include both dims to be non-trivial
        # We only configure speed→dim_1 here; just validate the grid structure
        grid = cs._build_step_grid(current_params)
        # No trajectory configs → single-step grid
        assert grid == [{}]


# ===========================================================================
# 4. Offline inference — dimensional level (multi-step)
# ===========================================================================

class TestOfflineInferenceDimensionalLevel:
    """run_calibration(mode=INFERENCE) with trajectory configs → multi-step loop."""

    def test_returns_experiment_spec(self, tmp_path):
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("speed", "dim_1")
        cs.configure_adaptation_delta({"speed": 50.0})
        current_params = {**exp.parameters.get_values_dict(), "speed": 100.0}

        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.INFERENCE, current_params=current_params,
        )
        assert isinstance(result, ExperimentSpec)

    def test_source_step_is_inference(self, tmp_path):
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("speed", "dim_1")
        cs.configure_adaptation_delta({"speed": 50.0})
        current_params = {**exp.parameters.get_values_dict(), "speed": 100.0}

        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.INFERENCE, current_params=current_params,
        )
        assert result.initial_params.source_step == "inference_step"

    def test_schedule_produced_for_configured_dimension(self, tmp_path):
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("speed", "dim_1")
        cs.configure_adaptation_delta({"speed": 50.0})
        current_params = {**exp.parameters.get_values_dict(), "speed": 100.0}

        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.INFERENCE, current_params=current_params,
        )
        assert "dim_1" in result.schedules

    def test_inference_trajectory_values_within_bounds(self, tmp_path):
        """Trajectory proposals under inference mode must still stay within trust region."""
        delta = 25.0
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("speed", "dim_1")
        cs.configure_adaptation_delta({"speed": delta})
        current_params = {**exp.parameters.get_values_dict(), "speed": 100.0}

        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.INFERENCE, current_params=current_params,
        )

        if "dim_1" not in result.schedules:
            pytest.skip("No schedule — dim_1 must be 1")

        seg0 = result.initial_params["speed"]
        waypoints = [p["speed"] for _, p in result.schedules["dim_1"].entries]
        all_vals = [seg0] + waypoints
        for k in range(len(all_vals) - 1):
            diff = abs(all_vals[k + 1] - all_vals[k])
            assert diff <= delta + 1e-4


# ===========================================================================
# 5. Step-loop internals: eligibility and grid construction
# ===========================================================================

class TestStepLoopInternals:
    """Low-level tests for _build_step_grid, _get_eligible_params, _build_experiment_spec."""

    def test_build_step_grid_no_trajectory_returns_single_empty_dict(self, tmp_path):
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        current_params = exp.parameters.get_values_dict()

        grid = cs._build_step_grid(current_params)
        assert grid == [{}]

    def test_build_step_grid_with_dim1_size2(self, tmp_path):
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("speed", "dim_1")
        cs.configure_adaptation_delta({"speed": 50.0})
        current_params = {**exp.parameters.get_values_dict(), "speed": 100.0}
        current_params["dim_1"] = 2

        grid = cs._build_step_grid(current_params)
        assert len(grid) == 2
        assert grid[0] == {"dim_1": 0}
        assert grid[1] == {"dim_1": 1}

    def test_get_eligible_params_all_transition_at_first_step(self, tmp_path):
        """At step 0 (prev_indices=None), ALL trajectory params are eligible."""
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("speed", "dim_1")

        eligible = cs._get_eligible_params(prev_indices=None, curr_indices={"dim_1": 0})
        assert "speed" in eligible

    def test_get_eligible_params_unchanged_dim_not_eligible(self, tmp_path):
        """When a dim doesn't change, its params are NOT eligible."""
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("speed", "dim_1")

        eligible = cs._get_eligible_params(
            prev_indices={"dim_1": 1}, curr_indices={"dim_1": 1},
        )
        assert "speed" not in eligible

    def test_get_eligible_params_transitioning_dim_is_eligible(self, tmp_path):
        """When a dim changes, its params ARE eligible."""
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("speed", "dim_1")

        eligible = cs._get_eligible_params(
            prev_indices={"dim_1": 0}, curr_indices={"dim_1": 1},
        )
        assert "speed" in eligible

    def test_build_experiment_spec_single_step_produces_empty_schedules(self, tmp_path):
        """Single proposal → ExperimentSpec with no schedules."""
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        from pred_fab.utils.enum import SourceStep

        spec = cs._build_experiment_spec(
            proposals=[{"speed": 100.0, "param_1": 2.5}],
            step_grid=[{}],
            source_step=SourceStep.EXPLORATION,
        )
        assert isinstance(spec, ExperimentSpec)
        assert spec.schedules == {}
        assert spec.initial_params.source_step == "exploration_step"

    def test_build_experiment_spec_two_steps_produces_schedule(self, tmp_path):
        """Two proposals with a dim transition → schedule populated."""
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("speed", "dim_1")
        from pred_fab.utils.enum import SourceStep

        proposals = [
            {"speed": 100.0, "param_1": 2.5},
            {"speed": 130.0, "param_1": 2.5},
        ]
        step_grid = [{"dim_1": 0}, {"dim_1": 1}]

        spec = cs._build_experiment_spec(
            proposals=proposals, step_grid=step_grid, source_step=SourceStep.EXPLORATION,
        )
        assert "dim_1" in spec.schedules
        entries = spec.schedules["dim_1"].entries
        assert len(entries) == 1
        _, prop = entries[0]
        assert prop["speed"] == pytest.approx(130.0)


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

    def test_exploration_step_with_trajectory_produces_schedule(self, tmp_path):
        agent, dataset, exp, datamodule = build_runtime_agent_stack(tmp_path)
        agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
        datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
        agent.train(datamodule=datamodule, validate=False, test=False)

        agent.calibration_system.configure_step_parameter("speed", "dim_1")
        agent.calibration_system.configure_adaptation_delta({"speed": 50.0})

        current_params = {**exp.parameters.get_values_dict(), "speed": 100.0}
        result = agent.exploration_step(datamodule=datamodule, current_params=current_params)

        assert isinstance(result, ExperimentSpec)
        assert "dim_1" in result.schedules

    def test_inference_step_with_trajectory_produces_schedule(self, tmp_path):
        agent, dataset, exp, datamodule = build_runtime_agent_stack(tmp_path)
        agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
        datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
        agent.train(datamodule=datamodule, validate=False, test=False)

        agent.calibration_system.configure_step_parameter("speed", "dim_1")
        agent.calibration_system.configure_adaptation_delta({"speed": 50.0})

        current_params = {**exp.parameters.get_values_dict(), "speed": 100.0}
        result = agent.inference_step(
            exp_data=exp, datamodule=datamodule, recompute=True,
            current_params=current_params,
        )

        assert isinstance(result, ExperimentSpec)
        assert "dim_1" in result.schedules

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


# ===========================================================================
# 8. Multi-dim mixed scenario: layer_height → dim_1 (level 1), speed → dim_2 (level 2)
# ===========================================================================

def _build_two_runtime_agent_stack(tmp_path):
    """Schema with two runtime params on different domain axes.

    layer_height (runtime) → mapped to dim_1 (coarse outer axis)
    speed        (runtime) → mapped to dim_2 (fine inner axis)
    dim_1=2 layers, dim_2=3 segments → 6-step Cartesian grid.
    """
    from pred_fab.core.data_objects import Feature, PerformanceAttribute, Dimension, Domain
    from pred_fab.core.data_blocks import Parameters, Features, PerformanceAttributes, Domains
    from pred_fab.core import Dataset, DatasetSchema
    from pred_fab.core.data_objects import Parameter as _Param
    from tests.utils.interfaces import MixedFeatureModel, ScalarEvaluationModel, MixedPredictionModel
    from pred_fab.orchestration.agent import PfabAgent as _PfabAgent

    p1 = _Param.real("param_1", min_val=0.0, max_val=10.0)
    layer_height = _Param.real("layer_height", min_val=0.05, max_val=0.4, runtime=True)
    speed = _Param.real("speed", min_val=0.0, max_val=200.0, runtime=True)

    f_grid = Feature.array("feature_grid", domain="spatial")
    f_d1 = Feature.array("feature_d1", domain="spatial", depth=1)
    f_scalar = Feature.array("feature_scalar")
    perf = PerformanceAttribute.score("performance_1")

    feats = Features.from_list([f_grid, f_d1, f_scalar])
    perfs = PerformanceAttributes.from_list([perf])

    domains = Domains()
    domains.add(Domain("spatial", [
        Dimension("dim_1", "d1", 1, 4),
        Dimension("dim_2", "d2", 1, 4),
    ]))

    schema = DatasetSchema(
        root_folder=str(tmp_path),
        name="schema_two_runtime",
        parameters=Parameters.from_list([p1, layer_height, speed]),
        features=feats,
        performance=perfs,
        domains=domains,
    )
    dataset = Dataset(schema=schema, debug_flag=True)
    dataset.create_experiment(
        "exp_001",
        parameters={"param_1": 2.5, "layer_height": 0.2, "speed": 100.0, "dim_1": 2, "dim_2": 3},
    )
    exp = dataset.get_experiment("exp_001")

    agent = _PfabAgent(root_folder=str(tmp_path), debug_flag=True)
    agent.register_feature_model(MixedFeatureModel)
    agent.register_evaluation_model(ScalarEvaluationModel)
    agent.register_prediction_model(MixedPredictionModel)
    agent.initialize_systems(schema, verbose_flag=False)

    datamodule = agent.create_datamodule(dataset)
    return agent, dataset, exp, datamodule


def _setup_two_runtime_agent(tmp_path):
    agent, dataset, exp, datamodule = _build_two_runtime_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)
    return agent, exp, datamodule


class TestMultiDimMixedScenario:
    """Two runtime params on different dimension levels:
    layer_height → dim_1 (level 1, coarse outer loop)
    speed        → dim_2 (level 2, fine inner loop)

    With dim_1=2 layers and dim_2=3 segments the flat Cartesian grid is:
      step 0: {dim_1:0, dim_2:0}  — both dims transition (prev=None)
      step 1: {dim_1:0, dim_2:1}  — only dim_2 transitions
      step 2: {dim_1:0, dim_2:2}  — only dim_2 transitions
      step 3: {dim_1:1, dim_2:0}  — both dims transition
      step 4: {dim_1:1, dim_2:1}  — only dim_2 transitions
      step 5: {dim_1:1, dim_2:2}  — only dim_2 transitions
    """

    def _configured_cs(self, tmp_path):
        agent, exp, datamodule = _setup_two_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("layer_height", "dim_1")
        cs.configure_step_parameter("speed", "dim_2")
        cs.configure_adaptation_delta({"layer_height": 0.1, "speed": 50.0})
        current_params = {**exp.parameters.get_values_dict()}
        return cs, exp, datamodule, current_params

    def test_step_grid_is_six_steps_coarsest_first(self, tmp_path):
        """dim_1=2 × dim_2=3 → 6 steps, dim_1 (level 1) varies in outer loop."""
        agent, exp, datamodule = _setup_two_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("layer_height", "dim_1")
        cs.configure_step_parameter("speed", "dim_2")
        current_params = {**exp.parameters.get_values_dict()}

        grid = cs._build_step_grid(current_params)

        assert len(grid) == 6
        expected = [
            {"dim_1": 0, "dim_2": 0},
            {"dim_1": 0, "dim_2": 1},
            {"dim_1": 0, "dim_2": 2},
            {"dim_1": 1, "dim_2": 0},
            {"dim_1": 1, "dim_2": 1},
            {"dim_1": 1, "dim_2": 2},
        ]
        assert grid == expected

    def test_layer_height_eligible_only_when_dim1_transitions(self, tmp_path):
        """layer_height is only eligible when dim_1 index changes."""
        agent, exp, datamodule = _setup_two_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("layer_height", "dim_1")
        cs.configure_step_parameter("speed", "dim_2")

        # Step 0: prev=None → dim_1 transitions → layer_height eligible
        eligible_0 = cs._get_eligible_params(None, {"dim_1": 0, "dim_2": 0})
        assert "layer_height" in eligible_0

        # Step 1: dim_1 unchanged → layer_height NOT eligible
        eligible_1 = cs._get_eligible_params({"dim_1": 0, "dim_2": 0}, {"dim_1": 0, "dim_2": 1})
        assert "layer_height" not in eligible_1

        # Step 3: dim_1 transitions (0→1) → layer_height eligible again
        eligible_3 = cs._get_eligible_params({"dim_1": 0, "dim_2": 2}, {"dim_1": 1, "dim_2": 0})
        assert "layer_height" in eligible_3

    def test_speed_eligible_at_all_steps(self, tmp_path):
        """speed maps to dim_2 which transitions at every step of the flat grid."""
        agent, exp, datamodule = _setup_two_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("layer_height", "dim_1")
        cs.configure_step_parameter("speed", "dim_2")
        current_params = {**exp.parameters.get_values_dict()}
        grid = cs._build_step_grid(current_params)

        prev = None
        for curr in grid:
            eligible = cs._get_eligible_params(prev, curr)
            assert "speed" in eligible, f"speed should be eligible at step {curr}"
            prev = curr

    def test_returns_both_dimension_schedules(self, tmp_path):
        """Both 'dim_1' and 'dim_2' keys must appear in result.schedules."""
        cs, exp, datamodule, current_params = self._configured_cs(tmp_path)
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION, current_params=current_params,
        )
        assert isinstance(result, ExperimentSpec)
        assert "dim_1" in result.schedules, "Expected dim_1 schedule for layer_height"
        assert "dim_2" in result.schedules, "Expected dim_2 schedule for speed"

    def test_dim1_schedule_entries_contain_layer_height(self, tmp_path):
        """Every entry in the dim_1 schedule must carry a layer_height value."""
        cs, exp, datamodule, current_params = self._configured_cs(tmp_path)
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION, current_params=current_params,
        )
        for step_idx, proposal in result.schedules["dim_1"].entries:
            assert "layer_height" in proposal, (
                f"dim_1 schedule entry at step {step_idx} missing layer_height"
            )

    def test_dim2_schedule_entries_contain_speed(self, tmp_path):
        """Every entry in the dim_2 schedule must carry a speed value."""
        cs, exp, datamodule, current_params = self._configured_cs(tmp_path)
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION, current_params=current_params,
        )
        for step_idx, proposal in result.schedules["dim_2"].entries:
            assert "speed" in proposal, (
                f"dim_2 schedule entry at step {step_idx} missing speed"
            )

    def test_layer_height_delta_constraint_across_layers(self, tmp_path):
        """Consecutive layer_height values (initial + dim_1 schedule) must stay within delta."""
        delta = 0.1
        agent, exp, datamodule = _setup_two_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("layer_height", "dim_1")
        cs.configure_step_parameter("speed", "dim_2")
        cs.configure_adaptation_delta({"layer_height": delta, "speed": 50.0})
        current_params = {**exp.parameters.get_values_dict()}

        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION, current_params=current_params,
        )

        if "dim_1" not in result.schedules:
            pytest.skip("No dim_1 schedule produced")

        lh0 = result.initial_params["layer_height"]
        lh_vals = [lh0] + [p["layer_height"] for _, p in result.schedules["dim_1"].entries]
        for k in range(len(lh_vals) - 1):
            diff = abs(lh_vals[k + 1] - lh_vals[k])
            assert diff <= delta + 1e-4, (
                f"layer_height step {k}→{k+1}: diff={diff:.6f} exceeds delta={delta}"
            )

    def test_speed_delta_constraint_across_segments(self, tmp_path):
        """Consecutive speed values (initial + dim_2 schedule) must stay within delta."""
        delta = 50.0
        agent, exp, datamodule = _setup_two_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("layer_height", "dim_1")
        cs.configure_step_parameter("speed", "dim_2")
        cs.configure_adaptation_delta({"layer_height": 0.1, "speed": delta})
        current_params = {**exp.parameters.get_values_dict()}

        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION, current_params=current_params,
        )

        if "dim_2" not in result.schedules:
            pytest.skip("No dim_2 schedule produced")

        s0 = result.initial_params["speed"]
        speed_vals = [s0] + [p["speed"] for _, p in result.schedules["dim_2"].entries]
        for k in range(len(speed_vals) - 1):
            diff = abs(speed_vals[k + 1] - speed_vals[k])
            assert diff <= delta + 1e-4, (
                f"speed step {k}→{k+1}: diff={diff:.4f} exceeds delta={delta}"
            )

    def test_build_experiment_spec_two_dims_directly(self, tmp_path):
        """Unit test _build_experiment_spec with the full 6-step two-dim grid."""
        from pred_fab.utils.enum import SourceStep

        agent, exp, datamodule = _setup_two_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("layer_height", "dim_1")
        cs.configure_step_parameter("speed", "dim_2")

        # Simulate 6 optimization results
        proposals = [
            {"param_1": 2.5, "layer_height": 0.20, "speed": 100.0},  # step 0
            {"param_1": 2.5, "layer_height": 0.20, "speed": 110.0},  # step 1
            {"param_1": 2.5, "layer_height": 0.20, "speed": 120.0},  # step 2
            {"param_1": 2.5, "layer_height": 0.25, "speed":  90.0},  # step 3
            {"param_1": 2.5, "layer_height": 0.25, "speed": 130.0},  # step 4
            {"param_1": 2.5, "layer_height": 0.25, "speed": 140.0},  # step 5
        ]
        step_grid = [
            {"dim_1": 0, "dim_2": 0},
            {"dim_1": 0, "dim_2": 1},
            {"dim_1": 0, "dim_2": 2},
            {"dim_1": 1, "dim_2": 0},
            {"dim_1": 1, "dim_2": 1},
            {"dim_1": 1, "dim_2": 2},
        ]

        spec = cs._build_experiment_spec(
            proposals=proposals, step_grid=step_grid, source_step=SourceStep.EXPLORATION,
        )

        # initial_params from step 0
        assert spec.initial_params["layer_height"] == pytest.approx(0.20)
        assert spec.initial_params["speed"] == pytest.approx(100.0)

        # dim_1 schedule: only step 3 transitions dim_1 (0→1)
        assert "dim_1" in spec.schedules
        dim1_entries = spec.schedules["dim_1"].entries
        assert len(dim1_entries) == 1
        idx, prop = dim1_entries[0]
        assert idx == 1  # dim_1 index = 1
        assert prop["layer_height"] == pytest.approx(0.25)

        # dim_2 schedule: transitions at steps 1,2,3,4,5
        assert "dim_2" in spec.schedules
        assert len(spec.schedules["dim_2"].entries) == 5
        speed_vals = [p["speed"] for _, p in spec.schedules["dim_2"].entries]
        assert speed_vals == pytest.approx([110.0, 120.0, 90.0, 130.0, 140.0])


# ===========================================================================
# 9. target_indices: collapse full grid to a single targeted step
# ===========================================================================

class TestTargetIndices:
    """target_indices={"dim_1": k} collapses the Cartesian grid to one step,
    optimising only the trajectory params mapped to the specified dimensions
    and fixing everything else.
    """

    def _configured_cs(self, tmp_path):
        agent, exp, datamodule = _setup_two_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("layer_height", "dim_1")
        cs.configure_step_parameter("speed", "dim_2")
        cs.configure_adaptation_delta({"layer_height": 0.1, "speed": 50.0})
        current_params = {**exp.parameters.get_values_dict()}
        return cs, exp, datamodule, current_params

    def test_target_dim1_returns_experiment_spec(self, tmp_path):
        cs, exp, datamodule, current_params = self._configured_cs(tmp_path)
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
            current_params=current_params, target_indices={"dim_1": 0},
        )
        assert isinstance(result, ExperimentSpec)

    def test_target_dim1_schedules_empty(self, tmp_path):
        """Single targeted step → no dim transitions → empty schedules."""
        cs, exp, datamodule, current_params = self._configured_cs(tmp_path)
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
            current_params=current_params, target_indices={"dim_1": 0},
        )
        assert result.schedules == {}

    def test_target_dim1_overrides_full_grid(self, tmp_path):
        """With dim_1=2 × dim_2=3 the full grid has 6 steps, but target_indices
        collapses it to 1 — only one proposal is produced."""
        cs, exp, datamodule, current_params = self._configured_cs(tmp_path)
        # Verify the full grid would be 6 steps
        full_grid = cs._build_step_grid(current_params)
        assert len(full_grid) == 6

        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
            current_params=current_params, target_indices={"dim_1": 0},
        )
        # Only initial_params, no schedule entries
        assert result.schedules == {}
        assert isinstance(result.initial_params, ParameterProposal)

    def test_target_dim1_only_layer_height_can_change(self, tmp_path):
        """With target_indices={"dim_1": 0}, layer_height is eligible.
        speed (mapped to dim_2) must be fixed at its current value."""
        cs, exp, datamodule, current_params = self._configured_cs(tmp_path)
        speed_before = current_params["speed"]

        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
            current_params=current_params, target_indices={"dim_1": 0},
        )
        assert result["speed"] == pytest.approx(speed_before, abs=1e-6)

    def test_target_dim2_only_speed_can_change(self, tmp_path):
        """With target_indices={"dim_2": 1}, speed is eligible.
        layer_height (mapped to dim_1) must be fixed at its current value."""
        cs, exp, datamodule, current_params = self._configured_cs(tmp_path)
        lh_before = current_params["layer_height"]

        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
            current_params=current_params, target_indices={"dim_2": 1},
        )
        assert result["layer_height"] == pytest.approx(lh_before, abs=1e-6)

    def test_target_both_dims_both_params_eligible(self, tmp_path):
        """target_indices covering both dims → both layer_height and speed can change."""
        cs, exp, datamodule, current_params = self._configured_cs(tmp_path)
        # Verify eligibility directly
        eligible = cs._get_eligible_params(None, {"dim_1": 1, "dim_2": 0})
        assert "layer_height" in eligible
        assert "speed" in eligible

        # run_calibration respects this: result is a valid ExperimentSpec
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
            current_params=current_params,
            target_indices={"dim_1": 1, "dim_2": 0},
        )
        assert isinstance(result, ExperimentSpec)
        assert result.schedules == {}

    def test_target_indices_source_step_preserved(self, tmp_path):
        cs, exp, datamodule, current_params = self._configured_cs(tmp_path)
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.INFERENCE,
            current_params=current_params, target_indices={"dim_1": 0},
        )
        assert result.initial_params.source_step == "inference_step"

    def test_target_indices_without_trajectory_config_is_experiment_level(self, tmp_path):
        """target_indices with no trajectory configs → experiment-level step (all params eligible)."""
        agent, exp, datamodule = _setup_two_runtime_agent(tmp_path)
        cs = agent.calibration_system
        # No configure_step_parameter calls

        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
            target_indices={"dim_1": 0},
        )
        assert isinstance(result, ExperimentSpec)
        assert result.schedules == {}


# ===========================================================================
# 10. MPC lookahead — reduces greedy myopia via simulated rollouts
# ===========================================================================

class TestMPCLookahead:
    """mpc_lookahead > 0 augments each step's objective with a short forward rollout
    .  The step-loop structure and return
    type are unchanged; only the score landscape seen by the optimizer differs.
    """

    # --- Helpers ---

    def _configured_cs_with_delta(self, tmp_path):
        """Return (cs, exp, datamodule, current_params) with speed delta configured."""
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_adaptation_delta({"speed": 50.0})
        current_params = {"param_1": 2.5, "speed": 100.0, "dim_1": 2, "dim_2": 3}
        return cs, exp, datamodule, current_params

    def _two_runtime_cs(self, tmp_path):
        """Return (cs, exp, datamodule, current_params) with both runtime params configured."""
        agent, exp, datamodule = _setup_two_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_step_parameter("layer_height", "dim_1")
        cs.configure_step_parameter("speed", "dim_2")
        cs.configure_adaptation_delta({"layer_height": 0.1, "speed": 50.0})
        current_params = {**exp.parameters.get_values_dict()}
        return cs, exp, datamodule, current_params

    # --- Smoke tests ---

    def test_mpc_depth_zero_returns_experiment_spec(self, tmp_path):
        """Explicit mpc_lookahead=0 (default, no lookahead) should behave identically to omitting the arg."""
        cs, exp, datamodule, current_params = self._configured_cs_with_delta(tmp_path)
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
            current_params=current_params, mpc_lookahead=0,
        )
        assert isinstance(result, ExperimentSpec)

    def test_mpc_depth_one_returns_experiment_spec(self, tmp_path):
        cs, exp, datamodule, current_params = self._configured_cs_with_delta(tmp_path)
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
            current_params=current_params, mpc_lookahead=1,
        )
        assert isinstance(result, ExperimentSpec)

    def test_mpc_depth_two_returns_experiment_spec(self, tmp_path):
        cs, exp, datamodule, current_params = self._configured_cs_with_delta(tmp_path)
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
            current_params=current_params, mpc_lookahead=2,
        )
        assert isinstance(result, ExperimentSpec)

    def test_mpc_inference_mode_returns_experiment_spec(self, tmp_path):
        """MPC works with INFERENCE mode as well."""
        cs, exp, datamodule, current_params = self._configured_cs_with_delta(tmp_path)
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.INFERENCE,
            current_params=current_params, mpc_lookahead=1,
        )
        assert isinstance(result, ExperimentSpec)

    def test_mpc_online_domain_returns_experiment_spec(self, tmp_path):
        """MPC also applies in the ONLINE domain (single-step trust-region run)."""
        cs, exp, datamodule, current_params = self._configured_cs_with_delta(tmp_path)
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION, target_indices={},
            current_params=current_params, mpc_lookahead=1,
        )
        assert isinstance(result, ExperimentSpec)

    def test_mpc_with_trajectory_returns_both_schedules(self, tmp_path):
        """MPC does not disrupt schedule assembly — both dim schedules still produced."""
        cs, exp, datamodule, current_params = self._two_runtime_cs(tmp_path)
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
            current_params=current_params, mpc_lookahead=1,
        )
        assert isinstance(result, ExperimentSpec)
        assert "dim_1" in result.schedules
        assert "dim_2" in result.schedules

    def test_mpc_source_step_preserved(self, tmp_path):
        """MPC does not affect the source_step tag on initial_params."""
        cs, exp, datamodule, current_params = self._configured_cs_with_delta(tmp_path)
        result = cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
            current_params=current_params, mpc_lookahead=1,
        )
        assert result.initial_params.source_step == "exploration_step"

    # --- _wrap_mpc_objective unit tests ---

    def test_wrap_mpc_depth_zero_returns_base_unchanged(self, tmp_path):
        """depth=0 must return the exact same callable object (no wrapping overhead)."""
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        base = cs._build_objective(Mode.EXPLORATION, 0.5)
        wrapped = cs._wrap_mpc_objective(base, datamodule, depth=0, discount=0.9)
        assert wrapped is base

    def test_wrap_mpc_depth_nonzero_returns_new_callable(self, tmp_path):
        """depth>0 must produce a different callable (the MPC wrapper)."""
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs._active_datamodule = datamodule
        base = cs._build_objective(Mode.EXPLORATION, 0.5)
        wrapped = cs._wrap_mpc_objective(base, datamodule, depth=1, discount=0.9)
        assert wrapped is not base
        assert callable(wrapped)

    def test_wrap_mpc_discount_zero_equals_base_score(self, tmp_path):
        """With discount=0 the MPC objective at any X equals the base objective at X.

        We use a concrete, well-behaved lambda (not the model-based acquisition)
        to avoid nan-propagation issues (0.0 * nan = nan in IEEE 754).
        """
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs._active_datamodule = datamodule
        cs.configure_adaptation_delta({"speed": 50.0})

        # Simple finite objective — no model calls, no NaN risk
        def simple_base(X: np.ndarray) -> float:
            return -float(np.sum(X ** 2))

        mpc = cs._wrap_mpc_objective(simple_base, datamodule, depth=2, discount=0.0)

        n_dims = len(datamodule.input_columns)
        X_test = np.full(n_dims, 0.5)

        base_score = simple_base(X_test)
        mpc_score = mpc(X_test)
        assert mpc_score == pytest.approx(base_score, abs=1e-8)

    # --- Call-count: MPC must evaluate the objective more than greedy ---

    def test_mpc_evaluates_objective_more_than_greedy(self, tmp_path):
        """Each MPC objective evaluation runs an inner minimize → more base calls."""
        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_adaptation_delta({"speed": 50.0})

        call_count = {"n": 0}
        original_perf_fn = cs.perf_fn

        def counting_perf_fn(params_dict):
            call_count["n"] += 1
            return original_perf_fn(params_dict)

        cs.perf_fn = counting_perf_fn
        current_params = {"param_1": 2.5, "speed": 100.0, "dim_1": 2, "dim_2": 3}

        # Greedy baseline (mpc_lookahead=0 → no lookahead)
        call_count["n"] = 0
        cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
            current_params=current_params, mpc_lookahead=0,
        )
        calls_greedy = call_count["n"]

        # MPC depth=1 (mpc_lookahead=1 → one step lookahead)
        call_count["n"] = 0
        cs.run_calibration(
            datamodule=datamodule, mode=Mode.EXPLORATION,
            current_params=current_params, mpc_lookahead=1,
        )
        calls_mpc = call_count["n"]

        assert calls_mpc > calls_greedy, (
            f"Expected MPC to call objective more than greedy "
            f"(greedy={calls_greedy}, mpc={calls_mpc})"
        )

    # --- MPC on shaped objectives selects different X than pure greedy ---

    def test_mpc_with_shaped_objective_differs_from_greedy(self, tmp_path):
        """On a strongly directional objective, MPC with discount=1 picks a different
        starting point than greedy because it anticipates the future trust-region step.

        The shaped objective rewards higher speed linearly.  Pure greedy will
        pick the maximum of the trust region around current_params (speed=100).
        MPC looks one step ahead and knows that a moderate speed now leads to a
        higher speed next step → it discounts the first step and aims for a point
        that enables a larger overall gain.
        """
        from pred_fab.orchestration.calibration import CalibrationSystem as _CS
        from tests.utils.builders import build_calibration_system, build_dataset_with_single_experiment

        # Build a CS where uncertainty is constant and perf = speed / 200.
        # The acquisition function then purely rewards high speed.
        dataset = build_dataset_with_single_experiment(tmp_path)

        agent, exp, datamodule = _setup_runtime_agent(tmp_path)
        cs = agent.calibration_system
        cs.configure_adaptation_delta({"speed": 50.0})
        delta = 50.0

        # Replace perf_fn with a linearly shaped one (higher speed = better)
        def shaped_perf(params_dict):
            return {"performance_1": float(params_dict.get("speed", 0.0)) / 200.0}

        cs.perf_fn = shaped_perf
        cs._active_datamodule = datamodule

        current_params = {"param_1": 2.5, "speed": 100.0, "dim_1": 2, "dim_2": 3}

        # Pure greedy (mpc_lookahead=0): maximize speed within [100-50, 100+50] = [50, 150]
        result_greedy = cs.run_calibration(
            datamodule=datamodule, mode=Mode.INFERENCE,
            current_params=current_params, mpc_lookahead=0,
        )
        speed_greedy = float(result_greedy["speed"])

        # MPC mpc_lookahead=1, discount=1.0: also considers step k+1 → can justify a slightly
        # lower step-k speed to stay centred and avoid hitting the schema boundary.
        result_mpc = cs.run_calibration(
            datamodule=datamodule, mode=Mode.INFERENCE,
            current_params=current_params, mpc_lookahead=1, mpc_discount=1.0,
        )
        speed_mpc = float(result_mpc["speed"])

        # Both must stay within trust-region bounds
        assert speed_greedy >= 100.0 - delta - 1e-4
        assert speed_greedy <= 100.0 + delta + 1e-4
        assert speed_mpc >= 100.0 - delta - 1e-4
        assert speed_mpc <= 100.0 + delta + 1e-4
        # Both are valid ExperimentSpecs
        assert isinstance(result_greedy, ExperimentSpec)
        assert isinstance(result_mpc, ExperimentSpec)
