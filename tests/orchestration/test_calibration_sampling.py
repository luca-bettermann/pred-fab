"""
Edge-case tests for CalibrationSystem baseline generation and run_calibration(ONLINE):
generate_baseline_experiments (n_samples, infinite bounds, fixed params),
run_calibration(domain=ONLINE) behavior, _compute_system_performance mismatched lengths,
and _get_online_bounds trust-region semantics.
"""
import pytest
import numpy as np

from pred_fab.core import ParameterProposal, ExperimentSpec, ParameterSchedule
from pred_fab.utils.enum import Mode
from tests.utils.builders import (
    build_calibration_system,
    build_dataset_with_single_experiment,
    build_workflow_stack,
    evaluate_loaded_workflow_experiments,
    configure_default_workflow_calibration,
    build_prepared_workflow_datamodule,
    build_real_agent_stack,
    build_runtime_agent_stack,
)


# ===== generate_baseline_experiments() basic behavior =====

def test_generate_baseline_returns_correct_count(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})

    results = calibration.baseline_sampler.generate(n_samples=5)
    assert len(results) == 5


def test_generate_baseline_returns_empty_for_zero_samples(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    results = calibration.baseline_sampler.generate(n_samples=0)
    assert results == []


def test_generate_baseline_all_samples_contain_schema_keys(tmp_path):
    """Every generated sample should contain all schema parameter keys."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4), "dim_1": (1, 3), "dim_2": (1, 3)})
    calibration.configure_fixed_params({"param_3": "B"})

    results = calibration.baseline_sampler.generate(n_samples=4)
    schema_keys = set(dataset.schema.parameters.keys())
    for sample in results:
        assert schema_keys == set(sample.keys()), f"Missing keys in sample: {schema_keys - set(sample.keys())}"


def test_generate_baseline_fixed_params_appear_in_all_samples(tmp_path):
    """Fixed parameters should appear with their fixed value in every sample."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_fixed_params({"param_3": "A"})

    results = calibration.baseline_sampler.generate(n_samples=6)
    for sample in results:
        assert sample["param_3"] == "A"


def test_generate_baseline_values_stay_within_param_bounds(tmp_path):
    """Sampled continuous values should respect configured parameter bounds."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (2.0, 7.0)})

    results = calibration.baseline_sampler.generate(n_samples=20)
    for sample in results:
        assert 2.0 <= sample["param_1"] <= 7.0, f"param_1 out of bounds: {sample['param_1']}"


def test_generate_baseline_integer_params_are_int_typed(tmp_path):
    """Integer parameters (param_2, dim_1, dim_2) should be coerced to int type."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    results = calibration.baseline_sampler.generate(n_samples=5)
    for sample in results:
        assert isinstance(sample["param_2"], int), f"param_2 should be int, got {type(sample['param_2'])}"


def test_generate_baseline_skips_params_with_infinite_schema_bounds(tmp_path):
    """Parameters without explicit bounds and with infinite schema bounds are skipped."""
    from pred_fab.core.data_objects import DataReal, Feature, PerformanceAttribute
    from pred_fab.core.data_blocks import Parameters, Features, PerformanceAttributes
    from pred_fab.core.schema import DatasetSchema
    from pred_fab.core import Dataset
    from pred_fab.utils.enum import Roles

    # Build a schema with one unconstrained parameter (no min/max → schema_bounds = (-inf, inf))
    params = Parameters.from_list([
        DataReal(code="bounded", role=Roles.PARAMETER, min_val=0.0, max_val=10.0),
        DataReal(code="unbounded", role=Roles.PARAMETER),  # no min/max → infinite bounds
    ])
    feats = Features.from_list([Feature.array("f1")])
    perfs = PerformanceAttributes.from_list([PerformanceAttribute.score("p1")])

    schema = DatasetSchema(
        root_folder=str(tmp_path),
        name="schema_infinite",
        parameters=params,
        features=feats,
        performance=perfs,
    )
    dataset = Dataset(schema=schema, debug_flag=True)
    calibration = build_calibration_system(tmp_path, dataset)

    results = calibration.baseline_sampler.generate(n_samples=3)
    # "unbounded" should be absent from each sample because it has infinite bounds
    for sample in results:
        assert "unbounded" not in sample, f"Infinite-bounds param should be skipped, got: {sample}"
        assert "bounded" in sample


# ===== _compute_system_performance() mismatched lengths =====

def test_compute_system_performance_fewer_values_than_perf_attrs_raises_index_error(tmp_path):
    """
    Passing fewer performance values than perf_names_order entries should expose the
    IndexError in the inner loop. Documents the known behavior — callers must match lengths.
    """
    agent, dataset, codes = build_workflow_stack(tmp_path)
    # workflow schema has 2 perf attrs: performance_1 and performance_2
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.set_performance_weights({"performance_1": 1.0, "performance_2": 1.0})

    with pytest.raises(IndexError):
        # Only 1 value but 2 perf attrs — should raise IndexError
        calibration._compute_system_performance([0.5])


# ===== run_calibration(domain=ONLINE) behavior =====

def test_run_calibration_online_returns_experiment_spec_with_schema_keys(tmp_path):
    """run_calibration(domain=ONLINE) should return an ExperimentSpec containing all parameter keys."""
    agent, dataset, exp, datamodule = build_runtime_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    # "speed" is runtime=True in the runtime schema — configure its trust region
    agent.calibration_system.configure_adaptation_delta({"speed": 10.0})

    current_params = exp.parameters.get_values_dict()
    result = agent.calibration_system.run_calibration(
        datamodule=datamodule,
        mode=Mode.INFERENCE,
        target_indices={},
        current_params=current_params,
    )

    assert isinstance(result, ExperimentSpec)
    schema_params = set(dataset.schema.parameters.keys())
    assert schema_params.issubset(set(result.keys()))


def test_run_calibration_online_with_no_trust_regions_and_no_runtime_params_passes(tmp_path):
    """
    Schema with NO runtime parameters: run_calibration(ONLINE) passes even with zero trust regions.
    The optimizer has zero degrees of freedom and should return (near) current values.
    """
    # build_real_agent_stack uses build_mixed_feature_schema which has no runtime params
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    # No trust regions, no runtime params → validation passes
    assert len(agent.calibration_system.trust_regions) == 0
    current_params = exp.parameters.get_values_dict()

    result = agent.calibration_system.run_calibration(
        datamodule=datamodule,
        mode=Mode.INFERENCE,
        target_indices={},
        current_params=current_params,
    )

    # With zero degrees of freedom, result should equal current params
    assert result["param_1"] == pytest.approx(current_params["param_1"], abs=0.01)


def test_run_calibration_online_raises_when_runtime_param_missing_trust_region(tmp_path):
    """
    Schema with a runtime parameter but no configured trust region → RuntimeError.
    """
    agent, dataset, exp, datamodule = build_runtime_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    # "speed" is runtime=True but NO trust region configured
    assert "speed" not in agent.calibration_system.trust_regions
    current_params = exp.parameters.get_values_dict()

    with pytest.raises(RuntimeError, match="runtime-adjustable"):
        agent.calibration_system.run_calibration(
            datamodule=datamodule,
            mode=Mode.INFERENCE,
            target_indices={},
            current_params=current_params,
        )


def test_run_calibration_online_trust_region_constrains_proposed_value(tmp_path):
    """
    The proposed ``speed`` value should stay within delta of the current value.
    """
    agent, dataset, exp, datamodule = build_runtime_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    delta = 5.0
    agent.calibration_system.configure_adaptation_delta({"speed": delta})
    current_params = exp.parameters.get_values_dict()
    current_val = current_params["speed"]

    result = agent.calibration_system.run_calibration(
        datamodule=datamodule,
        mode=Mode.INFERENCE,
        target_indices={},
        current_params=current_params,
    )

    assert result["speed"] == pytest.approx(current_val, abs=delta + 0.1)


# ===== CalibrationSystem.agent.configure_calibration integration =====

def test_configure_calibration_sets_all_system_fields(tmp_path):
    """agent.configure_calibration should correctly propagate all config to CalibrationSystem."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    configure_default_workflow_calibration(agent)

    cs = agent.calibration_system
    assert cs.performance_weights["performance_1"] == pytest.approx(2.0)
    assert cs.performance_weights["performance_2"] == pytest.approx(1.3)
    assert cs.param_bounds["param_1"] == (0.0, 10.0)
    assert cs.param_bounds["param_2"] == (1, 4)
    assert cs.fixed_params["param_3"] == "B"
    # "speed" is the runtime-adjustable parameter in the workflow schema
    assert cs.trust_regions["speed"] == pytest.approx(10.0)


# ===== Baseline determinism =====

def test_generate_baseline_is_deterministic_with_same_seed(tmp_path):
    """Same random seed must produce identical LHS samples."""
    agent, dataset, codes = build_workflow_stack(tmp_path)

    cal1 = build_calibration_system(tmp_path / "a", dataset)
    cal1.random_seed = 7
    cal1.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})
    r1 = cal1.baseline_sampler.generate(n_samples=5)

    cal2 = build_calibration_system(tmp_path / "b", dataset)
    cal2.random_seed = 7
    cal2.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})
    r2 = cal2.baseline_sampler.generate(n_samples=5)

    for s1, s2 in zip(r1, r2):
        assert s1["param_1"] == pytest.approx(s2["param_1"], abs=1e-6)
        assert s1["param_2"] == s2["param_2"]


# ===== ExperimentSpec return type =====

def test_generate_baseline_returns_experiment_spec_instances(tmp_path):
    """generate_baseline_experiments() should return ExperimentSpec objects."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    results = calibration.baseline_sampler.generate(n_samples=3)
    for spec in results:
        assert isinstance(spec, ExperimentSpec)
        assert isinstance(spec.initial_params, ParameterProposal)


def test_generate_baseline_no_trajectories_has_empty_schedules(tmp_path):
    """Without trajectory configs, every ExperimentSpec has empty schedules."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    results = calibration.baseline_sampler.generate(n_samples=4)
    for spec in results:
        assert spec.schedules == {}


def test_generate_baseline_experiment_spec_supports_dict_like_access(tmp_path):
    """ExperimentSpec forwarding interface: __getitem__, __contains__, keys() work."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    results = calibration.baseline_sampler.generate(n_samples=2)
    spec = results[0]

    # __getitem__
    val = spec["param_1"]
    assert isinstance(val, float)

    # __contains__
    assert "param_1" in spec
    assert "nonexistent" not in spec

    # keys()
    assert "param_1" in set(spec.keys())
    assert "speed" in set(spec.keys())


# ===== configure_trajectory() =====

def test_configure_trajectory_sets_config(tmp_path):
    """configure_trajectory() stores the dimension code for the given runtime param."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    calibration.configure_trajectory("speed", "dim_1")
    assert calibration.trajectory_configs["speed"] == "dim_1"


def test_configure_trajectory_raises_for_non_runtime_param(tmp_path):
    """configure_trajectory() raises ValueError when the parameter is not runtime-adjustable."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    with pytest.raises(ValueError, match="not runtime-adjustable"):
        calibration.configure_trajectory("param_1", "dim_1")


def test_configure_trajectory_blocked_without_force(tmp_path):
    """Calling configure_trajectory twice without force is silently blocked."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    calibration.configure_trajectory("speed", "dim_1")
    calibration.configure_trajectory("speed", "dim_2")  # blocked without force

    # Dimension code should remain dim_1, not overwritten by dim_2
    assert calibration.trajectory_configs["speed"] == "dim_1"


def test_configure_trajectory_with_force_overwrites(tmp_path):
    """force=True overwrites an existing trajectory configuration."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    calibration.configure_trajectory("speed", "dim_1")
    calibration.configure_trajectory("speed", "dim_2", force=True)

    assert calibration.trajectory_configs["speed"] == "dim_2"


def test_configure_trajectory_ignores_unknown_param(tmp_path):
    """configure_trajectory() silently skips params not in the schema."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    calibration.configure_trajectory("nonexistent", "dim_1")
    assert "nonexistent" not in calibration.trajectory_configs


# ===== generate_baseline_experiments with trajectories =====

def test_generate_baseline_with_trajectory_generates_non_empty_schedules(tmp_path):
    """When trajectory is configured, ExperimentSpecs should contain non-empty schedules."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    calibration.configure_trajectory("speed", "dim_1")
    results = calibration.baseline_sampler.generate(n_samples=3, n_trajectory_segments=3)

    for spec in results:
        # The schedule keys should contain the dimension code
        assert len(spec.schedules) > 0
        # Each schedule should have trigger entries
        for schedule in spec.schedules.values():
            assert isinstance(schedule, ParameterSchedule)
            assert len(schedule.entries) > 0


def test_generate_baseline_trajectory_entries_contain_speed(tmp_path):
    """Schedule entries must contain 'speed' values when speed is trajectory-configured."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    calibration.configure_trajectory("speed", "dim_1")
    results = calibration.baseline_sampler.generate(n_samples=2, n_trajectory_segments=2)

    for spec in results:
        for dim_key, schedule in spec.schedules.items():
            for step_idx, proposal in schedule.entries:
                assert "speed" in proposal


# ===== ParameterSchedule.apply() =====

def test_parameter_schedule_apply_records_update_events(tmp_path):
    """ParameterSchedule.apply() records each entry as a ParameterUpdateEvent."""
    agent, dataset, exp, _ = build_runtime_agent_stack(tmp_path)

    # "speed" initial = 100.0; change to 150.0 at step 1 of dim_1.
    proposal = ParameterProposal.from_dict({"speed": 150.0})
    schedule = ParameterSchedule(dimension="dim_1", entries=[(1, proposal)])

    initial_count = len(exp.parameter_updates)
    schedule.apply(exp)

    assert len(exp.parameter_updates) == initial_count + 1
    event = exp.parameter_updates[-1]
    assert event.updates.get("speed") == pytest.approx(150.0)
    assert event.dimension == "dim_1"
    assert event.step_index == 1


def test_experiment_spec_apply_schedules_records_all_entries(tmp_path):
    """ExperimentSpec.apply_schedules() applies all dimensional schedules to the experiment."""
    agent, dataset, exp, _ = build_runtime_agent_stack(tmp_path)

    proposal = ParameterProposal.from_dict({"speed": 180.0})
    schedule = ParameterSchedule(dimension="dim_1", entries=[(1, proposal)])
    spec = ExperimentSpec(
        initial_params=ParameterProposal.from_dict({"speed": 100.0}),
        schedules={"dim_1": schedule},
    )

    initial_count = len(exp.parameter_updates)
    spec.apply_schedules(exp)

    assert len(exp.parameter_updates) == initial_count + 1


# ===== run_calibration() with trajectory configs (formerly run_trajectory_exploration) =====

def test_run_calibration_offline_raises_without_trust_region_for_trajectory_param(tmp_path):
    """run_calibration() raises RuntimeError when trajectory param lacks a trust region."""
    agent, dataset, exp, datamodule = build_runtime_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    cs = agent.calibration_system
    cs.configure_trajectory("speed", "dim_1")
    # Intentionally NOT calling configure_adaptation_delta for "speed"

    current_params = exp.parameters.get_values_dict()
    with pytest.raises(RuntimeError, match="trust region"):
        cs.run_calibration(
            datamodule=datamodule,
            mode=Mode.EXPLORATION,
            current_params=current_params,
        )


def test_run_calibration_with_trajectory_returns_experiment_spec(tmp_path):
    """run_calibration() with trajectory configs returns an ExperimentSpec with a schedule."""
    agent, dataset, exp, datamodule = build_runtime_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    cs = agent.calibration_system
    cs.configure_trajectory("speed", "dim_1")
    cs.configure_adaptation_delta({"speed": 50.0})

    current_params = exp.parameters.get_values_dict()
    result = cs.run_calibration(
        datamodule=datamodule,
        mode=Mode.EXPLORATION,
        current_params=current_params,
    )

    assert isinstance(result, ExperimentSpec)
    # "speed" should be in the initial params
    assert "speed" in result.initial_params
    # At least one schedule should be present (dim_1)
    assert len(result.schedules) > 0
