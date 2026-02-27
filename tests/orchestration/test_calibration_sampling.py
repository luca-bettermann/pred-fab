"""
Edge-case tests for CalibrationSystem baseline generation and run_adaptation:
generate_baseline_experiments (n_samples, infinite bounds, fixed params),
run_adaptation behavior, _compute_system_performance mismatched lengths,
and _get_online_bounds trust-region semantics.
"""
import pytest
import numpy as np

from pred_fab.core import ParameterProposal
from pred_fab.utils.enum import Mode
from tests.utils.builders import (
    build_calibration_system_with_capturing_surrogate,
    build_dataset_with_single_experiment,
    build_workflow_stack,
    evaluate_loaded_workflow_experiments,
    configure_default_workflow_calibration,
    build_prepared_workflow_datamodule,
    build_real_agent_stack,
)


# ===== generate_baseline_experiments() basic behavior =====

def test_generate_baseline_returns_correct_count(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})

    results = calibration.generate_baseline_experiments(n_samples=5)
    assert len(results) == 5


def test_generate_baseline_returns_empty_for_zero_samples(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    results = calibration.generate_baseline_experiments(n_samples=0)
    assert results == []


def test_generate_baseline_all_samples_contain_schema_keys(tmp_path):
    """Every generated sample should contain all schema parameter keys."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4), "dim_1": (1, 3), "dim_2": (1, 3)})
    calibration.configure_fixed_params({"param_3": "B"})

    results = calibration.generate_baseline_experiments(n_samples=4)
    schema_keys = set(dataset.schema.parameters.keys())
    for sample in results:
        assert schema_keys == set(sample.keys()), f"Missing keys in sample: {schema_keys - set(sample.keys())}"


def test_generate_baseline_fixed_params_appear_in_all_samples(tmp_path):
    """Fixed parameters should appear with their fixed value in every sample."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)
    calibration.configure_fixed_params({"param_3": "A"})

    results = calibration.generate_baseline_experiments(n_samples=6)
    for sample in results:
        assert sample["param_3"] == "A"


def test_generate_baseline_values_stay_within_param_bounds(tmp_path):
    """Sampled continuous values should respect configured parameter bounds."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (2.0, 7.0)})

    results = calibration.generate_baseline_experiments(n_samples=20)
    for sample in results:
        assert 2.0 <= sample["param_1"] <= 7.0, f"param_1 out of bounds: {sample['param_1']}"


def test_generate_baseline_integer_params_are_int_typed(tmp_path):
    """Integer parameters (param_2, dim_1, dim_2) should be coerced to int type."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    results = calibration.generate_baseline_experiments(n_samples=5)
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
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    results = calibration.generate_baseline_experiments(n_samples=3)
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
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)
    calibration.set_performance_weights({"performance_1": 1.0, "performance_2": 1.0})

    with pytest.raises(IndexError):
        # Only 1 value but 2 perf attrs — should raise IndexError
        calibration._compute_system_performance([0.5])


# ===== run_adaptation() behavior =====

def test_run_adaptation_returns_dict_with_schema_keys(tmp_path):
    """run_adaptation should return a dict containing all parameter keys from the schema."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    # Configure a trust region on param_1 so adaptation can move
    agent.calibration_system.configure_adaptation_delta({"param_1": 1.0})

    current_params = exp.parameters.get_values_dict()
    result = agent.calibration_system.run_adaptation(
        datamodule=datamodule,
        mode=Mode.INFERENCE,
        current_params=current_params,
    )

    assert isinstance(result, dict)
    schema_params = set(dataset.schema.parameters.keys())
    assert schema_params.issubset(set(result.keys()))


def test_run_adaptation_with_no_trust_regions_returns_current_params(tmp_path):
    """
    With no trust regions configured, all params are fixed to current_params.
    The optimizer has zero degrees of freedom and should return (near) current values.
    """
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    # No trust regions — all params fixed
    assert len(agent.calibration_system.trust_regions) == 0
    current_params = exp.parameters.get_values_dict()

    result = agent.calibration_system.run_adaptation(
        datamodule=datamodule,
        mode=Mode.INFERENCE,
        current_params=current_params,
    )

    # With zero degrees of freedom, result should equal current params (or be very close)
    assert result["param_1"] == pytest.approx(current_params["param_1"], abs=0.01)


def test_run_adaptation_trust_region_constrains_proposed_value(tmp_path):
    """
    The proposed param_1 should stay within delta of the current value.
    """
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    delta = 0.5
    agent.calibration_system.configure_adaptation_delta({"param_1": delta})
    current_params = exp.parameters.get_values_dict()
    current_val = current_params["param_1"]

    result = agent.calibration_system.run_adaptation(
        datamodule=datamodule,
        mode=Mode.INFERENCE,
        current_params=current_params,
    )

    assert result["param_1"] == pytest.approx(current_val, abs=delta + 0.1)


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
    assert cs.trust_regions["param_1"] == pytest.approx(0.1)


# ===== Baseline determinism =====

def test_generate_baseline_is_deterministic_with_same_seed(tmp_path):
    """Same random seed must produce identical LHS samples."""
    agent, dataset, codes = build_workflow_stack(tmp_path)

    cal1, _ = build_calibration_system_with_capturing_surrogate(tmp_path / "a", dataset)
    cal1.random_seed = 7
    cal1.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})
    r1 = cal1.generate_baseline_experiments(n_samples=5)

    cal2, _ = build_calibration_system_with_capturing_surrogate(tmp_path / "b", dataset)
    cal2.random_seed = 7
    cal2.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})
    r2 = cal2.generate_baseline_experiments(n_samples=5)

    for s1, s2 in zip(r1, r2):
        assert s1["param_1"] == pytest.approx(s2["param_1"], abs=1e-6)
        assert s1["param_2"] == s2["param_2"]
