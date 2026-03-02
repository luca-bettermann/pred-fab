"""
Tests for CalibrationSystem configuration methods:
set_performance_weights, configure_param_bounds, configure_fixed_params,
configure_adaptation_delta, and _compute_system_performance.
"""
import pytest
import numpy as np

from tests.utils.builders import (
    build_calibration_system_with_capturing_surrogate,
    build_dataset_with_single_experiment,
    build_workflow_stack,
    evaluate_loaded_workflow_experiments,
    configure_default_workflow_calibration,
    build_prepared_workflow_datamodule,
)


# ===== set_performance_weights() =====

def test_set_performance_weights_updates_known_attribute(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.set_performance_weights({"performance_1": 2.5})
    assert calibration.performance_weights["performance_1"] == 2.5


def test_set_performance_weights_ignores_unknown_attribute(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.set_performance_weights({"nonexistent_metric": 3.0})
    assert "nonexistent_metric" not in calibration.performance_weights


def test_set_performance_weights_partial_update_preserves_others(tmp_path):
    """Setting weight for one metric should not affect others."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    original_w2 = calibration.performance_weights.get("performance_2", 1.0)
    calibration.set_performance_weights({"performance_1": 5.0})

    assert calibration.performance_weights["performance_1"] == 5.0
    assert calibration.performance_weights.get("performance_2", 1.0) == original_w2


# ===== configure_param_bounds() =====

def test_configure_param_bounds_sets_bounds_for_real_parameter(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.configure_param_bounds({"param_1": (1.0, 8.0)})
    assert calibration.param_bounds["param_1"] == (1.0, 8.0)


def test_configure_param_bounds_sets_bounds_for_integer_parameter(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.configure_param_bounds({"param_2": (1, 4)})
    assert calibration.param_bounds["param_2"] == (1, 4)


def test_configure_param_bounds_ignores_nonexistent_parameter(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.configure_param_bounds({"nonexistent": (0.0, 5.0)})
    assert "nonexistent" not in calibration.param_bounds


def test_configure_param_bounds_raises_when_exceeding_schema_max(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    with pytest.raises(ValueError, match="exceed schema constraints"):
        calibration.configure_param_bounds({"param_1": (0.0, 20.0)})


def test_configure_param_bounds_raises_when_below_schema_min(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    with pytest.raises(ValueError, match="exceed schema constraints"):
        calibration.configure_param_bounds({"param_1": (-5.0, 10.0)})


def test_configure_param_bounds_silently_ignores_categorical_parameter(tmp_path):
    """Categorical type fails type check — should be silently skipped."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.configure_param_bounds({"param_3": (0.0, 1.0)})
    assert "param_3" not in calibration.param_bounds


def test_configure_param_bounds_blocked_when_fixed_without_force(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.configure_fixed_params({"param_1": 5.0})
    calibration.configure_param_bounds({"param_1": (1.0, 9.0)})

    assert "param_1" not in calibration.param_bounds


def test_configure_param_bounds_with_force_removes_fixed_conflict(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.configure_fixed_params({"param_1": 5.0})
    calibration.configure_param_bounds({"param_1": (1.0, 9.0)}, force=True)

    assert "param_1" in calibration.param_bounds
    assert "param_1" not in calibration.fixed_params


# ===== configure_fixed_params() =====

def test_configure_fixed_params_sets_value_for_real(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.configure_fixed_params({"param_1": 5.0})
    assert calibration.fixed_params["param_1"] == 5.0


def test_configure_fixed_params_sets_value_for_categorical(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.configure_fixed_params({"param_3": "A"})
    assert calibration.fixed_params["param_3"] == "A"


def test_configure_fixed_params_ignores_unknown_parameter(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.configure_fixed_params({"nonexistent": "value"})
    assert "nonexistent" not in calibration.fixed_params


def test_configure_fixed_params_blocked_when_in_bounds_without_force(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.configure_param_bounds({"param_1": (1.0, 9.0)})
    calibration.configure_fixed_params({"param_1": 5.0})

    assert "param_1" not in calibration.fixed_params


def test_configure_fixed_params_with_force_removes_bounds_conflict(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.configure_param_bounds({"param_1": (1.0, 9.0)})
    calibration.configure_fixed_params({"param_1": 5.0}, force=True)

    assert "param_1" in calibration.fixed_params
    assert "param_1" not in calibration.param_bounds


def test_configure_fixed_params_blocked_when_in_trust_regions_without_force(tmp_path):
    """When a runtime param already has a trust region, trying to fix it is blocked."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    # "speed" is runtime=True, so delta config succeeds
    calibration.configure_adaptation_delta({"speed": 10.0})
    # Trying to fix a param that has a trust region (without force) should be blocked
    calibration.configure_fixed_params({"speed": 80.0})

    assert "speed" not in calibration.fixed_params


# ===== configure_adaptation_delta() =====

def test_configure_adaptation_delta_sets_trust_region(tmp_path):
    """Trust region is recorded for a runtime-adjustable parameter."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    # "speed" is declared runtime=True in build_workflow_schema
    calibration.configure_adaptation_delta({"speed": 10.0})
    assert calibration.trust_regions["speed"] == 10.0


def test_configure_adaptation_delta_raises_for_non_runtime_parameter(tmp_path):
    """Configuring a trust region for a static (non-runtime) parameter must raise ValueError."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    with pytest.raises(ValueError, match="not runtime-adjustable"):
        calibration.configure_adaptation_delta({"param_1": 0.5})


def test_configure_adaptation_delta_silently_ignores_categorical(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.configure_adaptation_delta({"param_3": 0.5})
    assert "param_3" not in calibration.trust_regions


def test_configure_adaptation_delta_blocked_when_fixed_without_force(tmp_path):
    """When a runtime param is already fixed, configuring a delta without force is blocked."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    # Fix "speed" first, then try to add trust region without force — should be blocked
    calibration.configure_fixed_params({"speed": 80.0})
    calibration.configure_adaptation_delta({"speed": 10.0})

    assert "speed" not in calibration.trust_regions


def test_configure_adaptation_delta_with_force_removes_fixed_conflict(tmp_path):
    """force=True removes the fixed-param conflict and stores the trust region."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.configure_fixed_params({"speed": 80.0})
    calibration.configure_adaptation_delta({"speed": 10.0}, force=True)

    assert "speed" in calibration.trust_regions
    assert "speed" not in calibration.fixed_params


def test_configure_adaptation_delta_ignores_unknown_parameter(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.configure_adaptation_delta({"nonexistent": 0.1})
    assert "nonexistent" not in calibration.trust_regions


# ===== _compute_system_performance() =====

def test_compute_system_performance_single_metric_equal_weight(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    score = calibration._compute_system_performance([0.8])
    assert score == pytest.approx(0.8)


def test_compute_system_performance_empty_list_returns_zero(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    score = calibration._compute_system_performance([])
    assert score == 0.0


def test_compute_system_performance_weighted_average(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.set_performance_weights({"performance_1": 2.0, "performance_2": 1.0})
    # performance_1=1.0, performance_2=0.0
    # Expected: (2.0 * 1.0 + 1.0 * 0.0) / (2.0 + 1.0) = 0.667
    score = calibration._compute_system_performance([1.0, 0.0])
    assert score == pytest.approx(2.0 / 3.0, abs=1e-5)


def test_compute_system_performance_all_zero_returns_zero(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    score = calibration._compute_system_performance([0.0])
    assert score == pytest.approx(0.0)


def test_compute_system_performance_all_one_returns_one(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    score = calibration._compute_system_performance([1.0])
    assert score == pytest.approx(1.0)


# ===== _get_hierarchical_bounds_for_code() =====

def test_hierarchical_bounds_returns_schema_bounds_by_default(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    low, high = calibration._get_hierarchical_bounds_for_code("param_1")
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(10.0)


def test_hierarchical_bounds_returns_param_bounds_when_configured(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.configure_param_bounds({"param_1": (2.0, 8.0)})
    low, high = calibration._get_hierarchical_bounds_for_code("param_1")
    assert low == pytest.approx(2.0)
    assert high == pytest.approx(8.0)


def test_hierarchical_bounds_returns_fixed_value_when_fixed(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    calibration.configure_fixed_params({"param_1": 5.0})
    low, high = calibration._get_hierarchical_bounds_for_code("param_1")
    assert low == high == pytest.approx(5.0)


def test_hierarchical_bounds_raises_for_unknown_code(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    calibration, _ = build_calibration_system_with_capturing_surrogate(tmp_path, dataset)

    with pytest.raises(ValueError):
        calibration._get_hierarchical_bounds_for_code("totally_unknown_param")
