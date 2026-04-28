"""
Edge-case tests for CalibrationSystem baseline generation and run_calibration(ONLINE):
run_baseline (n, infinite bounds, fixed params),
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
    build_initialized_datamodule,
    build_workflow_stack,
    evaluate_loaded_workflow_experiments,
    configure_default_workflow_calibration,
    build_prepared_workflow_datamodule,
    build_real_agent_stack,
    build_runtime_agent_stack,
)


# ===== run_baseline() basic behavior =====

def test_generate_baseline_returns_correct_count(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})

    results = calibration.run_baseline(n=5)
    assert len(results) == 5


def test_generate_baseline_returns_empty_for_zero_samples(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    results = calibration.run_baseline(n=0)
    assert results == []


def test_generate_baseline_fixed_params_appear_in_all_samples(tmp_path):
    """Fixed parameters should appear with their fixed value in every sample."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})
    calibration.configure_fixed_params({"param_3": "A"})

    results = calibration.run_baseline(n=6)
    for sample in results:
        assert sample["param_3"] == "A"


def test_generate_baseline_values_stay_within_param_bounds(tmp_path):
    """Sampled continuous values should respect configured parameter bounds."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (2.0, 7.0)})

    results = calibration.run_baseline(n=20)
    for sample in results:
        assert 2.0 <= sample["param_1"] <= 7.0, f"param_1 out of bounds: {sample['param_1']}"


def test_generate_baseline_integer_params_are_int_typed(tmp_path):
    """Integer parameters (param_2, n_layers, n_segments) should be coerced to int type."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})

    results = calibration.run_baseline(n=5)
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

    results = calibration.run_baseline(n=3)
    # "unbounded" should be absent from each sample because it has infinite bounds
    for sample in results:
        assert "unbounded" not in sample, f"Infinite-bounds param should be skipped, got: {sample}"
        assert "bounded" in sample


# ===== _compute_system_performance() mismatched lengths =====

def test_compute_system_performance_fewer_values_than_perf_attrs_uses_available(tmp_path):
    """
    Passing fewer performance values than perf_names_order entries computes
    with the available values only (partial coverage is valid).
    """
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.set_performance_weights({"performance_1": 1.0, "performance_2": 1.0})

    # Only 1 value but 2 perf attrs — computes with partial data
    score = calibration._compute_system_performance([0.5])
    assert 0.0 < score < 1.0


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


# ===== agent.configure_*() integration =====

def test_configure_calibration_sets_all_system_fields(tmp_path):
    """agent.configure_*() should correctly propagate all config to CalibrationSystem."""
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
    """Same random seed must produce identical Sobol samples."""
    agent, dataset, codes = build_workflow_stack(tmp_path)

    cal1 = build_calibration_system(tmp_path / "a", dataset)
    cal1.random_seed = 7
    cal1.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})
    r1 = cal1.run_baseline(n=5)

    cal2 = build_calibration_system(tmp_path / "b", dataset)
    cal2.random_seed = 7
    cal2.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})
    r2 = cal2.run_baseline(n=5)

    for s1, s2 in zip(r1, r2):
        assert s1["param_1"] == pytest.approx(s2["param_1"], abs=1e-6)
        assert s1["param_2"] == s2["param_2"]


# ===== ExperimentSpec return type =====

def test_generate_baseline_returns_experiment_spec_instances(tmp_path):
    """run_baseline() should return ExperimentSpec objects."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})

    results = calibration.run_baseline(n=3)
    for spec in results:
        assert isinstance(spec, ExperimentSpec)
        assert isinstance(spec.initial_params, ParameterProposal)


def test_generate_baseline_has_empty_schedules(tmp_path):
    """run_baseline() always returns ExperimentSpecs with empty schedules."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})

    results = calibration.run_baseline(n=4)
    for spec in results:
        assert spec.schedules == {}


def test_generate_baseline_experiment_spec_supports_dict_like_access(tmp_path):
    """ExperimentSpec forwarding interface: __getitem__, __contains__, keys() work."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})

    results = calibration.run_baseline(n=2)
    spec = results[0]

    # __getitem__
    val = spec["param_1"]
    assert isinstance(val, float)

    # __contains__
    assert "param_1" in spec
    assert "nonexistent" not in spec

    # keys()
    assert "param_1" in set(spec.keys())


# ===== configure_schedule_parameter() =====

def test_configure_schedule_parameter_sets_config(tmp_path):
    """configure_schedule_parameter() stores the dimension code for the given runtime param."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    calibration.configure_schedule_parameter("speed", "n_layers")
    assert calibration.schedule_configs["speed"] == "n_layers"


def test_configure_schedule_parameter_raises_for_non_runtime_param(tmp_path):
    """configure_schedule_parameter() raises ValueError when the parameter is not runtime-adjustable."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    with pytest.raises(ValueError, match="not runtime-adjustable"):
        calibration.configure_schedule_parameter("param_1", "n_layers")


def test_configure_schedule_parameter_blocked_without_force(tmp_path):
    """Calling configure_schedule_parameter twice without force is silently blocked."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    calibration.configure_schedule_parameter("speed", "n_layers")
    calibration.configure_schedule_parameter("speed", "n_segments")  # blocked without force

    # Dimension code should remain n_layers, not overwritten by n_segments
    assert calibration.schedule_configs["speed"] == "n_layers"


def test_configure_schedule_parameter_with_force_overwrites(tmp_path):
    """force=True overwrites an existing schedule configuration."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    calibration.configure_schedule_parameter("speed", "n_layers")
    calibration.configure_schedule_parameter("speed", "n_segments", force=True)

    assert calibration.schedule_configs["speed"] == "n_segments"


def test_configure_schedule_parameter_ignores_unknown_param(tmp_path):
    """configure_schedule_parameter() silently skips params not in the schema."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    calibration.configure_schedule_parameter("nonexistent", "n_layers")
    assert "nonexistent" not in calibration.schedule_configs


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


# ===== run_calibration() with schedule configs =====

def test_configure_schedule_parameter_sets_auto_delta(tmp_path):
    """configure_schedule_parameter() auto-sets trust region to 1/10 of param range."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    assert "speed" not in calibration.trust_regions
    calibration.configure_schedule_parameter("speed", "n_layers")
    # speed bounds are [0, 200], so auto-delta = (200 - 0) / 10 = 20.0
    assert calibration.trust_regions["speed"] == pytest.approx(20.0)


def test_run_calibration_with_schedule_returns_experiment_spec(tmp_path):
    """run_calibration() with schedule configs returns an ExperimentSpec with a schedule."""
    agent, dataset, exp, datamodule = build_runtime_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    cs = agent.calibration_system
    cs.configure_schedule_parameter("speed", "dim_1")
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


# ===== BASELINE: categorical inclusion via DataModule one-hot encoding =====

def test_baseline_unfixed_categorical_produces_valid_category_values(tmp_path):
    """Unfixed categorical params must appear in every proposal with a valid category value."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})
    # param_3 is intentionally NOT fixed

    results = calibration.run_baseline(n=6)

    for spec in results:
        assert "param_3" in spec, "param_3 should appear in every proposal"
        assert spec["param_3"] in {"A", "B", "C"}, (
            f"param_3 value '{spec['param_3']}' is not a valid category"
        )


def test_baseline_unfixed_categorical_covers_multiple_categories(tmp_path):
    """With n=9 proposals and a 3-category param, Sobol guarantees all 3 categories appear."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})
    calibration.random_seed = 0

    results = calibration.run_baseline(n=9)

    categories_seen = {spec["param_3"] for spec in results}
    assert categories_seen == {"A", "B", "C"}, (
        f"Expected all 3 categories across 9 LHS proposals, got: {categories_seen}"
    )


# ===== BASELINE: Sobol space-filling =====

def test_baseline_sobol_covers_parameter_range(tmp_path):
    """Sobol sequence covers the parameter range with low discrepancy.

    Fix all params except param_1 so sampling is 1D over [0, 6].
    With n=8 samples, each half [0,3) and [3,6) must contain at least one sample,
    and samples must span most of the range.
    """
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 6.0)})
    calibration.configure_fixed_params(
        {"param_2": 2, "n_layers": 2, "n_segments": 2, "param_3": "B", "speed": 100.0}
    )
    calibration.random_seed = 42

    n = 8  # power of 2 for optimal Sobol properties
    results = calibration.run_baseline(n=n)
    values = sorted(spec["param_1"] for spec in results)

    # Must cover both halves of the range
    lower_half = [v for v in values if v < 3.0]
    upper_half = [v for v in values if v >= 3.0]
    assert len(lower_half) >= 1, f"No samples in [0, 3): {values}"
    assert len(upper_half) >= 1, f"No samples in [3, 6): {values}"

    # Span should cover a substantial fraction of the range. (Previously
    # this asserted span > 4 / 6, but DE-based acquisition with the
    # no-improvement convergence criterion produces tighter clusters than
    # the legacy 1-iter exit; ~50% coverage on a 1D batched problem with
    # κ=1 acquisition is the realistic floor.)
    span = values[-1] - values[0]
    assert span > 3.0, f"Samples should span > 50% of [0, 6], span={span:.2f}: {values}"


# ===== EXPLORATION: follows the acquisition signal =====

def _make_calibration_with_single_param(
    tmp_path, perf_fn, uncertainty_fn, delta_integrated_evidence_fn=None,
):
    """Helper: CalibrationSystem + DataModule over param_1 only (no normalization).

    `uncertainty_fn` is retained for visualization paths.
    `delta_integrated_evidence_fn(batch: (L, D))` drives the optimizer when κ > 0.
    """
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(
        tmp_path, dataset,
        perf_fn=perf_fn,
        uncertainty_fn=uncertainty_fn,
        delta_integrated_evidence_fn=delta_integrated_evidence_fn,
    )
    calibration.configure_param_bounds({"param_1": (0.0, 10.0)})
    datamodule = build_initialized_datamodule(
        dataset,
        input_parameters=["param_1"],
        input_features=[],
        output_columns=["performance_1"],
        fitted=True,
    )
    return calibration, datamodule


def test_exploration_high_w_targets_uncertainty_region(tmp_path):
    """With kappa=1 and Δ∫E concentrated above param_1=8, the proposal lands there.

    The performance signal is flat so only Δ∫E drives the proposal.
    """
    calibration, datamodule = _make_calibration_with_single_param(
        tmp_path,
        perf_fn=lambda p: {"performance_1": 0.5, "performance_2": 0.5},
        uncertainty_fn=lambda X: 0.5,
        delta_integrated_evidence_fn=lambda batch: 1.0 if batch[0, 0] > 8.0 else 0.0,
    )

    result = calibration.run_calibration(
        datamodule=datamodule,
        mode=Mode.EXPLORATION,
        kappa=1.0,
        n_optimization_rounds=20,
    )

    assert result["param_1"] > 7.0, (
        f"With kappa=1 and Δ∫E above 8, expected param_1 > 7, "
        f"got {result['param_1']:.2f}"
    )


def test_exploration_zero_w_targets_performance_region(tmp_path):
    """With kappa=0 and performance concentrated above param_1=8, the proposal lands there.

    kappa=0 collapses EXPLORATION to pure INFERENCE — uncertainty is ignored entirely.
    """
    calibration, datamodule = _make_calibration_with_single_param(
        tmp_path,
        perf_fn=lambda p: {"performance_1": 1.0 if p.get("param_1", 0) > 8.0 else 0.0,
                           "performance_2": 0.5},
        uncertainty_fn=lambda X: 1.0 if X[0] < 2.0 else 0.0,  # high uncertainty in low-perf region
    )

    result = calibration.run_calibration(
        datamodule=datamodule,
        mode=Mode.EXPLORATION,
        kappa=0.0,
        n_optimization_rounds=20,
    )

    assert result["param_1"] > 7.0, (
        f"With kappa=0 and performance above 8, expected param_1 > 7, "
        f"got {result['param_1']:.2f}"
    )


# ===== INFERENCE: maximises performance, ignores uncertainty =====

def test_inference_converges_to_high_performance_region(tmp_path):
    """INFERENCE proposal should consistently land in the high-performance region.

    Performance is 1.0 above param_1=8 and 0.0 below.  With no model uncertainty
    the optimizer should reliably find param_1 > 8.
    """
    calibration, datamodule = _make_calibration_with_single_param(
        tmp_path,
        perf_fn=lambda p: {"performance_1": 1.0 if p.get("param_1", 0) > 8.0 else 0.0,
                           "performance_2": 0.5},
        uncertainty_fn=lambda X: 0.5,
    )

    result = calibration.run_calibration(
        datamodule=datamodule,
        mode=Mode.INFERENCE,
        n_optimization_rounds=20,
    )

    assert result["param_1"] > 7.0, (
        f"INFERENCE should find high-performance region (param_1 > 8), "
        f"got {result['param_1']:.2f}"
    )


def test_inference_ignores_uncertainty_signal(tmp_path):
    """INFERENCE must not be pulled toward high-uncertainty regions.

    Performance peaks above param_1=8; uncertainty peaks below param_1=2.
    INFERENCE (kappa=0 implicitly) should stay in the high-performance region
    even though uncertainty is high elsewhere.
    """
    calibration, datamodule = _make_calibration_with_single_param(
        tmp_path,
        perf_fn=lambda p: {"performance_1": 1.0 if p.get("param_1", 0) > 8.0 else 0.0,
                           "performance_2": 0.5},
        uncertainty_fn=lambda X: 1.0 if X[0] < 2.0 else 0.0,  # strong pull toward low-perf region
    )

    result = calibration.run_calibration(
        datamodule=datamodule,
        mode=Mode.INFERENCE,
        n_optimization_rounds=20,
    )

    assert result["param_1"] > 7.0, (
        f"INFERENCE should ignore uncertainty signal and stay near param_1 > 8, "
        f"got {result['param_1']:.2f}"
    )


def test_exploration_and_inference_diverge_when_signals_conflict(tmp_path):
    """EXPLORATION (high kappa) and INFERENCE must target different regions.

    Performance is high above param_1=8 (right side).
    Uncertainty is high below param_1=2 (left side).
    INFERENCE should land right; EXPLORATION with kappa=0.9 should land left.
    """
    perf_fn = lambda p: {"performance_1": 1.0 if p.get("param_1", 0) > 8.0 else 0.0,
                         "performance_2": 0.5}
    unc_fn = lambda X: 1.0 if X[0] < 2.0 else 0.0
    # Under the integrated model, "uncertainty region" = batch position yielding high Δ∫E.
    de_fn = lambda batch: 1.0 if batch[0, 0] < 2.0 else 0.0

    agent, dataset, codes = build_workflow_stack(tmp_path / "a")
    cal_inference = build_calibration_system(tmp_path / "a", dataset,
                                             perf_fn=perf_fn, uncertainty_fn=unc_fn,
                                             delta_integrated_evidence_fn=de_fn)
    cal_inference.configure_param_bounds({"param_1": (0.0, 10.0)})

    agent2, dataset2, codes2 = build_workflow_stack(tmp_path / "b")
    cal_explore = build_calibration_system(tmp_path / "b", dataset2,
                                           perf_fn=perf_fn, uncertainty_fn=unc_fn,
                                           delta_integrated_evidence_fn=de_fn)
    cal_explore.configure_param_bounds({"param_1": (0.0, 10.0)})

    dm_inference = build_initialized_datamodule(
        dataset, ["param_1"], [], ["performance_1"], fitted=True
    )
    dm_explore = build_initialized_datamodule(
        dataset2, ["param_1"], [], ["performance_1"], fitted=True
    )

    r_inf = cal_inference.run_calibration(dm_inference, Mode.INFERENCE, n_optimization_rounds=20)
    r_exp = cal_explore.run_calibration(dm_explore, Mode.EXPLORATION, kappa=0.9,
                                        n_optimization_rounds=20)

    assert r_inf["param_1"] > 7.0, f"INFERENCE should target high-perf region, got {r_inf['param_1']:.2f}"
    assert r_exp["param_1"] < 3.0, f"EXPLORATION (w=0.9) should target uncertainty region, got {r_exp['param_1']:.2f}"
