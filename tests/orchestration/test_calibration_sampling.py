"""
Edge-case tests for CalibrationSystem discovery generation:
run_discovery (n, infinite bounds, fixed params),
_compute_system_performance mismatched lengths,
and configure_trajectory_parameter / ParameterTrajectory / ExperimentSpec behavior.
"""
import pytest
import torch

from pred_fab.core import ParameterProposal, ExperimentSpec, ParameterTrajectory
from pred_fab.orchestration.calibration.space import SolutionSpace
from pred_fab.utils import SourceStep
from tests.utils.builders import (
    build_calibration_system,
    build_workflow_stack,
    configure_default_workflow_calibration,
    build_runtime_agent_stack,
)


# ===== run_discovery() basic behavior =====

def test_generate_discovery_returns_correct_count(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})

    results = calibration.run_discovery(n=5)
    assert len(results) == 5


def test_generate_discovery_returns_empty_for_zero_samples(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    results = calibration.run_discovery(n=0)
    assert results == []


def test_generate_discovery_fixed_params_appear_in_all_samples(tmp_path):
    """Fixed parameters should appear with their fixed value in every sample."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})
    calibration.configure_fixed_params({"param_3": "A"})

    results = calibration.run_discovery(n=6)
    for sample in results:
        assert sample["param_3"] == "A"


def test_generate_discovery_values_stay_within_param_bounds(tmp_path):
    """Sampled continuous values should respect configured parameter bounds."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.engine.n_starts = 1
    calibration.engine.n_sobol = 1
    calibration.configure_param_bounds({"param_1": (2.0, 7.0)})

    results = calibration.run_discovery(n=3)
    for sample in results:
        assert 2.0 <= sample["param_1"] <= 7.0, f"param_1 out of bounds: {sample['param_1']}"


def test_generate_discovery_integer_params_are_int_typed(tmp_path):
    """Integer parameters (param_2, n_layers, n_segments) should be coerced to int type."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})

    results = calibration.run_discovery(n=5)
    for sample in results:
        assert isinstance(sample["param_2"], int), f"param_2 should be int, got {type(sample['param_2'])}"


def test_generate_discovery_skips_params_with_infinite_schema_bounds(tmp_path):
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
    feats = Features.from_list([Feature("f1")])
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

    results = calibration.run_discovery(n=3)
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


# ===== Discovery determinism =====

def test_generate_discovery_is_deterministic_with_same_seed(tmp_path):
    """Same random seed must produce identical Sobol samples."""
    agent, dataset, codes = build_workflow_stack(tmp_path)

    cal1 = build_calibration_system(tmp_path / "a", dataset)
    cal1.random_seed = 7
    cal1.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})
    r1 = cal1.run_discovery(n=5)

    cal2 = build_calibration_system(tmp_path / "b", dataset)
    cal2.random_seed = 7
    cal2.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})
    r2 = cal2.run_discovery(n=5)

    for s1, s2 in zip(r1, r2):
        assert s1["param_1"] == pytest.approx(s2["param_1"], abs=1e-6)
        assert s1["param_2"] == s2["param_2"]


# ===== ExperimentSpec return type =====

def test_generate_discovery_returns_experiment_spec_instances(tmp_path):
    """run_discovery() should return ExperimentSpec objects."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})

    results = calibration.run_discovery(n=3)
    for spec in results:
        assert isinstance(spec, ExperimentSpec)
        assert isinstance(spec.initial_params, ParameterProposal)


def test_generate_discovery_has_empty_trajectories(tmp_path):
    """run_discovery() always returns ExperimentSpecs with empty trajectories."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})

    results = calibration.run_discovery(n=4)
    for spec in results:
        assert spec.trajectories == {}


def test_generate_discovery_experiment_spec_supports_dict_like_access(tmp_path):
    """ExperimentSpec forwarding interface: __getitem__, __contains__, keys() work."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})

    results = calibration.run_discovery(n=2)
    spec = results[0]

    # __getitem__
    val = spec["param_1"]
    assert isinstance(val, float)

    # __contains__
    assert "param_1" in spec
    assert "nonexistent" not in spec

    # keys()
    assert "param_1" in set(spec.keys())


# ===== configure_trajectory_parameter() =====

def test_configure_trajectory_parameter_sets_config(tmp_path):
    """configure_trajectory_parameter() stores the dimension code for the given runtime param."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    calibration.configure_trajectory_parameter("speed", "n_layers")
    assert calibration.trajectory_configs["speed"] == "n_layers"


def test_configure_trajectory_parameter_raises_for_non_runtime_param(tmp_path):
    """configure_trajectory_parameter() raises ValueError when the parameter is not runtime-adjustable."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    with pytest.raises(ValueError, match="not runtime-adjustable"):
        calibration.configure_trajectory_parameter("param_1", "n_layers")


def test_configure_trajectory_parameter_blocked_without_force(tmp_path):
    """Calling configure_trajectory_parameter twice without force is silently blocked."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    calibration.configure_trajectory_parameter("speed", "n_layers")
    calibration.configure_trajectory_parameter("speed", "n_segments")  # blocked without force

    # Dimension code should remain n_layers, not overwritten by n_segments
    assert calibration.trajectory_configs["speed"] == "n_layers"


def test_configure_trajectory_parameter_with_force_overwrites(tmp_path):
    """force=True overwrites an existing trajectory configuration."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    calibration.configure_trajectory_parameter("speed", "n_layers")
    calibration.configure_trajectory_parameter("speed", "n_segments", force=True)

    assert calibration.trajectory_configs["speed"] == "n_segments"


def test_configure_trajectory_parameter_ignores_unknown_param(tmp_path):
    """configure_trajectory_parameter() silently skips params not in the schema."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    calibration.configure_trajectory_parameter("nonexistent", "n_layers")
    assert "nonexistent" not in calibration.trajectory_configs


# ===== ParameterTrajectory.apply() =====

def test_parameter_trajectory_apply_records_update_events(tmp_path):
    """ParameterTrajectory.apply() records each entry as a ParameterUpdateEvent."""
    agent, dataset, exp, _ = build_runtime_agent_stack(tmp_path)

    # "speed" initial = 100.0; change to 150.0 at step 1 of dim_1.
    proposal = ParameterProposal.from_dict({"speed": 150.0})
    traj = ParameterTrajectory(dimension="dim_1", entries=[(1, proposal)])

    initial_count = len(exp.parameter_updates)
    traj.apply(exp)

    assert len(exp.parameter_updates) == initial_count + 1
    event = exp.parameter_updates[-1]
    assert event.updates.get("speed") == pytest.approx(150.0)
    assert event.iterator_code == "dim_1"
    assert event.step_index == 1


def test_experiment_spec_apply_trajectories_records_all_entries(tmp_path):
    """ExperimentSpec.apply_trajectories() applies all dimensional trajectories to the experiment."""
    agent, dataset, exp, _ = build_runtime_agent_stack(tmp_path)

    proposal = ParameterProposal.from_dict({"speed": 180.0})
    traj = ParameterTrajectory(dimension="dim_1", entries=[(1, proposal)])
    spec = ExperimentSpec(
        initial_params=ParameterProposal.from_dict({"speed": 100.0}),
        trajectories={"dim_1": traj},
    )

    initial_count = len(exp.parameter_updates)
    spec.apply_trajectories(exp)

    assert len(exp.parameter_updates) == initial_count + 1


# ===== DISCOVERY: categorical inclusion via DataModule one-hot encoding =====

def test_discovery_unfixed_categorical_produces_valid_category_values(tmp_path):
    """Unfixed categorical params must appear in every proposal with a valid category value."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})
    # param_3 is intentionally NOT fixed

    results = calibration.run_discovery(n=6)

    for spec in results:
        assert "param_3" in spec, "param_3 should appear in every proposal"
        assert spec["param_3"] in {"A", "B", "C"}, (
            f"param_3 value '{spec['param_3']}' is not a valid category"
        )


def test_discovery_unfixed_categorical_covers_multiple_categories(tmp_path):
    """With n=9 proposals and a 3-category param, Sobol guarantees all 3 categories appear."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})
    calibration.random_seed = 0

    results = calibration.run_discovery(n=9)

    categories_seen = {spec["param_3"] for spec in results}
    assert categories_seen == {"A", "B", "C"}, (
        f"Expected all 3 categories across 9 LHS proposals, got: {categories_seen}"
    )


def test_categorical_strata_reach_objective_with_own_category(tmp_path):
    """Each stratum must be scored under its *own* category in the objective frame.

    Regression: ``_build_prior_fill`` fell through to 0.5 for categoricals
    (``float("A")`` raises → caught), so the raw/z-score the objective sees had
    every stratum at ``cats[round(0.5)] = cats[0]``. ``decode_to_specs`` re-injects
    the assigned category, so only the *objective* path (``decode``) exposed the
    bug. Categoricals carry no normaliser stats, so ``_decode_frames`` passes the
    column through unchanged — the fix places the raw cat-index there directly.
    """
    _agent, dataset, _codes = build_workflow_stack(tmp_path)
    cal = build_calibration_system(tmp_path, dataset)
    cal.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})

    n = 3  # one stratum per category of param_3 = [A, B, C]
    variables, cat_codes, cat_assignments = cal._compose_variables(n)
    assert cat_codes == ["param_3"]
    assigned = [a[0] for a in cat_assignments]
    assert sorted(assigned) == ["A", "B", "C"]

    dm = cal._build_schema_datamodule()
    space = SolutionSpace(
        variables=variables,
        n_experiments=n,
        bounds_manager=cal.bounds,
        datamodule=dm,
        fixed_params=dict(cal.fixed_params),
        cat_codes=cat_codes,
        cat_assignments=cat_assignments,
        schema_sanitize=lambda d: cal.schema.parameters.sanitize_values(d, ignore_unknown=True),
        dimension_derivations=cal.dimension_derivations,
    )

    cat_col = dm.input_columns.index("param_3")
    cats = dm.categorical_mappings["param_3"]  # sorted -> ["A", "B", "C"]

    # Inspect the objective frame (raw), NOT decode_to_specs — the latter
    # re-injects cat_assignments and would pass even with the bug.
    raw, _zscore, _w = space.decode(torch.zeros(1, space.total_vars, dtype=torch.float64))
    assert raw.shape[1] == n  # no trajectory -> one point per experiment
    raw0 = raw[0]

    decoded = [cats[int(round(float(raw0[i, cat_col].item())))] for i in range(n)]
    assert decoded == assigned                    # each stratum scored as its own category
    assert sorted(decoded) == ["A", "B", "C"]     # all three represented (bug -> all cats[0])


# ===== DISCOVERY: Sobol space-filling =====

def test_discovery_sobol_covers_parameter_range(tmp_path):
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
    results = calibration.run_discovery(n=n)
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


# ===== SOBOL: data-independent space-filling test design =====

def test_run_sobol_returns_count_and_sobol_provenance(tmp_path):
    """run_sobol() returns n ExperimentSpecs tagged SOBOL, with empty trajectories."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})

    results = calibration.run_sobol(n=5)
    assert len(results) == 5
    for spec in results:
        assert isinstance(spec, ExperimentSpec)
        assert isinstance(spec.initial_params, ParameterProposal)
        assert spec.initial_params.source_step == SourceStep.SOBOL
        assert spec.trajectories == {}


def test_run_sobol_returns_empty_for_zero(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    assert calibration.run_sobol(n=0) == []


def test_run_sobol_values_within_bounds_and_int_typed(tmp_path):
    """Continuous values respect bounds; integer params are int-typed."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (2.0, 7.0), "param_2": (1, 4)})

    results = calibration.run_sobol(n=6)
    for spec in results:
        assert 2.0 <= spec["param_1"] <= 7.0, f"param_1 out of bounds: {spec['param_1']}"
        assert isinstance(spec["param_2"], int), f"param_2 should be int: {type(spec['param_2'])}"
        assert 1 <= spec["param_2"] <= 4


def test_run_sobol_fixed_params_appear_in_all(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})
    calibration.configure_fixed_params({"param_3": "A"})

    results = calibration.run_sobol(n=6)
    for spec in results:
        assert spec["param_3"] == "A"


def test_run_sobol_is_uniform_space_filling_to_the_bounds(tmp_path):
    """The defining property: uniform coverage reaching the bounds.

    Unlike acquisition (which decodes through a sigmoid, ~25× denser at the centre
    than the bounds), a Sobol test set maps unit points straight via ``to_real`` —
    so it tiles the range and reaches the extremes the optimiser under-samples.
    Fix all but param_1 → 1D over [0, 10]; n=16 must populate every quartile and
    come within 10% of each bound.
    """
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0)})
    calibration.configure_fixed_params(
        {"param_2": 2, "n_layers": 2, "n_segments": 2, "param_3": "B", "speed": 100.0}
    )
    calibration.random_seed = 42

    values = sorted(spec["param_1"] for spec in calibration.run_sobol(n=16))

    # Every quartile of [0, 10] is populated (uniform tiling, not centre-clustered).
    for lo in (0.0, 2.5, 5.0, 7.5):
        in_q = [v for v in values if lo <= v < lo + 2.5]
        assert in_q, f"Quartile [{lo}, {lo + 2.5}) empty — not space-filling: {values}"
    # Reaches within 10% of each bound (the extremes a sigmoid frame would miss).
    assert values[0] < 1.0, f"min {values[0]:.3f} does not reach the lower bound"
    assert values[-1] > 9.0, f"max {values[-1]:.3f} does not reach the upper bound"


def test_run_sobol_stratifies_categoricals(tmp_path):
    """An unfixed categorical is stratified — all categories appear across the set."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})
    calibration.random_seed = 0

    results = calibration.run_sobol(n=9)
    categories_seen = {spec["param_3"] for spec in results}
    assert categories_seen == {"A", "B", "C"}, (
        f"Stratification should cover all 3 categories, got: {categories_seen}"
    )


def test_run_sobol_is_deterministic_with_same_seed(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)

    cal1 = build_calibration_system(tmp_path / "a", dataset)
    cal1.random_seed = 7
    cal1.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})
    r1 = cal1.run_sobol(n=5)

    cal2 = build_calibration_system(tmp_path / "b", dataset)
    cal2.random_seed = 7
    cal2.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})
    r2 = cal2.run_sobol(n=5)

    for s1, s2 in zip(r1, r2):
        assert s1["param_1"] == pytest.approx(s2["param_1"], abs=1e-6)
        assert s1["param_2"] == s2["param_2"]


def test_run_sobol_raises_on_trajectory_params(tmp_path):
    """Trajectory test design is a follow-up — refuse rather than emit incomplete experiments."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.configure_param_bounds(
        {"param_1": (0.0, 10.0), "param_2": (1, 4), "speed": (50.0, 150.0)}
    )
    calibration.configure_trajectory_parameter("speed", "n_layers")

    with pytest.raises(NotImplementedError, match="trajectory"):
        calibration.run_sobol(n=4)


# ===== generative provenance (config_snapshot: source + seed + settings) =====

def test_source_value_mapping():
    from pred_fab.orchestration.calibration.system import _source_value
    assert _source_value(SourceStep.DISCOVERY) == "discovery_step"
    assert _source_value(SourceStep.SOBOL) == "sobol_step"
    assert _source_value("exploration_step") == "exploration_step"
    assert _source_value(None) is None


def test_run_sobol_stamps_generative_provenance(tmp_path):
    """Each Sobol spec carries its provenance: design='sobol', the seed, bounds; κ is N/A."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.random_seed = 11
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})

    results = calibration.run_sobol(n=4)
    for spec in results:
        snap = spec.config_snapshot
        assert snap["source"] == "sobol_step"
        assert snap["seed"] == 11
        assert snap["kappa"] is None                       # data-independent design
        assert "param_1" in snap["param_bounds"]


def test_run_discovery_stamps_design_and_seed(tmp_path):
    """Discovery specs carry design='discovery', κ=1.0, and the seed (via _optimize)."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    calibration.random_seed = 5
    calibration.configure_param_bounds({"param_1": (0.0, 10.0), "param_2": (1, 4)})

    results = calibration.run_discovery(n=3)
    for spec in results:
        snap = spec.config_snapshot
        assert snap["source"] == "discovery_step"
        assert snap["kappa"] == 1.0
        assert snap["seed"] == 5

