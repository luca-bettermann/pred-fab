"""
Tests for Dataset creation, experiment management, ExperimentData helpers,
and ParameterUpdateEvent behavior.
"""
import pytest
import numpy as np

from pred_fab.core import Dataset
from pred_fab.core.dataset import (
    ExperimentData,
    ParameterProposal,
    ParameterTrajectory,
    ParameterUpdateEvent,
    events_to_trajectory,
    trajectory_to_events,
)
from tests.utils.builders import (
    build_mixed_feature_schema,
    build_dataset_with_single_experiment,
    populate_single_experiment_features,
)


# ===== Dataset.create_experiment() =====

def test_create_experiment_raises_if_already_exists_without_recompute(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        dataset.create_experiment(
            "exp_001", parameters={"param_1": 1.0, "dim_1": 2, "dim_2": 3}
        )


def test_create_experiment_succeeds_with_recompute_flag(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.create_experiment(
        "exp_001",
        parameters={"param_1": 9.0, "dim_1": 2, "dim_2": 3},
        recompute=True,
    )
    assert exp.parameters.get_value("param_1") == pytest.approx(9.0)


def test_create_experiment_initializes_feature_arrays(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    grid_val = exp.features.get_value("feature_grid")
    assert grid_val.shape == (2, 3)  # dim_1=2, dim_2=3


def test_create_experiment_feature_arrays_are_nan(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    grid_val = exp.features.get_value("feature_grid")
    assert np.all(np.isnan(grid_val))


def test_create_experiment_with_optional_performance(tmp_path):
    schema = build_mixed_feature_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    exp = dataset.create_experiment(
        "exp_perf",
        parameters={"param_1": 5.0, "dim_1": 2, "dim_2": 3},
        performance={"performance_1": 0.75},
    )
    assert exp.performance.get_value("performance_1") == pytest.approx(0.75)


def test_create_experiment_returns_experiment_data_object(tmp_path):
    schema = build_mixed_feature_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    exp = dataset.create_experiment(
        "new_exp", parameters={"param_1": 1.0, "dim_1": 2, "dim_2": 3}
    )
    assert isinstance(exp, ExperimentData)
    assert exp.code == "new_exp"


# ===== Dataset has/get experiment =====

def test_has_experiment_returns_true_after_create(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    assert dataset.has_experiment("exp_001") is True


def test_has_experiment_returns_false_for_missing(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    assert dataset.has_experiment("nonexistent") is False


def test_get_experiment_raises_for_missing(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    with pytest.raises(KeyError):
        dataset.get_experiment("nonexistent_exp")


def test_get_experiment_codes_lists_all(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    codes = dataset.get_experiment_codes()
    assert "exp_001" in codes


# ===== get_populated_experiment_codes() =====

def test_get_populated_experiment_codes_excludes_unpopulated(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    populated = dataset.get_populated_experiment_codes()
    assert "exp_001" not in populated


def test_get_populated_experiment_codes_includes_populated(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    populate_single_experiment_features(dataset)
    populated = dataset.get_populated_experiment_codes()
    assert "exp_001" in populated


# ===== Dataset.add_experiment() =====

def test_add_experiment_stores_in_dataset(tmp_path):
    schema = build_mixed_feature_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)

    other = Dataset(schema=schema, debug_flag=True)
    other.create_experiment("exp_manual", parameters={"param_1": 3.0, "dim_1": 2, "dim_2": 3})
    exp = other.get_experiment("exp_manual")

    dataset.add_experiment(exp)
    assert dataset.has_experiment("exp_manual")


def test_add_experiment_raises_if_already_exists(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    with pytest.raises(ValueError, match="already exists"):
        dataset.add_experiment(exp, recompute=False)


def test_add_experiment_succeeds_with_recompute(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    dataset.add_experiment(exp, recompute=True)  # Should not raise


# ===== ExperimentData.is_complete() =====

def test_is_complete_returns_false_for_all_nan_array(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    assert exp.is_complete("feature_grid", 0, None) is False


def test_is_complete_returns_true_after_full_population(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    populate_single_experiment_features(dataset)
    exp = dataset.get_experiment("exp_001")
    assert exp.is_complete("feature_grid", 0, None) is True


def test_is_complete_with_partial_range_checks_only_slice(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")

    # Populate only first row (dim_2=3 cells)
    tensor = exp.features.get_value("feature_grid")
    tensor[0, :] = 5.0
    exp.features.set_value("feature_grid", tensor)

    # First 3 flat cells should be complete (row 0, all segments)
    assert exp.is_complete("feature_grid", 0, 3) is True
    # But not the full array
    assert exp.is_complete("feature_grid", 0, None) is False


def test_is_complete_raises_for_unknown_feature(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    with pytest.raises(KeyError):
        exp.is_complete("nonexistent_feature", 0, None)


def test_is_complete_with_evaluate_from_nonzero(tmp_path):
    """is_complete starting from a non-zero offset."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")

    tensor = exp.features.get_value("feature_grid")
    tensor[1, :] = 5.0  # populate second row only
    exp.features.set_value("feature_grid", tensor)

    # Rows 3-5 (second dim_1 slice) should be populated
    assert exp.is_complete("feature_grid", 3, 6) is True
    # Full array not complete
    assert exp.is_complete("feature_grid", 0, None) is False


# ===== ExperimentData.get_effective_parameters_for_row() =====

def test_get_effective_params_returns_base_params_without_updates(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    params = exp.get_effective_parameters_for_row(0)
    assert params["param_1"] == pytest.approx(2.5)


def test_get_effective_params_applies_update_at_correct_row(tmp_path):
    """dim_1=2, dim_2=3 → stride(dim_1)=3. Step 1 of dim_1 starts at row 3."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")

    proposal = ParameterProposal.from_dict({"param_1": 9.9})
    exp.record_parameter_update(proposal, iterator_code="dim_1", step_index=1)

    # Rows 0-2 should have base param_1=2.5
    assert exp.get_effective_parameters_for_row(0)["param_1"] == pytest.approx(2.5)
    assert exp.get_effective_parameters_for_row(2)["param_1"] == pytest.approx(2.5)

    # Rows 3+ should have updated param_1=9.9
    assert exp.get_effective_parameters_for_row(3)["param_1"] == pytest.approx(9.9)
    assert exp.get_effective_parameters_for_row(5)["param_1"] == pytest.approx(9.9)


def test_get_effective_params_multiple_sequential_updates(tmp_path):
    """Multiple updates should layer in order."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")

    prop1 = ParameterProposal.from_dict({"param_1": 5.0})
    exp.record_parameter_update(prop1, iterator_code="dim_1", step_index=0)

    prop2 = ParameterProposal.from_dict({"param_1": 8.0})
    exp.record_parameter_update(prop2, iterator_code="dim_1", step_index=1)

    assert exp.get_effective_parameters_for_row(0)["param_1"] == pytest.approx(5.0)
    assert exp.get_effective_parameters_for_row(3)["param_1"] == pytest.approx(8.0)


# ===== ExperimentData.get_num_rows() =====

def test_get_num_rows_with_two_dimensions(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    # dim_1=2, dim_2=3 → 6 rows
    assert exp.get_num_rows() == 6


def test_get_num_rows_with_unit_dimensions(tmp_path):
    schema = build_mixed_feature_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    dataset.create_experiment("exp_unit", parameters={"param_1": 1.0, "dim_1": 1, "dim_2": 1})
    exp = dataset.get_experiment("exp_unit")
    assert exp.get_num_rows() == 1


def test_get_num_rows_scales_with_dimensions(tmp_path):
    """Use max allowed values: dim_1=2, dim_2=3 → 6 total rows confirms multiplication."""
    schema = build_mixed_feature_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    dataset.create_experiment("exp_big", parameters={"param_1": 1.0, "dim_1": 2, "dim_2": 3})
    exp = dataset.get_experiment("exp_big")
    assert exp.get_num_rows() == 6  # 2 * 3


# ===== ParameterUpdateEvent serialization =====

def test_parameter_update_event_to_dict_from_dict_roundtrip():
    event = ParameterUpdateEvent(
        updates={"param_1": 7.5},
        iterator_code="dim_1",
        step_index=2,
        source_step="adaptation",
    )
    d = event.to_dict()
    restored = ParameterUpdateEvent.from_dict(d)

    assert restored.updates == {"param_1": 7.5}
    assert restored.iterator_code == "dim_1"
    assert restored.step_index == 2
    assert restored.source_step == "adaptation"


def test_parameter_update_event_from_dict_handles_missing_optional_fields():
    d = {"updates": {"param_1": 5.0}}
    event = ParameterUpdateEvent.from_dict(d)
    assert event.iterator_code is None
    assert event.step_index is None
    assert event.source_step is None


def test_parameter_update_event_to_dict_produces_serializable_output():
    event = ParameterUpdateEvent(
        updates={"param_1": 3.0},
        iterator_code="dim_1",
        step_index=0,
    )
    d = event.to_dict()
    import json
    json.dumps(d)  # Should not raise


# ===== record_parameter_update() edge cases =====

def test_record_parameter_update_step_index_without_dimension_raises(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    proposal = ParameterProposal.from_dict({"param_1": 5.0})
    with pytest.raises(ValueError):
        exp.record_parameter_update(proposal, iterator_code=None, step_index=1)


def test_record_parameter_update_dimension_without_step_index_raises(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    proposal = ParameterProposal.from_dict({"param_1": 5.0})
    with pytest.raises(ValueError):
        exp.record_parameter_update(proposal, iterator_code="dim_1", step_index=None)


def test_record_parameter_update_empty_proposal_returns_none(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    proposal = ParameterProposal.from_dict({})
    result = exp.record_parameter_update(proposal)
    assert result is None


def test_record_parameter_update_noop_delta_returns_none(tmp_path):
    """If proposed value equals current value, no event should be recorded."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    # param_1 is already 2.5 — same value proposal is a no-op
    proposal = ParameterProposal.from_dict({"param_1": 2.5})
    result = exp.record_parameter_update(proposal)
    assert result is None


def test_record_parameter_update_rejects_dimension_parameter(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    proposal = ParameterProposal.from_dict({"dim_1": 3})
    with pytest.raises(ValueError):
        exp.record_parameter_update(proposal, iterator_code="dim_1", step_index=0)


def test_record_parameter_update_appends_to_list(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    assert len(exp.parameter_updates) == 0

    proposal = ParameterProposal.from_dict({"param_1": 7.0})
    exp.record_parameter_update(proposal, iterator_code="dim_1", step_index=0)

    assert len(exp.parameter_updates) == 1


# ===== ParameterProposal =====

def test_parameter_proposal_mapping_interface():
    proposal = ParameterProposal.from_dict({"a": 1, "b": 2})
    assert proposal["a"] == 1
    assert "a" in proposal
    assert proposal.get("c", 99) == 99
    assert len(proposal) == 2


def test_parameter_proposal_to_dict_returns_copy():
    proposal = ParameterProposal.from_dict({"a": 1})
    d = proposal.to_dict()
    d["a"] = 999
    assert proposal["a"] == 1  # Original unchanged


# ===== Trajectory ↔ events round-trip helpers =====

def test_trajectory_to_events_preserves_ordering_and_payload():
    """trajectory_to_events lifts each (step_index, proposal) into a tagged event."""
    traj = ParameterTrajectory(
        dimension="layer",
        entries=[
            (0, ParameterProposal({"speed": 30.0}, source_step="explore")),
            (3, ParameterProposal({"speed": 45.0}, source_step="explore")),
        ],
    )
    events = trajectory_to_events(traj)
    assert len(events) == 2
    assert events[0].iterator_code == "layer"
    assert events[0].step_index == 0
    assert events[0].updates == {"speed": 30.0}
    assert events[0].source_step == "explore"
    assert events[1].step_index == 3
    assert events[1].updates == {"speed": 45.0}


def test_events_to_trajectory_filters_by_dimension_and_sorts():
    """events_to_trajectory keeps only matching-dimension events; sorts by step_index."""
    events = [
        ParameterUpdateEvent({"speed": 50.0}, iterator_code="layer", step_index=2),
        ParameterUpdateEvent({"feed": 1.0}, iterator_code="segment", step_index=1),  # filtered out
        ParameterUpdateEvent({"speed": 30.0}, iterator_code="layer", step_index=0),  # earlier
    ]
    traj = events_to_trajectory(events, dimension="layer")
    assert traj.dimension == "layer"
    assert len(traj.entries) == 2
    assert [step for step, _ in traj.entries] == [0, 2]   # sorted
    assert traj.entries[0][1].values == {"speed": 30.0}
    assert traj.entries[1][1].values == {"speed": 50.0}


def test_events_to_trajectory_skips_events_without_step_index():
    """Events without step_index (e.g. initial-state events) are skipped."""
    events = [
        ParameterUpdateEvent({"speed": 30.0}, iterator_code="layer", step_index=None),  # skipped
        ParameterUpdateEvent({"speed": 50.0}, iterator_code="layer", step_index=1),
    ]
    traj = events_to_trajectory(events, dimension="layer")
    assert len(traj.entries) == 1
    assert traj.entries[0][0] == 1


def test_trajectory_events_round_trip():
    """traj → events → traj produces an equivalent trajectory."""
    original = ParameterTrajectory(
        dimension="layer",
        entries=[
            (0, ParameterProposal({"speed": 30.0})),
            (5, ParameterProposal({"speed": 45.0, "water": 0.4})),
        ],
    )
    events = trajectory_to_events(original)
    restored = events_to_trajectory(events, dimension="layer")
    assert restored.dimension == original.dimension
    assert len(restored.entries) == len(original.entries)
    for (s_orig, p_orig), (s_rest, p_rest) in zip(original.entries, restored.entries):
        assert s_orig == s_rest
        assert p_orig.values == p_rest.values


def test_apply_uses_trajectory_to_events(tmp_path):
    """ParameterTrajectory.apply records events on the experiment via the helper path."""
    schema = build_mixed_feature_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    dataset.create_experiment(
        "exp_traj",
        parameters={"param_1": 1.0, "dim_1": 2, "dim_2": 3},
    )
    exp = dataset.get_experiment("exp_traj")

    traj = ParameterTrajectory(
        dimension="dim_1",
        entries=[(1, ParameterProposal({"param_1": 9.0}))],
    )
    traj.apply(exp)
    assert len(exp.parameter_updates) == 1
    assert exp.parameter_updates[0].iterator_code == "dim_1"
    assert exp.parameter_updates[0].step_index == 1
    assert exp.parameter_updates[0].updates == {"param_1": 9.0}
