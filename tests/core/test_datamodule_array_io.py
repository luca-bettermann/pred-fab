"""
Edge-case tests for DataModule array-level IO:
params_to_array, array_to_params (including categorical roundtrip),
denormalize_values, normalization overrides, copy, get_onehot_column_map,
and get_split_codes.
"""
import pytest
import numpy as np

from pred_fab.core import DataModule
from pred_fab.utils.enum import NormMethod, SplitType
from tests.utils.builders import (
    build_dataset_with_single_experiment,
    build_workflow_stack,
    evaluate_loaded_workflow_experiments,
    build_prepared_workflow_datamodule,
)


# ===== params_to_array() / array_to_params() error handling =====

def test_params_to_array_raises_when_not_fitted(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)
    dm.initialize(input_parameters=["param_1", "dim_1", "dim_2"], input_features=[], output_columns=[])

    with pytest.raises(RuntimeError):
        dm.params_to_array({"param_1": 2.5, "dim_1": 2, "dim_2": 3})


def test_array_to_params_raises_when_not_fitted(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)
    dm.initialize(input_parameters=["param_1", "dim_1", "dim_2"], input_features=[], output_columns=[])

    with pytest.raises(RuntimeError):
        dm.array_to_params(np.array([2.5, 2.0, 3.0]))


def test_array_to_params_raises_for_shape_mismatch(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    dm = build_prepared_workflow_datamodule(agent, dataset)

    wrong_shape = np.zeros(len(dm.input_columns) + 5)
    with pytest.raises(ValueError, match="shape"):
        dm.array_to_params(wrong_shape)


# ===== params_to_array() / array_to_params() continuous roundtrip =====

def test_params_to_array_produces_correct_width(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    dm = build_prepared_workflow_datamodule(agent, dataset)

    arr = dm.params_to_array({"param_1": 5.0, "param_2": 3, "dim_1": 2, "dim_2": 3, "param_3": "B"})
    assert arr.shape == (len(dm.input_columns),)


def test_params_to_array_array_to_params_continuous_roundtrip(tmp_path):
    """Continuous params should survive a params->array->params roundtrip (up to coercion)."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    dm = build_prepared_workflow_datamodule(agent, dataset)

    params_in = {"param_1": 3.5, "param_2": 2, "dim_1": 2, "dim_2": 3, "param_3": "B"}
    arr = dm.params_to_array(params_in)
    params_out = dm.array_to_params(arr)

    assert params_out["param_1"] == pytest.approx(params_in["param_1"], abs=0.01)
    assert params_out["param_2"] == params_in["param_2"]


# ===== array_to_params() categorical roundtrip =====

def test_array_to_params_decodes_categorical_correctly(tmp_path):
    """Category 'B' encoded as one-hot should survive array_to_params decode."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    # Manually build a DataModule that includes param_3 (workflow agent excludes it)
    dm = DataModule(dataset)
    dm.initialize(
        input_parameters=["param_1", "param_2", "dim_1", "dim_2", "param_3"],
        input_features=[],
        output_columns=["feature_1"],
    )
    dm.prepare(val_size=0.0, test_size=0.0, recompute=True)

    for cat in ["A", "B", "C"]:
        params_in = {"param_1": 4.0, "param_2": 2, "dim_1": 2, "dim_2": 3, "param_3": cat}
        arr = dm.params_to_array(params_in)
        params_out = dm.array_to_params(arr)
        assert params_out["param_3"] == cat, f"Expected category '{cat}', got '{params_out['param_3']}'"


# ===== denormalize_values() =====

def test_denormalize_values_returns_unchanged_when_not_fitted(tmp_path):
    """When the datamodule is not fitted, denormalize_values should return a copy unchanged."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)
    dm.initialize(input_parameters=["param_1"], input_features=[], output_columns=["feature_grid"])

    values = np.array([[0.5, 0.5, 0.5]])
    result = dm.denormalize_values(values, ["feature_grid"])
    np.testing.assert_array_equal(result, values)


def test_denormalize_values_1d_input(tmp_path):
    """1D input (single vector) should be denormalized per feature."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    dm = build_prepared_workflow_datamodule(agent, dataset)

    # Normalize a known value then reverse it
    y_1d = np.array([0.0, 0.0])  # normalized zeros
    result = dm.denormalize_values(y_1d, ["feature_1", "feature_2"])
    # Result should be a valid array (not crash), same length
    assert result.shape == (2,)


def test_denormalize_output_delegates_to_denormalize_values(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    dm = build_prepared_workflow_datamodule(agent, dataset)

    arr = np.zeros((3, len(dm.output_columns)), dtype=np.float32)
    r1 = dm.denormalize_output(arr)
    r2 = dm.denormalize_values(arr, dm.output_columns)
    np.testing.assert_array_equal(r1, r2)


# ===== Normalization overrides =====

def test_set_feature_normalize_override_affects_stats(tmp_path):
    """Overriding a feature's normalization method before fitting should use the override."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    dm = agent.create_datamodule(dataset)
    dm.set_feature_normalize("feature_1", NormMethod.MIN_MAX)
    dm.prepare(recompute=True)

    # feature_1 should use min_max stats (has 'min' key), not 'mean'/'std'
    assert "feature_1" in dm._feature_stats
    assert "min" in dm._feature_stats["feature_1"]
    assert "mean" not in dm._feature_stats["feature_1"]


def test_set_parameter_normalize_override_affects_stats(tmp_path):
    """Overriding a parameter's normalization method before fitting should use the override."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    dm = agent.create_datamodule(dataset)
    dm.set_parameter_normalize("param_1", NormMethod.ROBUST)
    dm.prepare(recompute=True)

    assert "param_1" in dm._parameter_stats
    assert "median" in dm._parameter_stats["param_1"]


# ===== get_onehot_column_map() =====

def test_get_onehot_column_map_returns_correct_mapping(tmp_path):
    """One-hot column map should map 'param_3_B' -> ('param_3', 'B') etc."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    # Manually build a DataModule that includes param_3 (workflow agent excludes it)
    dm = DataModule(dataset)
    dm.initialize(
        input_parameters=["param_1", "param_2", "dim_1", "dim_2", "param_3"],
        input_features=[],
        output_columns=["feature_1"],
    )
    dm.prepare(val_size=0.0, test_size=0.0, recompute=True)

    col_map = dm.get_onehot_column_map()

    # param_3 has categories A, B, C
    assert "param_3_A" in col_map
    assert "param_3_B" in col_map
    assert "param_3_C" in col_map
    assert col_map["param_3_B"] == ("param_3", "B")


def test_get_onehot_column_map_is_empty_when_no_categoricals(tmp_path):
    """Datamodule with no categorical inputs should produce an empty map."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)
    dm.initialize(input_parameters=["param_1", "dim_1", "dim_2"], input_features=[], output_columns=[])

    col_map = dm.get_onehot_column_map()
    assert col_map == {}


# ===== copy() =====

def test_copy_produces_independent_datamodule(tmp_path):
    """Mutating the copy should not affect the original."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    dm = build_prepared_workflow_datamodule(agent, dataset)

    dm_copy = dm.copy()
    original_batch_size = dm.batch_size
    dm_copy.batch_size = 9999

    assert dm.batch_size == original_batch_size


def test_copy_preserves_fitted_state(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    dm = build_prepared_workflow_datamodule(agent, dataset)

    dm_copy = dm.copy()
    assert dm_copy._is_fitted is True
    assert dm_copy.input_columns == dm.input_columns
    assert dm_copy.output_columns == dm.output_columns


# ===== get_split_codes() =====

def test_get_split_codes_returns_train_codes(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    dm = build_prepared_workflow_datamodule(agent, dataset)

    train_codes = dm.get_split_codes(SplitType.TRAIN)
    assert set(train_codes) == set(codes)


def test_get_split_codes_returns_empty_val_when_no_val_split(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    dm = build_prepared_workflow_datamodule(agent, dataset, val_size=0.0)

    assert dm.get_split_codes(SplitType.VAL) == []


# ===== __repr__() =====

def test_datamodule_repr_returns_non_empty_string(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)
    dm.initialize(input_parameters=["param_1"], input_features=[], output_columns=[])
    r = repr(dm)
    assert isinstance(r, str) and len(r) > 0


# ===== Standard normalization degenerate std=0 =====

def test_apply_normalization_standard_degenerate_std_zero_does_not_produce_inf(tmp_path):
    """When all values are identical, std=0. The formula uses 1e-8 epsilon — result should be finite."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)

    data = np.array([3.0, 3.0, 3.0])
    stats = dm._compute_normalization_stats(data, NormMethod.STANDARD)
    normalized = dm._apply_normalization(data, stats)

    assert not np.any(np.isinf(normalized))
    assert not np.any(np.isnan(normalized))
    # All values equal → normalized to 0 / 1e-8 = 0
    assert np.all(normalized == pytest.approx(0.0))


def test_reverse_normalization_standard_degenerate_std_zero(tmp_path):
    """Reverse of degenerate standard normalization should return the mean."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)

    data = np.array([7.0, 7.0, 7.0])
    stats = dm._compute_normalization_stats(data, NormMethod.STANDARD)
    normalized = dm._apply_normalization(data, stats)
    recovered = dm._reverse_normalization(normalized, stats)

    assert np.allclose(recovered, 7.0, atol=1e-3)
