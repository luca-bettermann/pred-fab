"""
Tests for DataModule normalization modes, split creation, prepare/update,
one-hot encoding, and get_batches behavior.
"""
import pytest
import numpy as np

from pred_fab.core import DataModule, Dataset
from pred_fab.utils.enum import NormMethod, SplitType
from tests.utils.builders import (
    build_dataset_with_single_experiment,
    build_workflow_stack,
    build_workflow_schema,
    evaluate_loaded_workflow_experiments,
    build_prepared_workflow_datamodule,
)


# ===== prepare() error handling =====

def test_prepare_raises_when_splits_exist_and_recompute_false(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    datamodule = agent.create_datamodule(dataset)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)

    with pytest.raises(RuntimeError, match="already has defined splits"):
        datamodule.prepare(val_size=0.0, test_size=0.0, recompute=False)


def test_prepare_with_recompute_true_does_not_raise(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    datamodule = agent.create_datamodule(dataset)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)  # should not raise


def test_prepare_sets_is_fitted(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    datamodule = agent.create_datamodule(dataset)
    assert datamodule._is_fitted is False
    datamodule.prepare(recompute=True)
    assert datamodule._is_fitted is True


# ===== _create_splits() =====

def test_create_splits_empty_dataset_produces_empty_splits(tmp_path):
    schema = build_workflow_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    datamodule = DataModule(dataset)
    datamodule.initialize(input_parameters=["param_1"], input_features=[], output_columns=[])

    datamodule._create_splits(0.2, 0.2)
    sizes = datamodule.get_split_sizes()
    assert sizes[SplitType.TRAIN] == 0
    assert sizes[SplitType.VAL] == 0
    assert sizes[SplitType.TEST] == 0


def test_create_splits_no_val_no_test_puts_all_in_train(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    datamodule = agent.create_datamodule(dataset)
    datamodule._create_splits(0.0, 0.0)

    sizes = datamodule.get_split_sizes()
    assert sizes[SplitType.VAL] == 0
    assert sizes[SplitType.TEST] == 0
    assert sizes[SplitType.TRAIN] == 3


def test_create_splits_with_test_only(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    datamodule = agent.create_datamodule(dataset)
    datamodule._create_splits(0.0, 0.33)

    sizes = datamodule.get_split_sizes()
    assert sizes[SplitType.TEST] >= 1
    assert sizes[SplitType.VAL] == 0
    assert sizes[SplitType.TRAIN] + sizes[SplitType.TEST] == 3


def test_create_splits_with_val_and_test_accounts_for_all_experiments(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    datamodule = agent.create_datamodule(dataset)
    datamodule._create_splits(0.33, 0.33)

    sizes = datamodule.get_split_sizes()
    total = sizes[SplitType.TRAIN] + sizes[SplitType.VAL] + sizes[SplitType.TEST]
    assert total == 3


def test_create_splits_is_deterministic_with_same_seed(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    dm1 = agent.create_datamodule(dataset)
    dm1.random_seed = 42
    dm1._create_splits(0.33, 0.0)
    split1 = sorted(dm1._split_codes[SplitType.TRAIN])

    dm2 = agent.create_datamodule(dataset)
    dm2.random_seed = 42
    dm2._create_splits(0.33, 0.0)
    split2 = sorted(dm2._split_codes[SplitType.TRAIN])

    assert split1 == split2


# ===== _compute_normalization_stats() =====

def test_compute_normalization_stats_standard(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = dm._compute_normalization_stats(data, NormMethod.STANDARD)

    assert stats['method'] == NormMethod.STANDARD
    assert stats['mean'] == pytest.approx(3.0)
    assert stats['std'] == pytest.approx(np.std(data))


def test_compute_normalization_stats_minmax(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)

    data = np.array([1.0, 3.0, 5.0])
    stats = dm._compute_normalization_stats(data, NormMethod.MIN_MAX)

    assert stats['min'] == pytest.approx(1.0)
    assert stats['max'] == pytest.approx(5.0)


def test_compute_normalization_stats_robust(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = dm._compute_normalization_stats(data, NormMethod.ROBUST)

    assert 'median' in stats
    assert 'q1' in stats
    assert 'q3' in stats
    assert stats['median'] == pytest.approx(3.0)


def test_compute_normalization_stats_none_returns_method_only(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)

    data = np.array([1.0, 2.0])
    stats = dm._compute_normalization_stats(data, NormMethod.NONE)
    assert stats == {'method': NormMethod.NONE}


# ===== _apply_normalization() =====

def test_apply_normalization_standard_produces_near_zero_mean(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = dm._compute_normalization_stats(data, NormMethod.STANDARD)
    normalized = dm._apply_normalization(data, stats)

    assert np.abs(np.mean(normalized)) < 1e-5


def test_apply_normalization_minmax_maps_to_near_0_1(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)

    data = np.array([0.0, 5.0, 10.0])
    stats = dm._compute_normalization_stats(data, NormMethod.MIN_MAX)
    normalized = dm._apply_normalization(data, stats)

    assert normalized[0] == pytest.approx(0.0, abs=0.01)
    assert normalized[-1] == pytest.approx(1.0, abs=0.01)


def test_apply_normalization_robust_centers_median_near_zero(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = dm._compute_normalization_stats(data, NormMethod.ROBUST)
    normalized = dm._apply_normalization(data, stats)

    # Median value (3.0) should map to ~0
    assert np.abs(normalized[2]) < 0.01


def test_apply_normalization_none_returns_unchanged(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)

    data = np.array([1.0, 2.0, 3.0])
    stats = {'method': NormMethod.NONE}
    result = dm._apply_normalization(data, stats)

    np.testing.assert_array_equal(result, data)


def test_apply_normalization_degenerate_minmax_returns_zeros(tmp_path):
    """Degenerate min==max should not produce inf — must return zeros."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)

    data = np.array([5.0, 5.0, 5.0])
    stats = dm._compute_normalization_stats(data, NormMethod.MIN_MAX)
    normalized = dm._apply_normalization(data, stats)

    assert np.all(normalized == 0.0)
    assert not np.any(np.isinf(normalized))
    assert not np.any(np.isnan(normalized))


# ===== _reverse_normalization() roundtrips =====

def test_reverse_normalization_standard_roundtrip(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = dm._compute_normalization_stats(data, NormMethod.STANDARD)
    normalized = dm._apply_normalization(data, stats)
    recovered = dm._reverse_normalization(normalized, stats)

    np.testing.assert_allclose(recovered, data, atol=1e-5)


def test_reverse_normalization_minmax_roundtrip(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)

    data = np.array([0.0, 5.0, 10.0])
    stats = dm._compute_normalization_stats(data, NormMethod.MIN_MAX)
    normalized = dm._apply_normalization(data, stats)
    recovered = dm._reverse_normalization(normalized, stats)

    np.testing.assert_allclose(recovered, data, atol=1e-4)


def test_reverse_normalization_robust_roundtrip(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = dm._compute_normalization_stats(data, NormMethod.ROBUST)
    normalized = dm._apply_normalization(data, stats)
    recovered = dm._reverse_normalization(normalized, stats)

    np.testing.assert_allclose(recovered, data, atol=1e-5)


def test_reverse_normalization_degenerate_minmax_returns_min(tmp_path):
    """Degenerate range: reverse should return min value."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)

    data = np.array([5.0, 5.0])
    stats = dm._compute_normalization_stats(data, NormMethod.MIN_MAX)
    normalized = dm._apply_normalization(data, stats)
    recovered = dm._reverse_normalization(normalized, stats)

    # Should return the constant value, not NaN/inf
    assert not np.any(np.isnan(recovered))
    assert not np.any(np.isinf(recovered))


# ===== normalize_parameter_bounds() =====

def test_normalize_parameter_bounds_returns_raw_when_no_stats(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)

    low, high = dm.normalize_parameter_bounds("param_1", 2.0, 8.0)
    assert low == 2.0
    assert high == 8.0


def test_normalize_parameter_bounds_transforms_when_stats_present(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    datamodule = build_prepared_workflow_datamodule(agent, dataset)
    # param_1 should have normalization stats
    low_norm, high_norm = datamodule.normalize_parameter_bounds("param_1", 0.0, 10.0)
    # Values should differ from raw since normalization is applied
    assert (low_norm, high_norm) != (0.0, 10.0)


# ===== _one_hot_encode() =====

def test_one_hot_encode_includes_categorical_columns(tmp_path):
    """Manually initializing a datamodule with param_3 as input should produce one-hot columns."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    from pred_fab.core import DataModule
    # Explicitly include param_3 (categorical) as an input parameter
    datamodule = DataModule(dataset)
    datamodule.initialize(
        input_parameters=["param_1", "param_2", "n_layers", "n_segments", "param_3"],
        input_features=[],
        output_columns=["feature_1", "feature_2"],
    )
    datamodule._create_splits(0.0, 0.0)
    datamodule._fit_normalize()

    X_df, _ = dataset.export_to_dataframe(codes)
    X_arr = datamodule._one_hot_encode(X_df)

    # One-hot columns for param_3 categories (A, B, C) should be present
    assert any("param_3" in col for col in datamodule.input_columns)
    assert X_arr.ndim == 2
    assert X_arr.shape[1] == len(datamodule.input_columns)


def test_one_hot_encode_handles_missing_categorical_column(tmp_path):
    """When categorical column is absent from DataFrame, should fill with zeros."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    from pred_fab.core import DataModule
    datamodule = DataModule(dataset)
    datamodule.initialize(
        input_parameters=["param_1", "param_2", "n_layers", "n_segments", "param_3"],
        input_features=[],
        output_columns=["feature_1", "feature_2"],
    )
    datamodule._create_splits(0.0, 0.0)
    datamodule._fit_normalize()

    X_df, _ = dataset.export_to_dataframe(codes[:1])
    X_df_missing = X_df.drop(columns=["param_3"], errors="ignore")

    # Should not raise; fills missing with 0
    X_arr = datamodule._one_hot_encode(X_df_missing)
    assert X_arr is not None
    assert X_arr.shape[1] == len(datamodule.input_columns)


def test_one_hot_encode_output_is_float32(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    from pred_fab.core import DataModule
    datamodule = DataModule(dataset)
    datamodule.initialize(
        input_parameters=["param_1", "param_2", "n_layers", "n_segments", "param_3"],
        input_features=[],
        output_columns=["feature_1", "feature_2"],
    )
    datamodule._create_splits(0.0, 0.0)
    datamodule._fit_normalize()

    X_df, _ = dataset.export_to_dataframe(codes)
    X_arr = datamodule._one_hot_encode(X_df)
    assert X_arr.dtype == np.float32


# ===== update() =====

def test_update_returns_zero_when_no_new_experiments(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    datamodule = agent.create_datamodule(dataset)
    datamodule.prepare(recompute=True)

    n_new = datamodule.update()
    assert n_new == 0


def test_update_is_fitted_after_adding_experiments(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    datamodule = agent.create_datamodule(dataset)
    # Manually set split to first experiment only
    datamodule._split_codes = {SplitType.TRAIN: [codes[0]], SplitType.VAL: [], SplitType.TEST: []}
    datamodule._fit_normalize()

    # Now add all remaining experiments
    n_new = datamodule.update()
    assert datamodule._is_fitted is True


# ===== get_batches() =====

def test_get_batches_with_batch_size_1_returns_one_batch_per_row(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    datamodule = agent.create_datamodule(dataset)
    datamodule.batch_size = 1
    datamodule.prepare(recompute=True)

    batches = datamodule.get_batches(SplitType.TRAIN)
    assert len(batches) >= 1
    for X_b, y_b in batches:
        assert X_b.shape[0] == 1


def test_get_batches_without_batch_size_returns_single_batch(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    datamodule = agent.create_datamodule(dataset)
    datamodule.batch_size = None
    datamodule.prepare(recompute=True)

    batches = datamodule.get_batches(SplitType.TRAIN)
    assert len(batches) == 1


def test_get_batches_returns_empty_for_empty_split(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    datamodule = agent.create_datamodule(dataset)
    datamodule.prepare(recompute=True)

    # VAL split is empty (no val_size specified)
    batches = datamodule.get_batches(SplitType.VAL)
    assert batches == []


def test_get_batches_x_has_correct_number_of_columns(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    datamodule = agent.create_datamodule(dataset)
    datamodule.prepare(recompute=True)

    batches = datamodule.get_batches(SplitType.TRAIN)
    assert len(batches) > 0
    X_b, _ = batches[0]
    assert X_b.shape[1] == len(datamodule.input_columns)


# ===== normalization_state export/import =====

def test_get_normalization_state_raises_when_not_fitted(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    dm = DataModule(dataset)
    with pytest.raises(RuntimeError):
        dm.get_normalization_state()


def test_get_and_set_normalization_state_roundtrip(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    datamodule = build_prepared_workflow_datamodule(agent, dataset)
    state = datamodule.get_normalization_state()

    # Apply to a fresh datamodule
    from pred_fab.core import DataModule
    dm2 = DataModule(dataset)
    dm2.initialize(
        input_parameters=["param_1", "param_2", "n_layers", "n_segments", "param_3"],
        input_features=[],
        output_columns=["feature_1", "feature_2"],
    )
    dm2.set_normalization_state(state)

    assert dm2._is_fitted is True
    assert dm2.input_columns == datamodule.input_columns
    assert dm2.output_columns == datamodule.output_columns
