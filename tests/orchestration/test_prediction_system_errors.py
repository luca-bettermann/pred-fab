"""
Edge-case tests for PredictionSystem error paths:
train/validate/tune boundary conditions and predict_experiment basic behavior.
"""
import pytest
import numpy as np

from pred_fab.utils import SplitType
from tests.utils.builders import (
    build_real_agent_stack,
    build_dataset_with_single_experiment,
    build_initialized_datamodule,
    build_shape_checking_prediction_system,
    populate_single_experiment_features,
)


# ===== PredictionSystem.train() errors =====

def test_train_raises_when_train_split_is_empty(tmp_path):
    """Training on an empty train split should raise ValueError."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)

    # Override split so train set is empty
    datamodule._split_codes = {SplitType.TRAIN: [], SplitType.VAL: [], SplitType.TEST: []}
    datamodule._is_fitted = True

    with pytest.raises(ValueError, match="empty training set"):
        agent.train(datamodule=datamodule, validate=False, test=False)


# ===== PredictionSystem.validate() errors =====

def test_validate_raises_when_not_trained(tmp_path):
    """Calling validate before training should raise RuntimeError."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    populate_single_experiment_features(dataset)

    datamodule = build_initialized_datamodule(
        dataset=dataset,
        input_parameters=["param_1", "dim_1", "dim_2"],
        input_features=[],
        output_columns=["feature_grid"],
        fitted=True,
        split_codes={SplitType.TRAIN: ["exp_001"], SplitType.VAL: ["exp_001"], SplitType.TEST: []},
    )
    pred_system, _ = build_shape_checking_prediction_system(
        tmp_path=tmp_path,
        dataset=dataset,
        datamodule=datamodule,
        model_specs=[(["param_1"], ["feature_grid"])],
    )
    # Builder sets datamodule; clear it to simulate untrained state
    pred_system.datamodule = None
    with pytest.raises(RuntimeError):
        pred_system.validate(use_test=False)


def test_validate_raises_when_val_split_is_empty(tmp_path):
    """validate(use_test=False) should raise ValueError when VAL split is empty."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    populate_single_experiment_features(dataset)

    datamodule = build_initialized_datamodule(
        dataset=dataset,
        input_parameters=["param_1", "dim_1", "dim_2"],
        input_features=[],
        output_columns=["feature_grid", "feature_d1", "feature_scalar"],
        fitted=True,
        split_codes={SplitType.TRAIN: ["exp_001"], SplitType.VAL: [], SplitType.TEST: []},
    )
    pred_system, _ = build_shape_checking_prediction_system(
        tmp_path=tmp_path,
        dataset=dataset,
        datamodule=datamodule,
        model_specs=[(["param_1", "dim_1", "dim_2"], ["feature_grid", "feature_d1", "feature_scalar"])],
    )
    pred_system.datamodule = datamodule  # set datamodule as if trained

    with pytest.raises(ValueError, match="empty"):
        pred_system.validate(use_test=False)


def test_validate_raises_when_test_split_is_empty(tmp_path):
    """validate(use_test=True) should raise ValueError when TEST split is empty."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    populate_single_experiment_features(dataset)

    datamodule = build_initialized_datamodule(
        dataset=dataset,
        input_parameters=["param_1", "dim_1", "dim_2"],
        input_features=[],
        output_columns=["feature_grid", "feature_d1", "feature_scalar"],
        fitted=True,
        split_codes={SplitType.TRAIN: ["exp_001"], SplitType.VAL: ["exp_001"], SplitType.TEST: []},
    )
    pred_system, _ = build_shape_checking_prediction_system(
        tmp_path=tmp_path,
        dataset=dataset,
        datamodule=datamodule,
        model_specs=[(["param_1", "dim_1", "dim_2"], ["feature_grid", "feature_d1", "feature_scalar"])],
    )
    pred_system.datamodule = datamodule

    with pytest.raises(ValueError, match="empty"):
        pred_system.validate(use_test=True)


# ===== PredictionSystem.tune() errors =====

def test_tune_raises_when_not_trained(tmp_path):
    """Calling tune before training should raise RuntimeError."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    populate_single_experiment_features(dataset)
    exp = dataset.get_experiment("exp_001")

    datamodule = build_initialized_datamodule(
        dataset=dataset,
        input_parameters=["param_1", "dim_1", "dim_2"],
        input_features=[],
        output_columns=["feature_grid", "feature_d1", "feature_scalar"],
    )
    pred_system, _ = build_shape_checking_prediction_system(
        tmp_path=tmp_path,
        dataset=dataset,
        datamodule=datamodule,
        model_specs=[(["param_1", "dim_1", "dim_2"], ["feature_grid", "feature_d1", "feature_scalar"])],
    )
    # pred_system.datamodule is None — not trained

    with pytest.raises(RuntimeError, match="not trained"):
        pred_system.tune(exp_data=exp, start=0, end=3)


def test_tune_raises_for_start_out_of_bounds(tmp_path):
    """Tune start index beyond the available rows should raise ValueError."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    total_rows = exp.get_num_rows()
    with pytest.raises(ValueError, match="out of bounds"):
        agent.pred_system.tune(exp_data=exp, start=total_rows + 5, end=total_rows + 6)


def test_tune_raises_for_end_equal_to_start(tmp_path):
    """end <= start is an invalid slice and should raise ValueError."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    with pytest.raises(ValueError):
        agent.pred_system.tune(exp_data=exp, start=2, end=2)  # end == start


def test_tune_raises_for_end_before_start(tmp_path):
    """end < start should raise ValueError."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    with pytest.raises(ValueError):
        agent.pred_system.tune(exp_data=exp, start=4, end=2)


def test_tune_raises_for_negative_start(tmp_path):
    """Negative start index should raise ValueError."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    with pytest.raises(ValueError, match="out of bounds"):
        agent.pred_system.tune(exp_data=exp, start=-1, end=3)


# ===== PredictionSystem.predict_experiment() =====

def test_predict_experiment_returns_dict_of_feature_arrays(tmp_path):
    """predict_experiment should return a dict with one array per model output."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    predictions = agent.pred_system.predict_experiment(exp_data=exp)
    assert isinstance(predictions, dict)
    # Should have arrays for each model output
    for key, arr in predictions.items():
        assert isinstance(arr, np.ndarray)


def test_predict_experiment_arrays_have_correct_shape(tmp_path):
    """Prediction arrays should have the same shape as the dimensional structure."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    predictions = agent.pred_system.predict_experiment(exp_data=exp)
    # The mixed schema has dim_1=2, dim_2=3, so shape (2, 3)
    for arr in predictions.values():
        assert arr.shape == (2, 3)


def test_predict_experiment_raises_before_training(tmp_path):
    """predict_experiment should raise RuntimeError if called before train()."""
    agent, dataset, exp, _ = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)

    with pytest.raises(RuntimeError, match="not trained"):
        agent.pred_system.predict_experiment(exp_data=exp)


def test_predict_experiment_range_subset_does_not_fill_all(tmp_path):
    """Predicting only a partial range leaves the rest as NaN."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    total = exp.get_num_rows()
    predictions = agent.pred_system.predict_experiment(exp_data=exp, predict_from=0, predict_to=2)
    # Only first 2 positions filled; rest should be NaN
    for arr in predictions.values():
        flat = arr.flatten()
        assert not np.isnan(flat[0])
        assert np.isnan(flat[total - 1])
