import pytest
import numpy as np

from pred_fab.core import DataModule, ParameterProposal
from pred_fab.utils import SplitType
from tests.utils.builders import build_dataset_with_single_experiment


def test_build_calibration_training_arrays_returns_ordered_xy(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    exp.performance.set_values_from_dict({"performance_1": 0.75}, logger=dataset.logger)

    datamodule = DataModule(dataset)
    datamodule.initialize(input_parameters=["param_1", "dim_1", "dim_2"], input_features=[], output_columns=[])
    datamodule._is_fitted = True
    datamodule._split_codes = {SplitType.TRAIN: ["exp_001"], SplitType.VAL: [], SplitType.TEST: []}

    X, y = datamodule.build_calibration_training_arrays(["performance_1"])

    assert X.shape == (1, 3)
    assert y.shape == (1, 1)
    assert np.isclose(y[0, 0], 0.75)


def test_build_calibration_training_arrays_raises_on_missing_performance_when_strict(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)

    datamodule = DataModule(dataset)
    datamodule.initialize(input_parameters=["param_1", "dim_1", "dim_2"], input_features=[], output_columns=[])
    datamodule._is_fitted = True
    datamodule._split_codes = {SplitType.TRAIN: ["exp_001"], SplitType.VAL: [], SplitType.TEST: []}

    with pytest.raises(ValueError):
        datamodule.build_calibration_training_arrays(["performance_1"], strict=True)


def test_build_calibration_training_arrays_returns_empty_shapes_for_empty_split(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)

    datamodule = DataModule(dataset)
    datamodule.initialize(input_parameters=["param_1", "dim_1", "dim_2"], input_features=[], output_columns=[])
    datamodule._is_fitted = True
    datamodule._split_codes = {SplitType.TRAIN: [], SplitType.VAL: [], SplitType.TEST: []}

    X, y = datamodule.build_calibration_training_arrays(["performance_1"])

    assert X.shape == (0, 3)
    assert y.shape == (0, 1)


def test_build_calibration_training_arrays_uses_latest_effective_parameters(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    exp.performance.set_values_from_dict({"performance_1": 0.5}, logger=dataset.logger)
    exp.record_parameter_update(
        ParameterProposal.from_dict({"param_1": 7.25}, source_step="adaptation_step"),
        dimension="dim_1",
        step_index=1,
    )

    datamodule = DataModule(dataset)
    datamodule.initialize(input_parameters=["param_1", "dim_1", "dim_2"], input_features=[], output_columns=[])
    datamodule._is_fitted = True
    datamodule._split_codes = {SplitType.TRAIN: ["exp_001"], SplitType.VAL: [], SplitType.TEST: []}

    X, _ = datamodule.build_calibration_training_arrays(["performance_1"])
    assert np.isclose(X[0, 0], 7.25)
