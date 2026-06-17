"""Lagged-parameter injection — per-experiment shift, no cross-experiment leakage.

``DataModule.add_lagged_params`` registers a ``prev_{col}`` input column;
``_inject_lagged_params`` fills it by shifting each column by ``lag`` rows
*within each experiment* (keyed by ``cell_meta[:, 0]``). The first ``lag``
rows of each experiment copy their own value (no change signal), and the
shift never crosses an experiment boundary.
"""

import pytest
import torch

from pred_fab.core import Dataset
from tests.utils.builders import (
    build_dataset_with_single_experiment,
    build_initialized_datamodule,
    build_workflow_schema,
    populate_single_experiment_features,
)


def _datamodule(tmp_path, input_parameters):
    dataset = Dataset(schema=build_workflow_schema(tmp_path))
    return build_initialized_datamodule(
        dataset,
        input_parameters=input_parameters,
        input_features=[],
        output_columns=["feature_3"],
    )


def test_inject_shifts_within_experiment_no_leakage(tmp_path):
    """Two experiments, lag=1: each row's prev value is the prior row of the
    SAME experiment; the first row of each copies itself — never the previous
    experiment's last row."""
    dm = _datamodule(tmp_path, ["param_1"])
    dm.add_lagged_params(lag=1)

    # exp 0 rows: 10, 20, 30 ; exp 1 rows: 100, 200
    X_dict = {"param_1": torch.tensor([10.0, 20.0, 30.0, 100.0, 200.0])}
    cell_meta = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]])

    out = dm._inject_lagged_params(X_dict, cell_meta)

    expected = torch.tensor([10.0, 10.0, 20.0, 100.0, 100.0])
    assert torch.equal(out["prev_param_1"], expected)
    # The boundary row (exp 1, row 0) carries its own value, not exp 0's last (30).
    assert out["prev_param_1"][3].item() == 100.0


def test_inject_lag_two(tmp_path):
    """lag=2: first two rows of each experiment copy themselves, then shift by 2."""
    dm = _datamodule(tmp_path, ["param_1"])
    dm.add_lagged_params(lag=2)

    X_dict = {"param_1": torch.tensor([1.0, 2.0, 3.0, 4.0])}  # single experiment
    cell_meta = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3]])

    out = dm._inject_lagged_params(X_dict, cell_meta)
    assert torch.equal(out["prev_param_1"], torch.tensor([1.0, 2.0, 1.0, 2.0]))


def test_inject_single_row_experiment_copies_self(tmp_path):
    """An experiment with n <= lag rows leaves every row carrying its own value."""
    dm = _datamodule(tmp_path, ["param_1"])
    dm.add_lagged_params(lag=1)

    X_dict = {"param_1": torch.tensor([5.0, 7.0])}  # two one-row experiments
    cell_meta = torch.tensor([[0, 0], [1, 0]])

    out = dm._inject_lagged_params(X_dict, cell_meta)
    assert torch.equal(out["prev_param_1"], torch.tensor([5.0, 7.0]))


def test_add_lagged_params_registers_column_and_inherits_norm(tmp_path):
    """A prev_ column is appended for each non-categorical input, inheriting its
    normalisation method; categoricals are skipped."""
    dm = _datamodule(tmp_path, ["param_1", "param_3"])  # param_3 is categorical
    dm.add_lagged_params(lag=1)

    assert dm._lagged_params == {"param_1": 1}
    assert "prev_param_1" in dm.input_columns
    assert "prev_param_3" not in dm.input_columns
    assert dm._col_norm_methods["prev_param_1"] == dm._col_norm_methods["param_1"]


def test_add_lagged_params_after_fit_raises(tmp_path):
    """Lagged columns reshape the input; adding them after fitting is rejected."""
    dm = _datamodule(tmp_path, ["param_1"])
    dm.fit_without_data()
    with pytest.raises(RuntimeError):
        dm.add_lagged_params(lag=1)


def test_lagged_column_in_batches_widens_input(tmp_path):
    """The lagged column is wired through the export → get_batches path."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    populate_single_experiment_features(dataset)
    dm = build_initialized_datamodule(dataset, ["param_1"], [], ["feature_scalar"])
    width_before = len(dm.input_columns)
    dm.add_lagged_params(lag=1)
    dm.set_split_codes(train_codes=["exp_001"])
    dm.fit_normalization()

    assert len(dm.input_columns) == width_before + 1
    batches = dm.get_batches()
    assert batches  # at least one non-empty batch produced
    X_t, _y, _meta = batches[0]
    assert X_t.shape[0] > 0
    assert X_t.shape[1] == len(dm.input_columns)
