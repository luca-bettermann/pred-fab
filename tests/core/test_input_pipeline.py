"""Shared input-pipeline decisions (column slicing + categorical encoding)."""

import pytest

from pred_fab.core.input_pipeline import categorical_to_index, column_indices


def test_column_indices_maps_in_order():
    cols = ["a", "b", "c", "d"]
    assert column_indices(cols, ["c", "a"]) == [2, 0]


def test_column_indices_raises_on_missing():
    with pytest.raises(ValueError, match="not found in input_columns"):
        column_indices(["a", "b"], ["a", "x"])


def test_column_indices_skip_missing():
    assert column_indices(["a", "b"], ["a", "x", "b"], skip_missing=True) == [0, 1]


def test_categorical_to_index_known():
    assert categorical_to_index("B", ["A", "B", "C"], code="mat") == 1


def test_categorical_to_index_unknown_raises():
    with pytest.raises(ValueError, match="Unknown categorical value"):
        categorical_to_index("Z", ["A", "B"], code="mat")


def test_datamodule_get_input_indices_delegates(tmp_path):
    """DataModule.get_input_indices uses the shared helper (same result)."""
    from pred_fab.core import DataModule, Dataset
    from tests.utils.builders import build_workflow_schema
    schema = build_workflow_schema(tmp_path)
    dm = DataModule(Dataset(schema=schema, debug_flag=True))
    dm.initialize(input_parameters=["param_1", "param_2"], input_features=[],
                  output_columns=["feature_1"])
    assert dm.get_input_indices(["param_2"]) == column_indices(dm.input_columns, ["param_2"])
