import numpy as np
import pandas as pd

from tests.utils.builders import build_dataset_with_single_experiment


def test_feature_storage_shapes_follow_column_dimensionality(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")

    assert exp.features.get_value("feature_grid").shape == (2, 3)
    assert exp.features.get_value("feature_d1").shape == (2,)
    assert exp.features.get_value("feature_scalar").shape == ()


def test_set_values_from_df_is_transform_boundary_for_feature_tables(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")

    rows = []
    for i in range(2):
        for j in range(3):
            rows.append([i, j, i * 10 + j])

    df = pd.DataFrame(rows, columns=["d1", "d2", "feature_grid"], dtype=np.float64)
    exp.features.set_values_from_df(df, logger=dataset.logger, parameters=exp.parameters)

    stored = exp.features.get_value("feature_grid")
    assert stored.shape == (2, 3)
    assert float(stored[1, 2]) == 12.0
    assert exp.features.get("feature_grid").columns == ["d1", "d2", "feature_grid"]


def test_tensor_to_table_roundtrip_for_mixed_features(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")

    grid_table = np.array(
        [[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 0, 10], [1, 1, 11], [1, 2, 12]],
        dtype=np.float64,
    )
    d1_table = np.array([[0, 100], [1, 101]], dtype=np.float64)
    scalar_table = np.array([[7.0]], dtype=np.float64)

    grid_tensor = exp.features.table_to_tensor("feature_grid", grid_table, exp.parameters)
    d1_tensor = exp.features.table_to_tensor("feature_d1", d1_table, exp.parameters)
    scalar_tensor = exp.features.table_to_tensor("feature_scalar", scalar_table, exp.parameters)

    assert np.array_equal(exp.features.tensor_to_table("feature_grid", grid_tensor, exp.parameters), grid_table)
    assert np.array_equal(exp.features.tensor_to_table("feature_d1", d1_tensor, exp.parameters), d1_table)
    assert np.array_equal(exp.features.tensor_to_table("feature_scalar", scalar_tensor, exp.parameters), scalar_table)
