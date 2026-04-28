import numpy as np
import pytest

from pred_fab.core import Dataset, DatasetSchema
from pred_fab.core.data_blocks import Features, Parameters, PerformanceAttributes
from pred_fab.core.data_objects import Feature, Parameter, PerformanceAttribute
from tests.utils.builders import build_dataset_with_single_experiment, populate_single_experiment_features, sample_feature_tables


def test_export_to_dataframe_handles_mixed_feature_dimensionality(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")

    grid, d1_only, scalar = sample_feature_tables()
    exp.features.set_value("feature_grid", exp.features.table_to_tensor("feature_grid", grid, exp.parameters))
    exp.features.set_value("feature_d1", exp.features.table_to_tensor("feature_d1", d1_only, exp.parameters))
    exp.features.set_value("feature_scalar", exp.features.table_to_tensor("feature_scalar", scalar, exp.parameters))

    X_df, y_df = dataset.export_to_dataframe(["exp_001"])

    assert len(X_df) == 6
    assert len(y_df) == 6
    assert set(["param_1", "dim_1", "dim_2"]).issubset(set(X_df.columns))
    assert set(["feature_grid", "feature_d1", "feature_scalar"]).issubset(set(y_df.columns))

    # dim_* columns now hold the experiment's actual sizes (constant per row),
    # not the iteration index. Position info is reconstructed from row order.
    for i in range(len(X_df)):
        d1_val = i // 3
        d2_val = i % 3

        assert float(X_df.iloc[i]["param_1"]) == 2.5
        assert int(X_df.iloc[i]["dim_1"]) == 2
        assert int(X_df.iloc[i]["dim_2"]) == 3
        assert float(y_df.iloc[i]["feature_grid"]) == d1_val * 10 + d2_val
        assert float(y_df.iloc[i]["feature_d1"]) == 100 + d1_val
        assert float(y_df.iloc[i]["feature_scalar"]) == 7.0


def test_export_empty_experiment_codes_returns_empty_dataframes(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    X_df, y_df = dataset.export_to_dataframe([])
    assert X_df.empty
    assert y_df.empty


def test_export_unpopulated_experiment_excludes_nan_from_y_columns(tmp_path):
    """NaN feature values must not appear as y columns; X must still have all dim-combination rows."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    # Features are all NaN (not populated)
    X_df, y_df = dataset.export_to_dataframe(["exp_001"])
    assert len(X_df) == 6        # one row per dim_1 × dim_2 combination
    assert y_df.shape == (6, 0)  # NaN values filtered out → no columns


def test_export_multiple_experiments_stacks_all_rows(tmp_path):
    """Exporting two experiments concatenates their rows in order."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    populate_single_experiment_features(dataset)

    dataset.create_experiment("exp_002", parameters={"param_1": 5.0, "dim_1": 2, "dim_2": 3})
    exp2 = dataset.get_experiment("exp_002")
    grid, d1_only, scalar = sample_feature_tables()
    exp2.features.set_value("feature_grid", exp2.features.table_to_tensor("feature_grid", grid, exp2.parameters))
    exp2.features.set_value("feature_d1", exp2.features.table_to_tensor("feature_d1", d1_only, exp2.parameters))
    exp2.features.set_value("feature_scalar", exp2.features.table_to_tensor("feature_scalar", scalar, exp2.parameters))

    X_df, y_df = dataset.export_to_dataframe(["exp_001", "exp_002"])
    assert len(X_df) == 12  # 6 rows × 2 experiments
    assert len(y_df) == 12
    # First 6 rows come from exp_001 (param_1=2.5), next 6 from exp_002 (param_1=5.0)
    assert np.allclose(X_df.iloc[:6]["param_1"].to_numpy(), 2.5)
    assert np.allclose(X_df.iloc[6:]["param_1"].to_numpy(), 5.0)


def test_export_scalar_experiment_with_no_dimension_parameters(tmp_path):
    """An experiment schema without dimension parameters exports as a single row per experiment."""
    params = Parameters.from_list([Parameter.real("param_a", 0.0, 10.0)])
    feats = Features.from_list([Feature.array("feat_scalar")])
    feats.get("feat_scalar").set_columns(["feat_scalar"])
    perfs = PerformanceAttributes.from_list([PerformanceAttribute.score("perf_1")])

    schema = DatasetSchema(
        root_folder=str(tmp_path),
        name="schema_no_dims",
        parameters=params,
        features=feats,
        performance=perfs,
    )
    dataset = Dataset(schema=schema, debug_flag=True)
    dataset.create_experiment("scalar_exp", parameters={"param_a": 3.5})

    X_df, y_df = dataset.export_to_dataframe(["scalar_exp"])
    assert len(X_df) == 1
    assert X_df.iloc[0]["param_a"] == pytest.approx(3.5)
    assert len(y_df) == 1
