from tests.utils.builders import build_dataset_with_single_experiment, sample_feature_tables


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
    assert set(["param_1", "dim_1", "dim_2", "d1", "d2"]).issubset(set(X_df.columns))
    assert set(["feature_grid", "feature_d1", "feature_scalar"]).issubset(set(y_df.columns))

    for i in range(len(X_df)):
        d1_val = int(X_df.iloc[i]["d1"])
        d2_val = int(X_df.iloc[i]["d2"])

        assert float(X_df.iloc[i]["param_1"]) == 2.5
        assert int(X_df.iloc[i]["dim_1"]) == 2
        assert int(X_df.iloc[i]["dim_2"]) == 3
        assert float(y_df.iloc[i]["feature_grid"]) == d1_val * 10 + d2_val
        assert float(y_df.iloc[i]["feature_d1"]) == 100 + d1_val
        assert float(y_df.iloc[i]["feature_scalar"]) == 7.0
