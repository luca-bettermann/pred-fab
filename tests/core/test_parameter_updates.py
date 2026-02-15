import numpy as np
import pytest

from pred_fab.core import ParameterProposal
from tests.utils.builders import build_dataset_with_single_experiment, sample_feature_tables


def _populate_features(exp):
    """Populate deterministic feature tensors for export tests."""
    grid, d1_only, scalar = sample_feature_tables()
    exp.features.set_value("feature_grid", exp.features.table_to_tensor("feature_grid", grid, exp.parameters))
    exp.features.set_value("feature_d1", exp.features.table_to_tensor("feature_d1", d1_only, exp.parameters))
    exp.features.set_value("feature_scalar", exp.features.table_to_tensor("feature_scalar", scalar, exp.parameters))


def test_export_to_dataframe_applies_recorded_parameter_updates_by_step(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    _populate_features(exp)

    proposal = ParameterProposal.from_dict({"param_1": 9.0}, source_step="adaptation_step")
    exp.record_parameter_update(proposal, dimension="dim_1", step_index=1)

    X_df, y_df = dataset.export_to_dataframe(["exp_001"])
    assert len(X_df) == 6
    assert len(y_df) == 6
    assert "dim_1" in X_df.columns and "dim_2" in X_df.columns
    assert "d1" not in X_df.columns and "d2" not in X_df.columns

    before = X_df.iloc[:3]["param_1"].to_numpy()
    after = X_df.iloc[3:]["param_1"].to_numpy()
    assert np.allclose(before, np.array([2.5, 2.5, 2.5]))
    assert np.allclose(after, np.array([9.0, 9.0, 9.0]))


def test_record_parameter_update_rejects_dimension_updates(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")

    proposal = ParameterProposal.from_dict({"dim_1": 1}, source_step="adaptation_step")
    with pytest.raises(ValueError, match="dimension parameter"):
        exp.record_parameter_update(proposal, dimension="dim_1", step_index=1)


def test_record_parameter_update_skips_no_change_delta(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")

    proposal = ParameterProposal.from_dict({"param_1": 2.5}, source_step="adaptation_step")
    event = exp.record_parameter_update(proposal, dimension="dim_1", step_index=0)
    assert event is None
    assert exp.parameter_updates == []


def test_parameter_update_events_roundtrip_via_save_load(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path / "run")
    exp = dataset.get_experiment("exp_001")
    _populate_features(exp)
    exp.record_parameter_update(
        ParameterProposal.from_dict({"param_1": 8.5}, source_step="adaptation_step"),
        dimension="dim_1",
        step_index=1,
    )

    dataset.save_all(recompute_flag=True, verbose_flag=False)

    schema = dataset.schema
    dataset_reloaded = type(dataset)(schema=schema, debug_flag=True)
    dataset_reloaded.load_experiments(["exp_001"], recompute_flag=False, verbose=False)
    reloaded_exp = dataset_reloaded.get_experiment("exp_001")

    assert len(reloaded_exp.parameter_updates) == 1
    X_df, _ = dataset_reloaded.export_to_dataframe(["exp_001"])
    assert np.allclose(X_df.iloc[3:]["param_1"].to_numpy(), np.array([8.5, 8.5, 8.5]))
