"""Tests for `Feature.iterator()` — auto-derived row-position context features."""

import pytest

from pred_fab.core import Dataset, DatasetSchema
from pred_fab.core.data_blocks import (
    Domains, Features, Parameters, PerformanceAttributes,
)
from pred_fab.core.data_objects import (
    Dimension, Domain, Feature, Parameter, PerformanceAttribute,
)


def _build_schema_with_iterator(tmp_path) -> DatasetSchema:
    spatial = Domain("spatial", [
        Dimension("n_layers", "layer_idx", 1, 4),
        Dimension("n_segments", "segment_idx", 1, 3),
    ])
    layer_dim, _ = spatial.axes
    return DatasetSchema(
        root_folder=str(tmp_path),
        name="iterator_schema",
        parameters=Parameters.from_list([
            Parameter.real("param_1", min_val=0.0, max_val=1.0),
        ]),
        features=Features.from_list([
            Feature.array("feature_grid", domain=spatial),
            Feature.iterator(spatial, layer_dim),
        ]),
        performance=PerformanceAttributes.from_list([
            PerformanceAttribute.score("perf_1"),
        ]),
        domains=Domains([spatial]),
    )


def test_iterator_factory_marks_feature_as_context(tmp_path):
    schema = _build_schema_with_iterator(tmp_path)
    feat = schema.features.get("layer_idx_pos")
    assert feat.context is True
    assert feat.is_iterator is True
    assert feat.iterator_axis_code == "layer_idx"


def test_iterator_rejects_dim_outside_domain(tmp_path):
    spatial = Domain("spatial", [Dimension("n_layers", "layer_idx", 1, 4)])
    other = Domain("other", [Dimension("n_other", "other_idx", 1, 4)])
    other_dim = other.axes[0]
    with pytest.raises(ValueError):
        Feature.iterator(spatial, other_dim)


def test_iterator_values_normalised_per_row(tmp_path):
    schema = _build_schema_with_iterator(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    dataset.create_experiment(
        "exp_001",
        parameters={"param_1": 0.5, "n_layers": 4, "n_segments": 3},
    )

    _, y_df = dataset.export_to_dataframe(["exp_001"])

    assert "layer_idx_pos" in y_df.columns
    # 4 layers × 3 segments = 12 rows, layer index varies across the L axis.
    # Values at layer k = k / (4 - 1) ∈ {0, 1/3, 2/3, 1}.
    expected = sorted({0.0, 1 / 3, 2 / 3, 1.0})
    actual = sorted(set(round(float(v), 6) for v in y_df["layer_idx_pos"]))
    assert all(abs(a - e) < 1e-6 for a, e in zip(actual, expected))


def test_iterator_excluded_from_input_columns_active_mask(tmp_path):
    """KDE active mask is parameter-only; iterator features must not be active."""
    from pred_fab.core.datamodule import DataModule

    schema = _build_schema_with_iterator(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    dataset.create_experiment(
        "exp_001",
        parameters={"param_1": 0.5, "n_layers": 4, "n_segments": 3},
    )

    dm = DataModule(dataset=dataset)
    dm.initialize(
        input_parameters=["param_1", "n_layers", "n_segments"],
        input_features=["layer_idx_pos"],
        output_columns=["feature_grid"],
    )

    assert "layer_idx_pos" in dm.input_columns
    # Confirms the column is *present* (model input) but is not a parameter.
    assert "layer_idx_pos" not in schema.parameters.data_objects
