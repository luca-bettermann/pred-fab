"""Tests for implicit iterator inputs (one per Domain axis).

Iterators are no longer schema-declared features — every ``Dimension`` exposes
a ``f"{ic}_pos"`` positional input automatically. Models reference them by
name; the framework populates from row coordinate at export / batch time.
"""

import pytest

from pred_fab.core import Dataset, DatasetSchema
from pred_fab.core.data_blocks import (
    Domains, Features, Parameters, PerformanceAttributes,
)
from pred_fab.core.data_objects import (
    Dimension, Domain, Feature, Parameter, PerformanceAttribute,
)


def _build_schema(tmp_path) -> DatasetSchema:
    spatial = Domain("spatial", [
        Dimension("n_layers", "layer_idx", 1, 4),
        Dimension("n_segments", "segment_idx", 1, 3),
    ])
    return DatasetSchema(
        root_folder=str(tmp_path),
        name="iterator_schema",
        parameters=Parameters.from_list([
            Parameter.real("param_1", min_val=0.0, max_val=1.0),
        ]),
        features=Features.from_list([
            Feature("feature_grid", domain=spatial),
        ]),
        performance=PerformanceAttributes.from_list([
            PerformanceAttribute.score("perf_1"),
        ]),
        domains=Domains([spatial]),
    )


def test_domain_exposes_iterator_input_codes(tmp_path):
    schema = _build_schema(tmp_path)
    spatial = schema.domains.get("spatial")
    assert spatial.iterator_input_codes == ["layer_idx_pos", "segment_idx_pos"]


def test_iterator_input_not_in_features_block(tmp_path):
    """Iterators are domain-implicit — not declared as features."""
    schema = _build_schema(tmp_path)
    assert "layer_idx_pos" not in schema.features.keys()
    assert "segment_idx_pos" not in schema.features.keys()


def test_iterator_values_normalised_per_row(tmp_path):
    schema = _build_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    dataset.create_experiment(
        "exp_001",
        parameters={"param_1": 0.5, "n_layers": 4, "n_segments": 3},
    )

    X_df, _ = dataset.export_to_dataframe(["exp_001"])

    # Iterator positions are implicit *inputs* (X), not features (y).
    assert "layer_idx_pos" in X_df.columns
    # 4 layers × 3 segments = 12 rows; layer values at k ∈ {0, 1/3, 2/3, 1}.
    expected = sorted({0.0, 1 / 3, 2 / 3, 1.0})
    actual = sorted(set(round(float(v), 6) for v in X_df["layer_idx_pos"]))
    assert all(abs(a - e) < 1e-6 for a, e in zip(actual, expected))


def test_iterator_input_accepted_by_datamodule(tmp_path):
    """KDE active mask is parameter-only; iterator inputs must not be active."""
    from pred_fab.core.datamodule import DataModule

    schema = _build_schema(tmp_path)
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
    assert "layer_idx_pos" not in schema.parameters.data_objects
