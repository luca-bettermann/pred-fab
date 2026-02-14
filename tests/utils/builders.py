"""Shared test builders for schema/dataset setup.

Add reusable factories here as the new test suite grows.
"""

import numpy as np

from pred_fab.core import DatasetSchema, Dataset
from pred_fab.core.data_objects import Parameter, Feature, PerformanceAttribute
from pred_fab.core.data_blocks import Parameters, Features, PerformanceAttributes


def build_mixed_feature_schema(tmp_path, name: str = "schema_test") -> DatasetSchema:
    """Create a schema with mixed feature dimensionality.

    Features:
    - feature_grid: depends on d1, d2
    - feature_d1: depends on d1 only
    - feature_scalar: scalar feature
    """
    p1 = Parameter.real("param_1", min_val=0.0, max_val=10.0)
    d1 = Parameter.dimension("dim_1", iterator_code="d1", level=1, max_val=2)
    d2 = Parameter.dimension("dim_2", iterator_code="d2", level=2, max_val=3)

    f_grid = Feature.array("feature_grid")
    f_d1 = Feature.array("feature_d1")
    f_scalar = Feature.array("feature_scalar")

    perf = PerformanceAttribute.score("performance_1")

    params = Parameters.from_list([p1, d1, d2])
    feats = Features.from_list([f_grid, f_d1, f_scalar])
    perfs = PerformanceAttributes.from_list([perf])

    # Mirrors FeatureSystem behavior: feature table columns are iterator(s) + output code.
    feats.get("feature_grid").set_columns(["d1", "d2", "feature_grid"])
    feats.get("feature_d1").set_columns(["d1", "feature_d1"])
    feats.get("feature_scalar").set_columns(["feature_scalar"])

    return DatasetSchema(
        root_folder=str(tmp_path),
        name=name,
        parameters=params,
        features=feats,
        performance=perfs,
    )


def build_dataset_with_single_experiment(tmp_path) -> Dataset:
    """Create a dataset with one experiment shell initialized."""
    schema = build_mixed_feature_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    dataset.create_experiment(
        "exp_001",
        parameters={"param_1": 2.5, "dim_1": 2, "dim_2": 3},
    )
    return dataset


def sample_feature_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic feature tables for mixed dimensionality tests."""
    grid_rows = []
    for i in range(2):
        for j in range(3):
            grid_rows.append([i, j, i * 10 + j])

    d1_rows = [[i, 100 + i] for i in range(2)]
    scalar_rows = [[7.0]]

    return (
        np.array(grid_rows, dtype=np.float64),
        np.array(d1_rows, dtype=np.float64),
        np.array(scalar_rows, dtype=np.float64),
    )
