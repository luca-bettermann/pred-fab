import numpy as np
import pytest

from pred_fab.core import DataModule
from tests.utils.builders import build_dataset_with_single_experiment, build_mixed_feature_schema


def test_parameters_sanitize_values_casts_and_rounds_but_does_not_clip(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    params = dataset.schema.parameters

    sanitized = params.sanitize_values(
        {
            "param_1": 1.23456,
            "dim_1": 1.2,
            "dim_2": 2.8,
        }
    )

    assert sanitized["param_1"] == 1.235
    assert sanitized["dim_1"] == 1
    assert sanitized["dim_2"] == 3
    assert isinstance(sanitized["dim_1"], int)
    assert isinstance(sanitized["dim_2"], int)


def test_array_to_params_uses_parameter_sanitization_boundary(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    datamodule = DataModule(dataset)
    datamodule.initialize(input_parameters=["param_1", "dim_1", "dim_2"], input_features=[], output_columns=[])
    datamodule._is_fitted = True

    raw = np.array([1.23456, 1.2, 2.8], dtype=np.float64)
    restored = datamodule.array_to_params(raw)

    assert restored["param_1"] == 1.235
    assert restored["dim_1"] == 1
    assert restored["dim_2"] == 3
    assert isinstance(restored["dim_1"], int)
    assert isinstance(restored["dim_2"], int)


def test_parameters_sanitize_values_raises_for_constraint_violations(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    params = dataset.schema.parameters

    with pytest.raises(ValueError):
        params.sanitize_values({"dim_1": 0.2})


def test_schema_compatibility_uses_block_compatibility_check(tmp_path):
    schema_a = build_mixed_feature_schema(tmp_path / "a", name="schema_a")
    schema_b = build_mixed_feature_schema(tmp_path / "b", name="schema_b")
    assert schema_a.is_compatible_with(schema_b)
