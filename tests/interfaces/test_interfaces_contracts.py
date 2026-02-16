import pytest

from tests.utils.builders import build_mixed_feature_schema, build_test_logger
from tests.utils.interfaces import (
    ContractEvaluationModelOk,
    ContractFeatureModelBadOutputType,
    ContractFeatureModelInvalidProps,
    ContractFeatureModelOk,
    ContractPredictionModelDefaults,
)


def test_feature_interface_compute_features_orders_outputs_and_dimensions(tmp_path):
    schema = build_mixed_feature_schema(tmp_path, name="schema_interfaces_feature")
    model = ContractFeatureModelOk(build_test_logger(tmp_path))
    model.set_ref_parameters(list(schema.parameters.data_objects.values()))

    params = schema.parameters.from_dict(schema.parameters.to_dict())
    params.set_values_from_dict({"param_1": 2.0, "dim_1": 2, "dim_2": 3}, logger=build_test_logger(tmp_path))

    arr = model.compute_features(params, evaluate_from=0, evaluate_to=None)
    assert arr.shape == (6, 3)
    assert float(arr[-1, -1]) == 12.0


def test_feature_interface_rejects_non_numeric_output_values(tmp_path):
    schema = build_mixed_feature_schema(tmp_path, name="schema_interfaces_bad_output")
    model = ContractFeatureModelBadOutputType(build_test_logger(tmp_path))
    model.set_ref_parameters(list(schema.parameters.data_objects.values()))

    params = schema.parameters.from_dict(schema.parameters.to_dict())
    params.set_values_from_dict({"param_1": 2.0, "dim_1": 2, "dim_2": 3}, logger=build_test_logger(tmp_path))

    with pytest.raises(TypeError):
        model.compute_features(params, evaluate_from=0, evaluate_to=1)


def test_base_interface_validates_property_types(tmp_path):
    with pytest.raises(TypeError):
        ContractFeatureModelInvalidProps(build_test_logger(tmp_path))


def test_evaluation_interface_wrappers_expose_singleton_lists(tmp_path):
    model = ContractEvaluationModelOk(build_test_logger(tmp_path))
    assert model.input_features == ["feature_scalar"]
    assert model.outputs == ["performance_1"]


def test_prediction_interface_default_optional_methods_raise(tmp_path):
    model = ContractPredictionModelDefaults(build_test_logger(tmp_path))
    with pytest.raises(NotImplementedError):
        model.tuning([])
    with pytest.raises(NotImplementedError):
        model._get_model_artifacts()
    with pytest.raises(NotImplementedError):
        model._set_model_artifacts({})
