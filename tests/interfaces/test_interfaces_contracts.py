import numpy as np
import pytest

from pred_fab.interfaces import IFeatureModel, IEvaluationModel, IPredictionModel
from pred_fab.utils import PfabLogger
from tests.utils.builders import build_mixed_feature_schema


class _FeatureModelOk(IFeatureModel):
    @property
    def input_parameters(self):
        return ["dim_1", "dim_2"]

    @property
    def outputs(self):
        return ["feature_grid"]

    def _load_data(self, params, **dimensions):
        return (params, dimensions)

    def _compute_feature_logic(self, data, params, visualize=False, **dimensions):
        return {"feature_grid": dimensions["d1"] * 10 + dimensions["d2"]}


class _FeatureModelBadOutputType(IFeatureModel):
    @property
    def input_parameters(self):
        return ["dim_1"]

    @property
    def outputs(self):
        return ["feature_d1"]

    def _load_data(self, params, **dimensions):
        return None

    def _compute_feature_logic(self, data, params, visualize=False, **dimensions):
        return {"feature_d1": "non_numeric"}


class _FeatureModelInvalidProps(IFeatureModel):
    @property
    def input_parameters(self):
        return "dim_1"

    @property
    def outputs(self):
        return ["feature_d1"]

    def _load_data(self, params, **dimensions):
        return None

    def _compute_feature_logic(self, data, params, visualize=False, **dimensions):
        return {"feature_d1": 1.0}


class _EvalModelOk(IEvaluationModel):
    @property
    def input_parameters(self):
        return []

    @property
    def input_feature(self) -> str:
        return "feature_scalar"

    @property
    def output_performance(self) -> str:
        return "performance_1"

    def _compute_target_value(self, params, **dimensions):
        return 1.0


class _PredModelDefaults(IPredictionModel):
    @property
    def input_parameters(self):
        return ["param_1"]

    @property
    def input_features(self):
        return []

    @property
    def outputs(self):
        return ["feature_scalar"]

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((X.shape[0], 1), dtype=np.float64)

    def train(self, train_batches, val_batches, **kwargs) -> None:
        return None


def _logger(tmp_path):
    return PfabLogger.get_logger(str(tmp_path / "logs"))


def test_feature_interface_compute_features_orders_outputs_and_dimensions(tmp_path):
    schema = build_mixed_feature_schema(tmp_path, name="schema_interfaces_feature")
    model = _FeatureModelOk(_logger(tmp_path))
    model.set_ref_parameters(list(schema.parameters.data_objects.values()))

    params = schema.parameters.from_dict(schema.parameters.to_dict())
    params.set_values_from_dict({"param_1": 2.0, "dim_1": 2, "dim_2": 3}, logger=_logger(tmp_path))

    arr = model.compute_features(params, evaluate_from=0, evaluate_to=None)
    assert arr.shape == (6, 3)
    assert float(arr[-1, -1]) == 12.0


def test_feature_interface_rejects_non_numeric_output_values(tmp_path):
    schema = build_mixed_feature_schema(tmp_path, name="schema_interfaces_bad_output")
    model = _FeatureModelBadOutputType(_logger(tmp_path))
    model.set_ref_parameters(list(schema.parameters.data_objects.values()))

    params = schema.parameters.from_dict(schema.parameters.to_dict())
    params.set_values_from_dict({"param_1": 2.0, "dim_1": 2, "dim_2": 3}, logger=_logger(tmp_path))

    with pytest.raises(TypeError):
        model.compute_features(params, evaluate_from=0, evaluate_to=1)


def test_base_interface_validates_property_types(tmp_path):
    with pytest.raises(TypeError):
        _FeatureModelInvalidProps(_logger(tmp_path))


def test_evaluation_interface_wrappers_expose_singleton_lists(tmp_path):
    model = _EvalModelOk(_logger(tmp_path))
    assert model.input_features == ["feature_scalar"]
    assert model.outputs == ["performance_1"]


def test_prediction_interface_default_optional_methods_raise(tmp_path):
    model = _PredModelDefaults(_logger(tmp_path))
    with pytest.raises(NotImplementedError):
        model.tuning([])
    with pytest.raises(NotImplementedError):
        model._get_model_artifacts()
    with pytest.raises(NotImplementedError):
        model._set_model_artifacts({})
