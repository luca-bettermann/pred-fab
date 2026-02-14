import numpy as np

from pred_fab.core import DataModule
from pred_fab.interfaces import IPredictionModel, ISurrogateModel
from pred_fab.orchestration.calibration import CalibrationSystem
from pred_fab.orchestration.inference_bundle import InferenceBundle
from pred_fab.orchestration.prediction import PredictionSystem
from pred_fab.utils import PfabLogger, SplitType, LocalData
from tests.utils.builders import build_dataset_with_single_experiment, sample_feature_tables, build_mixed_feature_schema


class _ShapeCheckingPredictionModel(IPredictionModel):
    def __init__(self, logger, in_params, outputs):
        self._in_params = in_params
        self._outputs = outputs
        self.seen_widths = []
        super().__init__(logger)

    @property
    def input_parameters(self):
        return self._in_params

    @property
    def input_features(self):
        return []

    @property
    def outputs(self):
        return self._outputs

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        self.seen_widths.append(X.shape[1])
        return np.zeros((X.shape[0], len(self._outputs)), dtype=np.float64)

    def train(self, train_batches, val_batches, **kwargs):
        return None

    def _get_model_artifacts(self):
        return {"dummy": True}

    def _set_model_artifacts(self, artifacts):
        return None


class _DummySurrogate(ISurrogateModel):
    def __init__(self, logger):
        super().__init__(logger)
        self.fit_calls = 0
        self.last_shapes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.fit_calls += 1
        self.last_shapes = (X.shape, y.shape)

    def predict(self, X: np.ndarray):
        return np.zeros((X.shape[0], 1)), np.zeros((X.shape[0], 1))


def _logger(tmp_path):
    return PfabLogger.get_logger(str(tmp_path / "logs"))


def _populate_single_experiment_features(dataset):
    exp = dataset.get_experiment("exp_001")
    grid, d1_only, scalar = sample_feature_tables()
    exp.features.set_value("feature_grid", exp.features.table_to_tensor("feature_grid", grid, exp.parameters))
    exp.features.set_value("feature_d1", exp.features.table_to_tensor("feature_d1", d1_only, exp.parameters))
    exp.features.set_value("feature_scalar", exp.features.table_to_tensor("feature_scalar", scalar, exp.parameters))
    return exp


def test_prediction_validate_uses_model_specific_input_slices(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    _populate_single_experiment_features(dataset)

    datamodule = DataModule(dataset)
    datamodule.initialize(
        input_parameters=["param_1", "dim_1", "dim_2"],
        input_features=[],
        output_columns=["feature_grid", "feature_d1", "feature_scalar"],
    )
    datamodule._is_fitted = True
    datamodule._split_codes = {SplitType.TRAIN: ["exp_001"], SplitType.VAL: ["exp_001"], SplitType.TEST: []}

    schema = dataset.schema
    pred_system = PredictionSystem(_logger(tmp_path), schema=schema, local_data=LocalData(str(tmp_path)))
    m_a = _ShapeCheckingPredictionModel(_logger(tmp_path), in_params=["param_1"], outputs=["feature_grid"])
    m_b = _ShapeCheckingPredictionModel(_logger(tmp_path), in_params=["dim_1", "dim_2"], outputs=["feature_d1"])
    pred_system.models = [m_a, m_b]
    pred_system.datamodule = datamodule

    pred_system.validate(use_test=False)
    assert m_a.seen_widths and m_a.seen_widths[0] == 1
    assert m_b.seen_widths and m_b.seen_widths[0] == 2


def test_calibration_surrogate_training_skips_missing_performance_rows(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    schema = dataset.schema

    datamodule = DataModule(dataset)
    datamodule.initialize(input_parameters=["param_1", "dim_1", "dim_2"], input_features=[], output_columns=[])
    datamodule._is_fitted = True
    datamodule._split_codes = {SplitType.TRAIN: ["exp_001"], SplitType.VAL: [], SplitType.TEST: []}

    surrogate = _DummySurrogate(_logger(tmp_path))
    calibration = CalibrationSystem(
        schema=schema,
        logger=_logger(tmp_path),
        predict_fn=lambda x: {"feature_scalar": np.zeros((len(x), 1))},
        residual_predict_fn=lambda x: np.zeros((len(x), 1)),
        evaluate_fn=lambda x: {"performance_1": np.zeros((len(x), 1))},
        surrogate_model=surrogate,
    )

    calibration.train_surrogate_model(datamodule)
    assert surrogate.fit_calls == 0

    exp = dataset.get_experiment("exp_001")
    exp.performance.set_values_from_dict({"performance_1": 0.5}, logger=dataset.logger)
    calibration.train_surrogate_model(datamodule)
    assert surrogate.fit_calls == 1
    assert surrogate.last_shapes == ((1, 3), (1, 1))


def test_inference_bundle_handles_degenerate_minmax():
    bundle = InferenceBundle(prediction_models=[], normalization_state={"method": "none"}, schema_dict={})
    stats = {"method": "minmax", "min": 2.0, "max": 2.0}

    x = np.array([1.0, 2.0, 3.0])
    x_norm = bundle._apply_normalization(x, stats)
    x_denorm = bundle._reverse_normalization(np.array([0.1, 0.9]), stats)

    assert np.allclose(x_norm, np.zeros_like(x))
    assert np.allclose(x_denorm, np.array([2.0, 2.0]))
