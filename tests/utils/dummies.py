"""Reusable dummy/test-double implementations for orchestration tests."""

from typing import List
import numpy as np

from pred_fab.interfaces import IFeatureModel, IEvaluationModel, IPredictionModel, ISurrogateModel, IResidualModel


class ShapeCheckingPredictionModel(IPredictionModel):
    """Prediction model test double that records observed input width."""

    def __init__(self, logger, in_params: List[str], outputs: List[str]):
        self._in_params = in_params
        self._outputs = outputs
        self.seen_widths: List[int] = []
        self.seen_batch_sizes: List[int] = []
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
        self.seen_batch_sizes.append(X.shape[0])
        return np.zeros((X.shape[0], len(self._outputs)), dtype=np.float64)

    def train(self, train_batches, val_batches, **kwargs):
        return None

    def _get_model_artifacts(self):
        return {"dummy": True}

    def _set_model_artifacts(self, artifacts):
        return None


class MixedFeatureModel(IFeatureModel):
    """Feature-model test double for mixed-dimensional schemas."""

    @property
    def input_parameters(self):
        return ["dim_1", "dim_2"]

    @property
    def outputs(self):
        return ["feature_grid", "feature_d1", "feature_scalar"]

    def _load_data(self, params, **dimensions):
        return None

    def _compute_feature_logic(self, data, params, visualize: bool = False, **dimensions):
        d1 = float(dimensions["d1"])
        d2 = float(dimensions["d2"])
        return {
            "feature_grid": d1 * 10.0 + d2,
            "feature_d1": 100.0 + d1,
            "feature_scalar": 7.0,
        }


class ScalarEvaluationModel(IEvaluationModel):
    """Evaluation-model test double producing one scalar performance metric."""

    @property
    def input_parameters(self):
        return []

    @property
    def input_feature(self):
        return "feature_scalar"

    @property
    def output_performance(self):
        return "performance_1"

    def _compute_target_value(self, params, **dimensions):
        return 7.0


class MixedPredictionModel(IPredictionModel):
    """Prediction-model test double aligned with mixed-dimensional test schema."""

    def __init__(self, logger):
        super().__init__(logger)
        self.weights = np.array(
            [
                [0.2, 0.1, 0.0],
                [0.1, 0.2, 0.0],
                [0.3, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

    @property
    def input_parameters(self):
        return ["param_1", "dim_1", "dim_2"]

    @property
    def input_features(self):
        return []

    @property
    def outputs(self):
        return ["feature_grid", "feature_d1", "feature_scalar"]

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X[:, :3], self.weights)

    def train(self, train_batches, val_batches, **kwargs):
        return None


class DummySurrogateModel(ISurrogateModel):
    """Surrogate model test double that captures fit calls and shapes."""

    def __init__(self, logger):
        super().__init__(logger)
        self.fit_calls = 0
        self.last_shapes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.fit_calls += 1
        self.last_shapes = (X.shape, y.shape)

    def predict(self, X: np.ndarray):
        return np.zeros((X.shape[0], 1)), np.zeros((X.shape[0], 1))


class CapturingResidualModel(IResidualModel):
    """Residual model test double that captures fit input/output shapes."""

    def __init__(self):
        self.last_fit_shapes = None

    def fit(self, X: np.ndarray, residuals: np.ndarray, **kwargs):
        self.last_fit_shapes = (X.shape, residuals.shape)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((X.shape[0], 1), dtype=np.float64)
