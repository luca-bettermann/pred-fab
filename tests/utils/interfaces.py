"""Reusable interface implementations for tests.

These classes are lightweight implementations of framework interfaces
used to compose real orchestration systems in tests.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from pred_fab.interfaces import (
    IExternalData,
    IEvaluationModel,
    IFeatureModel,
    IPredictionModel,
    IResidualModel,
    ISurrogateModel,
)


class ShapeCheckingPredictionModel(IPredictionModel):
    """Prediction interface that captures observed batch widths and sizes."""

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
        return {"interface_model": True}

    def _set_model_artifacts(self, artifacts):
        return None


class MixedFeatureModel(IFeatureModel):
    """Feature interface aligned with mixed-dimensional schema test fixtures."""

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
    """Evaluation interface returning one deterministic scalar metric."""

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
    """Prediction interface aligned with mixed-dimensional schema fixtures."""

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


class CapturingSurrogateModel(ISurrogateModel):
    """Surrogate interface that records fit calls and observed shapes."""

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
    """Residual interface that captures fit input/output shapes."""

    def __init__(self):
        self.last_fit_shapes = None

    def fit(self, X: np.ndarray, residuals: np.ndarray, **kwargs):
        self.last_fit_shapes = (X.shape, residuals.shape)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((X.shape[0], 1), dtype=np.float64)


class WorkflowFeatureModelA(IFeatureModel):
    """Workflow feature interface for array-valued features."""

    @property
    def input_parameters(self) -> List[str]:
        return ["param_1", "dim_1", "dim_2"]

    @property
    def outputs(self) -> List[str]:
        return ["feature_1", "feature_2"]

    def _load_data(self, params: Dict, **dimensions) -> Any:
        return {"base_val": params.get("param_1", 0.0)}

    def _compute_feature_logic(self, data: Any, params: Dict, visualize: bool = False, **dimensions) -> Dict[str, float]:
        d1 = dimensions.get("d1", 0)
        d2 = dimensions.get("d2", 0)
        base = float(data["base_val"])
        return {
            "feature_1": base + 0.1 * d1 + 0.01 * d2,
            "feature_2": 0.5 * base - 0.05 * d1,
        }


class WorkflowFeatureModelB(IFeatureModel):
    """Workflow feature interface for scalar derived feature."""

    @property
    def input_parameters(self) -> List[str]:
        return ["param_2"]

    @property
    def outputs(self) -> List[str]:
        return ["feature_3"]

    def _load_data(self, params: Dict, **dimensions) -> Any:
        return params

    def _compute_feature_logic(self, data: Any, params: Dict, visualize: bool = False, **dimensions) -> Dict[str, float]:
        return {"feature_3": 2.0 + 0.1 * float(params["param_2"])}


class WorkflowEvaluationModelA(IEvaluationModel):
    """Workflow evaluation interface for performance_1."""

    @property
    def input_parameters(self) -> List[str]:
        return ["param_1"]

    @property
    def input_feature(self) -> str:
        return "feature_1"

    @property
    def output_performance(self) -> str:
        return "performance_1"

    def _compute_target_value(self, params: Dict, **dimensions) -> float:
        return float(params.get("param_1", 5.0)) * 1.5


class WorkflowEvaluationModelB(IEvaluationModel):
    """Workflow evaluation interface for performance_2."""

    @property
    def input_parameters(self) -> List[str]:
        return []

    @property
    def input_feature(self) -> str:
        return "feature_2"

    @property
    def output_performance(self) -> str:
        return "performance_2"

    def _compute_target_value(self, params: Dict, **dimensions) -> float:
        return 1.0


class WorkflowPredictionModel(IPredictionModel):
    """Workflow prediction interface driven only by parameters/dimensions."""

    def __init__(self, logger):
        super().__init__(logger)
        self.weights = np.array(
            [
                [0.6, 0.2],
                [0.3, 0.1],
                [0.1, 0.7],
                [0.2, 0.4],
            ],
            dtype=np.float64,
        )

    @property
    def input_parameters(self) -> List[str]:
        return ["param_1", "param_2", "dim_1", "dim_2"]

    @property
    def input_features(self) -> List[str]:
        return []

    @property
    def outputs(self) -> List[str]:
        return ["feature_1", "feature_2"]

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != self.weights.shape[0]:
            return np.dot(X[:, : self.weights.shape[0]], self.weights)
        return np.dot(X, self.weights)

    def train(self, train_batches: List[Tuple[np.ndarray, np.ndarray]], val_batches: List[Tuple[np.ndarray, np.ndarray]], **kwargs) -> None:
        self.weights += 0.01


class WorkflowExternalData(IExternalData):
    """External-data interface providing deterministic workflow parameters."""

    def pull_parameters(self, exp_codes: List[str]) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
        found: Dict[str, Dict[str, Any]] = {}
        for i, code in enumerate(exp_codes):
            found[code] = {
                "param_1": float(i + 1),
                "param_2": int(2 + i),
                "dim_1": 2,
                "dim_2": 3,
            }
        return [], found


class ContractFeatureModelOk(IFeatureModel):
    """Feature interface for contract tests with valid output behavior."""

    @property
    def input_parameters(self):
        return ["dim_1", "dim_2"]

    @property
    def outputs(self):
        return ["feature_grid"]

    def _load_data(self, params, **dimensions):
        return params, dimensions

    def _compute_feature_logic(self, data, params, visualize=False, **dimensions):
        return {"feature_grid": dimensions["d1"] * 10 + dimensions["d2"]}


class ContractFeatureModelBadOutputType(IFeatureModel):
    """Feature interface for contract tests with invalid non-numeric output."""

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


class ContractFeatureModelInvalidProps(IFeatureModel):
    """Feature interface for contract tests with invalid property type."""

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


class ContractEvaluationModelOk(IEvaluationModel):
    """Evaluation interface for contract tests."""

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


class ContractPredictionModelDefaults(IPredictionModel):
    """Prediction interface for contract tests of default optional methods."""

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
