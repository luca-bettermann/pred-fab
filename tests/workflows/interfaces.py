"""Workflow mock interfaces for integration demonstration."""

from typing import Any, Dict, List, Tuple
import numpy as np

from pred_fab.interfaces import IFeatureModel, IEvaluationModel, IPredictionModel, IExternalData
from pred_fab.utils import PfabLogger


class WorkflowFeatureModelA(IFeatureModel):
    def __init__(self, logger: PfabLogger):
        super().__init__(logger)

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
    def __init__(self, logger: PfabLogger):
        super().__init__(logger)

    @property
    def input_parameters(self) -> List[str]:
        return ["param_2"]

    @property
    def outputs(self) -> List[str]:
        return ["feature_3"]

    def _load_data(self, params: Dict, **dimensions) -> Any:
        return params

    def _compute_feature_logic(self, data: Any, params: Dict, visualize: bool = False, **dimensions) -> Dict[str, float]:
        return {"feature_3": 2.0 + 0.1 * float(params["param_2"]) }


class WorkflowEvaluationModelA(IEvaluationModel):
    def __init__(self, logger: PfabLogger):
        super().__init__(logger)

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
    def __init__(self, logger: PfabLogger):
        super().__init__(logger)

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
    """Prediction model using only parameters/dimensions as inputs for clean calibration bounds."""

    def __init__(self, logger: PfabLogger):
        super().__init__(logger)
        self.weights = np.random.rand(4, 2)

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
