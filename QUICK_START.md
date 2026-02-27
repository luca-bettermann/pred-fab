# Quick Start

This guide shows the current `pred_fab` workflow with `PfabAgent`.

## 1. Define Schema

```python
import numpy as np
from pred_fab.core import DatasetSchema
from pred_fab.core.data_blocks import Parameters, Features, PerformanceAttributes
from pred_fab.core.data_objects import Parameter, Feature, PerformanceAttribute

params = Parameters.from_list([
    Parameter.real("param_1", min_val=0.0, max_val=10.0),
    Parameter.integer("param_2", min_val=1, max_val=5),
    Parameter.dimension("dim_1", iterator_code="d1", level=1, max_val=5),
    Parameter.dimension("dim_2", iterator_code="d2", level=2, max_val=5),
])

features = Features.from_list([
    Feature.array("feature_1"),
    Feature.array("feature_2"),
])

performance = PerformanceAttributes.from_list([
    PerformanceAttribute.score("performance_1"),
])

schema = DatasetSchema(
    root_folder=".",
    name="schema_001",
    parameters=params,
    features=features,
    performance=performance,
)
```

## 2. Implement Models

```python
from typing import Any, Dict, List, Tuple
import numpy as np
from pred_fab.interfaces import IFeatureModel, IEvaluationModel, IPredictionModel

class MyFeatureModel(IFeatureModel):
    @property
    def input_parameters(self) -> List[str]:
        return ["dim_1", "dim_2", "param_1"]

    @property
    def outputs(self) -> List[str]:
        return ["feature_1", "feature_2"]

    def _load_data(self, params: Dict, **dimensions) -> Any:
        return params

    def _compute_feature_logic(self, data: Any, params: Dict, visualize: bool = False, **dimensions) -> Dict[str, float]:
        d1 = float(dimensions["d1"])
        d2 = float(dimensions["d2"])
        base = float(params["param_1"])
        return {
            "feature_1": base + 0.1 * d1 + 0.01 * d2,
            "feature_2": 0.5 * base + d1,
        }

class MyEvaluationModel(IEvaluationModel):
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
        return float(params["param_1"])

class MyPredictionModel(IPredictionModel):
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
        out = np.zeros((X.shape[0], 2), dtype=np.float64)
        out[:, 0] = X[:, 0]
        out[:, 1] = X[:, 0] * 0.5
        return out

    def train(self, train_batches: List[Tuple[np.ndarray, np.ndarray]], val_batches: List[Tuple[np.ndarray, np.ndarray]], **kwargs) -> None:
        return None
```

## 3. Create Dataset + Agent

```python
from pred_fab.core import Dataset
from pred_fab.orchestration import PfabAgent

dataset = Dataset(schema=schema, debug_flag=True)

a = PfabAgent(root_folder=".", debug_flag=True)
a.register_feature_model(MyFeatureModel)
a.register_evaluation_model(MyEvaluationModel)
a.register_prediction_model(MyPredictionModel)
a.initialize_systems(schema)
```

## 4. Create/Load Experiment and Evaluate

```python
exp = dataset.create_experiment(
    "exp_001",
    parameters={"param_1": 2.0, "param_2": 2, "dim_1": 2, "dim_2": 3},
)

a.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
print(exp.performance.get_value("performance_1"))
```

## 5. Train and Predict

```python
dm = a.create_datamodule(dataset)
dm.prepare(val_size=0.0, test_size=0.0, recompute=True)

a.train(datamodule=dm, validate=False)
preds = a.predict(exp_data=exp)
```

## 6. Calibrate (Optional)

```python
a.configure_calibration(
    performance_weights={"performance_1": 1.0},
    bounds={"param_1": (0.0, 10.0), "param_2": (1, 5), "dim_1": (1, 4), "dim_2": (1, 4)},
    adaptation_delta={"param_1": 0.2},
)

proposal = a.exploration_step(datamodule=dm, w_explore=0.5)
print(proposal.to_dict())
```

## 7. Persist Local Artifacts

```python
dataset.save_all(recompute_flag=True)
# Later:
# dataset.load_experiments(["exp_001"], recompute_flag=False)
```
