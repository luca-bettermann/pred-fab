# pred-fab

A Python framework for iterative parameter optimization in digital fabrication systems.

`pred-fab` provides a structured loop between **feature extraction**, **performance evaluation**, **predictive modelling**, and **parameter calibration**. It supports offline experiment design, online process adaptation, and inference-guided proposals — all through a single orchestration agent.

**Package**: `pred-fab` v0.1.0
**Python**: 3.9+

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Quick Start](#quick-start)
- [Interfaces](#interfaces)
- [Agent API](#agent-api)
- [Calibration Configuration](#calibration-configuration)
- [Workflow Modes](#workflow-modes)
- [License](#license)

---

## Overview

The framework is built around a dataset-centric architecture with a clear separation between domain-specific logic (user-implemented interfaces) and orchestration (handled by the framework):

```
DatasetSchema          — defines parameters, features, and performance attributes
├── IFeatureModel      — extracts numerical features from raw fabrication data
├── IEvaluationModel   — scores features against targets to compute performance
└── IPredictionModel   — predicts features from parameters (enables calibration)

PfabAgent              — orchestrates the full loop
├── evaluate()         — run feature extraction + evaluation on recorded experiments
├── train()            — train prediction models on evaluated data
├── exploration_step() — propose next experiment parameters (UCB acquisition)
├── inference_step()   — evaluate + propose next parameters (performance-guided)
└── adaptation_step()  — online trust-region adaptation during fabrication
```

---

## Installation

```bash
git clone <repository-url>
cd pred-fab

uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## Core Concepts

### Schema

The `DatasetSchema` describes everything the framework needs to know about your process: which parameters are controllable, which features are measurable, and which performance attributes matter.

```python
from pred_fab.core.schema import DatasetSchema
from pred_fab.core.data_blocks import Parameters, Features, PerformanceAttributes
from pred_fab.core.data_objects import Parameter, Feature, PerformanceAttribute

schema = DatasetSchema(
    root_folder="./data",
    name="my_process",
    parameters=Parameters.from_list([
        Parameter.real("speed", min_val=20.0, max_val=120.0),
        Parameter.integer("layer_count", min_val=1, max_val=10),
        Parameter.categorical("material", categories=["A", "B", "C"]),
        Parameter.dimension("n_layers", iterator_code="layer_idx", level=0),
        Parameter.dimension("n_segments", iterator_code="seg_idx", level=1),
        Parameter.real("feed_rate", min_val=10.0, max_val=80.0, runtime=True),
    ]),
    features=Features.from_list([
        Feature.array("width_profile"),
        Feature.scalar("mean_width"),
    ]),
    performance=PerformanceAttributes.from_list([
        PerformanceAttribute.score("width_accuracy"),
    ]),
)
```

**Parameter types:**

| Type | Constructor | Notes |
|------|-------------|-------|
| Continuous | `Parameter.real(code, min_val, max_val)` | Float-valued |
| Integer | `Parameter.integer(code, min_val, max_val)` | Int-valued |
| Categorical | `Parameter.categorical(code, categories)` | One-hot encoded for prediction |
| Boolean | `Parameter.boolean(code)` | True/False |
| Dimension | `Parameter.dimension(code, iterator_code, level)` | Defines iteration structure |

Set `runtime=True` on any parameter that can be adjusted mid-fabrication. This enables online adaptation and trajectory-based exploration.

### Dataset and Experiments

```python
from pred_fab.core.dataset import Dataset

dataset = Dataset(schema=schema)
dataset.load_experiments(["exp_001", "exp_002", "exp_003"])

exp = dataset.get_experiment("exp_001")
params = exp.parameters.get_values_dict()
```

### DataModule

`DataModule` prepares data for training and prediction, handling normalization and train/val/test splits:

```python
datamodule = agent.create_datamodule(dataset)
datamodule.prepare(val_size=0.15, test_size=0.1)
```

---

## Quick Start

```python
from pred_fab import PfabAgent
from pred_fab.core.dataset import Dataset

# 1. Define schema (see above)
schema = DatasetSchema(...)

# 2. Create agent, register models, and initialize
agent = PfabAgent(root_folder="./data")
agent.register_feature_model(WidthFeatureModel)
agent.register_evaluation_model(WidthEvaluationModel)
agent.register_prediction_model(WidthPredictionModel)
agent.initialize_systems(schema)

# 3. Configure calibration bounds
agent.configure_calibration(
    bounds={"speed": (20.0, 100.0), "feed_rate": (10.0, 60.0)},
)

# 4. Sample initial experiments via Latin Hypercube Sampling
baseline_specs = agent.sample_baseline_experiments(n_samples=10)
# → run physical experiments for each spec, record results

# 5. Evaluate and train on recorded experiments
dataset = Dataset(schema=schema)
dataset.load_experiments([...])
datamodule = agent.create_datamodule(dataset)

for exp in dataset.experiments:
    agent.evaluate(exp_data=exp)
datamodule.prepare()
agent.train(datamodule)

# 6. Propose next experiment
spec = agent.exploration_step(datamodule, w_explore=0.6)
print(spec.initial_params)  # → {"speed": 72.4, "feed_rate": 38.1, ...}
```

---

## Interfaces

Implement these abstract classes to connect your domain logic to the framework. Each model declares its `input_parameters`, `input_features`, and `outputs` as class attributes so the framework can validate the data flow at initialization.

### IFeatureModel

Extracts numerical features from raw fabrication data.

```python
from pred_fab import IFeatureModel

class WidthFeatureModel(IFeatureModel):
    input_parameters = ["speed", "feed_rate", "n_layers", "n_segments"]
    input_features = []
    outputs = ["width_profile", "mean_width"]

    def _load_data(self, params: dict, **dimensions) -> Any:
        # Load raw data (files, sensor streams, etc.) — full control over format
        layer_idx = dimensions["layer_idx"]
        seg_idx = dimensions["seg_idx"]
        return load_scan(params, layer_idx, seg_idx)

    def _compute_feature_logic(self, data: Any, params: dict, **dimensions) -> dict:
        profile = compute_width_profile(data)
        return {
            "width_profile": profile,
            "mean_width": float(np.mean(profile)),
        }
```

### IEvaluationModel

Scores a single feature against a target to produce a normalized performance value in [0, 1].

```python
from pred_fab import IEvaluationModel

class WidthEvaluationModel(IEvaluationModel):
    input_parameters = ["speed"]

    @property
    def input_feature(self) -> str:
        return "mean_width"

    @property
    def output_performance(self) -> str:
        return "width_accuracy"

    def _compute_target_value(self, params: dict, **dimensions) -> float:
        return 2.0  # target width in mm

    def _compute_scaling_factor(self, params: dict, **dimensions) -> Optional[float]:
        return None  # use default scoring curve
```

### IPredictionModel

Predicts features from parameters. Enables the framework to score hypothetical parameter combinations without running physical experiments.

```python
from pred_fab import IPredictionModel

class WidthPredictionModel(IPredictionModel):
    input_parameters = ["speed", "feed_rate"]
    input_features = []
    outputs = ["mean_width"]

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        # X: (n_samples, n_inputs) normalized
        # Returns: (n_samples, n_outputs) normalized
        return self.model.predict(X)

    def train(self, train_batches, val_batches, **kwargs) -> None:
        X = np.vstack([x for x, _ in train_batches])
        y = np.vstack([y for _, y in train_batches])
        self.model.fit(X, y)
```

For online adaptation, implement `tuning()` to update the model on a small recent slice:

```python
    def tuning(self, tune_batches, **kwargs) -> None:
        # Fine-tune on recent measurements (residual learning, etc.)
        ...
```

---

## Agent API

### Initialization

```python
agent = PfabAgent(
    root_folder="./data",
    debug_flag=False,       # skip external data operations if True
    recompute_flag=False,   # force recomputation of cached results if True
    visualize_flag=True,    # enable model visualizations if True
)
agent.register_feature_model(MyFeatureModel)
agent.register_evaluation_model(MyEvaluationModel)
agent.register_prediction_model(MyPredictionModel)   # optional, needed for calibration
agent.initialize_systems(schema, verbose_flag=True)
```

### Core Methods

| Method | Description |
|--------|-------------|
| `evaluate(exp_data)` | Run feature extraction and performance evaluation on an experiment |
| `train(datamodule, validate, test)` | Train prediction models; returns validation metrics if requested |
| `predict(exp_data, dimension, step_index)` | Predict features for an experiment or a specific step slice |
| `sample_baseline_experiments(n_samples, param_bounds)` | Generate LHS-sampled initial experiment parameters |
| `exploration_step(datamodule, w_explore, n_optimization_rounds, current_params)` | UCB-acquisition proposal for the next experiment |
| `inference_step(exp_data, datamodule, w_explore, n_optimization_rounds, current_params)` | Evaluate current experiment + propose next parameters |
| `adaptation_step(dimension, step_index, exp_data, mode, w_explore, record)` | Online trust-region adaptation for a single fabrication step |
| `configure_calibration(bounds, fixed_params, adaptation_delta, performance_weights)` | Configure the calibration system |
| `state_report()` | Print a summary of registered models and their I/O |
| `calibration_state_report()` | Print current bounds and trust-region configuration |

---

## Calibration Configuration

```python
agent.configure_calibration(
    # Hard bounds for offline optimization
    bounds={"speed": (20.0, 100.0), "feed_rate": (10.0, 60.0)},

    # Parameters held constant during optimization
    fixed_params={"material": "A"},

    # Trust-region deltas for online/adaptation steps (runtime=True params only)
    adaptation_delta={"feed_rate": 5.0},

    # Performance weighting (default 1.0 for all)
    performance_weights={"width_accuracy": 2.0},
)
```

### Trajectory Exploration

For runtime parameters that change along a fabrication dimension (e.g. feed rate per layer), configure trajectories to have the framework optimize a per-step schedule:

```python
agent.calibration_system.configure_trajectory("feed_rate", dimension_code="n_layers")
agent.calibration_system.configure_adaptation_delta({"feed_rate": 5.0})
```

`exploration_step` and `inference_step` then return an `ExperimentSpec` with both `initial_params` and per-dimension `schedules`.

### MPC Lookahead

Reduce greedy myopia during trajectory optimization by simulating steps ahead:

```python
agent.calibration_system.default_mpc_lookahead_depth = 2   # simulate 2 steps ahead
agent.calibration_system.default_mpc_discount = 0.9        # geometric discount factor
```

---

## Workflow Modes

### Offline Exploration

Propose the next experiment using UCB acquisition (balances exploration vs. exploitation). Use when you want to explore broadly or have no trained model yet:

```python
spec = agent.exploration_step(
    datamodule=datamodule,
    w_explore=0.7,             # 0.0 = exploit, 1.0 = explore
    n_optimization_rounds=20,  # random restarts for global search
    current_params=None,       # optional: provide to initialize search from known state
)
```

### Inference-Guided Proposal

Evaluate the current experiment first, then propose the next parameters based on updated performance estimates:

```python
spec = agent.inference_step(
    exp_data=current_exp,
    datamodule=datamodule,
    w_explore=0.3,
)
```

### Online Adaptation

Adapt runtime parameters step-by-step during a live fabrication run. The prediction model is fine-tuned on the latest slice before proposing new values:

```python
spec = agent.adaptation_step(
    exp_data=current_exp,
    dimension="n_layers",
    step_index=3,
    mode=Mode.INFERENCE,  # or Mode.EXPLORATION
    w_explore=0.0,
    record=True,          # persist the proposed update to the experiment record
)
```

### ExperimentSpec

All step methods return an `ExperimentSpec`:

```python
spec = agent.exploration_step(datamodule)

# Experiment-level parameters
speed = spec["speed"]               # dict-like access on initial_params
params = spec.initial_params        # full ParameterProposal

# Trajectory schedules (when trajectory params are configured)
for dim_code, schedule in spec.schedules.items():
    for step_idx, proposal in schedule.entries:
        print(f"  layer {step_idx}: feed_rate = {proposal['feed_rate']:.2f}")
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
