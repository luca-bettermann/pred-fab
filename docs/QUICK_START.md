# Quick Start Guide

**Last Updated**: December 9, 2025  
**Status**: Current (PFAB Release)

---

## Installation

```bash
# Clone repository
git clone <repo-url>
cd lbp_package

# Install package
pip install -e .
```

---

## 5-Minute Workflow

### 1. Define Your Schema

First, define the structure of your dataset using `DatasetSchema`. This acts as the contract for your data.

```python
from lbp_package.core import DatasetSchema
from lbp_package.core.data_objects import Parameter, Feature, PerformanceAttribute
from lbp_package.core.data_blocks import Parameters, PerformanceAttributes, Features

# Define Parameters (Inputs)
p1 = Parameter.real("print_speed", min_val=10.0, max_val=200.0)
p2 = Parameter.real("layer_height", min_val=0.1, max_val=0.4)
p3 = Parameter.dimension("n_layers", iterator_code="layer_idx", level=1)

# Define Features (Intermediate Arrays)
f1 = Feature.array("energy_trace")
f2 = Feature.array("path_deviation")

# Define Performance Attributes (Scalar Outputs)
perf1 = PerformanceAttribute.score("energy_efficiency")
perf2 = PerformanceAttribute.score("geometric_accuracy")

# Create Blocks
param_block = Parameters.from_list([p1, p2, p3])
feat_block = Features.from_list([f1, f2])
perf_block = PerformanceAttributes.from_list([perf1, perf2])

# Initialize Schema
schema = DatasetSchema(
    name="my_schema_v1",
    parameters=param_block,
    features=feat_block,
    performance=perf_block,
)
```

---

### 2. Define Your Models

Implement the interfaces for Feature Extraction and Evaluation.

```python
from lbp_package.interfaces import IFeatureModel, IEvaluationModel
from lbp_package.utils import LBPLogger
from typing import List, Dict, Any

class EnergyFeatureModel(IFeatureModel):
    """Extract energy trace from raw sensor data."""
    
    def __init__(self, logger: LBPLogger):
        super().__init__(logger)
        
    @property
    def input_parameters(self) -> List[str]:
        return ["print_speed", "n_layers"]
        
    @property
    def input_features(self) -> List[str]:
        return []
        
    @property
    def outputs(self) -> List[str]:
        return ["energy_trace"]

    def _load_data(self, params: Dict, **dimensions) -> Any:
        # Load raw sensor data (e.g., from CSV or DB)
        return [0.1, 0.2, 0.3] # Mock data

    def _compute_feature_logic(self, data: Any, params: Dict, visualize: bool = False, **dimensions) -> Dict[str, float]:
        # Compute feature value for specific dimension index
        return {"energy_trace": sum(data) * params["print_speed"]}

class EnergyEvaluationModel(IEvaluationModel):
    """Calculate energy efficiency score."""
    
    def __init__(self, logger: LBPLogger):
        super().__init__(logger)
        
    @property
    def input_parameters(self) -> List[str]:
        return []
        
    @property
    def input_features(self) -> List[str]:
        return ["energy_trace"]
        
    @property
    def outputs(self) -> List[str]:
        return ["energy_efficiency"]
        
    def _evaluate_logic(self, features: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, float]:
        # Calculate score (0-1)
        energy = features["energy_trace"]
        score = 1.0 / (1.0 + energy)
        return {"energy_efficiency": score}
```

---

### 3. Initialize Agent & Register Models

```python
from lbp_package.orchestration.agent import LBPAgent

# Create Agent
agent = LBPAgent(
    root_folder="./my_project",
    debug_flag=True
)

# Register Models
agent.register_feature_model(EnergyFeatureModel)
agent.register_evaluation_model(EnergyEvaluationModel)

# Initialize Dataset (Validates models against schema)
dataset = agent.initialize(schema)

print(f"Dataset initialized with ID: {dataset.schema_id}")
```

---

### 4. Run Experiments

```python
from lbp_package.utils import StepType

# Load or Create Experiment
dataset.load_experiments(["test_001"]) # Tries to load from disk
exp_data = dataset.get_experiment("test_001")

# If new, set parameters
if not exp_data.parameters.has_value("print_speed"):
    exp_data.parameters.set_value("print_speed", 50.0)
    exp_data.parameters.set_value("layer_height", 0.2)
    exp_data.parameters.set_value("n_layers", 100)

# Run Offline Step (Feature Extraction -> Evaluation)
agent.step_offline(
    exp_data=exp_data,
    step_type=StepType.EVAL, # Only evaluate, don't train
    recompute=True
)

# Access Results
score = exp_data.performance.get_value("energy_efficiency")
print(f"Energy Efficiency: {score}")
```

---

### 5. Train Prediction Models

```python
from lbp_package.core import DataModule

# Create DataModule for ML
datamodule = DataModule(dataset)

# Run Full Step (Evaluation + Training)
agent.step_offline(
    exp_data=exp_data,
    datamodule=datamodule,
    step_type=StepType.FULL # Evaluate AND Train
)
```

---

## Key Concepts

### 1. Schema-First Design
You explicitly define the `DatasetSchema` (Parameters, Features, Performance) before doing anything else. This acts as a contract. The Agent validates that your registered models fulfill this contract (i.e., they produce the required outputs using the available inputs).

### 2. Stateless Agent
The `LBPAgent` orchestrates the workflow but does not own the data. It returns a `Dataset` object that you manage.

### 3. Explicit Interfaces
Models must implement `IFeatureModel` or `IEvaluationModel` and explicitly declare their `input_parameters`, `input_features`, and `outputs`.

### 4. Offline Stepping
The `agent.step_offline()` method is the main entry point for processing experiments. It handles the dependency chain:
1.  **Feature Extraction**: Computes arrays from raw data.
2.  **Evaluation**: Computes scores from features.
3.  **Training**: (Optional) Updates prediction models using `DataModule`.
