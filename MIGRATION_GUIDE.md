# Migration Guide: LBPManager → LBPAgent (AIXD Architecture)

This guide helps migrate from study-based `LBPManager` to dataset-centric `LBPAgent`.

## Key Architectural Changes

### Terminology
| LBPManager (Legacy) | LBPAgent (AIXD) | Description |
|---------------------|-----------------|-------------|
| `study_code` | `schema_id` | Identifier for dataset structure |
| `study_params` | `static_params` | Static/study-level parameters |
| `exp_params` | `exp_params` | Dynamic/experiment-level parameters |
| `@study_parameter` | `@parameter_static` | Static parameter decorator |
| `@exp_parameter` | `@parameter_dynamic` | Dynamic parameter decorator |
| `@dim_parameter` | `@parameter_dimensional` | Dimensional parameter decorator |

### Folder Structure
```
LBPManager (study-based):
local/
  study_001/
    study_001_001/
      ...

LBPAgent (schema-based):
local/
  .lbp/
    schema_registry.json
  schema_001/
    schema.json
    schema_001_001/
      ...
```

## Migration Steps

### 1. Replace Imports

**Before (LBPManager):**
```python
from lbp_package import LBPManager
```

**After (LBPAgent):**
```python
from lbp_package.core import LBPAgent
```

### 2. Update Model Decorators (Optional)

While legacy decorators still work, update to AIXD terminology:

**Before:**
```python
from lbp_package.utils import study_parameter, exp_parameter, dim_parameter

@dataclass
class MyModel(IEvaluationModel):
    material: str = study_parameter()
    print_speed: float = exp_parameter()
    n_layers: int = dim_parameter()
```

**After:**
```python
from lbp_package.utils import parameter_static, parameter_dynamic, parameter_dimensional

@dataclass
class MyModel(IEvaluationModel):
    material: str = parameter_static()
    print_speed: float = parameter_dynamic()
    n_layers: int = parameter_dimensional()
```

**Note:** Legacy decorators (`@study_parameter`, etc.) remain available for backward compatibility.

### 3. Update Initialization

**Before (LBPManager):**
```python
manager = LBPManager(
    root_folder="/path/to/project",
    local_folder="/path/to/local",
    log_folder="/path/to/logs"
)

# Register models
manager.add_evaluation_model(
    performance_code="energy",
    evaluation_class=EnergyModel,
    weight=0.7
)

# Initialize for study
manager.initialize_for_study(
    study_code="study_001",
    debug_flag=True
)
```

**After (LBPAgent):**
```python
agent = LBPAgent(
    root_folder="/path/to/project",
    local_folder="/path/to/local",
    log_folder="/path/to/logs",
    debug_flag=True  # Now constructor parameter
)

# Register models (same API)
agent.add_evaluation_model(
    performance_code="energy",
    evaluation_class=EnergyModel,
    weight=0.7
)

# Initialize for dataset
dataset = agent.initialize_for_dataset(
    performance_codes=["energy"],  # Explicit list
    static_params={
        "material": "PLA",
        "printer_type": "FDM"
    }
)
```

### 4. Update Experiment Execution

**Before (LBPManager):**
```python
manager.run_evaluation(
    study_code="study_001",
    exp_nr=1,
    visualize_flag=True,
    debug_flag=True,
    recompute_flag=False
)
```

**After (LBPAgent):**
```python
exp_code = f"{dataset.schema_id}_001"

# Initialize experiment
agent.initialize_for_exp(
    exp_code=exp_code,
    exp_params={
        "print_speed": 50.0,
        "layer_height": 0.2,
        "n_layers": 100
    }
)

# Run evaluation
results = agent.evaluate_experiment(
    exp_code=exp_code,
    exp_params=exp_params,
    visualize=True,
    recompute=False
)
```

### 5. Update Data Persistence

**Before (LBPManager):**
```python
# Implicit hierarchical load/save via run_evaluation
manager.run_evaluation(study_code, exp_nr=1)
```

**After (LBPAgent):**
```python
# Explicit hierarchical operations
# Load
missing = agent.load_experiments_hierarchical(
    exp_codes=[exp_code],
    recompute=False
)

# Save
agent.save_experiments_hierarchical(
    exp_codes=[exp_code],
    recompute=False
)
```

### 6. Load Existing Dataset

**Before (LBPManager):**
```python
manager.initialize_for_study("study_001")
```

**After (LBPAgent):**
```python
# Load by schema_id (with compatibility check)
dataset = agent.initialize_for_dataset(
    schema_id="schema_001"
)
```

## Key Benefits of AIXD Architecture

1. **Deterministic Schema IDs**: Same model configuration always gets same `schema_id`
2. **Schema Validation**: Automatic compatibility checking when loading datasets
3. **Type Safety**: DataObjects provide runtime validation of parameters
4. **Explicit Dependencies**: Clear separation of static vs dynamic parameters
5. **Human-Readable Folders**: `schema_001` instead of opaque study codes

## Backward Compatibility

The following remain unchanged:
- Model registration API (`add_evaluation_model`, `add_prediction_model`)
- Model interfaces (`IEvaluationModel`, `IPredictionModel`, etc.)
- Legacy decorators (`@study_parameter`, `@exp_parameter`, `@dim_parameter`)
- LocalData methods (internally use `schema_id` but `study_code` still works)

## Complete Example Comparison

### LBPManager (Legacy)
```python
from lbp_package import LBPManager

manager = LBPManager(
    root_folder=".",
    local_folder="./local",
    log_folder="./logs"
)

manager.add_evaluation_model("energy", EnergyModel, weight=0.7)
manager.initialize_for_study("study_001")
manager.run_evaluation("study_001", exp_nr=1, visualize_flag=True)
```

### LBPAgent (AIXD)
```python
from lbp_package.core import LBPAgent

agent = LBPAgent(
    root_folder=".",
    local_folder="./local",
    log_folder="./logs",
    debug_flag=True
)

agent.add_evaluation_model("energy", EnergyModel, weight=0.7)

dataset = agent.initialize_for_dataset(
    performance_codes=["energy"],
    static_params={"material": "PLA"}
)

exp_code = f"{dataset.schema_id}_001"
agent.initialize_for_exp(exp_code, {"print_speed": 50, "n_layers": 100})
results = agent.evaluate_experiment(exp_code, exp_params, visualize=True)
```

## Troubleshooting

### Schema Compatibility Errors
If you get "Schema incompatibility" errors:
1. Check that all registered models match the stored schema
2. Verify parameter names and types haven't changed
3. Use `schema.is_compatible_with()` for detailed error messages

### Missing Schema Registry
If loading fails with "Schema not found":
1. Ensure `.lbp/schema_registry.json` exists in local folder
2. Check schema_id matches registry entries
3. Registry is auto-created on first `initialize_for_dataset()`

### Parameter Validation Errors
If parameters fail validation:
1. Check DataObject constraints (min/max for DataReal, allowed values for DataCategorical)
2. Ensure dimensional params are positive integers
3. Verify all required static/dynamic params provided

## Getting Help

For issues during migration:
1. Check examples in `examples/aixd_example.py`
2. Review AIXD architecture in `src/lbp_package/core/`
3. Legacy `LBPManager` remains available for gradual migration
