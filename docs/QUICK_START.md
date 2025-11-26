# Quick Start Guide

**Last Updated**: November 25, 2025  
**Status**: Phase 7 API (Dataset-Centric Architecture)

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

### 1. Define Your Models

Create evaluation and feature models by implementing the interfaces:

```python
from lbp_package.interfaces import IEvaluationModel, IFeatureModel
from lbp_package.core import Parameter, Performance, Dimension, Dataset
from lbp_package.utils import LBPLogger
from dataclasses import dataclass, field
from typing import Any

@dataclass
class EnergyFeatureModel(IFeatureModel):
    """Compute energy consumption features."""
    
    dataset: Dataset  # Required for memoization
    logger: LBPLogger  # Required for logging
    
    def _load_data(self, **param_values) -> Any:
        """Load raw data for these parameters."""
        # Use param_values to locate/load unstructured data
        # (CAD files, sensor data, etc.)
        return raw_data  # Your custom data format
    
    def _compute_features(self, data: Any) -> float:
        """Extract feature value from loaded data."""
        # Your feature computation here
        return 123.45

@dataclass
class EnergyEvaluationModel(IEvaluationModel):
    """Evaluate energy consumption."""
    
    # Declare parameters as dataclass fields
    print_speed: Parameter = field(
        default_factory=lambda: Parameter.real(min_val=10.0, max_val=200.0, round_digits=2)
    )
    layer_height: Parameter = field(
        default_factory=lambda: Parameter.real(min_val=0.1, max_val=0.4, round_digits=3)
    )
    n_layers: Dimension = field(
        default_factory=lambda: Dimension.integer(
            param_name="n_layers",
            dim_name="layers", 
            iterator_name="layer_idx",
            min_val=1,
            max_val=1000
        )
    )
    
    @property
    def feature_model_type(self):
        return EnergyFeatureModel
    
    @property
    def dim_names(self):
        return ["layers"]
    
    @property
    def dim_param_names(self):
        return ["n_layers"]
    
    @property
    def dim_iterator_names(self):
        return ["layer_idx"]
    
    @property
    def target_value(self):
        return 0.0  # Minimize energy
    
    @property
    def scaling_factor(self):
        return 1.0
    
    def evaluate(self, dataset, exp_code, indices, features):
        # Your evaluation logic here
        return 100.0  # Energy value
```

---

### 2. Initialize Agent

```python
from lbp_package.core import LBPAgent

# Create agent (no round_digits - that's in DataObjects now)
agent = LBPAgent(
    root_folder="/path/to/project",
    local_folder="/path/to/local",
    log_folder="/path/to/logs",
    debug_flag=True  # Skip external database operations
)
```

---

### 3. Register Models

```python
# Register evaluation models (no weight, no round_digits)
agent.register_evaluation_model(
    performance_code="energy_consumption",
    evaluation_model_class=EnergyEvaluationModel
)

agent.register_evaluation_model(
    performance_code="path_deviation",
    evaluation_model_class=PathDeviationEvaluationModel
)

# Register prediction models (optional)
agent.register_prediction_model(
    performance_code="energy_consumption",
    prediction_model_class=EnergyPredictionModel
)
```

**Key Changes in Phase 7:**
- Use `register_evaluation_model()` (not `add_evaluation_model`)
- No `weight` parameter (that's for calibration, not evaluation)
- No `round_digits` parameter (configured in DataObjects)

---

### 4. Initialize Dataset

```python
# Define static parameters (study-level, shared across experiments)
static_params = {
    "material": "PLA",
    "printer_type": "FDM",
    "nozzle_diameter": 0.4
}

# Agent returns Dataset - user owns it
dataset = agent.initialize(static_params=static_params)

print(f"Dataset initialized: {dataset.schema_id}")
print(f"Parameters: {len(dataset.schema.parameters.data_objects)}")
print(f"Dimensions: {len(dataset.schema.dimensions.data_objects)}")
```

**Key Changes in Phase 7:**
- `initialize()` returns Dataset without storing it
- User owns and manages the dataset
- No more `schema_id` parameter (use `populate()` to load existing)

---

### 5. Run Experiments

```python
# Define experiment parameters
exp_params = {
    "print_speed": 50.0,      # Dynamic parameter
    "layer_height": 0.2,      # Dynamic parameter
    "n_layers": 100           # Dimensional parameter
}

exp_code = "test_001"

# Add experiment to dataset (hierarchical: tries to load first, then creates)
exp_data = dataset.add_experiment(exp_code, exp_params)

print(f"Experiment {exp_code} created")

# Evaluate experiment - mutates exp_data in place
agent.evaluate_experiment(
    dataset=dataset,
    exp_data=exp_data,
    visualize=True,
    recompute=False
)

print(f"Experiment {exp_code} evaluated")

# Access results from exp_data
for perf_code in exp_data.performance:
    metrics = exp_data.performance.get_value(perf_code)
    print(f"{perf_code}: {metrics}")
```

**Key Changes in Phase 7:**
- Add experiments via `dataset.add_experiment()` (not `agent.add_experiment()`)
- `evaluate_experiment()` takes `dataset` and `exp_data` parameters
- Results stored in `exp_data` (mutated in place), not returned
- No more initialization step before evaluation

---

### 6. Save Data

```python
# Save single experiment to local storage
dataset.save_experiment(exp_code, local=True, external=False, recompute=False)

print(f"Saved {exp_code} to local storage")

# Or save all experiments at once
dataset.save(local=True, external=False, recompute=False)

print(f"Saved all experiments + schema")
```

**Key Changes in Phase 7:**
- Save via `dataset.save()` or `dataset.save_experiment()` (not `agent.save_experiments_hierarchical()`)
- Clear control over local vs external storage

---

### 7. Load Existing Data

```python
# In a new session, create agent and dataset
agent2 = LBPAgent(
    root_folder="/path/to/project",
    local_folder="/path/to/local",
    log_folder="/path/to/logs",
    debug_flag=True
)

# Register same models (required for schema compatibility)
agent2.register_evaluation_model(
    performance_code="energy_consumption",
    evaluation_model_class=EnergyEvaluationModel
)

# Initialize with same static params (creates same schema)
dataset2 = agent2.initialize(static_params=static_params)

# Load all experiments from local storage
dataset2.populate(source="local")

print(f"Loaded dataset: {dataset2.schema_id}")

# Check what was loaded
all_exp_codes = dataset2.get_experiment_codes()
print(f"Found {len(all_exp_codes)} experiments")

for code in all_exp_codes:
    params = dataset2.get_experiment_params(code)
    print(f"{code}: {params}")
```

**Key Changes in Phase 7:**
- Use `dataset.populate()` to load all experiments (not `agent.load_existing_dataset()`)
- No `schema_id` parameter in `initialize()` - it's generated from models
- Hierarchical loading built into `add_experiment()` and `populate()`

---

### 8. Train Prediction Models (Optional)

If you have prediction models registered, you can train and validate them:

```python
from lbp_package.core import DataModule

# Configure DataModule with train/val/test splits
datamodule = DataModule(
    dataset,
    test_size=0.2,      # 20% for test set
    val_size=0.1,       # 10% of remaining for validation
    random_seed=42,     # Reproducible splits
    normalize='standard'  # Z-score normalization
)

# Check split sizes
sizes = datamodule.get_split_sizes()
print(f"Train: {sizes['train']}, Val: {sizes['val']}, Test: {sizes['test']}")

# Train models on training set
agent.train(datamodule, epochs=100, learning_rate=0.001)

# Validate on validation set
val_results = agent.validate(use_test=False)
for feature, metrics in val_results.items():
    print(f"{feature}: MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")

# Final evaluation on test set
test_results = agent.validate(use_test=True)
for feature, metrics in test_results.items():
    print(f"{feature} (test): MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")

# Make predictions on new data
import pandas as pd
X_new = pd.DataFrame({
    "print_speed": [50.0, 75.0],
    "layer_height": [0.2, 0.3],
    "n_layers": [100, 150]
})

predictions = agent.predict(X_new)
print(predictions)  # DataFrame with predicted features (auto-denormalized)
```

**Split Configuration:**
- `test_size=0.0, val_size=0.0` - Use all data for training (no validation)
- `test_size=0.2, val_size=0.0` - 80/20 train/test split (no validation)
- `test_size=0.2, val_size=0.1` - Approximately 70/10/20 train/val/test

**Validation Metrics:**
- `mae`: Mean Absolute Error
- `rmse`: Root Mean Squared Error
- `r2`: Coefficient of Determination (R²)
- `n_samples`: Number of samples in the split

---

## Alternative: Load Specific Experiments

```python
# Load (or create) specific experiment with hierarchical pattern
exp_data = dataset.add_experiment("test_002")  # No params = try to load

# Hierarchical loading pattern:
# 1. Check memory (already loaded?)
# 2. Try local storage
# 3. Try external storage
# 4. Create new if not found

if exp_data.parameters.has_value("print_speed"):
    print(f"Loaded existing experiment")
else:
    print(f"Created new experiment")
    # Set parameters...
    exp_data.parameters.set_value("print_speed", 60.0)
```

---

## Complete Example

See `examples/aixd_example.py` for a complete working example demonstrating:
- Model definition with dataclass fields
- Agent initialization
- Dataset creation and ownership
- Experiment evaluation with mutation pattern
- Hierarchical save/load operations

---

## Key Concepts

### 1. Dataset-Centric Architecture

**You own the dataset**, not the agent. Agent is a stateless orchestration layer.

```python
dataset = agent.initialize(...)  # Agent returns dataset, doesn't store it
# You manage dataset lifecycle
```

### 2. Hierarchical Load/Save Pattern

Pattern: **Memory → Local → External → Compute**

```python
# Try to load first, create if not found
exp_data = dataset.add_experiment("test_001")  # Smart loading

# Or explicitly create new
exp_data = dataset.add_experiment_manual("test_001", {...})  # Always create
```

### 3. Mutation Pattern

`evaluate_experiment()` mutates `exp_data` in place, doesn't return results.

```python
agent.evaluate_experiment(dataset, exp_data)  # Mutates exp_data
# Results now in exp_data.performance
```

### 4. Declarative Rounding

Configure rounding in DataObjects, not in agent or models.

```python
# In model definition
print_speed: Parameter = field(
    default_factory=lambda: Parameter.real(min_val=10, max_val=200, round_digits=2)
)

# Or in schema
schema = DatasetSchema(default_round_digits=3)
```

### 5. Schema Generation

Schema is generated from model dataclass fields, not decorators.

```python
@dataclass
class MyModel(IEvaluationModel):
    # These fields become parameters in schema
    param1: Parameter = field(default_factory=lambda: Parameter.real(...))
    param2: Parameter = field(default_factory=lambda: Parameter.integer(...))
```

---

## Common Operations

### Access Experiment Data

```python
# Get all experiment codes
codes = dataset.get_experiment_codes()

# Get parameters for experiment
params = dataset.get_experiment_params("test_001")

# Get complete ExperimentData
exp_data = dataset.get_experiment_data("test_001")

# Access values
speed = exp_data.parameters.get_value("print_speed")
energy = exp_data.performance.get_value("energy_consumption")
```

### Check if Experiment Exists

```python
if "test_001" in dataset.get_experiment_codes():
    print("Experiment exists in memory")

# Or try to load
exp_data = dataset.add_experiment("test_001")  # Loads if exists, creates if not
```

### Iterate Over Experiments

```python
for exp_code in dataset.get_experiment_codes():
    exp_data = dataset.get_experiment_data(exp_code)
    params = {name: exp_data.parameters.get_value(name) 
              for name in exp_data.parameters.data_objects.keys()}
    print(f"{exp_code}: {params}")
```

---

## Key API Patterns

**Important**: This guide shows the current Phase 7 API. If you see references to old patterns like `agent.add_experiment()` or `agent.save_experiments_hierarchical()`, those are outdated.

**Current patterns:**
- ✅ `dataset.add_experiment()` - experiments belong to dataset
- ✅ `agent.evaluate_experiment(dataset, exp_data)` - explicit dataset parameter, mutates exp_data
- ✅ `dataset.save()` or `dataset.save_experiment()` - dataset handles persistence
- ✅ `dataset.populate()` - dataset loads its own data
- ✅ No `weight` or `round_digits` in model registration

---

## Next Steps

1. **Define your models**: Implement `IEvaluationModel` and `IFeatureModel`
2. **Set up agent**: Create and configure `LBPAgent`
3. **Register models**: Call `register_evaluation_model()` for each metric
4. **Create dataset**: Call `agent.initialize()` with static params
5. **Run experiments**: Add experiments and evaluate them
6. **Save results**: Use `dataset.save()` for persistence
7. **Load data**: Use `dataset.populate()` to reload

For detailed architecture information, see:
- `docs/SEPARATION_OF_CONCERNS.md` - Component responsibilities
- `docs/CORE_DATA_STRUCTURES.md` - Data model details
- `docs/IMPLEMENTATION_SUMMARY.md` - Full implementation history
