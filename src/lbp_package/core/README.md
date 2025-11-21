# AIXD Architecture - Core Module

This module implements the AIXD (AI-Driven Experimental Design) architecture for the LBP package, providing a dataset-centric approach to managing learning-by-printing experiments.

## Architecture Overview

The AIXD architecture introduces structured data objects, schema validation, and deterministic dataset management:

```
┌─────────────────────────────────────────────────────────────┐
│                         LBPAgent                             │
│  (Orchestration: Model registration, initialization, eval)  │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─── Dataset ────┬─── DatasetSchema
             │                │     ├── ParametersStatic
             │                │     ├── ParametersDynamic
             │                │     ├── ParametersDimensional
             │                │     └── PerformanceAttributes
             │                │
             │                └─── Data Storage
             │                      ├── _aggr_metrics
             │                      ├── _metric_arrays
             │                      └── _exp_records
             │
             ├─── SchemaRegistry (hash → schema_id mapping)
             │
             └─── DataObjects (typed variables with validation)
                   ├── DataReal (float with min/max)
                   ├── DataInt (int with min/max)
                   ├── DataBool
                   ├── DataCategorical (with allowed values)
                   ├── DataString
                   └── DataDimension (three-aspect: name/param/iterator)
```

## Components

### 1. DataObjects (`data_objects.py`)
Type system for parameter definitions with runtime validation.

**Classes:**
- `DataObject` - Abstract base with validation interface
- `DataReal` - Float values with optional min/max constraints
- `DataInt` - Integer values with optional min/max constraints
- `DataBool` - Boolean values
- `DataCategorical` - String values from allowed set
- `DataString` - Arbitrary string values
- `DataDimension` - Dimensional parameters (name, param_name, iterator_name)

**Example:**
```python
from lbp_package.core import DataReal, DataInt, DataCategorical

# Define typed parameters
speed = DataReal("print_speed", min_value=10.0, max_value=200.0)
layers = DataInt("n_layers", min_value=1)
material = DataCategorical("material", allowed_values=["PLA", "ABS", "PETG"])

# Validate values
speed.validate(50.0)  # OK
speed.validate(250.0)  # Raises ValueError: exceeds max_value
```

### 2. DataBlocks (`data_blocks.py`)
Collections organizing related DataObjects.

**Classes:**
- `DataBlock` - Base container with validation
- `ParametersStatic` - Study-level fixed parameters
- `ParametersDynamic` - Per-experiment parameters
- `ParametersDimensional` - Iteration structure definition
- `PerformanceAttributes` - Performance metrics with optional weights

**Example:**
```python
from lbp_package.core import ParametersStatic, DataReal

static = ParametersStatic()
static.add("nozzle_diameter", DataReal("nozzle_diameter", min_value=0.1))
static.add("material", DataCategorical("material", allowed_values=["PLA", "ABS"]))

# Validate all at once
static.validate_all({"nozzle_diameter": 0.4, "material": "PLA"})
```

### 3. DatasetSchema (`schema.py`)
Structure definition with deterministic hashing and compatibility checking.

**Features:**
- **Hash Computation**: SHA256 of types + constraints (collision-resistant)
- **Compatibility Checking**: Structural comparison with detailed error messages
- **Serialization**: JSON export/import for persistence

**Example:**
```python
from lbp_package.core import DatasetSchema, ParametersStatic, DataReal

schema = DatasetSchema()
schema.static_params.add("material", DataCategorical("material", ["PLA", "ABS"]))
schema.dynamic_params.add("speed", DataReal("print_speed", min_value=10))

# Compute deterministic hash
hash_value = schema._compute_schema_hash()  # Always same for same structure

# Check compatibility
other_schema = DatasetSchema()
# ... define other_schema ...
try:
    schema.is_compatible_with(other_schema)
    print("Schemas compatible!")
except ValueError as e:
    print(f"Incompatible: {e}")
```

### 4. SchemaRegistry (`schema_registry.py`)
Deterministic mapping between schema hashes and human-readable IDs.

**Storage:** `{local_folder}/.lbp/schema_registry.json`

**Format:**
```json
{
  "abc123...": {
    "schema_id": "schema_001",
    "created": "2025-01-15T10:30:00",
    "structure": { ... }
  }
}
```

**Features:**
- Auto-increment schema IDs (`schema_001`, `schema_002`, ...)
- Deterministic: same hash always returns same ID
- Import/export for registry transfer

**Example:**
```python
from lbp_package.core import SchemaRegistry, DatasetSchema

registry = SchemaRegistry("/path/to/local")

# Get or create ID
schema = DatasetSchema()
# ... configure schema ...
schema_hash = schema._compute_schema_hash()
schema_id = registry.get_or_create_schema_id(schema_hash, schema.to_dict())
# Returns "schema_001" on first call, same ID on subsequent calls

# List all schemas
schemas = registry.list_schemas()
for sid, info in schemas:
    print(f"{sid}: created {info['created']}")
```

### 5. Dataset (`dataset.py`)
Container with master storage and comprehensive validation.

**Storage:**
- `_aggr_metrics`: Experiment → Performance → Metrics
- `_metric_arrays`: Performance → Experiment → Array
- `_exp_records`: Experiment → Record
- `_static_values`: Static parameter values

**Validation:**
- Required parameters present
- Dimensional params are positive integers
- Type validation via DataObjects
- Performance codes match schema

**Example:**
```python
from lbp_package.core import Dataset, DatasetSchema

dataset = Dataset(name="my_dataset", schema=schema, schema_id="schema_001")

# Set static values
dataset.set_static_values({"material": "PLA", "nozzle_diameter": 0.4})

# Add experiment (with validation)
dataset.add_experiment(
    exp_code="schema_001_001",
    exp_params={"print_speed": 50.0, "n_layers": 100},
    aggr_metrics={"energy": {"Performance_Avg": 123.45}}
)

# Access data
codes = dataset.get_experiment_codes()
params = dataset.get_experiment_params("schema_001_001")
```

### 6. LBPAgent (`agent.py`)
Main orchestration class integrating AIXD architecture with existing systems.

**Key Methods:**

**Initialization:**
```python
from lbp_package.core import LBPAgent

agent = LBPAgent(
    root_folder=".",
    local_folder="./local",
    log_folder="./logs",
    debug_flag=True
)

# Register models
agent.add_evaluation_model("energy", EnergyModel, weight=0.7)

# Initialize new dataset
dataset = agent.initialize_for_dataset(
    performance_codes=["energy"],
    static_params={"material": "PLA"}
)

# OR load existing dataset
dataset = agent.initialize_for_dataset(schema_id="schema_001")
```

**Experiment Execution:**
```python
exp_code = f"{dataset.schema_id}_001"

# Initialize experiment
agent.initialize_for_exp(exp_code, {"print_speed": 50, "n_layers": 100})

# Evaluate
results = agent.evaluate_experiment(
    exp_code=exp_code,
    exp_params=exp_params,
    visualize=True
)
```

**Hierarchical Load/Save:**
```python
# Load: Memory → Local → External
missing = agent.load_experiments_hierarchical([exp_code])

# Save: Memory → Local → External
agent.save_experiments_hierarchical([exp_code])
```

## Two-Phase Initialization

Resolves circular dependency between Dataset and EvaluationSystem:

**Phase 1: Model Setup**
1. Activate models for performance codes
2. Initialize feature models with static params
3. Generate schema from active models

**Phase 2: Dataset Setup**
4. Get/create schema_id via registry
5. Create dataset instance
6. Attach storage to systems
7. Initialize evaluation system

## Schema Generation

Schema is automatically generated from active models via introspection:

```python
# Agent inspects models using ParameterHandling helpers
static_params = model.get_param_names_by_type('model')
dynamic_params = model.get_param_names_by_type('experiment')

# Infers DataObject types from field annotations
field = model.get_param_field('print_speed')
if field.type == float:
    data_obj = DataReal('print_speed')

# Dimensional params from model.dim_param_names
# Performance attrs from model.performance_code
```

## File Structure

```
local/
├── .lbp/
│   └── schema_registry.json      # Hash → schema_id mapping
├── schema_001/
│   ├── schema.json               # Schema definition (future)
│   ├── schema_001_001/           # Experiment folder
│   │   ├── schema_001_001_record.json
│   │   ├── schema_001_001_energy_metrics.json
│   │   └── ...
│   └── schema_001_002/
│       └── ...
└── schema_002/
    └── ...
```

## Migration from LBPManager

See `MIGRATION_GUIDE.md` for detailed migration steps.

**Quick comparison:**
```python
# Legacy (LBPManager)
manager.initialize_for_study("study_001")
manager.run_evaluation("study_001", exp_nr=1)

# AIXD (LBPAgent)
dataset = agent.initialize_for_dataset(
    performance_codes=["energy"],
    static_params={"material": "PLA"}
)
agent.initialize_for_exp(exp_code, exp_params)
results = agent.evaluate_experiment(exp_code, exp_params)
```

## Backward Compatibility

- Legacy decorators (`@study_parameter`, etc.) still work
- `LBPManager` remains available
- `LocalData` supports both `study_code` and `schema_id`

## Future Extensions

1. **Schema Versioning**: Track schema evolution over time
2. **Constraint Inheritance**: DataObjects inherit from parent blocks
3. **Multi-Dataset Operations**: Compare across schemas
4. **Distributed Registry**: Sync schemas across teams
