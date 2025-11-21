# AIXD Quick Start Guide

Get started with the AIXD architecture in 5 minutes.

## Installation

```bash
# Clone and checkout dataObjects branch
git clone <repo-url>
cd lbp_package
git checkout dataObjects

# Install package
pip install -e .
```

## Basic Workflow

### 1. Initialize Agent

```python
from lbp_package.core import LBPAgent

agent = LBPAgent(
    root_folder=".",
    local_folder="./local",
    log_folder="./logs",
    debug_flag=True  # Skip external database operations
)
```

### 2. Register Models

```python
from your_models import EnergyModel, PathDeviationModel

agent.add_evaluation_model(
    performance_code="energy",
    evaluation_class=EnergyModel,
    weight=0.7  # For multi-objective calibration
)

agent.add_evaluation_model(
    performance_code="path_deviation",
    evaluation_class=PathDeviationModel,
    weight=0.3
)
```

### 3. Initialize Dataset

```python
# For NEW dataset
dataset = agent.initialize_for_dataset(
    performance_codes=["energy", "path_deviation"],
    static_params={
        "material": "PLA",
        "printer_type": "FDM",
        "nozzle_diameter": 0.4
    }
)

# For EXISTING dataset
dataset = agent.initialize_for_dataset(
    schema_id="schema_001"  # Load by ID
)
```

### 4. Run Experiments

```python
# Define experiment parameters
exp_params = {
    "print_speed": 50.0,
    "layer_height": 0.2,
    "infill_density": 20,
    "n_layers": 100  # Dimensional parameter
}

exp_code = f"{dataset.schema_id}_001"

# Initialize and evaluate
agent.initialize_for_exp(exp_code, exp_params)
results = agent.evaluate_experiment(
    exp_code=exp_code,
    exp_params=exp_params,
    visualize=True,
    recompute=False
)

print(f"Results: {results}")
```

### 5. Save & Load Data

```python
# Save experiments to local files
agent.save_experiments_hierarchical([exp_code])

# Later: load experiments
missing = agent.load_experiments_hierarchical([exp_code])
if not missing:
    print("All experiments loaded!")
```

## Complete Example

```python
from lbp_package.core import LBPAgent
from your_models import EnergyModel

# Setup
agent = LBPAgent(".", "./local", "./logs", debug_flag=True)
agent.add_evaluation_model("energy", EnergyModel, weight=1.0)

# Initialize dataset
dataset = agent.initialize_for_dataset(
    performance_codes=["energy"],
    static_params={"material": "PLA"}
)

# Run experiment
exp_code = f"{dataset.schema_id}_001"
exp_params = {"print_speed": 50.0, "n_layers": 100}

agent.initialize_for_exp(exp_code, exp_params)
results = agent.evaluate_experiment(exp_code, exp_params, visualize=True)

# Save
agent.save_experiments_hierarchical([exp_code])

print(f"✓ Experiment {exp_code} complete with results: {results}")
```

## Key Concepts

### Schema ID
- Auto-generated identifier (e.g., `schema_001`)
- Deterministic: same model configuration → same ID
- Human-readable folder names

### Static vs Dynamic Parameters
- **Static**: Study-level, constant across experiments (e.g., material)
- **Dynamic**: Vary per experiment (e.g., print_speed)
- **Dimensional**: Define iteration structure (e.g., n_layers)

### Two-Phase Initialization
1. **Phase 1**: Activate models → Generate schema
2. **Phase 2**: Create dataset → Attach storage

### Hierarchical Operations
Data flows through three tiers:
1. **Memory**: Fast access, volatile
2. **Local**: JSON files, persistent
3. **External**: Database, shared

## Common Tasks

### Check Dataset Contents
```python
exp_codes = dataset.get_experiment_codes()
for code in exp_codes:
    params = dataset.get_experiment_params(code)
    print(f"{code}: {params}")
```

### Multiple Experiments
```python
for i in range(10):
    exp_code = f"{dataset.schema_id}_{i+1:03d}"
    exp_params = {"print_speed": 10 + i*5, "n_layers": 100}
    
    agent.initialize_for_exp(exp_code, exp_params)
    agent.evaluate_experiment(exp_code, exp_params)

# Batch save
all_codes = [f"{dataset.schema_id}_{i+1:03d}" for i in range(10)]
agent.save_experiments_hierarchical(all_codes)
```

### Schema Compatibility
```python
try:
    dataset = agent.initialize_for_dataset(schema_id="schema_001")
    print("✓ Schema compatible with active models")
except ValueError as e:
    print(f"✗ Incompatible: {e}")
```

## Model Updates (AIXD Decorators)

Update your models to use AIXD terminology (optional - legacy works):

```python
from dataclasses import dataclass
from lbp_package.interfaces import IEvaluationModel
from lbp_package.utils import parameter_static, parameter_dynamic, parameter_dimensional

@dataclass
class MyModel(IEvaluationModel):
    # Static parameters
    material: str = parameter_static()
    printer_type: str = parameter_static()
    
    # Dynamic parameters
    print_speed: float = parameter_dynamic()
    layer_height: float = parameter_dynamic()
    
    # Dimensional parameters
    n_layers: int = parameter_dimensional()
    
    # Rest of model implementation...
```

## File Structure
```
your_project/
├── local/
│   ├── .lbp/
│   │   └── schema_registry.json    # Schema ID registry
│   └── schema_001/                  # Dataset folder
│       ├── schema_001_001/          # Experiment folders
│       │   ├── schema_001_001_record.json
│       │   └── schema_001_001_energy_metrics.json
│       └── schema_001_002/
│           └── ...
└── logs/
    └── LBPAgent_session_*.log
```

## Troubleshooting

### "Schema not found in registry"
- Registry created on first `initialize_for_dataset()`
- Check `.lbp/schema_registry.json` exists in local folder

### "Missing required parameters"
- Ensure all static params provided in `initialize_for_dataset()`
- Ensure all dynamic params provided in experiment

### "Schema incompatible"
- Models registered must match stored schema structure
- Check error message for specific parameter mismatches

### Performance Issues
- Use batch operations (`save_experiments_hierarchical` with multiple codes)
- Enable caching for repeated schema lookups
- Use `recompute=False` to skip re-evaluation

## Next Steps

1. **Read Full Docs**: See `src/lbp_package/core/README.md`
2. **Migration Guide**: See `MIGRATION_GUIDE.md` for LBPManager migration
3. **Example Code**: See `examples/aixd_example.py`
4. **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`

## Getting Help

- Check existing issues on GitHub
- Review error messages (they include context)
- Consult API documentation in module docstrings
- Ask in discussions or open an issue

---

**You're ready to use AIXD architecture!** 🚀
