# Core Data Structures

**Last Updated**: November 26, 2025  
**Status**: Current (Architecture Fix - DataObject Value Storage)

---

## Overview

The LBP package uses a hierarchical type system with validation and persistence. Data flows from atomic typed values (`DataObjects`) through collections (`DataBlocks`) up to complete experiments (`ExperimentData`) and datasets (`Dataset`).

```
Dataset
  └── ExperimentData (per experiment)
       ├── parameters: DataBlock
       ├── performance: DataBlock
       └── metric_arrays: DataBlock
            └── DataObjects (typed values with constraints)
```

---

## 1. DataObjects - Typed Primitives

**File**: `src/lbp_package/core/data_objects.py`

### Base Class

```python
class DataObject(ABC):
    """Abstract base for typed data with validation.
    
    Note: DataObject instances are immutable schema templates that define
    types and validation rules. They do NOT store values - values are
    stored in DataBlock.values dictionary. This separation ensures that
    shared DataObject instances (e.g., in DatasetSchema) don't cause
    data corruption across experiments.
    """
    
    def __init__(self, name: str, round_digits: Optional[int] = None):
        self.name = name
        self.round_digits = round_digits
        # No self.value attribute - values stored in DataBlock!
    
    @abstractmethod
    def validate(self, value: Any) -> None:
        """Raise ValueError if invalid."""
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
    
    @classmethod
    def from_dict(cls, name: str, data: Dict) -> 'DataObject':
        """Deserialize from dictionary."""
```

### Concrete Types

#### DataReal - Float Values

```python
DataReal(
    name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    round_digits: Optional[int] = None
)
```

**Features:**
- Optional min/max constraints
- Automatic rounding via `DataBlock.set_value()` if `round_digits` is set
- Preserves full precision if `round_digits=None`

**Example:**
```python
from lbp_package.core import DataBlock, DataReal

# Create schema template
energy = DataReal("energy_consumption", min_value=0.0, max_value=1000.0, round_digits=3)

# Store value in DataBlock (not in DataObject!)
block = DataBlock()
block.add("energy_consumption", energy)
block.set_value("energy_consumption", 123.456789)  # Stores 123.457 (rounded)

value = block.get_value("energy_consumption")  # Returns 123.457
```

#### Other Types

- **DataInt**: Integer values with optional min/max
- **DataBool**: Boolean values
- **DataCategorical**: Enumerated values from allowed list
- **DataString**: Arbitrary strings
- **DataArray**: Numpy arrays with optional shape/dtype validation

---

### Factory Classes

```python
from lbp_package.core import Parameter, Performance, Dimension

# Parameter factory
speed = Parameter.real(min_val=10.0, max_val=200.0, round_digits=2)
count = Parameter.integer(min_val=1, max_val=100)
material = Parameter.categorical(categories=["PLA", "ABS"])

# Performance factory
energy = Performance.real(min_val=0.0, max_val=1000.0, round_digits=3)

# Dimension factory
layers_dim = Dimension.integer(
    param_name="n_layers",
    dim_name="layers",
    iterator_name="layer_idx",
    min_val=1,
    max_val=1000
)
```

---

## 2. DataBlocks - Value Collections

**File**: `src/lbp_package/core/data_blocks.py`

### Base Class

```python
class DataBlock:
    """Container for related DataObjects with values.
    
    Architecture:
    - data_objects: Dict[str, DataObject] - Schema templates (shared/reusable)
    - values: Dict[str, Any] - Actual data values (per-experiment)
    
    This separation is critical: DataObject instances can be shared across
    experiments (e.g., stored in DatasetSchema), while each DataBlock owns
    its own values dictionary. This prevents data corruption.
    """
    
    def __init__(self):
        self.data_objects: Dict[str, DataObject] = {}
        self.values: Dict[str, Any] = {}
    
    def add(self, name: str, data_obj: DataObject) -> None:
        """Register a DataObject schema template."""
    
    def set_value(self, name: str, value: Any) -> None:
        """Validate and store a value.
        
        - Validates value against data_obj.validate()
        - Applies rounding if data_obj.round_digits is set
        - Stores in self.values (NOT in DataObject!)
        """
    
    def get_value(self, name: str) -> Any:
        """Retrieve a value from self.values."""
    
    def has_value(self, name: str) -> bool:
        """Check if value is set in self.values."""
    
    def to_numpy(self) -> np.ndarray:
        """Convert all values to numpy array."""
    
    def to_dict(self) -> Dict:
        """Serialize block + values."""
```

### Specialized Blocks

- **Parameters**: All experiment parameters (unified block)
- **Dimensions**: Dimensional metadata (name, param, iterator)
- **PerformanceAttributes**: Performance metrics
- **MetricArrays**: Array-valued metrics

### Usage

```python
from lbp_package.core import Parameters, Parameter

params = Parameters()
params.add("speed", Parameter.real(min_val=10.0, max_val=200.0))
params.set_value("speed", 50.0)  # Validates and stores in params.values
speed = params.get_value("speed")  # Returns 50.0
```

---

## 3. DatasetSchema - Structure Definition

**File**: `src/lbp_package/core/schema.py`

### Structure

```python
class DatasetSchema:
    """Defines dataset structure with deterministic hashing."""
    
    def __init__(self, default_round_digits: int = 3):
        self.parameters = Parameters()
        self.dimensions = Dimensions()
        self.performance_attrs = PerformanceAttributes()
        self.metric_arrays = MetricArrays()
        self.default_round_digits = default_round_digits
        self._schema_hash: Optional[str] = None
        self._schema_id: Optional[str] = None
```

### Key Methods

#### Hash Computation

```python
def _compute_schema_hash(self) -> str:
    """Compute SHA256 hash of schema structure (types, constraints, names).
    Deterministic and collision-resistant. Excludes values, IDs, timestamps.
    """
```

#### Compatibility Check

```python
def is_compatible_with(self, other: 'DatasetSchema') -> None:
    """Raises ValueError if schemas incompatible (missing params, type/constraint mismatches)."""
```

---

## 4. SchemaRegistry - Hash-to-ID Mapping

**File**: `src/lbp_package/core/schema_registry.py`

Maps schema hashes to human-readable IDs (schema_001, schema_002, ...) with auto-increment.

**Storage**: `{local_folder}/.lbp/schema_registry.json`

```python
registry = SchemaRegistry("/path/to/local")
schema_id = registry.get_or_create_schema_id(schema_hash, schema.to_dict())
# Returns "schema_001" for new hash, or existing ID for known hash
```

---

## 5. ExperimentData - Single Experiment

**File**: `src/lbp_package/core/dataset.py`

```python
@dataclass
class ExperimentData:
    exp_code: str
    parameters: DataBlock
    performance: DataBlock
    metric_arrays: DataBlock
```

---

## 6. Dataset - Experiment Collection

**File**: `src/lbp_package/core/dataset.py`

```python
class Dataset:
    def __init__(self, schema: DatasetSchema, local_data=None, external_data=None):
        self.schema = schema
        self.experiments: Dict[str, ExperimentData] = {}
```

### Key Methods

- **add_experiment(exp_code, exp_params=None)**: Hierarchical load (Memory → Local → External → Create)
- **add_experiment_manual(exp_code, exp_params)**: Always create new
- **populate(source="local")**: Load all experiments from storage
- **save(local=True, external=False)**: Persist experiments and schema
- **save_experiment(exp_code)**: Save single experiment

### Hierarchical Load Pattern

```
1. Check memory (already loaded?)
2. Try local storage
3. Try external storage
4. Create new if not found
```

---

## Data Flows

```python
def save(
    self,
    local: bool = True,
    external: bool = False,
### Creation Flow

```
DataObjects → DataBlocks → DatasetSchema → SchemaRegistry → Dataset → ExperimentData → Save
```

### Loading Flow

```
Load schema → Create Dataset → populate() → Hierarchical load (Memory → Local → External → Create)
```

---

## Serialization Examples

### DataObject (JSON)

```json
{
  "type": "DataReal",
  "constraints": {"min_value": 10.0, "max_value": 200.0},
  "round_digits": 3
}
```

### Experiment Record (JSON)

Stored: `{local_folder}/{schema_id}/{exp_code}/exp_record.json`

```json
{
  "exp_code": "test_001",
  "parameters": {"speed": 50.0, "material": "PLA", "n_layers": 100}
}
```

---

## Validation

All validation happens at data entry via `DataObject.validate()`:
- Type checks
- Constraint checks (min/max, allowed values)
- Raises `ValueError` if invalid
