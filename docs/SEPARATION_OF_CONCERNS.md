# Architecture: Separation of Concerns

**Last Updated**: November 25, 2025  
**Status**: Current (Phase 7 Complete)

---

## Core Principle

**Single Responsibility**: Each component has ONE clear purpose and handles it completely. No two components should share the same responsibility.

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────┐
│                    Application                       │
│                    (LBPAgent)                        │
│              Stateless Orchestration                 │
└─────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────┐
│                   Orchestration                      │
│        (EvaluationSystem, PredictionSystem)         │
│            Manage Model Execution                    │
└─────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────┐
│                    Interfaces                        │
│  (IEvaluationModel, IFeatureModel, IPredictionModel)│
│              User-Defined Models                     │
└─────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────┐
│                    Core Data                         │
│     (Dataset, ExperimentData, Schema, DataObjects)  │
│        User-Owned, Self-Contained                   │
└─────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────┐
│                     Utilities                        │
│          (Logger, LocalData, IExternalData)         │
│             Storage & Persistence                    │
└─────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

### 1. Core Data Layer (`src/lbp_package/core/`)

#### 1.1 DataObjects (`data_objects.py`)

**Responsibility:** Define individual data types with validation and rounding.

**Does:**
- Define typed primitives: `DataReal`, `DataInt`, `DataBool`, `DataCategorical`, `DataString`, `DataArray`
- Store type constraints (min/max, categories, array shapes)
- Store rounding configuration (`round_digits`)
- Validate values against constraints
- Apply rounding automatically via `set_value()`
- Serialize/deserialize with type preservation

**Does NOT:**
- Store actual experiment data (that's `DataBlock`)
- Know about schemas or experiments
- Handle I/O operations

**Factories:**
- `Parameter.real()`, `Parameter.integer()`, `Parameter.categorical()`, etc.
- `Performance.real()`, `Performance.integer()`
- `Dimension.integer()`

---

#### 1.2 DataBlocks (`data_blocks.py`)

**Responsibility:** Store and manage collections of related data values organized by name.

**Does:**
- Organize `DataObjects` into named collections
- Store actual values for each `DataObject`
- Validate values using `DataObject` validators
- Provide dict-like access (`set_value`, `get_value`)
- Serialize/deserialize collections
- Convert to numpy arrays for ML (`to_numpy`)

**Does NOT:**
- Define data types (that's `DataObjects`)
- Know about schema structure
- Perform domain logic
- Handle persistence

**Block Types:**
- `Parameters`: All experiment parameters (unified)
- `Dimensions`: Dimensional metadata
- `PerformanceAttributes`: Performance metrics
- `MetricArrays`: Array-valued metrics

---

#### 1.3 DatasetSchema (`schema.py`)

**Responsibility:** Define dataset structure with deterministic hashing.

**Does:**
- Define structure: parameters, dimensions, performance attributes
- Compute deterministic SHA256 hash (types + constraints)
- Store default rounding configuration (`default_round_digits`)
- Check structural compatibility between schemas
- Serialize/deserialize schema definitions
- Provide helper methods (`get_all_param_names()`, `get_dimension_names()`)

**Does NOT:**
- Store actual data values
- Manage experiments
- Handle persistence (that's SchemaRegistry)

**Key Methods:**
- `_compute_schema_hash()`: Collision-resistant hash
- `is_compatible_with()`: Structural validation
- `to_dict()` / `from_dict()`: Serialization

---

#### 1.4 SchemaRegistry (`schema_registry.py`)

**Responsibility:** Map schema hashes to human-readable IDs.

**Does:**
- Store hash → schema_id mapping
- Auto-increment schema IDs (`schema_001`, `schema_002`, ...)
- Ensure deterministic ID assignment (same hash = same ID)
- Persist registry to JSON file
- Provide import/export functionality

**Does NOT:**
- Validate schemas (that's `DatasetSchema`)
- Store experiment data
- Manage datasets

**Storage:** `{local_folder}/.lbp/schema_registry.json`

---

#### 1.5 Dataset (`dataset.py`)

**Responsibility:** Container for experiments with schema validation and persistence.

**Does:**
- Own and validate experiments against schema
- Store `ExperimentData` instances
- Manage hierarchical load/save (Memory → Local → External → Compute)
- Provide experiment CRUD operations
- Validate all data against schema
- Own storage interfaces (`local_data`, `external_data`)

**Does NOT:**
- Execute computations (that's Agent + Systems)
- Store schema definitions (that's `DatasetSchema`)
- Manage model registration

**Key Methods:**
- `add_experiment()`: Smart hierarchical add (tries to load first)
- `add_experiment_manual()`: Explicit creation
- `populate()`: Load all experiments from storage
- `save()`: Save all experiments + schema
- `save_experiment()`: Save single experiment
- `get_experiment_codes()`, `get_experiment_params()`: Data access

**Hierarchical Pattern:**
1. Check memory (already loaded?)
2. Try local storage
3. Try external storage
4. Compute/create new

---

#### 1.6 ExperimentData (`dataset.py`)

**Responsibility:** Container for single experiment's complete data.

**Does:**
- Store experiment code
- Store parameter values (`DataBlock`)
- Store performance metrics (`DataBlock`)
- Store metric arrays (`DataBlock`)
- Provide computed property for dimensions
- Validate completeness

**Does NOT:**
- Perform computations
- Handle persistence
- Know about other experiments

**Structure:**
```python
@dataclass
class ExperimentData:
    exp_code: str
    parameters: DataBlock              # All param values  
    features: Optional[DataBlock]       # Raw feature measurements
    performance: Optional[DataBlock]    # Aggregated performance (0-1)
    metric_arrays: Optional[DataBlock]  # Multi-dimensional arrays
    
    @property
    def dimensions(self) -> Dict[str, Any]:
        # View into dimensional params
```

---

### 2. Orchestration Layer (`src/lbp_package/orchestration/`)

#### 2.1 LBPAgent (`agent.py`)

**Responsibility:** Stateless orchestration - register models and coordinate execution.

**Does:**
- Register evaluation and prediction models (store specs)
- Generate schema from registered models (dataclass introspection)
- Initialize dataset and systems
- Coordinate experiment evaluation (delegate to EvaluationSystem)
- Coordinate prediction training/inference (delegate to PredictionSystem)
- Provide high-level workflow API

**Does NOT:**
- Store dataset (user owns it)
- Store experiment data
- Execute models directly (that's Systems)
- Manage persistence (that's Dataset)

**Key Change in Phase 7:** Agent is now stateless - returns Dataset without storing it.

**Methods:**
- `register_evaluation_model()`: Register model class
- `register_prediction_model()`: Register model class
- `initialize()`: Create dataset + systems, return dataset
- `evaluate_experiment(dataset, exp_data)`: Mutate exp_data with results
- `train(datamodule, **kwargs)`: Train prediction models on training split\n- `validate(use_test)`: Validate models on val/test split with metrics
- `predict(X_new)`: Predict features for new parameters (auto denormalized)

---

#### 2.2 EvaluationSystem (`orchestration/evaluation.py`)

**Responsibility:** Execute evaluation models and store results.

**Does:**
- Manage evaluation model instances
- Execute evaluation models in correct order
- Compute features via feature models
- Store feature values in `exp_data.features` DataBlock
- Store performance values in `exp_data.performance` DataBlock
- Store metric arrays in `exp_data.metric_arrays` DataBlock
- Pass visualization flags to feature models
- Handle recompute logic (clear feature cache)

**Does NOT:**
- Own dataset (references it)
- Define models (that's user via interfaces)
- Validate data (that's Dataset)

**Workflow:**
```
evaluate_experiment() →
  for each performance code:
    IEvaluationModel.run() →
      IFeatureModel.run() → compute feature → store in exp_data.features
      compute performance → store in exp_data.performance
      store metric_arrays
```

---

#### 2.3 PredictionSystem (`orchestration/prediction.py`)

**Responsibility:** Train and execute prediction models with DataModule integration.

**Does:**
- Manage prediction model instances
- Own DataModule lifecycle (stores copy after training)
- Coordinate training with normalization/batching
- Extract training data from DataModule (train split only)
- Fit normalization on features (y) from training data
- Train models on normalized training data
- Validate models on val/test splits with metrics (MAE, RMSE, R²)
- Validate input parameters for prediction
- Auto-denormalize predictions
- Return prediction results (DataFrames)
- Export models to InferenceBundle with validation

**Does NOT:**
- Own dataset (uses it for training data extraction)
- Store predictions permanently
- Normalize parameters (X) - only features (y)
- Predict performance (predicts features)

**Training Flow with DataModule:**
```
train(datamodule, **kwargs) →
  check train split not empty
  datamodule.extract_all(split='train') → X_train, y_train
  datamodule.fit_normalize(y_train) → compute mean/std on training data
  for each feature:
    y_norm = datamodule.normalize(y_feature)
    model.train(X_train, y_norm, **kwargs)  # Pass hyperparameters to model
```

**Validation Flow:**
```
validate(use_test=False) →
  check split not empty
  split = 'test' if use_test else 'val'
  datamodule.extract_all(split=split) → X_split, y_split
  for each model:
    y_pred_norm = model.forward_pass(X_split)
    y_pred = datamodule.denormalize(y_pred_norm)
    compute metrics: MAE, RMSE, R²
  return {'feature': {'mae': ..., 'rmse': ..., 'r2': ..., 'n_samples': ...}}
```

**Hyperparameter Support:**
The `**kwargs` allow users to pass custom hyperparameters to their model implementations:
```python
agent.train(
    datamodule,
    learning_rate=0.001,
    epochs=100,
    batch_size=32,
    verbose=True
)
```

**Prediction Flow:**
```
predict(X_new) →
  validate X_new columns against schema
  for each feature:
    y_norm = model.predict(X_new)
    y_pred = datamodule.denormalize(y_norm)
  return DataFrame with all features
```
predict(feature_name, X: DataFrame) →
  model.predict(X) → returns DataFrame with feature columns
```

**Export Flow:**
```
export_inference_bundle(filepath) →
  validate each model via round-trip test:
    artifacts = model._get_model_artifacts()
    fresh_model._set_model_artifacts(artifacts)
    verify fresh_model.feature_names matches
  create bundle dict:
    - prediction model specs + artifacts
    - normalization state from datamodule
    - schema for validation
  pickle.dump(bundle, file)
```

---

#### 2.4 InferenceBundle (`orchestration/inference_bundle.py`)

**Responsibility:** Lightweight production inference without Dataset/training dependencies.

**Does:**
- Load exported models from pickle file
- Reconstruct models from class paths and artifacts
- Validate input parameters against schema
- Run predictions through all models
- Auto-denormalize predictions using saved stats
- Return prediction DataFrames

**Does NOT:**
- Require Dataset, DataModule, or training code
- Support training or evaluation
- Store or manage experiment data
- Modify or update models

**Load Flow:**
```
InferenceBundle.load(filepath) →
  unpickle bundle dict
  for each model spec:
    import model class from class_path
    instantiate model
    model._set_model_artifacts(spec['artifacts'])
  store normalization state
  store schema for validation
```

**Predict Flow:**
```
bundle.predict(X) →
  validate X columns against schema.parameters
  for each model:
    y_pred_norm = model.forward_pass(X)
    y_pred = _denormalize(y_pred_norm, feature_names)
    collect predictions
  return DataFrame(predictions)
```

---

### 3. Interface Layer (`src/lbp_package/interfaces/`)

#### 3.1 IEvaluationModel (`interfaces/evaluation.py`)

**Responsibility:** Define contract for evaluation models.

**Does:**
- Declare required properties (feature_model_type, dim_names, target_value, etc.)
- Declare `evaluate()` method signature
- Use dataclass fields to declare parameters

**Does NOT:**
- Implement evaluation logic (user does)
- Store data
- Manage execution

---

#### 3.2 IFeatureModel (`interfaces/features.py`)

**Responsibility:** Load unstructured data and extract features with memoization.

**Does:**
- Declare `_load_data(**param_values)` for loading raw data
- Declare `_compute_features(data, visualize)` for feature extraction
- Implement `run()` method with dataset memoization
- Check cache before computing (avoid redundant work)
- Store computed features in dataset for reuse
- Pass `visualize` flag to `_compute_features()`

**Does NOT:**
- Implement data loading logic (user defines `_load_data()`)
- Implement feature computation (user defines `_compute_features()`)
- Store features permanently (that's Dataset's job)

**Required Fields:**
- `dataset: Dataset` - For memoization
- `logger: LBPLogger` - For logging

**Returns:**
- Feature values as floats (physical measurements)

---

#### 3.3 IPredictionModel (`interfaces/prediction.py`)

**Responsibility:** Define contract for predicting features from parameters using machine learning.

**Does:**
- Declare `feature_names` property (what features to predict)
- Declare `train(X: pd.DataFrame, y: pd.DataFrame, **kwargs)` method
- Declare `forward_pass(X: pd.DataFrame) -> pd.DataFrame` method

**Does NOT:**
- Store dataset reference (data passed explicitly)
- Implement ML logic (user does)
- Store predictions (returns DataFrames)
- Predict performance directly (predicts features)

**Key Pattern:**
```python
class MyPredictionModel(IPredictionModel):
    # Prediction models predict FEATURES, not performance
    @property
    def feature_names(self) -> List[str]:
        return ['filament_width', 'layer_height']
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs):
        # X: parameters, y: features (normalized by DataModule)
        # **kwargs: learning_rate, epochs, batch_size, verbose, etc.
        learning_rate = kwargs.get('learning_rate', 0.01)
        epochs = kwargs.get('epochs', 100)
        # Implement ML training
        pass
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        # Returns DataFrame with feature columns (floats)
        pass
```

**Data Flow:**
```
Parameters → PredictionModel.predict() → Features → EvaluationModel → Performance
```

**Why Features, Not Performance?**
- **Interpretability**: "Width will be 1.78mm" vs "Performance 90%"
- **Flexibility**: Same predictions, different objectives (multi-objective optimization)
- **Composability**: Calibration can combine predict() + evaluate() freely
- **Visualization**: Can plot predicted feature values

---

#### 3.4 IExternalData (`interfaces/external_data.py`)

**Responsibility:** Define contract for external data storage.

**Does:**
- Declare methods for experiment load/save
- Declare methods for schema load/save
- Provide default implementations (return False/None)

**Does NOT:**
- Implement actual storage (user does)
- Validate data
- Manage connections

**Methods:**
- `load_experiment_record()`, `save_experiment_record()`
- `load_performance_metrics()`, `save_performance_metrics()`
- `pull_schema()`, `push_schema()`

---

### 4. Utilities (`src/lbp_package/utils/`)

#### 4.1 LocalData (`local_data.py`)

**Responsibility:** Local filesystem persistence.

**Does:**
- Save/load experiment records to JSON
- Save/load performance metrics to JSON
- Save/load metric arrays to CSV
- Manage folder structure
- List available experiments

**Does NOT:**
- Validate data (that's Dataset)
- Know about schemas
- Execute computations

**Pattern:** Uses `schema_id` as folder name (not `study_code`).

---

#### 4.2 DataModule (`datamodule.py`)

**Responsibility:** ML preprocessing for train/val/test splitting, batching, and normalization.

**Does:**
- Split data into train/validation/test sets (reproducible with random_seed)
- Extract features and targets from Dataset experiments by split
- Apply normalization (standard, minmax, robust, or none)
- Batch data for training (default: single batch with all data)
- Store normalization statistics for inference
- Provide denormalization utilities

**Does NOT:**
- Store experiments (that's Dataset)
- Execute ML training (that's IPredictionModel)
- Define features (that's IFeatureModel/IPredictionModel)

**Design Pattern:**
- Configurable train/val/test splits (default: 80/10/10)
- Fits normalization on training data only
- Lazy evaluation - creates splits on first data extraction
- Inspired by PyTorch Lightning DataModule and AIXD paper

**Example:**
```python
# Configure splits and normalization
datamodule = DataModule(
    dataset,
    test_size=0.2,      # 20% for test
    val_size=0.1,       # 10% of remaining for validation
    random_seed=42,     # Reproducible splits
    normalize='standard'
)

# Get split sizes
sizes = datamodule.get_split_sizes()
# {'train': 72, 'val': 8, 'test': 20}

# Extract specific splits
X_train, y_train = datamodule.extract_all(split='train')
X_val, y_val = datamodule.extract_all(split='val')
X_test, y_test = datamodule.extract_all(split='test')
X_all, y_all = datamodule.extract_all()  # All data (no split)

# Train on training split
system.train(datamodule)  # Uses train split automatically

# Validate on validation/test sets
val_metrics = system.validate(use_test=False)  # Validation set
test_metrics = system.validate(use_test=True)  # Test set

# No splits (use all data for training)
datamodule = DataModule(dataset, test_size=0.0, val_size=0.0)
```

**Normalization Methods:**
- `'standard'`: Z-score normalization (mean=0, std=1)
- `'minmax'`: Min-max scaling to [0, 1]
- `'robust'`: Median and IQR-based (robust to outliers)
- `'none'`: No normalization

**Split Configuration:**
- `test_size`: Fraction for test set (0.0-1.0)
- `val_size`: Fraction of remaining data for validation (0.0-1.0)
- `random_seed`: Random seed for reproducibility (None = random)
- Setting both to 0.0 uses all data for training

---

#### 4.3 Logger (`logger.py`)

**Responsibility:** Centralized logging.

**Does:**
- Provide singleton logger instance
- Configure log formatting
- Write to file and console

**Does NOT:**
- Store data
- Execute logic

---

## Ownership & Dependencies

### Who Owns What?

- **User owns**: `Dataset` (returned from `agent.initialize()`)
- **Dataset owns**: `ExperimentData` instances, `local_data`, `external_data`
- **ExperimentData owns**: `DataBlocks` with values
- **Agent owns**: Model registries, systems (but not dataset)
- **Systems own**: Model instances (but reference dataset)

### Dependency Flow (Phase 7)

```
User
 └─> creates Agent
      └─> registers models (stores specs)
      └─> calls initialize() 
           └─> generates schema from model classes
           └─> creates Dataset (user owns)
           └─> creates Systems
           └─> instantiates models
           └─> returns Dataset to user

User
 └─> owns Dataset
      └─> calls dataset.add_experiment()
      └─> calls agent.evaluate_experiment(dataset, exp_data)
           └─> mutates exp_data
      └─> calls dataset.save()
```

---

## Key Design Patterns

### 1. Dataset-Centric Architecture
- Dataset is user-owned and self-contained
- Agent is stateless orchestration layer
- All data operations go through Dataset

### 2. Hierarchical Load/Save
Pattern: **Memory → Local → External → Compute**
- Check memory first (already loaded?)
- Try local storage
- Try external storage
- Compute/create new if not found

### 3. Mutation Pattern
- `evaluate_experiment(dataset, exp_data)` mutates `exp_data` in place
- No return value - results stored in exp_data
- Clear that data is modified

### 4. Declarative Configuration
- Rounding configured in `DataObject` (`round_digits`)
- Parameters declared in model dataclass fields
- Schema generated from declarations

### 5. Factory Pattern
- `Parameter.real()`, `Performance.real()`, `Dimension.integer()`
- Simplify common DataObject creation patterns

---

## Anti-Patterns (What NOT to Do)

❌ **Agent storing dataset** → Agent is stateless, user owns dataset  
❌ **Delegation methods** → Call `dataset.add_experiment()` directly, not through agent  
❌ **Passing round_digits everywhere** → Configure once in DataObject  
❌ **Mixing concerns** → Each class has single responsibility  
❌ **Hidden state** → All dependencies explicit  
❌ **Tight coupling** → Use interfaces, not concrete classes  

---

## Validation Points

Each layer validates its own concern:

1. **DataObject**: Type and constraint validation
2. **DataBlock**: Validates using DataObject validators
3. **Dataset**: Schema compliance, completeness
4. **Agent**: Model registration validity
5. **Systems**: Execution order, model availability

---

This separation ensures clean boundaries, testability, and maintainability.
