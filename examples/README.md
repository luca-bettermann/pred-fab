# LBP Examples Repository

This repository contains complete working implementations of the Learning by Printing (LbP) Framework. Each example demonstrates real domain-specific implementations showing how to integrate the framework with your manufacturing processes.

## Repository Structure

This examples repository demonstrates concrete implementations of the LBP Framework interfaces with working code for geometric evaluation and energy analysis:

```
examples/
├── main.py                    # Main execution script - complete workflow
├── external_data.py          # Mock database interface implementation
├── evaluation_geometry.py    # Path deviation evaluation model + feature model
├── evaluation_energy.py      # Energy consumption evaluation model + feature model
├── prediction.py             # ML prediction model implementation
├── calibration.py            # Optimization model implementations
├── utils/                    # Utility functions
│   ├── mock_data.py          # Test data generation utilities
│   └── visualize.py          # Visualization helper functions
└── UML_diagram.png          # Architecture diagram

# Created at runtime when executing main.py:
# ├── local/                  # Data storage directory
# │   └── test/               # Study data (JSON files and arrays)
# └── logs/                   # Execution logs
```

## Architecture Overview

![Examples Implementation](UML_diagram.png)

This diagram shows the concrete implementations of the LBP framework interfaces, demonstrating how abstract contracts are fulfilled with working code.

## Quick Start - Run the Complete Example

### Prerequisites
- Python 3.9+
- LBP Package framework installed

### Run the Example

```bash
# Navigate to examples directory
cd examples

# Run the complete workflow
python main.py
```

This will execute the complete learning workflow using the pre-configured test data, demonstrating evaluation, prediction, and calibration.

## Implementation Walkthrough

Let's examine each component of the example implementation:

### 1. Main Workflow (`main.py`)

The main script demonstrates the complete Learning by Printing workflow:

```python
from lbp_package import LBPManager
from examples.evaluation_energy import EnergyConsumption
from examples.evaluation_geometry import PathEvaluation
from examples.prediction import PredictExample
from examples.external_data import MockDataInterface
from examples.calibration import RandomSearchCalibration, DifferentialEvolutionCalibration

def main():
    # Initialize LBPManager with mock database interface
    lbp_manager = LBPManager(
        root_folder=str(root_dir),
        local_folder=str(local_dir), 
        log_folder=str(logs_dir),
        external_data_interface=MockDataInterface(str(local_dir))
    )
    
    # Register evaluation models with calibration weights
    lbp_manager.add_evaluation_model("energy_consumption", EnergyConsumption, weight=0.3)
    lbp_manager.add_evaluation_model("path_deviation", PathEvaluation, weight=0.7)
    
    # Register prediction model for multiple performance metrics
    lbp_manager.add_prediction_model(["energy_consumption", "path_deviation"], PredictExample)
    
    # Execute complete workflow
    lbp_manager.initialize_for_study("test")
    
    # Run evaluations (individual and batch)
    lbp_manager.run_evaluation("test", exp_nr=1)
    lbp_manager.run_evaluation("test", exp_nrs=[2, 3])
    
    # Train prediction models
    lbp_manager.run_training("test", exp_nrs=[1, 2, 3])
    
    # Calibration with different optimizers
    param_ranges = {"layerTime": (0.0, 1.0), "layerHeight": (10, 100)}
    
    lbp_manager.set_calibration_model(RandomSearchCalibration, n_evaluations=100)
    lbp_manager.run_calibration(exp_nr=4, param_ranges=param_ranges)
    
    lbp_manager.set_calibration_model(DifferentialEvolutionCalibration, maxiter=10, seed=42)
    lbp_manager.run_calibration(exp_nr=4, param_ranges=param_ranges)
```

**Key Learning**: See how the framework orchestrates multiple evaluation models, prediction, and calibration in a unified workflow.

### 2. Path Deviation Analysis (`evaluation_geometry.py`)

This implementation demonstrates **multi-dimensional evaluation** with automatic array management:

```python
@dataclass
class PathEvaluation(IEvaluationModel):
    # Study parameters (constant across experiments)
    target_deviation: float = study_parameter(default=0.1)
    max_deviation: float = study_parameter()
    
    # Experiment parameters (define array dimensions)
    n_layers: int = exp_parameter()
    n_segments: int = exp_parameter()
    
    # Dimension parameters (track current position during iteration)
    layer_id: int = dim_parameter()
    segment_id: int = dim_parameter()
    
    # Required dimension configuration
    @property
    def dim_names(self) -> List[str]:
        return ['layers', 'segments']
    
    @property
    def dim_param_names(self) -> List[str]:
        return ['n_layers', 'n_segments']  # Must match exp record fields
    
    @property  
    def dim_iterator_names(self) -> List[str]:
        return ['layer_id', 'segment_id']  # Must match dim_parameter names
    
    @property
    def feature_model_type(self) -> Type[IFeatureModel]:
        return PathDeviationFeature  # Associated feature model
    
    @property
    def target_value(self) -> float:
        return self.target_deviation  # Performance target
```

**Key Learning**: Framework automatically creates 2D arrays `[n_layers, n_segments]` and iterates through all combinations `(layer_id, segment_id)`.

### 3. Energy Consumption Analysis (`evaluation_energy.py`)

This implementation demonstrates **0-dimensional evaluation** (single aggregated metric):

```python
@dataclass
class EnergyConsumption(IEvaluationModel):
    # Study parameters
    target_energy: float = study_parameter(0.0)
    max_energy: float = study_parameter()
    
    # No dimension configuration (0-dimensional)
    @property
    def dim_names(self) -> List[str]:
        return []  # Empty = single aggregated value
    
    @property
    def dim_param_names(self) -> List[str]:
        return []  # No dimensions
    
    @property
    def dim_iterator_names(self) -> List[str]:
        return []  # No iteration needed
    
    @property
    def target_value(self) -> float:
        return self.target_energy
    
    @property
    def scaling_factor(self) -> float:
        return self.max_energy  # Optional: provides normalization
```

**Key Learning**: Compare with PathEvaluation - same interface, but 0-dimensional vs 2-dimensional evaluation.

### 4. Mock Database Interface (`external_data.py`)

This implementation shows how to integrate with external data sources:

```python
class MockDataInterface(IExternalData):
    def pull_study_record(self, study_code: str) -> Dict[str, Any]:
        # Return hardcoded study data with proper structure
        return {
            "id": 0,
            "Code": study_code,
            "Parameters": {        
                "target_deviation": 0.0,
                "max_deviation": 5.0,
                "target_energy": 0.0,
                "max_energy": 10000.0,
                "power_rating": 50.0
            },
            "Performance": ["path_deviation", "energy_consumption"]
        }
    
    def pull_exp_record(self, exp_code: str) -> Dict[str, Any]:
        # Map experiment codes to parameter combinations
        params = {
            'test_001': [2, 2, 30.0, 0.2],  # n_layers, n_segments, layerTime, layerHeight
            'test_002': [3, 4, 40.0, 0.3], 
            'test_003': [4, 3, 50.0, 0.4],
        }
        return {
            "id": int(''.join(filter(str.isdigit, exp_code))),
            "Code": exp_code,
            "Parameters": {
                "n_layers": params[exp_code][0],
                "n_segments": params[exp_code][1], 
                "layerTime": params[exp_code][2],
                "layerHeight": params[exp_code][3]
            }
        }
```

**Key Learning**: External data interface allows integration with existing databases/APIs while maintaining framework independence.

### 5. ML Prediction Model (`prediction.py`)

This implementation demonstrates machine learning integration for performance prediction:

```python
class PredictExample(IPredictionModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    @property
    def input(self) -> List[str]:
        return ["layerTime", "layerHeight", "temperature"]  # Input features
        
    @property
    def dataset_type(self) -> IPredictionModel.DatasetType:
        return IPredictionModel.DatasetType.AGGR_METRICS  # Use aggregated metrics
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        self.model.fit(X, y)
        r2_score = self.model.score(X, y)
        return {
            "training_score": r2_score,
            "training_samples": len(X)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
```

**Key Learning**: Framework handles feature extraction and data preparation. You focus on ML model training and prediction logic.

### 6. Optimization Models (`calibration.py`)

Two different optimization approaches are implemented:

#### Random Search Calibration
```python
class RandomSearchCalibration(ICalibrationModel):
    def optimize(self, param_ranges: Dict[str, Tuple[float, float]], 
                 objective_fn: Callable[[Dict[str, float]], float]) -> Dict[str, float]:
        # Try n_evaluations random points, return the best
        best_params = None
        best_objective = float('-inf')
        
        for params in ParameterSampler(param_dists, n_iter=self.n_evaluations):
            objective_val = objective_fn(params)
            if objective_val > best_objective:
                best_objective = objective_val
                best_params = params.copy()
        
        return best_params
```

#### Differential Evolution Calibration  
```python
class DifferentialEvolutionCalibration(ICalibrationModel):
    def optimize(self, param_ranges: Dict[str, Tuple[float, float]], 
                 objective_fn: Callable[[Dict[str, float]], float]) -> Dict[str, float]:
        # Use scipy's differential evolution directly
        def scipy_objective(x):
            params = dict(zip(param_ranges.keys(), x))
            return -objective_fn(params)  # Scipy minimizes, we maximize
            
        result = differential_evolution(
            scipy_objective, 
            bounds=list(param_ranges.values()),
            maxiter=self.maxiter
        )
        return dict(zip(param_ranges.keys(), result.x))
```

**Key Learning**: Framework provides objective function based on evaluation model weights. You implement the optimization algorithm of choice.

## Test Data Structure

The `local/test/` directory contains the complete data structure for standalone operation:

### Study Record (`local/test/study_record.json`)
```json
{
  "id": 0,
  "Code": "test",
  "Parameters": {
    "target_deviation": 0.0,
    "max_deviation": 5.0,
    "target_energy": 0.0,
    "max_energy": 10000.0,
    "power_rating": 50.0
  },
  "Performance": [
    "path_deviation",
    "energy_consumption"
  ]
}
```

### Experiment Records (`local/test/test_00X/exp_record.json`)
Each experiment folder contains:
- `exp_record.json` - Experiment parameters and metadata
- Raw data files (loaded by feature models) 
- `arrays/` - Generated feature and performance arrays
- Result files (aggregated metrics, predictions, etc.)

Example experiment record:
```json
{
  "id": 1,
  "Code": "test_001", 
  "Parameters": {
    "n_layers": 2,
    "n_segments": 2,
    "layerTime": 30.0,
    "layerHeight": 0.2
  }
}
```

**Key Learning**: This data structure demonstrates the framework's hierarchical organization and shows how parameters flow from records to model instances.

## Running Different Workflows

You can run individual components of the workflow:

```python
# Evaluation only
lbp_manager.run_evaluation("test", exp_nrs=[1])

# Training only (requires existing evaluation results)  
lbp_manager.run_training("test", exp_nrs=[1, 2, 3])

# Calibration only (requires trained prediction models)
lbp_manager.run_calibration(exp_nr=4, param_ranges={"layerTime": (0.0, 1.0)})

# Flag overrides
lbp_manager.run_evaluation("test", exp_nrs=[1], 
                          debug_flag=True,      # Skip external data
                          visualize_flag=False,  # No visualization
                          recompute_flag=True)   # Force recomputation
```

## Next Steps

1. **Adapt Feature Models**: Modify `PathDeviationFeature` and `EnergyFeature` to load your domain-specific data formats
2. **Customize Evaluation Logic**: Adjust target values and scaling factors in evaluation models
3. **Integrate Your Database**: Replace `MockDataInterface` with your actual database interface
4. **Experiment with ML Models**: Try different algorithms in the prediction model
5. **Optimize Your Process**: Implement domain-specific optimization algorithms in calibration models

For detailed API documentation and core concepts, see the [LBP Framework README](../README.md).

## Implementation Examples

### MockDataInterface
File-based data source that demonstrates proper interface implementation:

```python
class MockDataInterface(IExternalData):
    def pull_study_record(self, study_code: str) -> Dict[str, Any]:
        return {
            "id": 0,
            "Code": study_code,
            "Parameters": {        
                "target_deviation": 0.0,
                "max_deviation": 5.0,
                "target_energy": 0.0,
                "max_energy": 10000.0,
                "power_rating": 50.0
            },
            "Performance": ["path_deviation", "energy_consumption"]
        }
```

### PathEvaluation & PathDeviationFeature
Comprehensive geometric accuracy assessment:

```python
@dataclass
class PathEvaluation(IEvaluationModel):
    # Study parameters
    target_deviation: Optional[float] = study_parameter(default=0.1)
    max_deviation: Optional[float] = study_parameter()
    
    # Experiment parameters
    n_layers: Optional[int] = exp_parameter()
    n_segments: Optional[int] = exp_parameter()

    # Dimensionality parameters
    layer_id: Optional[int] = dim_parameter()
    segment_id: Optional[int] = dim_parameter()

    @property
    def dim_names(self) -> List[str]:
        return ['layers', 'segments']
```

### EnergyConsumption & EnergyFeature  
Power consumption analysis:

```python
@dataclass
class EnergyConsumption(IEvaluationModel):
    target_energy: Optional[float] = study_parameter(0.0)
    max_energy: Optional[float] = study_parameter()

    @property
    def dim_names(self) -> List[str]:
        return []  # No dimensions - scalar evaluation
```

### PredictExample
Machine learning prediction with RandomForest:

```python
class PredictExample(IPredictionModel):
    @property
    def input(self) -> List[str]:
        return ["layerTime", "layerHeight", "temperature"]
        
    @property
    def dataset_type(self) -> IPredictionModel.DatasetType:
        return IPredictionModel.DatasetType.AGGR_METRICS

    def train(self, X: ndarray, y: ndarray) -> Dict[str, Any]:
        self.model.fit(X, y)
        return {
            "training_score": round(self.model.score(X, y), 4),
            "training_samples": X.shape[0]
        }
```

### Calibration Examples
Two optimization approaches:

```python
class RandomSearchCalibration(ICalibrationModel):
    def __init__(self, n_evaluations: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.n_evaluations = n_evaluations
        
    def optimize(self, param_ranges, objective_fn):
        # Random search implementation
        pass

class DifferentialEvolutionCalibration(ICalibrationModel):
    def __init__(self, maxiter: int = 50, popsize: int = 15, **kwargs):
        super().__init__(**kwargs)
        self.maxiter = maxiter
        self.popsize = popsize
```

## Configuration

The framework uses **programmatic configuration** without configuration files. All models are registered directly in code:

```python
# Register evaluation models with calibration weights
lbp_manager.add_evaluation_model("energy_consumption", EnergyConsumption, weight=0.3)
lbp_manager.add_evaluation_model("path_deviation", PathEvaluation, weight=0.7)

# Register prediction model for multiple performance metrics
lbp_manager.add_prediction_model(["energy_consumption", "path_deviation"], PredictExample)

# Register calibration model with parameters
lbp_manager.set_calibration_model(RandomSearchCalibration, n_evaluations=100)
```

## Key Implementation Patterns

### Parameter Management
All models use the framework's parameter handling system:
- `@study_parameter()` - Constants across experiments
- `@exp_parameter()` - Varies between experiments  
- `@dim_parameter()` - Changes during execution

### Feature Model Integration
Evaluation models declare their feature dependencies:
```python
@property
def feature_model_type(self) -> Type[IFeatureModel]:
    return PathDeviationFeature
```

### Data Loading Patterns
Feature models handle domain-specific data:
```python
def _load_data(self, exp_code: str, exp_folder: str, debug_flag: bool):
    # Load geometry files, sensor data, etc.
    return loaded_data

def _compute_features(self, data: Any, visualize_flag: bool):
    # Extract quantitative features
    return {"feature_name": computed_value}
```

## Utility Functions

The examples include helper utilities in the `utils/` directory:
- `mock_data.generate_path_data()` - Creates mock geometry data with optional noise
- `mock_data.generate_temperature_data()` - Creates mock sensor data  
- `visualize.visualize_geometry()` - Visualization tools for path analysis
- `visualize.plot_temperature_data()` - Temperature data visualization

## Next Steps

1. **Customize Data Interface**: Replace `MockDataInterface` with your actual data source (database, API, files)
2. **Implement Domain Models**: Create evaluation and feature models for your specific use case
3. **Customize Data Loading**: Modify `_load_data()` methods to handle your data formats
4. **Add Prediction Models**: Implement ML models suited for your performance metrics
5. **Setup Calibration**: Choose and configure optimization algorithms for your parameter spaces
6. **Scale Dimensionality**: Extend dimension configuration to match your analysis structure

For detailed framework documentation, see the [LBP Package Framework](https://github.com/your-org/lbp-package) repository.