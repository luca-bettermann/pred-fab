# Learning by Printing (LbP) Framework

A Python framework for iterative manufacturing process improvement through automated performance evaluation and optimization.

## Project Structure

```
lbp_package/
├── src/lbp_package/           # Core framework code
│   ├── evaluation.py          # Base classes for evaluation models
│   ├── data_interface.py      # Data access interface
│   ├── orchestration.py       # System orchestration
│   └── utils/                 # Utility modules
├── examples/                  # Self-contained working example
│   ├── *.py                   # Implementation files
│   ├── config.yaml           # Configuration
│   ├── local/                # Auto-generated data files
│   └── logs/                 # Auto-generated log files
├── tests/                    # Comprehensive test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   ├── end_to_end/          # End-to-end tests
│   ├── test_data.py         # Shared test data utilities
│   └── conftest.py          # Test fixtures and configuration
├── README.md                # This file
└── configTEMPLATE.yaml      # Template for the config file

```


## 1. Introduction & Quick Start

### 1.1 Learning by Printing Overview

Learning by Printing is an iterative manufacturing optimization approach that systematically improves printing processes through automated performance evaluation and parameter adjustment. The framework enables closed-loop learning where each experiment provides feedback for process refinement.

For detailed methodology, see: [An Introduction to Learning by Printing](https://mediatum.ub.tum.de/doc/1781543/1781543.pdf)

### 1.2 Quick Start Guide

#### Prerequisites
- Python 3.9+
- `uv` package manager

#### Installation

```bash
# Clone repository
git clone <repository-url>
cd lbp_package

# Setup virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

#### Configuration

1. **Environment Setup**: Copy `.envTEMPLATE` to `.env` and configure paths:
   ```bash
   cp .envTEMPLATE .env
   # Edit .env with your local paths and database credentials
   ```

2. **System Configuration**: Copy `configTEMPLATE.yaml` to `config.yaml`:
   ```bash
   cp configTEMPLATE.yaml config.yaml
   # Edit config.yaml to map your performance codes to evaluation classes
   ```

#### 5-Minute Example
This code snippet will allow you to get your project started, although more implementation is needed before you get useable results.

```python
from src.lbp_package.orchestration import LBPManager
from examples.mock_data_interface import ExampleDataInterface

# Initialize system
data_interface = ExampleDataInterface("/tmp", {}, {})
manager = LBPManager(
    local_folder="/path/to/local",
    server_folder="/path/to/server", 
    log_folder="/path/to/logs",
    data_interface=data_interface
)

# Run evaluation
manager.initialize_study("MY_STUDY")
manager.run_evaluation(exp_nr=1, debug_flag=True)
```

### Running the Example

The `examples/` folder contains a complete, self-sufficient demonstration:

```bash
# Navigate to examples and run the main script
cd examples
python run_example.py
```

This will automatically:
- Generate example data files if they don't exist
- Initialize a study called "test" with 2 performance metrics
- Load experimental data from JSON files
- Execute path deviation and energy consumption evaluations
- Display results and performance summaries

#### Project Structure
```
lbp_package/
├── src/lbp_package/          # Core framework
├── examples/                 # Implementation examples  
├── tests/                    # Test suite
├── config.yaml              # System configuration
└── .env                     # Environment variables
```

## Data Structure

The framework expects data in a structured format. The examples folder demonstrates one possibility of this organization. The crucial part is that the DataInterface knows how to navigate the database structure or in the example case the file-based

```
examples/
├── main.py                     # Main execution script
├── file_data_interface.py      # File-based data interface implementation
├── path_deviation.py           # Path deviation evaluation model
├── energy_consumption.py       # Energy consumption evaluation model
├── config.yaml                 # System configuration file
├── setup_example_data.py       # Standalone data generation script
├── local/                      # Data storage directory
│   └── test/                   # Study folder (generated automatically)
│       ├── study_params.json           # Study-level parameters
│       ├── performance_records.json    # Performance metric definitions
│       └── test_001/                   # Experiment folder
│           ├── exp_params.json         # Experiment-specific parameters
│           ├── test_001_designed_paths.json   # Designed path data
│           └── test_001_measured_paths.json   # Measured path data
└── logs/                       # Log files (created automatically)
    └── LBPManager_session_*.log     # Execution logs
```

The data files are automatically generated when running `main.py`.

## 2. Architecture & Framework Design

### 2.1 System Overview

![LBP Base Classes](src/lbp_package/UML_base_classes.png)

The framework follows a modular architecture with clear separation between orchestration (system management, grey) and interface classes (extension points, highlighted in color).

### 2.2 Orchestration Classes

#### LBPManager
Main entry point that coordinates the complete workflow and acts as the central hub connecting all subsystems. The LBPManager is responsible for:
- **System Initialization**: Loads configuration files, establishes database connections through the DataInterface, and initializes all subsystems
- **Study Management**: Manages study-level configuration and parameters that persist across multiple experiments
- **Subsystem Coordination**: Distributes functionality to EvaluationSystem, PredictionSystem, and Calibration components
- **External Source Integration**: Handles connections to databases, file systems, and external APIs through configurable interfaces
- **Workflow Orchestration**: Coordinates the complete learning loop from evaluation through prediction to calibration

**Key Methods:**
- `initialize_study(study_code)`: Loads study configuration, initializes all subsystems, and prepares the framework for experiment execution
- `run_evaluation(exp_nr)`: Executes the complete evaluation pipeline for a specific experiment
- `_load_config()`: Parses configuration files and distributes settings to appropriate subsystems
- System configuration loading and management with runtime override capabilities

#### EvaluationSystem  
Orchestrates multiple evaluation models for comprehensive performance assessment. The system enables parallel evaluation of different performance metrics and supports dimensional evaluation that can be executed incrementally as data becomes available.

**Key Features:**
- **Multi-Performance Evaluation**: Manages multiple evaluation models simultaneously, each targeting different performance aspects
- **Dimensional Independence**: Processes evaluation dimensions (layers, segments, etc.) independently, enabling real-time evaluation during manufacturing
- **Feature Model Optimization**: Shares feature model instances across evaluation models to avoid redundant data processing
- **Incremental Processing**: Can evaluate performance metrics as soon as dimensional data is available, supporting online monitoring

**Key Methods:**
- `add_evaluation_model(evaluation_class, performance_code, study_params)`: Registers evaluation models for specific performance metrics
- `add_feature_model_instances(study_params)`: Initializes and optimizes feature model sharing across evaluations
- `run(exp_nr, exp_record, **exp_params)`: Executes complete evaluation pipeline with dimensional iteration and result aggregation

#### PredictionSystem *(to be added)*
ML model management and inference execution for predicting performance outcomes and optimizing process parameters.

#### Calibration *(to be added)*
Optimization system for automated parameter adjustment based on evaluation results and prediction models.

### 2.3 Interface Classes (Extension Points)

#### DataInterface
Abstract base class providing standardized database and data source integration. This interface abstracts away specific database implementations, allowing the framework to work with various data backends (SQL databases, NoSQL systems, APIs, files).

**Abstract Methods:**
- `get_study_record(study_code)` → `Dict`: Retrieves comprehensive study metadata including configuration parameters, active performance metrics, and study-level settings. Must return a dictionary with study identification, parameter and performance fields.
- `get_exp_record(exp_code)` → `Dict`: Fetches experiment-specific data including experimental conditions and execution parameters. Return dictionary must contain experiment identification and associated metadata.
- `get_study_parameters(study_record)` → `Dict`: Extracts study-level parameters that define model configurations, target values, and evaluation criteria. These parameters are typically constant across all experiments within a study.
- `get_performance_records(study_record)` → `List[Dict]`: Returns list of performance metric configurations for the study. Each dictionary must contain a 'Code' field that matches the performance codes used in the configuration mapping.
- `get_exp_variables(exp_record)` → `Dict`: Extracts experiment-specific variables including process parameters, environmental conditions, and dimensional settings (n_layers, n_segments, etc.).

**Optional Methods:**
- `push_to_database(exp_record, performance_code, value_dict)`: Stores evaluation results back to the database with performance metrics
- `update_system_performance(study_record)`: Updates aggregated system-wide performance metrics and study progress indicators

#### FeatureModel & EvaluationModel: Paired Architecture
FeatureModel and EvaluationModel work as paired structures with a one-to-one, or in certain cases, a many-to-one mapping relationship. In these instances, multiple EvaluationModels can share a single FeatureModel instance to optimize computational efficiency. This design recognizes that different performance metrics often require similar feature extraction processes, avoiding redundant computation while maintaining clean separation of evaluation logic. However, the simplest architecture case provides one FeatureModel per EvaluationModel.

#### FeatureModel
Abstract base class for extracting quantitative features from experimental data. FeatureModels handle data loading, preprocessing, and feature computation while supporting multi-dimensional evaluation and parameter-driven configuration.

**Abstract Methods:**
- `_load_data(exp_nr)` → `Any`: Loads and preprocesses raw experimental data for the specified experiment. Return type is flexible to accommodate various data formats (JSON, CSV, images, sensor data).
- `_compute_features(data, visualize_flag)` → `Dict[str, float]`: Extracts quantitative features from loaded data. Returns dictionary mapping performance codes to computed feature values. The visualize_flag enables optional visualization output for debugging and analysis.

**Optional Methods:**
- `_initialization_step(performance_code)`: Performs setup operations before feature extraction, such as initializing computational resources or loading calibration data
- `_cleanup_step(performance_code)`: Handles post-processing cleanup, resource deallocation, or temporary file removal
- `_fetch_data(exp_nr)`: Downloads or retrieves data from external sources if not locally available
- `_validate_parameters()`: Validates parameter configurations and ensures computational prerequisites are met

**Parameter Integration:** Inherits from `ParameterHandling` for seamless parameter management across model, experiment, and runtime scopes.

#### EvaluationModel  
Abstract base class for performance evaluation that compares extracted features against target criteria. Supports multi-dimensional analysis with automated aggregation and configurable scaling strategies.

**Abstract Methods:**
- `_compute_target_value()` → `float`: Defines the target value for performance evaluation. This represents the ideal or desired performance level for the specific metric.

**Optional Methods:**
- `_compute_scaling_factor()` → `Optional[float]`: Provides normalization factor for performance values, enabling consistent performance metrics across different scales and units
- `_aggregate_performance(performance_array)`: Defines custom aggregation strategy for combining performance values across dimensions (default: mean aggregation)
- `_initialization_step()`: Setup operations before performance computation
- `_cleanup_step()`: Post-processing cleanup after performance evaluation

**Key Features:**
- **Dimensional Evaluation**: Supports structured evaluation across multiple dimensions (layers, segments, time steps) with independent processing
- **Automatic Performance Aggregation**: Combines dimensional results into overall performance metrics
- **Feature Model Integration**: Seamlessly integrates with paired FeatureModel for optimized computation

### 2.4 Utility Classes

#### ParameterHandling
Sophisticated dataclass-based parameter management system supporting three distinct parameter categories with automatic type checking and IDE support. Each class inheriting from ParameterHandling automatically gains parameter management capabilities.

**Parameter Types:**
- `model_parameter(default_value)`: **Model Configuration Parameters** - Define the fundamental behavior and configuration of models. Set during class initialization and remain constant throughout the study. Examples: tolerance values, algorithm settings, model hyperparameters.
- `exp_parameter(default_value)`: **Experiment-Specific Parameters** - Values that vary between experiments within a study. Define experimental conditions and setup parameters. Examples: number of layers, process temperatures, material properties.
- `runtime_parameter()`: **Runtime Execution Parameters** - Dynamic values that change during execution, typically representing current processing state. Examples: current layer_id, segment_id, iteration counters.

**Built-in Functionality:**
All classes inheriting from ParameterHandling automatically receive:
- `set_model_parameters(**kwargs)`: Safely sets model parameters with automatic filtering
- `set_experiment_parameters(**kwargs)`: Updates experiment-specific parameters
- `set_runtime_parameters(**kwargs)`: Sets dynamic runtime values
- Automatic parameter validation and type checking
- Clean separation of parameter concerns with IDE autocomplete support

#### FolderNavigator
Centralized file system organization utility that enforces consistent naming conventions across studies, experiments, and results. Handles both local and server-side file management with automatic path generation and validation.

**Key Features:**
- Automatic experiment code generation with zero-padding
- Consistent directory structure across studies
- Server synchronization support
- File operation utilities with error handling

#### LBPLogger
Enhanced logging system providing dual output channels (file and console) with intelligent formatting, debug mode switching, and ANSI code handling for clean log files.

**Logging Methods:**
- File-only logging: `debug()`, `info()`, `warning()`, `error()`
- Console + file logging: `console_info()`, `console_success()`, `console_warning()`, `console_summary()`
- Debug mode switching with automatic file renaming
- Clean log file output with ANSI escape sequence removal

#### System Configuration
YAML-based configuration system supporting modular configuration loading with runtime override capabilities.

**Configuration Sections:**
- **evaluation**: Performance code to evaluation class mappings
- **prediction**: ML model configurations (to be added)
- **calibration**: Optimization algorithm settings (to be added)  
- **system**: Global system defaults (debug_mode, visualize_flag, round_digits, logging levels)

### 2.5 Critical Implementation Notes

#### ⚠️ Dimension Naming Conventions
The parameter name (third element) in `dimension_names` must exactly match your database field names to ensure proper parameter injection:
```python
dimension_names = [('layers', 'layer_id', 'n_layers')]
#                                           ^^^^^^^^^ 
#                                           Must match database parameter name
```

#### ⚠️ Runtime Parameter Naming  
Runtime parameter field names must match iterator names (second element) from `dimension_names` for automatic parameter setting:
```python
dimension_names = [('layers', 'layer_id', 'n_layers')]
#                             ^^^^^^^^
#                             This name must be used in runtime_parameter()

@dataclass
class MyFeatureModel(FeatureModel):
    layer_id: int = runtime_parameter()  # Must match iterator name exactly
```

#### ⚠️ Config File Class Paths
Configuration file must contain accurate, fully qualified class paths that can be imported by Python's import system:
```yaml
evaluation:
  path_deviation: 'examples.path_deviation.PathDeviationEvaluation'
  #               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  #               Must be importable Python path with correct module structure
```

#### ⚠️ Environment Configuration
Configure `.env` file with all required paths and credentials. Reference `.envTEMPLATE` for complete variable list including database tokens and folder paths.

## 3. Implementation Example

### 3.1 Example Overview

![LBP Test Example](examples/UML_test_example.png)

The `examples/` directory demonstrates a complete 3D printing case study implementing path deviation and energy consumption metrics. This example showcases the paired FeatureModel/EvaluationModel architecture and proper inheritance from framework base classes.

### 3.2 Implementation Walkthrough

#### PathDeviationEvaluation & PathDeviationFeature
Comprehensive geometric accuracy assessment comparing designed tool paths against measured execution paths. Demonstrates multi-dimensional evaluation across layers and segments.

```python
@dataclass
class PathDeviationEvaluation(EvaluationModel):
    target_deviation: float = model_parameter()      # Target accuracy level
    max_deviation: float = model_parameter()         # Maximum acceptable deviation
    n_layers: int = exp_parameter()                  # Number of manufacturing layers
    n_segments: int = exp_parameter()                # Segments per layer

    def __init__(self, performance_code, folder_navigator, logger, **study_params):
        dimension_names = [
            ('layers', 'layer_id', 'n_layers'),      # ← Critical naming convention
            ('segments', 'segment_id', 'n_segments') # ← Must match database fields
        ]
        super().__init__(performance_code, folder_navigator, dimension_names, 
                        PathDeviationFeature, logger, **study_params)
```

#### EnergyConsumption & EnergyFeature  
Power consumption analysis during manufacturing process. Demonstrates scalar evaluation (no dimensions) with simple feature computation.

```python
@dataclass
class EnergyFeature(FeatureModel):
    power_rating: float = model_parameter(50.0)     # Equipment power rating
    layerTime: float = exp_parameter()              # Time per layer

    def _compute_features(self, data, visualize_flag):
        energy_consumption = self.power_rating * self.layerTime  # P × t = Energy
        return {"energy_consumption": energy_consumption}
```

#### ExampleDataInterface
Mock implementation demonstrating proper database integration patterns and parameter naming consistency:

```python
class ExampleDataInterface(DataInterface):
    def get_study_parameters(self, study_record):
        return {
            "target_deviation": 0.0,
            "max_deviation": 0.5,
            "n_layers": 2,        # ← Matches dimension parameter name exactly
            "n_segments": 2       # ← Must align with EvaluationModel expectations  
        }

    def get_performance_records(self, study_record):
        return [
            {"Code": "path_deviation", "Active": True},      # ← 'Code' field required
            {"Code": "energy_consumption", "Active": True}   # ← Must match config.yaml keys
        ]
```

### 3.3 Configuration Example

The working `examples/config.yaml` demonstrates proper class path mapping and system configuration:

```yaml
evaluation:
  path_deviation: 'examples.path_deviation.PathDeviationEvaluation'
  energy_consumption: 'examples.energy_consumption.EnergyConsumption'

system:
  debug_mode: false      # Default execution mode
  visualize_flag: false  # Default visualization setting
  round_digits: 3        # Numerical precision for results
  log_level: 'INFO'      # Logging verbosity
```

### 3.4 Testing Infrastructure

The comprehensive test suite validates framework functionality and demonstrates proper usage patterns:

```bash
# Run all tests with detailed output
python tests/run_tests.py --verbose

# Test specific categories  
python tests/run_tests.py --unit          # Core functionality tests
python tests/run_tests.py --integration   # Component interaction tests
python tests/run_tests.py --end-to-end    # Complete workflow tests

# Generate coverage report
python tests/run_tests.py --coverage
```

**Test Categories:**
- **Unit Tests**: Validate individual component functionality, parameter handling, and utility classes
- **Integration Tests**: Verify proper interaction between FeatureModels, EvaluationModels, and the EvaluationSystem
- **End-to-End Tests**: Test complete workflows including configuration loading, study initialization, and evaluation execution

The test suite demonstrates proper framework usage patterns and validates all critical functionality including the naming convention requirements and parameter handling mechanisms highlighted throughout this documentation.
