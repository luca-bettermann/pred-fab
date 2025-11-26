"""
Example workflow using AIXD architecture with LBPAgent.

Demonstrates:
- Dataset-centric workflow (Phase 7 API)
- User owns Dataset, Agent is stateless
- Hierarchical load/save pattern
- ExperimentData mutation pattern
- Rounding configured in DataObjects
"""

from lbp_package.core import LBPAgent, Dataset
from lbp_package.interfaces import IEvaluationModel, IFeatureModel
from lbp_package.utils import LBPLogger
from dataclasses import dataclass, field
from typing import Any

# Example: Define custom evaluation and feature models
# (In practice, import your own implementations)

@dataclass
class ExampleFeatureModel(IFeatureModel):
    """Example feature model implementation."""
    
    dataset: Dataset  # Required for memoization
    logger: LBPLogger  # Required for logging
    
    def _load_data(self, **param_values) -> Any:
        """Load raw data for these parameter values."""
        # Use param_values to locate and load unstructured data
        # e.g., CAD files, sensor data, images, etc.
        return {}  # Placeholder: your custom data format
    
    def _compute_features(self, data: Any, visualize: bool = False) -> float:
        """Extract feature value from loaded data."""
        # Placeholder: compute features from data
        if visualize:
            # User can implement custom visualization logic
            print(f"Visualizing feature extraction...")
        return 1.0

class ExampleEvaluationModel(IEvaluationModel):
    """Example evaluation model implementation."""
    
    @property
    def feature_model_type(self):
        return ExampleFeatureModel
    
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
        return 0.0
    
    @property
    def scaling_factor(self):
        return 1.0


def main():
    """Main workflow demonstrating AIXD architecture with Phase 7 API."""
    
    # === SETUP ===
    
    # Initialize LBP Agent (no round_digits - now in schema/DataObjects)
    agent = LBPAgent(
        root_folder="/Users/TUM/Documents/repos/lbp_package/examples",
        local_folder="/Users/TUM/Documents/repos/lbp_package/examples/local",
        log_folder="/Users/TUM/Documents/repos/lbp_package/examples/logs",
        debug_flag=True,  # Skip external operations
        recompute_flag=False,
        visualize_flag=True
    )
    
    # Register models (no weight parameter - that's for calibration, not evaluation)
    agent.register_evaluation_model(
        performance_code="energy_consumption",
        evaluation_model_class=ExampleEvaluationModel
    )
    
    # === CREATE NEW DATASET ===
    
    # Define static parameters (study-level)
    static_params = {
        "material": "PLA",
        "printer_type": "FDM",
        "nozzle_diameter": 0.4,
    }
    
    # Agent returns Dataset - user owns it
    dataset = agent.initialize(static_params=static_params)
    
    print(f"✓ Dataset initialized with schema: {dataset.schema_id}")
    print(f"  Static params: {len(dataset.schema.static_params.data_objects)}")
    print(f"  Dynamic params: {len(dataset.schema.dynamic_params.data_objects)}")
    print(f"  Dimensional params: {len(dataset.schema.dimensional_params.data_objects)}")
    print(f"  Default rounding: {dataset.schema.default_round_digits} digits")
    
    # === RUN EXPERIMENTS ===
    
    # Define experiment parameters
    experiment_params = {
        "print_speed": 50.0,
        "layer_height": 0.2,
        "infill_density": 20,
        "n_layers": 100,  # Dimensional parameter
    }
    
    exp_code = "test_001"
    
    # Add experiment to dataset (hierarchical: tries local/external first, then creates)
    exp_data = dataset.add_experiment(exp_code, experiment_params)
    
    print(f"\n✓ Experiment {exp_code} created")
    
    # Run evaluation - mutates exp_data in place
    agent.evaluate_experiment(
        dataset=dataset,
        exp_data=exp_data,
        visualize=True,
        recompute=False
    )
    
    print(f"\n✓ Experiment {exp_code} evaluated")
    
    # Access results from exp_data
    for perf_code in exp_data.performance_metrics:
        metrics = exp_data.performance_metrics[perf_code].to_dict()
        print(f"  {perf_code}: {metrics}")
    
    # === SAVE DATA ===
    
    # Save single experiment (hierarchical: memory → local → external)
    dataset.save_experiment(exp_code, local=True, external=False, recompute=False)
    print(f"\n✓ Saved experiment {exp_code} to local storage")
    
    # Or save all experiments at once
    # dataset.save(local=True, external=False, recompute=False)
    
    # === LOAD EXISTING DATASET ===
    
    # In a new session, load existing dataset from local storage
    agent2 = LBPAgent(
        root_folder="/Users/TUM/Documents/repos/lbp_package/examples",
        local_folder="/Users/TUM/Documents/repos/lbp_package/examples/local",
        log_folder="/Users/TUM/Documents/repos/lbp_package/examples/logs",
        debug_flag=True
    )
    
    # Register same models (required for schema compatibility)
    agent2.register_evaluation_model(
        performance_code="energy_consumption",
        evaluation_model_class=ExampleEvaluationModel
    )
    
    # Create empty dataset with same schema
    dataset2 = agent2.initialize(static_params=static_params)
    
    # Load all experiments from local/external storage
    dataset2.populate(source="local")
    
    print(f"\n✓ Loaded dataset: {dataset2.schema_id}")
    
    # Check what was loaded
    all_exp_codes = dataset2.get_experiment_codes()
    print(f"✓ Found {len(all_exp_codes)} experiments")
    
    for code in all_exp_codes:
        params = dataset2.get_experiment_params(code)
        print(f"  {code}: print_speed={params.get('print_speed')}, layer_height={params.get('layer_height')}")
    
    # === ALTERNATIVE: LOAD SPECIFIC EXPERIMENTS ===
    
    # Or load specific experiments (hierarchical: memory → local → external → create)
    exp_data2 = dataset2.add_experiment("test_002")  # Will load from local/external if exists
    
    print(f"\n✓ Loaded or created experiment test_002")
    
    print("\n✓ AIXD workflow complete!")


if __name__ == "__main__":
    main()
