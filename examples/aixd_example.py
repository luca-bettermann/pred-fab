"""
Example workflow using AIXD architecture with LBPAgent.

Demonstrates:
- Dataset-centric workflow
- Schema generation from active models
- Two-phase initialization
- Hierarchical load/save
"""

from lbp_package.core import LBPAgent
from lbp_package.interfaces import IEvaluationModel, IFeatureModel

# Example: Define custom evaluation and feature models
# (In practice, import your own implementations)

class ExampleFeatureModel(IFeatureModel):
    """Example feature model implementation."""
    
    @property
    def feature_name(self):
        return "example_feature"
    
    def compute(self, exp_code, exp_folder, visualize_flag, debug_flag):
        # Placeholder: compute features
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
    """Main workflow demonstrating AIXD architecture."""
    
    # === SETUP ===
    
    # Initialize LBP Agent
    agent = LBPAgent(
        root_folder="/path/to/project",
        local_folder="/path/to/local_data",
        log_folder="/path/to/logs",
        debug_flag=True,  # Skip external operations
        recompute_flag=False,
        visualize_flag=True,
        round_digits=3
    )
    
    # Register models
    agent.add_evaluation_model(
        performance_code="energy_consumption",
        evaluation_class=ExampleEvaluationModel,
        weight=0.7  # Calibration weight
    )
    
    # === NEW DATASET WORKFLOW ===
    
    # Define static parameters (study-level)
    static_params = {
        "material": "PLA",
        "printer_type": "FDM",
        "nozzle_diameter": 0.4,
    }
    
    # Define which performance metrics to track
    performance_codes = ["energy_consumption"]
    
    # Initialize dataset (two-phase initialization)
    dataset = agent.initialize_for_dataset(
        performance_codes=performance_codes,
        static_params=static_params
    )
    
    print(f"✓ Dataset initialized with schema: {dataset.schema_id}")
    print(f"  Static params: {len(dataset.schema.static_params.data_objects)}")
    print(f"  Dynamic params: {len(dataset.schema.dynamic_params.data_objects)}")
    print(f"  Dimensional params: {len(dataset.schema.dimensional_params.data_objects)}")
    
    # === RUN EXPERIMENTS ===
    
    # Define experiment parameters
    experiment_params = {
        "print_speed": 50.0,
        "layer_height": 0.2,
        "infill_density": 20,
        "n_layers": 100,  # Dimensional parameter
    }
    
    exp_code = f"{dataset.schema_id}_001"
    
    # Initialize experiment
    agent.initialize_for_exp(exp_code, experiment_params)
    
    # Run evaluation
    results = agent.evaluate_experiment(
        exp_code=exp_code,
        exp_params=experiment_params,
        visualize=True,
        recompute=False
    )
    
    print(f"\n✓ Experiment {exp_code} evaluated")
    for perf_code, metrics in results.items():
        print(f"  {perf_code}: {metrics}")
    
    # === SAVE DATA ===
    
    # Hierarchical save: Memory → Local → External
    agent.save_experiments_hierarchical([exp_code], recompute=False)
    print(f"\n✓ Saved experiment {exp_code}")
    
    # === LOAD EXISTING DATASET ===
    
    # In a new session, load existing dataset by schema_id
    agent2 = LBPAgent(
        root_folder="/path/to/project",
        local_folder="/path/to/local_data",
        log_folder="/path/to/logs",
        debug_flag=True
    )
    
    # Register same models (required for compatibility check)
    agent2.add_evaluation_model(
        performance_code="energy_consumption",
        evaluation_class=ExampleEvaluationModel,
        weight=0.7
    )
    
    # Load dataset by schema_id
    loaded_dataset = agent2.initialize_for_dataset(
        schema_id=dataset.schema_id  # e.g., "schema_001"
    )
    
    print(f"\n✓ Loaded dataset: {loaded_dataset.schema_id}")
    
    # Load experiments from hierarchical storage
    missing = agent2.load_experiments_hierarchical(
        exp_codes=[exp_code],
        recompute=False
    )
    
    if not missing:
        print(f"✓ All experiments loaded successfully")
        exp_params_loaded = loaded_dataset.get_experiment_params(exp_code)
        print(f"  Loaded params: {exp_params_loaded}")
    else:
        print(f"⚠ Could not load experiments: {missing}")
    
    # === ACCESS DATASET CONTENTS ===
    
    # Get all experiments
    all_exp_codes = dataset.get_experiment_codes()
    print(f"\nDataset contains {len(all_exp_codes)} experiments")
    
    # Access specific experiment data
    for code in all_exp_codes:
        params = dataset.get_experiment_params(code)
        print(f"  {code}: {params}")
    
    print("\n✓ AIXD workflow complete!")


if __name__ == "__main__":
    main()
