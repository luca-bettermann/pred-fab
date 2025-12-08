"""
Example workflow using LBP Package.
"""

import os
import shutil
from lbp_package.core import DatasetSchema
from lbp_package.core.data_objects import DataReal, DataInt, DataDimension, DataArray, PerformanceAttribute
from lbp_package.core.data_blocks import Parameters, PerformanceAttributes, Features
from lbp_package.orchestration.agent import LBPAgent
from lbp_package.core.dataset import ExperimentData

# Import mock interfaces
from interfaces import (
    MockFeatureModelA,
    MockFeatureModelB,
    MockEvaluationModelA,
    MockEvaluationModelB,
    MockPredictionModel, 
    MockExternalData
)

def main():
    # 1. Setup Workspace
    root_folder = "./examples"
    # Clean up previous run if exists
    if os.path.exists(root_folder):
        shutil.rmtree(root_folder)
    os.makedirs(root_folder, exist_ok=True)

    # 2. Define Schema
    print("\n--- Defining Schema ---")
    
    # Define DataObjects
    parameters = [
        DataReal("param_1", min_val=0.0, max_val=10.0),
        DataInt("param_2", min_val=1, max_val=5),
        DataDimension("dim_1", iterator_code="d1", level=1),
        DataDimension("dim_2", iterator_code="d2", level=2)
    ]
    
    performance_attrs = [
        DataReal("performance_1", min_val=0, max_val=100),
        DataReal("performance_2", min_val=0, max_val=100)
    ]
    
    features_list = [
        DataArray("feature_1"),
        DataArray("feature_2"),
        DataArray("feature_3")
    ]
    
    # Build blocks
    parameter_block = Parameters.from_list(parameters)
    performance_block = PerformanceAttributes.from_list(performance_attrs)
    feature_block = Features.from_list(features_list)
    
    # Initialize Schema
    schema = DatasetSchema(
        parameters=parameter_block,
        performance=performance_block,
        features=feature_block
    )

    # 3. Initialize Agent
    print("\n--- Initializing Agent ---")
    # We need to assert external_data is not None for type checker if we use it later
    ext_data = MockExternalData()
    agent = LBPAgent(
        root_folder=root_folder,
        external_data=ext_data,
        debug_flag=True, # Use local data only (mock external data is used via interface)
        recompute_flag=True
    )

    # 4. Register Models
    print("\n--- Registering Models ---")
    agent.register_feature_model(MockFeatureModelA)
    agent.register_feature_model(MockFeatureModelB)
    agent.register_evaluation_model(MockEvaluationModelA)
    agent.register_evaluation_model(MockEvaluationModelB)
    agent.register_prediction_model(MockPredictionModel)

    # 5. Initialize Dataset (Schema Validation & Hashing)
    print("\n--- Initializing Dataset ---")
    dataset = agent.initialize(schema)

    # 6. Add Experiments
    print("\n--- Adding Experiments ---")
    exp_codes = ["exp_001", "exp_002", "exp_003"]
    
    # Pull parameters from "external" source (MockExternalData)
    missing, params_dict = ext_data.pull_parameters(exp_codes)
    
    for code in exp_codes:
        if code in params_dict:
            print(f"Adding experiment {code}...")
            # Use dataset.create_experiment to handle object creation
            dataset.create_experiment(
                exp_code=code,
                parameters=params_dict[code]
            )

    # 7. Run Agent Step
    print("\n--- Running Agent Step ---")
    
    # Set active experiment (e.g., the last one)
    last_exp = dataset.get_experiment("exp_003")
    agent.set_active_experiment(last_exp)
    
    # Run full step (Feature Extraction -> Evaluation -> Training)
    # We need to create a DataModule for training
    from lbp_package.core import DataModule
    datamodule = DataModule(dataset)
    
    agent.step(
        exp_data=last_exp,
        datamodule=datamodule,
        recompute=True,
        visualize=False,
        online=False, # Offline mode (training)
        epochs=1 # Kwarg for training
    )

    print("\n--- Workflow Complete ---")

if __name__ == "__main__":
    main()
