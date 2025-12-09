"""
Example workflow using LBP Package.
"""

import os
import shutil
from lbp_package.core import DatasetSchema
from lbp_package.core.data_objects import Parameter, Feature, PerformanceAttribute
from lbp_package.core.data_blocks import Parameters, PerformanceAttributes, Features
from lbp_package.orchestration.agent import LBPAgent
from lbp_package.core.dataset import ExperimentData
from lbp_package.core import DataModule

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
    root_folder = "./examples/test_workflow"
    # Clean up previous run if exists
    if os.path.exists(root_folder):
        shutil.rmtree(root_folder)
    os.makedirs(root_folder, exist_ok=True)

    # 2. Define Schema
    print("\n--- Defining Schema ---")

    p1 = Parameter.real("param_1", min_val=0.0, max_val=10.0)
    p2 = Parameter.integer("param_2", min_val=1, max_val=5)
    p3 = Parameter.dimension("dim_1", iterator_code="d1", level=1)
    p4 = Parameter.dimension("dim_2", iterator_code="d2", level=2)

    feat1 = Feature.array("feature_1")
    feat2 = Feature.array("feature_2")
    feat3 = Feature.array("feature_3")

    perf1 = PerformanceAttribute.score("performance_1")
    perf2 = PerformanceAttribute.score("performance_2")

    param_block = Parameters.from_list([p1, p2, p3, p4])
    feat_block = Features.from_list([feat1, feat2, feat3])
    perf_block = PerformanceAttributes.from_list([perf1, perf2])
    
    # Initialize Schema
    schema = DatasetSchema(
        parameters=param_block,
        features=feat_block,
        performance=perf_block,
    )

    # 3. Initialize Agent
    # We need to assert external_data is not None for type checker if we use it later
    ext_data = MockExternalData()
    agent = LBPAgent(
        root_folder=root_folder,
        external_data=ext_data,
        debug_flag=False, # Use local data only (mock external data is used via interface)
        recompute_flag=False
    )

    # 4. Register Models
    agent.register_feature_model(MockFeatureModelA)
    agent.register_feature_model(MockFeatureModelB)
    agent.register_evaluation_model(MockEvaluationModelA)
    agent.register_evaluation_model(MockEvaluationModelB)
    agent.register_prediction_model(MockPredictionModel)

    # 5. Initialize Dataset (Schema Validation & Hashing)
    dataset = agent.initialize(schema)
    dataset.load_experiments(["exp_001", "exp_002", "exp_003"])
    exp_1 = dataset.get_experiment("exp_001")
    exp_2 = dataset.get_experiment("exp_002")

    # 6. Run Feature Extraction Step
    agent.evaluation_step(exp_1)
    agent.evaluation_step(exp_2)

    last_exp = dataset.get_experiment("exp_003")

    # Run full step (Feature Extraction -> Evaluation -> Training)
    # We need to create a DataModule for training
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
