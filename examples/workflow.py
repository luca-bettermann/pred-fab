"""
Example workflow using LBP Package.
"""

import os
from pred_fab.core import DatasetSchema, Dataset
from pred_fab.core.data_objects import Parameter, Feature, PerformanceAttribute
from pred_fab.core.data_blocks import Parameters, PerformanceAttributes, Features
from pred_fab.orchestration.agent import PfabAgent
from pred_fab.core.dataset import ExperimentData
from pred_fab.core import DataModule

from pred_fab.utils import StepType

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
    os.makedirs(root_folder, exist_ok=True)

    # 2. Define Schema
    p1 = Parameter.real("param_1", min_val=0.0, max_val=10.0)
    p2 = Parameter.integer("param_2", min_val=1, max_val=5)
    p3 = Parameter.dimension("dim_1", iterator_code="d1", level=1)
    p4 = Parameter.dimension("dim_2", iterator_code="d2", level=2)
    p5 = Parameter.categorical("param_3", categories=["A", "B", "C"])

    feat1 = Feature.array("feature_1")
    feat2 = Feature.array("feature_2")
    feat3 = Feature.array("feature_3")

    perf1 = PerformanceAttribute.score("performance_1")
    perf2 = PerformanceAttribute.score("performance_2")

    param_block = Parameters.from_list([p1, p2, p3, p4])
    feat_block = Features.from_list([feat1, feat2, feat3])
    perf_block = PerformanceAttributes.from_list([perf1, perf2])
    
    # Initialize Schema and Dataset
    schema = DatasetSchema(
        name="schema_001",
        root_folder=root_folder,
        parameters=param_block,
        features=feat_block,
        performance=perf_block,
    )

    dataset = Dataset(
        schema=schema,
        external_data=MockExternalData(),
        debug_flag=False,
    )

    # 3. Initialize Agent
    agent = PfabAgent(
        root_folder=root_folder,
        debug_flag=False,
        recompute_flag=False
    )

    # 4. Register Models and Initialize Systems
    agent.register_feature_model(MockFeatureModelA)
    agent.register_feature_model(MockFeatureModelB)
    agent.register_evaluation_model(MockEvaluationModelA)
    agent.register_evaluation_model(MockEvaluationModelB)
    agent.register_prediction_model(MockPredictionModel)
    agent.initialize_systems(schema)

    # 5. Load Experiments
    dataset.load_experiments(["exp_001", "exp_002", "exp_003"])
    exp_1 = dataset.get_experiment("exp_001")
    exp_2 = dataset.get_experiment("exp_002")

    # 6. Run Feature Extraction Step
    agent.step_offline(exp_1, step_type=StepType.EVAL)
    agent.step_offline(exp_2, step_type=StepType.EVAL)

    last_exp = dataset.get_experiment("exp_003")

    # Run full step (Feature Extraction -> Evaluation -> Training)
    # We need to create a DataModule for training
    datamodule = DataModule(dataset)
    
    agent.step_offline(
        exp_data=last_exp,
        datamodule=datamodule
    )

if __name__ == "__main__":
    main()
