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
    MockFeatureModelC,
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
    feat4 = Feature.array("feature_4")

    perf1 = PerformanceAttribute.score("performance_1")
    perf2 = PerformanceAttribute.score("performance_2")

    param_block = Parameters.from_list([p1, p2, p3, p4, p5])
    feat_block = Features.from_list([feat1, feat2, feat3, feat4])
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
    agent.register_feature_model(MockFeatureModelC)
    agent.register_evaluation_model(MockEvaluationModelA)
    agent.register_evaluation_model(MockEvaluationModelB)
    agent.register_prediction_model(MockPredictionModel)
    agent.initialize_systems(schema, verbose_flag=False)
    # agent.state_report()

    # 5. Load Experiments
    # dataset.populate(verbose_flag=False)
    dataset.load_experiments(['exp_001', 'exp_002', 'exp_003'])

    # evlauate all loaded experiments (if needed)
    dataset.state_report()
    exps = dataset.get_all_experiments()
    for exp in exps:
        # evaluate features and performance attributes
        agent.evaluation_step(exp)

    # save all experiments
    dataset.save_all(recompute_flag=True, verbose_flag=False)

    # FIX SAVE HIERARCHICAL LOGGING
    dataset.state_report()

    # configure calibration settings
    agent.configure_calibration(
        performance_weights={
            "performance_1": 2.0,
            "performance_2": 1.3
        },
        bounds={
            "param_1": (0.0, 10.0), # same as before
            "param_2": (1, 4)       # narrower range
        },
        fixed_params={
            "param_3": "B"          # fix categorical parameter
        },
        adaptation_delta={
            "param_1": 0.1,
            "param_2": 0.5
        }
    )

    agent.calibration_state_report()

    datamodule = DataModule(dataset)
    new_params = agent.exploration_step(datamodule)


if __name__ == "__main__":
    main()
