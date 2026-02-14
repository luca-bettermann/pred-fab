"""Manual end-to-end workflow module.

Run directly for click-through debugging:
    python -m tests.workflows.manual_workflow
"""

from pathlib import Path

from pred_fab.core import DatasetSchema, Dataset, DataModule
from pred_fab.core.data_objects import Parameter, Feature, PerformanceAttribute
from pred_fab.core.data_blocks import Parameters, Features, PerformanceAttributes
from pred_fab.orchestration.agent import PfabAgent

from tests.workflows.interfaces import (
    WorkflowFeatureModelA,
    WorkflowFeatureModelB,
    WorkflowEvaluationModelA,
    WorkflowEvaluationModelB,
    WorkflowPredictionModel,
    WorkflowExternalData,
)


def run_workflow(root_folder: str) -> dict:
    # Schema
    p1 = Parameter.real("param_1", min_val=0.0, max_val=10.0)
    p2 = Parameter.integer("param_2", min_val=1, max_val=5)
    p3 = Parameter.dimension("dim_1", iterator_code="d1", level=1, max_val=5)
    p4 = Parameter.dimension("dim_2", iterator_code="d2", level=2, max_val=5)
    p5 = Parameter.categorical("param_3", categories=["A", "B", "C"])

    f1 = Feature.array("feature_1")
    f2 = Feature.array("feature_2")
    f3 = Feature.array("feature_3")

    perf1 = PerformanceAttribute.score("performance_1")
    perf2 = PerformanceAttribute.score("performance_2")

    schema = DatasetSchema(
        root_folder=root_folder,
        name="schema_001",
        parameters=Parameters.from_list([p1, p2, p3, p4, p5]),
        features=Features.from_list([f1, f2, f3]),
        performance=PerformanceAttributes.from_list([perf1, perf2]),
    )

    dataset = Dataset(schema=schema, external_data=WorkflowExternalData(), debug_flag=False)

    # Agent + systems
    agent = PfabAgent(root_folder=root_folder, debug_flag=False, recompute_flag=False)
    agent.register_feature_model(WorkflowFeatureModelA)
    agent.register_feature_model(WorkflowFeatureModelB)
    agent.register_evaluation_model(WorkflowEvaluationModelA)
    agent.register_evaluation_model(WorkflowEvaluationModelB)
    agent.register_prediction_model(WorkflowPredictionModel)
    agent.initialize_systems(schema, verbose_flag=False)

    # Load experiments and evaluate
    dataset.load_experiments(["exp_001", "exp_002", "exp_003"])
    for exp in dataset.get_all_experiments():
        exp.parameters.set_value("param_3", "B")
        agent.evaluation_step(exp)

    dataset.save_all(recompute_flag=True, verbose_flag=False)

    # Calibration config + exploration step
    agent.configure_calibration(
        performance_weights={"performance_1": 2.0, "performance_2": 1.3},
        bounds={"param_1": (0.0, 10.0), "param_2": (1, 4), "dim_1": (1, 3), "dim_2": (1, 3)},
        fixed_params={"param_3": "B"},
        adaptation_delta={"param_1": 0.1, "param_2": 0.5},
    )

    datamodule = agent.create_datamodule(dataset)
    datamodule.prepare()
    new_params = agent.exploration_step(datamodule)
    return new_params


def main() -> None:
    root = Path("./tests/workflows").resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_workflow(str(root))


if __name__ == "__main__":
    main()
