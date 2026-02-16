"""Shared builders for schema, interfaces, and real system test stacks."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pred_fab.core import DataModule, Dataset, DatasetSchema
from pred_fab.core.data_blocks import Features, Parameters, PerformanceAttributes
from pred_fab.core.data_objects import Feature, Parameter, PerformanceAttribute
from pred_fab.orchestration.agent import PfabAgent
from pred_fab.orchestration.calibration import CalibrationSystem
from pred_fab.orchestration.prediction import PredictionSystem
from pred_fab.utils import LocalData, PfabLogger, SplitType
from tests.utils.interfaces import (
    CapturingSurrogateModel,
    MixedFeatureModel,
    MixedPredictionModel,
    ScalarEvaluationModel,
    ShapeCheckingPredictionModel,
    WorkflowEvaluationModelA,
    WorkflowEvaluationModelB,
    WorkflowExternalData,
    WorkflowFeatureModelA,
    WorkflowFeatureModelB,
    WorkflowPredictionModel,
)


def build_test_logger(tmp_path) -> PfabLogger:
    """Create a test logger scoped to the temporary test folder."""
    return PfabLogger.get_logger(str(Path(tmp_path) / "logs"))


def build_mixed_feature_schema(tmp_path, name: str = "schema_test") -> DatasetSchema:
    """Create a mixed-dimensional schema for tensor/table conversion tests."""
    p1 = Parameter.real("param_1", min_val=0.0, max_val=10.0)
    d1 = Parameter.dimension("dim_1", iterator_code="d1", level=1, max_val=2)
    d2 = Parameter.dimension("dim_2", iterator_code="d2", level=2, max_val=3)

    f_grid = Feature.array("feature_grid")
    f_d1 = Feature.array("feature_d1")
    f_scalar = Feature.array("feature_scalar")

    perf = PerformanceAttribute.score("performance_1")

    params = Parameters.from_list([p1, d1, d2])
    feats = Features.from_list([f_grid, f_d1, f_scalar])
    perfs = PerformanceAttributes.from_list([perf])

    # Mirror FeatureSystem write paths for stable csv export/import boundaries.
    feats.get("feature_grid").set_columns(["d1", "d2", "feature_grid"])
    feats.get("feature_d1").set_columns(["d1", "feature_d1"])
    feats.get("feature_scalar").set_columns(["feature_scalar"])

    return DatasetSchema(
        root_folder=str(tmp_path),
        name=name,
        parameters=params,
        features=feats,
        performance=perfs,
    )


def build_workflow_schema(tmp_path, name: str = "schema_001") -> DatasetSchema:
    """Create the end-to-end workflow schema previously used by manual workflow."""
    p1 = Parameter.real("param_1", min_val=0.0, max_val=10.0)
    p2 = Parameter.integer("param_2", min_val=1, max_val=5)
    d1 = Parameter.dimension("dim_1", iterator_code="d1", level=1, max_val=5)
    d2 = Parameter.dimension("dim_2", iterator_code="d2", level=2, max_val=5)
    p3 = Parameter.categorical("param_3", categories=["A", "B", "C"])

    f1 = Feature.array("feature_1")
    f2 = Feature.array("feature_2")
    f3 = Feature.array("feature_3")

    perf1 = PerformanceAttribute.score("performance_1")
    perf2 = PerformanceAttribute.score("performance_2")

    return DatasetSchema(
        root_folder=str(tmp_path),
        name=name,
        parameters=Parameters.from_list([p1, p2, d1, d2, p3]),
        features=Features.from_list([f1, f2, f3]),
        performance=PerformanceAttributes.from_list([perf1, perf2]),
    )


def build_dataset_with_single_experiment(tmp_path) -> Dataset:
    """Create a dataset with one initialized experiment for focused unit tests."""
    schema = build_mixed_feature_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    dataset.create_experiment(
        "exp_001",
        parameters={"param_1": 2.5, "dim_1": 2, "dim_2": 3},
    )
    return dataset


def build_workflow_dataset(tmp_path, schema_name: str = "schema_001") -> Dataset:
    """Create workflow dataset with external parameter source enabled."""
    schema = build_workflow_schema(tmp_path, name=schema_name)
    return Dataset(schema=schema, external_data=WorkflowExternalData(), debug_flag=False)


def build_workflow_agent(tmp_path, schema: DatasetSchema) -> PfabAgent:
    """Create workflow agent with full feature/eval/prediction registration."""
    agent = PfabAgent(root_folder=str(tmp_path), debug_flag=False, recompute_flag=False)
    agent.register_feature_model(WorkflowFeatureModelA)
    agent.register_feature_model(WorkflowFeatureModelB)
    agent.register_evaluation_model(WorkflowEvaluationModelA)
    agent.register_evaluation_model(WorkflowEvaluationModelB)
    agent.register_prediction_model(WorkflowPredictionModel)
    agent.initialize_systems(schema, verbose_flag=False)
    return agent


def build_workflow_stack(tmp_path, exp_codes: Optional[List[str]] = None) -> Tuple[PfabAgent, Dataset, List[str]]:
    """Build workflow dataset + agent and load external experiment parameters."""
    dataset = build_workflow_dataset(tmp_path)
    agent = build_workflow_agent(tmp_path, dataset.schema)
    codes = exp_codes or ["exp_001", "exp_002", "exp_003"]
    dataset.load_experiments(codes, recompute_flag=False, verbose=False)
    return agent, dataset, codes


def evaluate_loaded_workflow_experiments(agent: PfabAgent, dataset: Dataset, category_value: str = "B") -> None:
    """Evaluate all loaded workflow experiments after setting categorical parameter."""
    for exp in dataset.get_all_experiments():
        exp.parameters.set_value("param_3", category_value)
        agent.evaluate(exp_data=exp, recompute_flag=False, visualize=False)


def configure_default_workflow_calibration(agent: PfabAgent) -> None:
    """Apply the workflow calibration configuration used across integration tests."""
    agent.configure_calibration(
        performance_weights={"performance_1": 2.0, "performance_2": 1.3},
        bounds={"param_1": (0.0, 10.0), "param_2": (1, 4), "dim_1": (1, 3), "dim_2": (1, 3)},
        fixed_params={"param_3": "B"},
        adaptation_delta={"param_1": 0.1, "param_2": 0.5},
    )


def build_prepared_workflow_datamodule(
    agent: PfabAgent,
    dataset: Dataset,
    val_size: float = 0.0,
    test_size: float = 0.0,
    recompute: bool = True,
) -> DataModule:
    """Create and prepare workflow datamodule for training/calibration."""
    datamodule = agent.create_datamodule(dataset)
    datamodule.prepare(val_size=val_size, test_size=test_size, recompute=recompute)
    return datamodule


def sample_feature_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic feature tables for mixed-dimensionality fixtures."""
    grid_rows = []
    for i in range(2):
        for j in range(3):
            grid_rows.append([i, j, i * 10 + j])

    d1_rows = [[i, 100 + i] for i in range(2)]
    scalar_rows = [[7.0]]

    return (
        np.array(grid_rows, dtype=np.float64),
        np.array(d1_rows, dtype=np.float64),
        np.array(scalar_rows, dtype=np.float64),
    )


def populate_single_experiment_features(dataset: Dataset):
    """Populate deterministic mixed-dimensional feature tensors in exp_001."""
    exp = dataset.get_experiment("exp_001")
    grid, d1_only, scalar = sample_feature_tables()
    exp.features.set_value("feature_grid", exp.features.table_to_tensor("feature_grid", grid, exp.parameters))
    exp.features.set_value("feature_d1", exp.features.table_to_tensor("feature_d1", d1_only, exp.parameters))
    exp.features.set_value("feature_scalar", exp.features.table_to_tensor("feature_scalar", scalar, exp.parameters))
    return exp


def build_initialized_datamodule(
    dataset: Dataset,
    input_parameters: List[str],
    input_features: List[str],
    output_columns: List[str],
    fitted: bool = False,
    split_codes: Optional[Dict[SplitType, List[str]]] = None,
) -> DataModule:
    """Build and initialize a datamodule with optional pre-fitted split metadata."""
    datamodule = DataModule(dataset)
    datamodule.initialize(
        input_parameters=input_parameters,
        input_features=input_features,
        output_columns=output_columns,
    )
    if split_codes is not None:
        datamodule._split_codes = split_codes
    if fitted:
        datamodule._is_fitted = True
    return datamodule


def build_shape_checking_prediction_system(
    tmp_path,
    dataset: Dataset,
    datamodule: DataModule,
    model_specs: Sequence[Tuple[List[str], List[str]]],
) -> Tuple[PredictionSystem, List[ShapeCheckingPredictionModel]]:
    """Build prediction system with reusable shape-checking prediction interfaces."""
    logger = build_test_logger(tmp_path)
    system = PredictionSystem(logger=logger, schema=dataset.schema, local_data=LocalData(str(tmp_path)))
    models = [
        ShapeCheckingPredictionModel(logger=logger, in_params=in_params, outputs=outputs)
        for in_params, outputs in model_specs
    ]
    system.models = models
    system.datamodule = datamodule
    return system, models


def build_calibration_system_with_capturing_surrogate(
    tmp_path,
    dataset: Dataset,
    predict_fn: Optional[Callable[[np.ndarray], Dict[str, np.ndarray]]] = None,
    residual_predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    evaluate_fn: Optional[Callable[[np.ndarray], Dict[str, np.ndarray]]] = None,
) -> Tuple[CalibrationSystem, CapturingSurrogateModel]:
    """Build calibration system using a reusable capturing surrogate interface."""
    logger = build_test_logger(tmp_path)
    surrogate = CapturingSurrogateModel(logger)
    calibration = CalibrationSystem(
        schema=dataset.schema,
        logger=logger,
        predict_fn=predict_fn or (lambda x: {"feature_scalar": np.zeros((len(x), 1))}),
        residual_predict_fn=residual_predict_fn or (lambda x: np.zeros((len(x), 1))),
        evaluate_fn=evaluate_fn or (lambda x: {"performance_1": np.zeros((len(x), 1))}),
        surrogate_model=surrogate,
    )
    return calibration, surrogate


def build_real_agent_stack(tmp_path):
    """Build real orchestration stack for step-level tests."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    schema = dataset.schema
    exp = dataset.get_experiment("exp_001")

    agent = PfabAgent(root_folder=str(tmp_path), debug_flag=True)
    agent.register_feature_model(MixedFeatureModel)
    agent.register_evaluation_model(ScalarEvaluationModel)
    agent.register_prediction_model(MixedPredictionModel)
    agent.initialize_systems(schema, verbose_flag=False)

    datamodule = agent.create_datamodule(dataset)
    return agent, dataset, exp, datamodule


def collect_workflow_local_artifact_paths(root_folder: str, schema_name: str, exp_codes: List[str]) -> List[Path]:
    """Return expected local artifact paths for workflow persistence checks."""
    local_root = Path(root_folder) / "local"
    schema_root = local_root / schema_name
    paths = [
        local_root / "schema_registry.json",
        schema_root / "schema.json",
    ]

    for code in exp_codes:
        exp_root = schema_root / code
        paths.extend(
            [
                exp_root / "parameters.json",
                exp_root / "performance_attrs.json",
                exp_root / "feature_1.csv",
                exp_root / "feature_2.csv",
                exp_root / "feature_3.csv",
            ]
        )
    return paths
