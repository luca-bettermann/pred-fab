"""Shared builders for schema, interfaces, and real system test stacks."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pred_fab.core import DataModule, Dataset, DatasetSchema
from pred_fab.core.data_blocks import Features, Parameters, PerformanceAttributes, Domains
from pred_fab.core.data_objects import Feature, Parameter, PerformanceAttribute, Dimension, Domain
from pred_fab.orchestration.agent import PfabAgent
from pred_fab.orchestration.calibration import CalibrationSystem
from pred_fab.orchestration.prediction import PredictionSystem
from pred_fab.utils import LocalData, PfabLogger, SplitType
from tests.utils.interfaces import (
    MixedFeatureModelGrid,
    MixedFeatureModelD1,
    MixedFeatureModelScalar,
    MixedPredictionModelGrid,
    MixedPredictionModelD1,
    MixedPredictionModelScalar,
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

    spatial = Domain("spatial", [
        Dimension("dim_1", "d1", 1, 2),
        Dimension("dim_2", "d2", 1, 3),
    ])

    f_grid = Feature.array("feature_grid", domain=spatial)
    f_d1 = Feature.array("feature_d1", domain=spatial, depth=1)
    f_scalar = Feature.array("feature_scalar")

    perf = PerformanceAttribute.score("performance_1")

    params = Parameters.from_list([p1])
    feats = Features.from_list([f_grid, f_d1, f_scalar])
    perfs = PerformanceAttributes.from_list([perf])

    # Mirror FeatureSystem write paths for stable csv export/import boundaries.
    feats.get("feature_scalar").set_columns(["feature_scalar"])

    domains = Domains([spatial])

    return DatasetSchema(
        root_folder=str(tmp_path),
        name=name,
        parameters=params,
        features=feats,
        performance=perfs,
        domains=domains,
    )


def build_workflow_schema(tmp_path, name: str = "schema_001") -> DatasetSchema:
    """Create the end-to-end workflow schema previously used by manual workflow."""
    p1 = Parameter.real("param_1", min_val=0.0, max_val=10.0)
    p2 = Parameter.integer("param_2", min_val=1, max_val=5)
    p3 = Parameter.categorical("param_3", categories=["A", "B", "C"])
    # Runtime-adjustable parameter for adaptation / schedule tests.
    speed = Parameter.real("speed", min_val=0.0, max_val=200.0, runtime=True)

    spatial = Domain("spatial", [
        Dimension("n_layers", "d1", 1, 5),
        Dimension("n_segments", "d2", 1, 5),
    ])

    f1 = Feature.array("feature_1", domain=spatial)
    f2 = Feature.array("feature_2", domain=spatial)
    f3 = Feature.array("feature_3")

    perf1 = PerformanceAttribute.score("performance_1")
    perf2 = PerformanceAttribute.score("performance_2")

    domains = Domains([spatial])

    return DatasetSchema(
        root_folder=str(tmp_path),
        name=name,
        parameters=Parameters.from_list([p1, p2, p3, speed]),
        features=Features.from_list([f1, f2, f3]),
        performance=PerformanceAttributes.from_list([perf1, perf2]),
        domains=domains,
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
    """Apply the workflow calibration configuration used across integration tests.

    Trust regions are configured only for ``speed`` — the one runtime-adjustable parameter
    in the workflow schema. Static parameters (``param_1``, ``param_2``) cannot receive
    trust regions as of Phase 2 validation.
    """
    agent.configure_performance(weights={"performance_1": 2.0, "performance_2": 1.3})
    agent.calibration_system.configure_param_bounds(
        {"param_1": (0.0, 10.0), "param_2": (1, 4), "n_layers": (1, 3), "n_segments": (1, 3)}
    )
    agent.calibration_system.configure_fixed_params({"param_3": "B"})
    agent.calibration_system.configure_adaptation_delta({"speed": 10.0})  # only runtime-adjustable params may have deltas


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
        datamodule.set_split_codes(
            train_codes=split_codes.get(SplitType.TRAIN, []),
            val_codes=split_codes.get(SplitType.VAL, []),
            test_codes=split_codes.get(SplitType.TEST, []),
        )
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
    system.models = models  # type: ignore[assignment]
    system.datamodule = datamodule
    return system, models


def build_calibration_system(
    tmp_path,
    dataset: Dataset,
    perf_fn: Optional[Callable] = None,
    uncertainty_fn: Optional[Callable] = None,
    delta_integrated_evidence_fn: Optional[Callable] = None,
) -> CalibrationSystem:
    """Build a CalibrationSystem with lightweight no-op callables for unit tests."""
    logger = build_test_logger(tmp_path)
    schema = dataset.schema
    perf_names = list(schema.performance_attrs.keys())

    def _default_perf_fn(params_dict):
        return {name: 0.5 for name in perf_names}

    cal = CalibrationSystem(
        schema=schema,
        logger=logger,
        perf_fn=perf_fn or _default_perf_fn,  # type: ignore[arg-type]
        uncertainty_fn=uncertainty_fn or (lambda x: 1.0),
        delta_integrated_evidence_fn=delta_integrated_evidence_fn,
    )
    # Fast DE settings for tests (production uses scipy defaults: 1000/15)
    cal.de_maxiter = 5
    cal.de_popsize = 2
    return cal


def build_real_agent_stack(tmp_path):
    """Build real orchestration stack for step-level tests."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    schema = dataset.schema
    exp = dataset.get_experiment("exp_001")

    agent = PfabAgent(root_folder=str(tmp_path), debug_flag=True)
    agent.register_feature_model(MixedFeatureModelGrid)
    agent.register_feature_model(MixedFeatureModelD1)
    agent.register_feature_model(MixedFeatureModelScalar)
    agent.register_evaluation_model(ScalarEvaluationModel)
    agent.register_prediction_model(MixedPredictionModelGrid)
    agent.register_prediction_model(MixedPredictionModelD1)
    agent.register_prediction_model(MixedPredictionModelScalar)
    agent.initialize_systems(schema, verbose_flag=False)

    datamodule = agent.create_datamodule(dataset)
    return agent, dataset, exp, datamodule


def build_runtime_agent_stack(tmp_path):
    """Build a real orchestration stack with a runtime-adjustable ``speed`` parameter.

    Used by adaptation / schedule tests that need a schema containing at least one
    ``runtime=True`` parameter.
    """
    from pred_fab.core.data_objects import Feature, PerformanceAttribute, Dimension, Domain
    from pred_fab.core.data_blocks import Parameters, Features, PerformanceAttributes, Domains
    from pred_fab.core import Dataset, DatasetSchema

    p1 = Parameter.real("param_1", min_val=0.0, max_val=10.0)
    speed = Parameter.real("speed", min_val=0.0, max_val=200.0, runtime=True)

    spatial = Domain("spatial", [
        Dimension("dim_1", "d1", 1, 2),
        Dimension("dim_2", "d2", 1, 3),
    ])

    f_grid = Feature.array("feature_grid", domain=spatial)
    f_d1 = Feature.array("feature_d1", domain=spatial, depth=1)
    f_scalar = Feature.array("feature_scalar")
    perf = PerformanceAttribute.score("performance_1")

    feats = Features.from_list([f_grid, f_d1, f_scalar])
    perfs = PerformanceAttributes.from_list([perf])
    feats.get("feature_scalar").set_columns(["feature_scalar"])

    domains = Domains([spatial])

    schema = DatasetSchema(
        root_folder=str(tmp_path),
        name="schema_runtime",
        parameters=Parameters.from_list([p1, speed]),
        features=feats,
        performance=perfs,
        domains=domains,
    )
    dataset = Dataset(schema=schema, debug_flag=True)
    dataset.create_experiment(
        "exp_001",
        parameters={"param_1": 2.5, "speed": 100.0, "dim_1": 2, "dim_2": 3},
    )
    exp = dataset.get_experiment("exp_001")

    from pred_fab.orchestration.agent import PfabAgent as _PfabAgent
    agent = _PfabAgent(root_folder=str(tmp_path), debug_flag=True)
    agent.register_feature_model(MixedFeatureModelGrid)
    agent.register_feature_model(MixedFeatureModelD1)
    agent.register_feature_model(MixedFeatureModelScalar)
    agent.register_evaluation_model(ScalarEvaluationModel)
    agent.register_prediction_model(MixedPredictionModelGrid)
    agent.register_prediction_model(MixedPredictionModelD1)
    agent.register_prediction_model(MixedPredictionModelScalar)
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
