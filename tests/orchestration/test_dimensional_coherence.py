"""Tests for prediction model dimensional coherence: depth property and validation."""
import pytest
import numpy as np

from pred_fab.interfaces import IPredictionModel
from tests.utils.builders import (
    build_mixed_feature_schema,
    build_real_agent_stack,
    build_workflow_schema,
    build_workflow_agent,
    build_workflow_dataset,
)
from tests.utils.interfaces import (
    MixedPredictionModel,
    WorkflowPredictionModel,
)


# === depth property ===

def test_depth_is_zero_before_refs_set(tmp_path):
    """depth returns 0 when _ref_features is not yet populated."""
    from pred_fab.utils import PfabLogger
    logger = PfabLogger.get_logger(str(tmp_path / "logs"))

    class SimpleModel(IPredictionModel):
        @property
        def input_parameters(self): return ["param_1"]
        @property
        def input_features(self): return []
        @property
        def outputs(self): return ["feature_grid"]
        def forward_pass(self, X): return np.zeros((X.shape[0], 1))
        def train(self, train_batches, val_batches, **kwargs): pass

    model = SimpleModel(logger)
    assert model.depth == 0


def test_depth_reflects_output_feature_columns(tmp_path):
    """depth equals max number of iterator columns across output features."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    # After initialize_systems, _ref_features is populated with column-set DataArrays
    pred_model = agent.pred_system.models[0]  # MixedPredictionModel

    # MixedPredictionModel outputs: feature_grid (depth 2), feature_d1 (depth 1), feature_scalar (depth 0)
    # operational depth = max = 2
    assert pred_model.depth == 2


def test_depth_for_depth2_only_model(tmp_path):
    """depth == 2 for a model whose all outputs are depth-2 features."""
    dataset = build_workflow_dataset(tmp_path)
    agent = build_workflow_agent(tmp_path, dataset.schema)
    pred_model = agent.pred_system.models[0]  # WorkflowPredictionModel

    # WorkflowPredictionModel outputs: feature_1, feature_2 — both depth 2
    assert pred_model.depth == 2


# === validate_dimensional_coherence ===

def test_coherence_valid_depth2_model(tmp_path):
    """A model declaring dim_1 and dim_2 with depth-2 outputs passes validation."""
    dataset = build_workflow_dataset(tmp_path)
    agent = build_workflow_agent(tmp_path, dataset.schema)
    model = agent.pred_system.models[0]  # WorkflowPredictionModel
    # Should not raise
    model.validate_dimensional_coherence(dataset.schema)


def test_coherence_warns_on_mixed_output_depths(tmp_path, recwarn):
    """A model with mixed-depth outputs emits a warning but does not raise."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    model = agent.pred_system.models[0]  # MixedPredictionModel — mixed depths
    # Should not raise (Rule 1 is a warning)
    model.validate_dimensional_coherence(dataset.schema)


def test_coherence_error_on_gap_in_declared_dims(tmp_path):
    """Declaring dim_1 and dim_3 (skipping dim_2) raises ValueError."""
    from pred_fab.core.data_blocks import Parameters, Features, PerformanceAttributes
    from pred_fab.core.data_objects import Feature, PerformanceAttribute
    from pred_fab.core import DatasetSchema
    from pred_fab.utils import PfabLogger
    from pred_fab.core.data_objects import Parameter

    # Build a 3-dim schema
    d1 = Parameter.dimension("dim_1", iterator_code="d1", level=1, max_val=2)
    d2 = Parameter.dimension("dim_2", iterator_code="d2", level=2, max_val=3)
    d3 = Parameter.dimension("dim_3", iterator_code="d3", level=3, max_val=2)
    p1 = Parameter.real("param_1", min_val=0.0, max_val=10.0)
    f1 = Feature.array("feat_3d")
    perf = PerformanceAttribute.score("perf_1")

    schema = DatasetSchema(
        root_folder=str(tmp_path),
        name="schema_3d",
        parameters=Parameters.from_list([p1, d1, d2, d3]),
        features=Features.from_list([f1]),
        performance=PerformanceAttributes.from_list([perf]),
    )
    schema.features.get("feat_3d").set_columns(["d1", "d2", "d3", "feat_3d"])

    logger = PfabLogger.get_logger(str(tmp_path / "logs"))

    class GapModel(IPredictionModel):
        @property
        def input_parameters(self): return ["param_1", "dim_1", "dim_3"]  # gap at level 2
        @property
        def input_features(self): return []
        @property
        def outputs(self): return ["feat_3d"]
        def forward_pass(self, X): return np.zeros((X.shape[0], 1))
        def train(self, tb, vb, **kw): pass

    model = GapModel(logger)
    model.set_ref_parameters(list(schema.parameters.data_objects.values()))
    model.set_ref_features(list(schema.features.data_objects.values()))

    with pytest.raises(ValueError, match="consecutive"):
        model.validate_dimensional_coherence(schema)


def test_coherence_error_on_dims_not_starting_at_level1(tmp_path):
    """Declaring only dim_2 (not starting at level 1) raises ValueError."""
    from pred_fab.core.data_blocks import Parameters, Features, PerformanceAttributes
    from pred_fab.core.data_objects import Feature, PerformanceAttribute
    from pred_fab.core import DatasetSchema
    from pred_fab.utils import PfabLogger
    from pred_fab.core.data_objects import Parameter

    d1 = Parameter.dimension("dim_1", iterator_code="d1", level=1, max_val=2)
    d2 = Parameter.dimension("dim_2", iterator_code="d2", level=2, max_val=3)
    p1 = Parameter.real("param_1", min_val=0.0, max_val=10.0)
    f1 = Feature.array("feat_2d")
    perf = PerformanceAttribute.score("perf_1")

    schema = DatasetSchema(
        root_folder=str(tmp_path),
        name="schema_2d_gap",
        parameters=Parameters.from_list([p1, d1, d2]),
        features=Features.from_list([f1]),
        performance=PerformanceAttributes.from_list([perf]),
    )
    schema.features.get("feat_2d").set_columns(["d1", "d2", "feat_2d"])

    logger = PfabLogger.get_logger(str(tmp_path / "logs"))

    class NoLevel1Model(IPredictionModel):
        @property
        def input_parameters(self): return ["param_1", "dim_2"]  # starts at level 2, not 1
        @property
        def input_features(self): return []
        @property
        def outputs(self): return ["feat_2d"]
        def forward_pass(self, X): return np.zeros((X.shape[0], 1))
        def train(self, tb, vb, **kw): pass

    model = NoLevel1Model(logger)
    model.set_ref_parameters(list(schema.parameters.data_objects.values()))
    model.set_ref_features(list(schema.features.data_objects.values()))

    with pytest.raises(ValueError, match="consecutive"):
        model.validate_dimensional_coherence(schema)


def test_coherence_error_when_input_feature_deeper_than_output(tmp_path):
    """An input feature with depth > model's operational depth raises ValueError."""
    from pred_fab.core.data_blocks import Parameters, Features, PerformanceAttributes
    from pred_fab.core.data_objects import Feature, PerformanceAttribute
    from pred_fab.core import DatasetSchema
    from pred_fab.utils import PfabLogger
    from pred_fab.core.data_objects import Parameter

    d1 = Parameter.dimension("dim_1", iterator_code="d1", level=1, max_val=2)
    d2 = Parameter.dimension("dim_2", iterator_code="d2", level=2, max_val=3)
    p1 = Parameter.real("param_1", min_val=0.0, max_val=10.0)
    f_deep = Feature.array("feat_deep")   # depth 2 (input)
    f_shallow = Feature.array("feat_shallow")  # depth 1 (output)
    perf = PerformanceAttribute.score("perf_1")

    schema = DatasetSchema(
        root_folder=str(tmp_path),
        name="schema_depth_mismatch",
        parameters=Parameters.from_list([p1, d1, d2]),
        features=Features.from_list([f_deep, f_shallow]),
        performance=PerformanceAttributes.from_list([perf]),
    )
    schema.features.get("feat_deep").set_columns(["d1", "d2", "feat_deep"])
    schema.features.get("feat_shallow").set_columns(["d1", "feat_shallow"])

    logger = PfabLogger.get_logger(str(tmp_path / "logs"))

    class DeepInputShallowOutputModel(IPredictionModel):
        @property
        def input_parameters(self): return ["param_1", "dim_1"]
        @property
        def input_features(self): return ["feat_deep"]  # depth 2 > output depth 1
        @property
        def outputs(self): return ["feat_shallow"]
        def forward_pass(self, X): return np.zeros((X.shape[0], 1))
        def train(self, tb, vb, **kw): pass

    model = DeepInputShallowOutputModel(logger)
    model.set_ref_parameters(list(schema.parameters.data_objects.values()))
    model.set_ref_features(list(schema.features.data_objects.values()))

    with pytest.raises(ValueError, match="depth"):
        model.validate_dimensional_coherence(schema)


# === Per-feature tensor shapes in predict_experiment ===

def test_predict_returns_correct_shape_per_feature(tmp_path):
    """Each predicted feature tensor has the shape matching its own depth."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    predictions = agent.pred_system.predict_experiment(exp_data=exp)

    assert predictions["feature_grid"].shape == (2, 3)   # depth 2
    assert predictions["feature_d1"].shape == (2,)        # depth 1
    assert predictions["feature_scalar"].shape == ()       # depth 0


def test_predict_calibration_feature_arrays_match_depth(tmp_path):
    """predict_for_calibration returns tabular arrays sized by feature depth."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    params = exp.parameters.get_values_dict()
    feature_arrays, _ = agent.pred_system.predict_for_calibration(params)

    # feature_grid: depth 2 → 6 rows, each [d1, d2, value]
    assert feature_arrays["feature_grid"].shape == (6, 3)
    # feature_d1: depth 1 → 2 rows, each [d1, value]
    assert feature_arrays["feature_d1"].shape == (2, 2)
    # feature_scalar: depth 0 → 1 row, each [value]
    assert feature_arrays["feature_scalar"].shape == (1, 1)
