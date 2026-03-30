"""Tests for prediction model dimensional coherence: depth property and validation."""
import pytest
import numpy as np
from typing import Optional

from pred_fab.interfaces import IPredictionModel
from pred_fab.core.data_blocks import Parameters, Features, PerformanceAttributes, Domains
from pred_fab.core.data_objects import Feature, PerformanceAttribute, Dimension, Domain
from pred_fab.core import DatasetSchema
from pred_fab.utils import PfabLogger
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
    logger = PfabLogger.get_logger(str(tmp_path / "logs"))

    class SimpleModel(IPredictionModel):
        @property
        def input_parameters(self): return ["param_1"]
        @property
        def input_features(self): return []
        @property
        def outputs(self): return ["feature_grid"]
        @property
        def input_domain(self) -> Optional[str]: return "spatial"
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
    """A model with depth-2 spatial domain outputs passes validation."""
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


def test_coherence_error_when_input_feature_deeper_than_output(tmp_path):
    """An input feature with depth > model's operational depth raises ValueError."""
    from pred_fab.core.data_objects import Parameter

    p1 = Parameter.real("param_1", min_val=0.0, max_val=10.0)
    spatial = Domain("spatial", [Dimension("dim_1", "d1", 1, 2), Dimension("dim_2", "d2", 1, 3)])
    f_deep = Feature.array("feat_deep", domain=spatial)    # depth 2 (input)
    f_shallow = Feature.array("feat_shallow", domain=spatial, depth=1)  # depth 1 (output)
    perf = PerformanceAttribute.score("perf_1")

    domains = Domains([spatial])

    schema = DatasetSchema(
        root_folder=str(tmp_path),
        name="schema_depth_mismatch",
        parameters=Parameters.from_list([p1]),
        features=Features.from_list([f_deep, f_shallow]),
        performance=PerformanceAttributes.from_list([perf]),
        domains=domains,
    )

    logger = PfabLogger.get_logger(str(tmp_path / "logs"))

    class DeepInputShallowOutputModel(IPredictionModel):
        @property
        def input_parameters(self): return ["param_1"]
        @property
        def input_features(self): return ["feat_deep"]  # depth 2 > output depth 1
        @property
        def outputs(self): return ["feat_shallow"]
        @property
        def input_domain(self) -> Optional[str]: return "spatial"
        def forward_pass(self, X): return np.zeros((X.shape[0], 1))
        def train(self, tb, vb, **kw): pass

    model = DeepInputShallowOutputModel(logger)
    model.set_ref_parameters(list(schema.parameters.data_objects.values()))
    model.set_ref_features(list(schema.features.data_objects.values()))

    with pytest.raises(ValueError, match="depth"):
        model.validate_dimensional_coherence(schema)


def test_coherence_error_when_input_domain_mismatches_output_domain(tmp_path):
    """input_domain that doesn't match the output feature domain raises ValueError."""
    from pred_fab.core.data_objects import Parameter

    p1 = Parameter.real("param_1", min_val=0.0, max_val=10.0)
    spatial = Domain("spatial", [Dimension("dim_1", "d1", 1, 2), Dimension("dim_2", "d2", 1, 3)])
    f1 = Feature.array("feat_spatial", domain=spatial)
    perf = PerformanceAttribute.score("perf_1")

    domains = Domains([spatial, Domain("temporal", [Dimension("t_step", "t", 1, 5)])])

    schema = DatasetSchema(
        root_folder=str(tmp_path),
        name="schema_domain_mismatch",
        parameters=Parameters.from_list([p1]),
        features=Features.from_list([f1]),
        performance=PerformanceAttributes.from_list([perf]),
        domains=domains,
    )

    logger = PfabLogger.get_logger(str(tmp_path / "logs"))

    class WrongDomainModel(IPredictionModel):
        @property
        def input_parameters(self): return ["param_1"]
        @property
        def input_features(self): return []
        @property
        def outputs(self): return ["feat_spatial"]
        @property
        def input_domain(self) -> Optional[str]: return "temporal"  # mismatch
        def forward_pass(self, X): return np.zeros((X.shape[0], 1))
        def train(self, tb, vb, **kw): pass

    model = WrongDomainModel(logger)
    model.set_ref_parameters(list(schema.parameters.data_objects.values()))
    model.set_ref_features(list(schema.features.data_objects.values()))

    with pytest.raises(ValueError, match="input_domain"):
        model.validate_dimensional_coherence(schema)


def test_coherence_error_when_outputs_span_multiple_named_domains(tmp_path):
    """A model outputting features from two different named domains raises ValueError."""
    from pred_fab.core.data_objects import Parameter

    p1 = Parameter.real("param_1", min_val=0.0, max_val=10.0)
    spatial = Domain("spatial", [Dimension("dim_1", "d1", 1, 2)])
    temporal = Domain("temporal", [Dimension("t_step", "t", 1, 5)])
    f_spatial = Feature.array("feat_spatial", domain=spatial)
    f_temporal = Feature.array("feat_temporal", domain=temporal)
    perf = PerformanceAttribute.score("perf_1")

    domains = Domains([spatial, temporal])

    schema = DatasetSchema(
        root_folder=str(tmp_path),
        name="schema_multi_domain_model",
        parameters=Parameters.from_list([p1]),
        features=Features.from_list([f_spatial, f_temporal]),
        performance=PerformanceAttributes.from_list([perf]),
        domains=domains,
    )

    logger = PfabLogger.get_logger(str(tmp_path / "logs"))

    class MultiDomainModel(IPredictionModel):
        @property
        def input_parameters(self): return ["param_1"]
        @property
        def input_features(self): return []
        @property
        def outputs(self): return ["feat_spatial", "feat_temporal"]
        @property
        def input_domain(self) -> Optional[str]: return "spatial"
        def forward_pass(self, X): return np.zeros((X.shape[0], 2))
        def train(self, tb, vb, **kw): pass

    model = MultiDomainModel(logger)
    model.set_ref_parameters(list(schema.parameters.data_objects.values()))
    model.set_ref_features(list(schema.features.data_objects.values()))

    with pytest.raises(ValueError, match="multiple named domains"):
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
