"""Tests for context feature flag."""
import pytest
import numpy as np

from pred_fab.core.data_objects import DataArray, Domain, Dimension, Feature, Parameter, PerformanceAttribute
from pred_fab.core.data_blocks import Parameters, Features, PerformanceAttributes, Domains
from pred_fab.core import DatasetSchema, DataModule, Dataset
from tests.utils.builders import (
    build_workflow_dataset,
    build_workflow_agent,
)


# === DataArray.context flag ===

def test_feature_array_context_defaults_to_false():
    """Feature.array creates a non-context DataArray by default."""
    feat = Feature.array("temperature")
    assert feat.context is False


def test_feature_context_factory_sets_flag():
    """Feature.context creates a DataArray with context=True."""
    feat = Feature.context("temperature")
    assert feat.context is True


def test_feature_context_with_domain():
    """Feature.context accepts a Domain object."""
    domain = Domain("spatial", [Dimension("n_layers", "layer_idx", 1, 5)])
    feat = Feature.context("temperature", domain=domain)
    assert feat.context is True
    assert feat.domain_code == "spatial"


def test_dataarray_context_serialization_roundtrip():
    """Context flag survives to_dict / from_dict round-trip."""
    from pred_fab.utils.enum import Roles
    feat = Feature.context("humidity")
    d = feat.to_dict()
    assert d["constraints"].get("context") is True

    restored = DataArray._from_json_impl("humidity", Roles.FEATURE, d["constraints"])
    assert restored.context is True


def test_dataarray_non_context_not_in_constraints():
    """Non-context DataArray does not write 'context' key to constraints."""
    feat = Feature.array("path_deviation")
    assert "context" not in feat.to_dict()["constraints"]


# === DataModule context feature handling ===

def _build_schema_with_context(tmp_path) -> DatasetSchema:
    """Schema with one regular feature and one context feature."""
    p1 = Parameter.real("param_1", min_val=0.0, max_val=10.0)
    spatial = Domain("spatial", [Dimension("n_layers", "layer_idx", 1, 3)])
    f_output = Feature.array("path_deviation", domain=spatial)
    f_context = Feature.context("temperature", domain=spatial)
    perf = PerformanceAttribute.score("accuracy")
    return DatasetSchema(
        root_folder=str(tmp_path),
        name="schema_ctx",
        parameters=Parameters([p1]),
        features=Features([f_output, f_context]),
        performance=PerformanceAttributes([perf]),
        domains=Domains([spatial]),
    )


def test_datamodule_tracks_context_feature_codes(tmp_path):
    """DataModule._context_feature_codes populated when context feature in input_features."""
    schema = _build_schema_with_context(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    dm = DataModule(dataset)
    dm.initialize(
        input_parameters=["param_1", "n_layers"],
        input_features=["temperature"],
        output_columns=["path_deviation"],
    )
    assert "temperature" in dm._context_feature_codes


def test_datamodule_non_context_feature_not_tracked(tmp_path):
    """Non-context features are not added to _context_feature_codes."""
    schema = _build_schema_with_context(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    dm = DataModule(dataset)
    dm.initialize(
        input_parameters=["param_1", "n_layers"],
        input_features=["temperature"],
        output_columns=["path_deviation"],
    )
    assert "path_deviation" not in dm._context_feature_codes


def test_inject_context_features_copies_column(tmp_path):
    """_inject_context_features copies context columns from y_df into X_df."""
    import pandas as pd
    schema = _build_schema_with_context(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    dm = DataModule(dataset)
    dm.initialize(
        input_parameters=["param_1", "n_layers"],
        input_features=["temperature"],
        output_columns=["path_deviation"],
    )
    X_df = pd.DataFrame({"param_1": [1.0, 2.0], "n_layers": [2, 3]})
    y_df = pd.DataFrame({"path_deviation": [0.1, 0.2], "temperature": [22.5, 23.0]})
    result = dm._inject_context_features(X_df, y_df)
    assert "temperature" in result.columns
    np.testing.assert_array_almost_equal(result["temperature"].values, [22.5, 23.0])  # type: ignore[arg-type]


def test_inject_context_features_no_context_noop(tmp_path):
    """_inject_context_features is a no-op when no context features are configured."""
    import pandas as pd
    schema = _build_schema_with_context(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    dm = DataModule(dataset)
    dm.initialize(
        input_parameters=["param_1", "n_layers"],
        input_features=[],
        output_columns=["path_deviation"],
    )
    X_df = pd.DataFrame({"param_1": [1.0]})
    y_df = pd.DataFrame({"path_deviation": [0.1], "temperature": [22.5]})
    result = dm._inject_context_features(X_df, y_df)
    assert "temperature" not in result.columns


# === PfabAgent context snapshot ===

def test_agent_update_context_snapshot_updates_dict(tmp_path):
    """update_context_snapshot replaces the context snapshot."""
    dataset = build_workflow_dataset(tmp_path)
    agent = build_workflow_agent(tmp_path, dataset.schema)
    agent.update_context_snapshot({"temperature": 22.5, "humidity": 0.6})
    assert agent._context_snapshot == {"temperature": 22.5, "humidity": 0.6}


def test_agent_update_context_snapshot_replaces_old(tmp_path):
    """Calling update_context_snapshot again replaces the previous snapshot."""
    dataset = build_workflow_dataset(tmp_path)
    agent = build_workflow_agent(tmp_path, dataset.schema)
    agent.update_context_snapshot({"temperature": 22.5})
    agent.update_context_snapshot({"temperature": 25.0, "humidity": 0.7})
    assert "temperature" in agent._context_snapshot
    assert agent._context_snapshot["temperature"] == 25.0
    assert agent._context_snapshot.get("humidity") == 0.7
