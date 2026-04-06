"""Tests for context feature flag and OFAT trajectory strategy."""
import pytest
import numpy as np
from typing import Any, Dict, List, Optional

from pred_fab.core.data_objects import DataArray, Domain, Dimension, Feature, Parameter, PerformanceAttribute
from pred_fab.core.data_blocks import Parameters, Features, PerformanceAttributes, Domains
from pred_fab.core import DatasetSchema, DataModule, Dataset
from pred_fab.utils import PfabLogger
from tests.utils.builders import (
    build_test_logger,
    build_calibration_system,
    build_workflow_schema,
    build_workflow_dataset,
    build_workflow_agent,
    configure_default_workflow_calibration,
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


# === CalibrationSystem OFAT strategy ===

def test_ofat_configure_requires_trust_regions(tmp_path):
    """configure_ofat_strategy raises if parameter has no trust region."""
    dataset = build_workflow_dataset(tmp_path)
    agent = build_workflow_agent(tmp_path, dataset.schema)
    configure_default_workflow_calibration(agent)
    with pytest.raises(ValueError, match="trust region"):
        agent.configure(ofat_strategy=["param_1"])  # param_1 has bounds, not a trust region


def test_ofat_configure_succeeds_for_trust_region_param(tmp_path):
    """configure_ofat_strategy succeeds when parameter has a trust region."""
    dataset = build_workflow_dataset(tmp_path)
    agent = build_workflow_agent(tmp_path, dataset.schema)
    configure_default_workflow_calibration(agent)
    agent.configure(ofat_strategy=["speed"])  # speed has trust region (adaptation_delta)
    assert agent.calibration_system._ofat_codes == ["speed"]
    assert agent.calibration_system._ofat_index == 0


def test_ofat_active_code_cycles(tmp_path):
    """_get_active_ofat_code returns codes in order and wraps around."""
    dataset = build_workflow_dataset(tmp_path)
    agent = build_workflow_agent(tmp_path, dataset.schema)
    configure_default_workflow_calibration(agent)
    cs = agent.calibration_system
    # Manually inject two OFAT codes (pretend both have trust regions)
    cs.trust_regions["speed"] = 10.0  # already set
    cs._ofat_codes = ["speed"]
    cs._ofat_index = 0
    assert cs._get_active_ofat_code() == "speed"
    cs._advance_ofat()
    assert cs._ofat_index == 0  # wraps with single element


def test_ofat_empty_codes_returns_none(tmp_path):
    """_get_active_ofat_code returns None when OFAT is not configured."""
    dataset = build_workflow_dataset(tmp_path)
    cs = build_calibration_system(tmp_path, dataset)
    assert cs._get_active_ofat_code() is None


def test_ofat_trust_region_bounds_fixes_inactive_param(tmp_path):
    """With OFAT active, inactive trust-region param is fixed (low==high) in bounds."""
    from pred_fab.core.data_objects import Parameter as P
    from pred_fab.core.data_blocks import Parameters as Params, Features as Feats, PerformanceAttributes as Perfs
    from pred_fab.core.data_objects import PerformanceAttribute
    from pred_fab.core import DatasetSchema, DataModule, Dataset
    from pred_fab.utils.enum import NormMethod

    # Build schema with two runtime params
    p1 = P.real("speed_a", min_val=0.0, max_val=100.0, runtime=True)
    p2 = P.real("speed_b", min_val=0.0, max_val=100.0, runtime=True)
    perf = PerformanceAttribute.score("perf_1")
    schema = DatasetSchema(
        root_folder=str(tmp_path),
        name="ofat_test",
        parameters=Params([p1, p2]),
        features=Feats([]),
        performance=Perfs([perf]),
    )
    dataset = Dataset(schema=schema, debug_flag=True)
    logger = build_test_logger(tmp_path)

    from pred_fab.orchestration.calibration import CalibrationSystem
    cs = CalibrationSystem(
        schema=schema,
        logger=logger,
        perf_fn=lambda p: {"perf_1": 0.5},
        uncertainty_fn=lambda x: 1.0,
    )
    cs.configure_adaptation_delta({"speed_a": 10.0, "speed_b": 10.0})
    cs.configure_ofat_strategy(["speed_a", "speed_b"])

    # Build a minimal fitted datamodule for bounds computation
    datamodule = DataModule(dataset, normalize=NormMethod.NONE)
    datamodule.initialize(
        input_parameters=["speed_a", "speed_b"],
        input_features=[],
        output_columns=["perf_1"],
    )
    datamodule.fit_without_data()
    cs._active_datamodule = datamodule

    current = {"speed_a": 50.0, "speed_b": 50.0}
    bounds = cs._get_trust_region_bounds(datamodule, current)

    # speed_a is active (index 0) → free; speed_b is inactive → fixed
    speed_a_idx = datamodule.input_columns.index("speed_a")
    speed_b_idx = datamodule.input_columns.index("speed_b")
    assert bounds[speed_a_idx, 0] < bounds[speed_a_idx, 1], "Active param should have non-zero range"
    assert bounds[speed_b_idx, 0] == bounds[speed_b_idx, 1], "Inactive param should be fixed (low==high)"


def test_ofat_advance_on_online_calibration(tmp_path):
    """OFAT index advances automatically after each online run_calibration call."""
    from pred_fab.core.data_objects import Parameter as P
    from pred_fab.core.data_blocks import Parameters as Params, Features as Feats, PerformanceAttributes as Perfs
    from pred_fab.core.data_objects import PerformanceAttribute
    from pred_fab.core import DatasetSchema, DataModule, Dataset
    from pred_fab.utils.enum import NormMethod, Mode

    p1 = P.real("speed_a", min_val=0.0, max_val=100.0, runtime=True)
    p2 = P.real("speed_b", min_val=0.0, max_val=100.0, runtime=True)
    perf = PerformanceAttribute.score("perf_1")
    schema = DatasetSchema(
        root_folder=str(tmp_path),
        name="ofat_advance",
        parameters=Params([p1, p2]),
        features=Feats([]),
        performance=Perfs([perf]),
    )
    dataset = Dataset(schema=schema, debug_flag=True)
    logger = build_test_logger(tmp_path)

    from pred_fab.orchestration.calibration import CalibrationSystem
    cs = CalibrationSystem(
        schema=schema,
        logger=logger,
        perf_fn=lambda p: {"perf_1": 0.5},
        uncertainty_fn=lambda x: 1.0,
    )
    cs.configure_adaptation_delta({"speed_a": 10.0, "speed_b": 10.0})
    cs.configure_ofat_strategy(["speed_a", "speed_b"])

    datamodule = DataModule(dataset, normalize=NormMethod.NONE)
    datamodule.initialize(
        input_parameters=["speed_a", "speed_b"],
        input_features=[],
        output_columns=["perf_1"],
    )
    datamodule.fit_without_data()

    assert cs._ofat_index == 0
    cs.run_calibration(
        datamodule=datamodule,
        mode=Mode.INFERENCE,
        current_params={"speed_a": 50.0, "speed_b": 50.0},
        target_indices={},
    )
    assert cs._ofat_index == 1  # advanced to speed_b
    cs.run_calibration(
        datamodule=datamodule,
        mode=Mode.INFERENCE,
        current_params={"speed_a": 50.0, "speed_b": 50.0},
        target_indices={},
    )
    assert cs._ofat_index == 0  # wrapped back to speed_a
