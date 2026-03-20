"""
Tests for DatasetSchema and SchemaRegistry: hash determinism, compatibility
checks, registration validation, and serialization roundtrip.
"""
import pytest

from pred_fab.core.schema import DatasetSchema, SchemaRegistry
from pred_fab.core.data_blocks import Parameters, Features, PerformanceAttributes
from pred_fab.core.data_objects import Parameter, Feature, PerformanceAttribute
from tests.utils.builders import build_mixed_feature_schema, build_workflow_schema


# ===== Schema hash =====

def test_schema_hash_is_deterministic(tmp_path):
    schema = build_mixed_feature_schema(tmp_path / "a", name="schema_a")
    hash_a = schema._compute_schema_hash()
    hash_b = schema._compute_schema_hash()
    assert hash_a == hash_b


def test_identical_structures_produce_same_hash(tmp_path):
    schema_a = build_mixed_feature_schema(tmp_path / "a", name="schema_a")
    schema_b = build_mixed_feature_schema(tmp_path / "b", name="schema_b")
    assert schema_a._compute_schema_hash() == schema_b._compute_schema_hash()


def test_different_structures_produce_different_hashes(tmp_path):
    schema_a = build_mixed_feature_schema(tmp_path / "a", name="schema_a")
    schema_b = build_workflow_schema(tmp_path / "b", name="schema_b")
    assert schema_a._compute_schema_hash() != schema_b._compute_schema_hash()


# ===== SchemaRegistry: registration rules =====

def test_same_structure_different_names_raises_on_second_registration(tmp_path):
    """Same schema structure cannot be registered under two different names."""
    build_mixed_feature_schema(tmp_path, name="schema_original")

    with pytest.raises(ValueError):
        build_mixed_feature_schema(tmp_path, name="schema_duplicate")


def test_same_name_same_structure_does_not_raise(tmp_path):
    """Re-registering same name with identical structure is idempotent."""
    build_mixed_feature_schema(tmp_path, name="schema_test")
    # Second call with same name + same structure must not raise
    build_mixed_feature_schema(tmp_path, name="schema_test")


def test_different_structure_same_name_raises(tmp_path):
    build_mixed_feature_schema(tmp_path, name="schema_test")

    params = Parameters.from_list([Parameter.real("param_extra", 0.0, 1.0)])
    feats = Features()
    perfs = PerformanceAttributes.from_list([PerformanceAttribute.score("perf_1")])

    with pytest.raises(ValueError, match="different structure"):
        DatasetSchema(
            root_folder=str(tmp_path),
            name="schema_test",
            parameters=params,
            features=feats,
            performance=perfs,
        )


# ===== is_compatible_with() =====

def test_schema_is_compatible_with_identical_schema(tmp_path):
    schema_a = build_mixed_feature_schema(tmp_path / "a", name="schema_a")
    schema_b = build_mixed_feature_schema(tmp_path / "b", name="schema_b")
    assert schema_a.is_compatible_with(schema_b) is True


def test_schema_is_compatible_raises_for_different_schema(tmp_path):
    schema_a = build_mixed_feature_schema(tmp_path / "a", name="schema_a")
    schema_b = build_workflow_schema(tmp_path / "b", name="schema_b")

    with pytest.raises(ValueError):
        schema_a.is_compatible_with(schema_b)


# ===== to_dict / from_dict roundtrip =====

def test_schema_to_dict_from_dict_roundtrip(tmp_path):
    schema = build_mixed_feature_schema(tmp_path, name="test_schema")
    d = schema.to_dict()

    restored = DatasetSchema.from_dict(d, str(tmp_path / "restored"))
    assert restored.name == schema.name
    assert set(restored.parameters.keys()) == set(schema.parameters.keys())
    assert set(restored.features.keys()) == set(schema.features.keys())
    assert set(restored.performance_attrs.keys()) == set(schema.performance_attrs.keys())


def test_schema_from_dict_raises_for_missing_schema_id(tmp_path):
    schema = build_mixed_feature_schema(tmp_path, name="test_schema")
    d = schema.to_dict()
    del d["schema_id"]

    with pytest.raises(ValueError, match="schema_id"):
        DatasetSchema.from_dict(d, str(tmp_path / "restored"))


# ===== Domain-based schema initialization =====

def test_schema_raises_for_feature_with_unknown_domain(tmp_path):
    """Feature referencing an unregistered domain should raise ValueError during init."""
    feats = Features.from_list([Feature.array("f1", domain="nonexistent")])
    perfs = PerformanceAttributes.from_list([PerformanceAttribute.score("p1")])
    params = Parameters.from_list([Parameter.real("param_1", 0.0, 10.0)])

    with pytest.raises(ValueError, match="not registered"):
        DatasetSchema(
            root_folder=str(tmp_path),
            name="schema_bad_domain",
            parameters=params,
            features=feats,
            performance=perfs,
        )


def test_schema_registers_domain_axes_into_parameters(tmp_path):
    """Domain axes should be automatically added to Parameters during init."""
    schema = build_mixed_feature_schema(tmp_path, name="schema_domain_axes")
    assert schema.parameters.has("dim_1")
    assert schema.parameters.has("dim_2")


# ===== SchemaRegistry internals =====

def test_registry_get_hash_by_id_returns_none_for_unknown(tmp_path):
    import os
    os.makedirs(str(tmp_path / "local"), exist_ok=True)
    registry = SchemaRegistry(str(tmp_path / "local"))
    assert registry.get_hash_by_id("nonexistent_id") is None


def test_registry_stores_and_retrieves_schema_id(tmp_path):
    schema = build_mixed_feature_schema(tmp_path, name="my_schema")
    registry = SchemaRegistry(str(schema.local_data.local_folder))
    hash_val = registry.get_hash_by_id("my_schema")
    assert hash_val is not None
    assert len(hash_val) == 64  # SHA-256 hex digest length
