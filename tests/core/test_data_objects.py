"""
Tests for DataObject subclasses: validation, coercion, serialization, and type contracts.

Covers DataReal, DataInt, DataBool, DataCategorical, DataDomainAxis, Domain, DataArray,
and the factory classes Parameter, Feature, PerformanceAttribute.
"""
import pytest
import numpy as np

from pred_fab.core.data_objects import (
    DataReal,
    DataInt,
    DataBool,
    DataCategorical,
    DataDomainAxis,
    Dimension,
    Domain,
    DataArray,
    Parameter,
    Feature,
    PerformanceAttribute,
)
from pred_fab.utils.enum import Roles


# ===== DataReal =====

def test_datareal_validate_accepts_value_at_min_bound():
    obj = DataReal("x", Roles.PARAMETER, min_val=0.0, max_val=10.0)
    assert obj.validate(0.0) is True


def test_datareal_validate_accepts_value_at_max_bound():
    obj = DataReal("x", Roles.PARAMETER, min_val=0.0, max_val=10.0)
    assert obj.validate(10.0) is True


def test_datareal_validate_raises_for_value_below_min():
    obj = DataReal("x", Roles.PARAMETER, min_val=0.0, max_val=10.0)
    with pytest.raises(ValueError, match="below minimum"):
        obj.validate(-0.001)


def test_datareal_validate_raises_for_value_above_max():
    obj = DataReal("x", Roles.PARAMETER, min_val=0.0, max_val=10.0)
    with pytest.raises(ValueError, match="above maximum"):
        obj.validate(10.001)


def test_datareal_validate_raises_for_non_numeric():
    obj = DataReal("x", Roles.PARAMETER)
    with pytest.raises(TypeError):
        obj.validate("not_a_number")


def test_datareal_validate_accepts_int_as_numeric():
    obj = DataReal("x", Roles.PARAMETER, min_val=0.0, max_val=10.0)
    assert obj.validate(5) is True


def test_datareal_coerce_applies_rounding():
    obj = DataReal("x", Roles.PARAMETER, round_digits=2)
    assert obj.coerce(3.14159) == pytest.approx(3.14)


def test_datareal_coerce_no_rounding_when_not_configured():
    obj = DataReal("x", Roles.PARAMETER)
    assert obj.coerce(3.14159) == pytest.approx(3.14159)


def test_datareal_coerce_converts_int_to_float():
    obj = DataReal("x", Roles.PARAMETER)
    result = obj.coerce(5)
    assert isinstance(result, float)
    assert result == 5.0


def test_datareal_validate_no_constraints_accepts_any_numeric():
    obj = DataReal("x", Roles.PARAMETER)
    assert obj.validate(-1e10) is True
    assert obj.validate(1e10) is True


def test_datareal_to_dict_from_dict_roundtrip():
    obj = DataReal("speed", Roles.PARAMETER, min_val=0.0, max_val=100.0, round_digits=3)
    d = obj.to_dict()
    from pred_fab.core.data_objects import DataObject
    restored = DataObject.from_dict(d)
    assert restored.code == obj.code
    assert restored.constraints == obj.constraints
    assert restored.round_digits == obj.round_digits


def test_datareal_validate_exactly_at_boundary_inclusive():
    """Boundary values min and max should be valid (inclusive)."""
    obj = DataReal("x", Roles.PARAMETER, min_val=5.0, max_val=5.0)
    assert obj.validate(5.0) is True


# ===== DataInt =====

def test_dataint_validate_accepts_int():
    obj = DataInt("n", Roles.PARAMETER, min_val=1, max_val=10)
    assert obj.validate(5) is True


def test_dataint_validate_raises_for_float():
    obj = DataInt("n", Roles.PARAMETER)
    with pytest.raises(TypeError):
        obj.validate(3.0)


def test_dataint_validate_raises_for_bool():
    """bool is subclass of int in Python; DataInt explicitly rejects it."""
    obj = DataInt("n", Roles.PARAMETER)
    with pytest.raises(TypeError, match="must be int"):
        obj.validate(True)


def test_dataint_validate_raises_below_min():
    obj = DataInt("n", Roles.PARAMETER, min_val=1, max_val=10)
    with pytest.raises(ValueError):
        obj.validate(0)


def test_dataint_validate_raises_above_max():
    obj = DataInt("n", Roles.PARAMETER, min_val=1, max_val=10)
    with pytest.raises(ValueError):
        obj.validate(11)


def test_dataint_validate_accepts_boundary_values():
    obj = DataInt("n", Roles.PARAMETER, min_val=1, max_val=10)
    assert obj.validate(1) is True
    assert obj.validate(10) is True


def test_dataint_coerce_rounds_0_4_down():
    obj = DataInt("n", Roles.PARAMETER)
    assert obj.coerce(0.4) == 0


def test_dataint_coerce_rounds_0_6_up():
    obj = DataInt("n", Roles.PARAMETER)
    assert obj.coerce(0.6) == 1


def test_dataint_coerce_0_5_uses_python_rounding():
    """Python 3 uses banker's rounding: round(0.5) == 0, round(1.5) == 2."""
    obj = DataInt("n", Roles.PARAMETER)
    result = obj.coerce(0.5)
    assert isinstance(result, int)


def test_dataint_coerce_float_to_int():
    obj = DataInt("n", Roles.PARAMETER)
    assert obj.coerce(3.9) == 4
    assert obj.coerce(2.1) == 2
    assert isinstance(obj.coerce(2.1), int)


def test_dataint_coerce_negative_float():
    obj = DataInt("n", Roles.PARAMETER)
    assert obj.coerce(-1.6) == -2
    assert obj.coerce(-1.4) == -1


def test_dataint_coerce_returns_int_type():
    obj = DataInt("n", Roles.PARAMETER)
    result = obj.coerce(5.7)
    assert isinstance(result, int)


# ===== DataBool =====

def test_databool_validate_accepts_true():
    obj = DataBool("flag", Roles.PARAMETER)
    assert obj.validate(True) is True


def test_databool_validate_accepts_false():
    obj = DataBool("flag", Roles.PARAMETER)
    assert obj.validate(False) is True


def test_databool_validate_raises_for_int():
    obj = DataBool("flag", Roles.PARAMETER)
    with pytest.raises(TypeError):
        obj.validate(1)


def test_databool_validate_raises_for_string():
    obj = DataBool("flag", Roles.PARAMETER)
    with pytest.raises(TypeError):
        obj.validate("true")


def test_databool_validate_raises_for_float():
    obj = DataBool("flag", Roles.PARAMETER)
    with pytest.raises(TypeError):
        obj.validate(1.0)


def test_databool_coerce_0_5_is_true():
    obj = DataBool("flag", Roles.PARAMETER)
    assert obj.coerce(0.5) is True


def test_databool_coerce_0_49_is_false():
    obj = DataBool("flag", Roles.PARAMETER)
    assert obj.coerce(0.49) is False


def test_databool_coerce_0_51_is_true():
    obj = DataBool("flag", Roles.PARAMETER)
    assert obj.coerce(0.51) is True


def test_databool_coerce_zero_is_false():
    obj = DataBool("flag", Roles.PARAMETER)
    assert obj.coerce(0) is False


def test_databool_coerce_one_is_true():
    obj = DataBool("flag", Roles.PARAMETER)
    assert obj.coerce(1) is True


def test_databool_coerce_passes_through_true():
    obj = DataBool("flag", Roles.PARAMETER)
    assert obj.coerce(True) is True


def test_databool_coerce_passes_through_false():
    obj = DataBool("flag", Roles.PARAMETER)
    assert obj.coerce(False) is False


def test_databool_coerce_raises_for_string():
    obj = DataBool("flag", Roles.PARAMETER)
    with pytest.raises(TypeError):
        obj.coerce("yes")


def test_databool_coerce_returns_bool_type():
    obj = DataBool("flag", Roles.PARAMETER)
    result = obj.coerce(0.8)
    assert isinstance(result, bool)


# ===== DataCategorical =====

def test_datacategorical_validate_accepts_known_category():
    obj = DataCategorical("material", ["PLA", "ABS", "PETG"], Roles.PARAMETER)
    assert obj.validate("PLA") is True
    assert obj.validate("ABS") is True


def test_datacategorical_validate_raises_for_unknown_category():
    obj = DataCategorical("material", ["PLA", "ABS"], Roles.PARAMETER)
    with pytest.raises(ValueError, match="not in allowed categories"):
        obj.validate("UNKNOWN")


def test_datacategorical_validate_raises_for_non_string():
    obj = DataCategorical("material", ["PLA", "ABS"], Roles.PARAMETER)
    with pytest.raises(TypeError):
        obj.validate(1)


def test_datacategorical_raises_for_empty_categories():
    with pytest.raises(ValueError, match="cannot be empty"):
        DataCategorical("material", [], Roles.PARAMETER)


def test_datacategorical_coerce_converts_number_to_str():
    obj = DataCategorical("material", ["1", "2"], Roles.PARAMETER)
    assert obj.coerce(1) == "1"


def test_datacategorical_coerce_passes_through_string():
    obj = DataCategorical("material", ["PLA", "ABS"], Roles.PARAMETER)
    assert obj.coerce("PLA") == "PLA"


def test_datacategorical_to_dict_from_dict_roundtrip():
    obj = DataCategorical("mat", ["A", "B", "C"], Roles.PARAMETER)
    from pred_fab.core.data_objects import DataObject
    d = obj.to_dict()
    restored = DataObject.from_dict(d)
    assert restored.constraints["categories"] == obj.constraints["categories"]


def test_datacategorical_case_sensitive():
    """Validation should be case-sensitive."""
    obj = DataCategorical("mat", ["PLA", "ABS"], Roles.PARAMETER)
    with pytest.raises(ValueError):
        obj.validate("pla")


# ===== DataDomainAxis =====

def test_datadomainaxis_stores_iterator_code():
    d = DataDomainAxis("n_layers", "layer_idx", Roles.PARAMETER, min_val=1, max_val=10)
    assert d.iterator_code == "layer_idx"


def test_datadomainaxis_iterator_code_in_constraints():
    d = DataDomainAxis("n_layers", "layer_idx", Roles.PARAMETER)
    assert d.constraints["domain_iterator_code"] == "layer_idx"


def test_datadomainaxis_normalize_strategy_is_min_max():
    from pred_fab.utils.enum import NormMethod
    d = DataDomainAxis("n_layers", "layer_idx", Roles.PARAMETER)
    assert d.normalize_strategy == NormMethod.MIN_MAX


def test_datadomainaxis_to_dict_from_dict_roundtrip():
    from pred_fab.core.data_objects import DataObject
    obj = DataDomainAxis("n_layers", "layer_idx", Roles.PARAMETER, min_val=1, max_val=10)
    d = obj.to_dict()
    restored = DataObject.from_dict(d)
    assert restored.iterator_code == "layer_idx"  # type: ignore
    assert restored.constraints.get("min") == 1
    assert restored.constraints.get("max") == 10


def test_datadomainaxis_min_val_default_is_1():
    d = DataDomainAxis("n_layers", "layer_idx", Roles.PARAMETER)
    assert d.constraints.get("min") == 1


def test_datadomainaxis_runtime_adjustable_false():
    obj = DataDomainAxis("n_layers", "layer_idx", Roles.PARAMETER)
    assert obj.runtime_adjustable is False


def test_datadomainaxis_validates_int():
    obj = DataDomainAxis("n_layers", "layer_idx", Roles.PARAMETER, min_val=1, max_val=10)
    assert obj.validate(5) is True


def test_datadomainaxis_validates_rejects_non_int():
    obj = DataDomainAxis("n_layers", "layer_idx", Roles.PARAMETER)
    with pytest.raises(TypeError):
        obj.validate(3.0)


# ===== Domain =====

def test_domain_stores_code():
    d = Domain("spatial", [Dimension("n_layers", "layer_idx", 1, 5)])
    assert d.code == "spatial"


def test_domain_axes_length():
    d = Domain("spatial", [
        Dimension("n_layers", "layer_idx", 1, 5),
        Dimension("n_segments", "seg_idx", 1, 4),
    ])
    assert len(d.axes) == 2


def test_domain_depth():
    d = Domain("spatial", [
        Dimension("n_layers", "layer_idx", 1, 5),
        Dimension("n_segments", "seg_idx", 1, 4),
    ])
    assert d.depth == 2


def test_domain_get_param_codes():
    d = Domain("spatial", [
        Dimension("n_layers", "layer_idx", 1, 5),
        Dimension("n_segments", "seg_idx", 1, 4),
    ])
    assert d.get_param_codes() == ["n_layers", "n_segments"]


def test_domain_get_iterator_codes():
    d = Domain("spatial", [
        Dimension("n_layers", "layer_idx", 1, 5),
        Dimension("n_segments", "seg_idx", 1, 4),
    ])
    assert d.get_iterator_codes() == ["layer_idx", "seg_idx"]


def test_domain_create_axis_params_types():
    d = Domain("spatial", [Dimension("n_layers", "layer_idx", 1, 5)])
    params = d.create_axis_params(Roles.PARAMETER)
    assert len(params) == 1
    assert isinstance(params[0], DataDomainAxis)
    assert params[0].code == "n_layers"
    assert params[0].iterator_code == "layer_idx"


def test_domain_to_dict_from_dict_roundtrip():
    d = Domain("spatial", [
        Dimension("n_layers", "layer_idx", 1, 5),
        Dimension("n_segments", "seg_idx", 1, 4),
    ])
    d2 = Domain.from_dict(d.to_dict())
    assert d2.code == d.code
    assert len(d2.axes) == len(d.axes)
    assert d2.axes[0].code == "n_layers"
    assert d2.axes[1].iterator_code == "seg_idx"


def test_dimension_max_val_none_by_default():
    """Dimension with no max_val should have max_val=None."""
    dim = Dimension("n_layers", "layer_idx", 1)
    assert dim.max_val is None


def test_dimension_max_val_none_explicit():
    dim = Dimension("n_layers", "layer_idx", 1, None)
    assert dim.max_val is None


# ===== DataArray =====

def test_dataarray_validate_accepts_correct_dtype():
    obj = DataArray("features", Roles.FEATURE, dtype=np.float64)
    assert obj.validate(np.array([1.0, 2.0], dtype=np.float64)) is True


def test_dataarray_validate_raises_for_wrong_dtype():
    obj = DataArray("features", Roles.FEATURE, dtype=np.float64)
    with pytest.raises(ValueError, match="dtype mismatch"):
        obj.validate(np.array([1, 2], dtype=np.int32))


def test_dataarray_validate_raises_for_list_input():
    obj = DataArray("features", Roles.FEATURE, dtype=np.float64)
    with pytest.raises(TypeError):
        obj.validate([1.0, 2.0])


def test_dataarray_coerce_converts_list_to_ndarray():
    obj = DataArray("features", Roles.FEATURE, dtype=np.float64)
    result = obj.coerce([1, 2, 3])
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_dataarray_coerce_preserves_shape():
    obj = DataArray("features", Roles.FEATURE, dtype=np.float64)
    result = obj.coerce(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert result.shape == (2, 2)


def test_dataarray_set_columns_updates_constraints():
    obj = DataArray("feat", Roles.FEATURE, dtype=np.float64)
    obj.set_columns(["d1", "d2", "feat"])
    assert obj.columns == ["d1", "d2", "feat"]
    assert obj.constraints["columns"] == ["d1", "d2", "feat"]


def test_dataarray_to_dict_from_dict_roundtrip():
    from pred_fab.core.data_objects import DataObject
    obj = DataArray("feat", Roles.FEATURE, dtype=np.float64)
    obj.set_columns(["d1", "feat"])
    d = obj.to_dict()
    restored = DataObject.from_dict(d)
    assert restored.constraints.get("columns") == ["d1", "feat"]


def test_dataarray_stores_domain_code():
    obj = DataArray("feat", Roles.FEATURE, dtype=np.float64, domain_code="spatial")
    assert obj.domain_code == "spatial"
    assert obj.constraints.get("domain_code") == "spatial"


def test_dataarray_stores_feature_depth():
    obj = DataArray("feat", Roles.FEATURE, dtype=np.float64, domain_code="spatial", feature_depth=1)
    assert obj.feature_depth == 1
    assert obj.constraints.get("feature_depth") == 1


def test_dataarray_roundtrip_with_domain_code():
    from pred_fab.core.data_objects import DataObject
    obj = DataArray("feat", Roles.FEATURE, dtype=np.float64, domain_code="spatial", feature_depth=1)
    restored = DataObject.from_dict(obj.to_dict())
    assert restored.domain_code == "spatial"  # type: ignore
    assert restored.feature_depth == 1  # type: ignore


# ===== Factories =====

def test_parameter_factory_real_creates_correct_type():
    p = Parameter.real("speed", min_val=0.0, max_val=100.0)
    assert isinstance(p, DataReal)
    assert p.role == Roles.PARAMETER


def test_parameter_factory_real_defaults_round_digits_3():
    p = Parameter.real("speed", min_val=0.0, max_val=100.0)
    assert p.round_digits == 3


def test_parameter_factory_integer_creates_correct_type():
    p = Parameter.integer("n_layers", min_val=1, max_val=50)
    assert isinstance(p, DataInt)
    assert p.role == Roles.PARAMETER


def test_parameter_factory_categorical_creates_correct_type():
    p = Parameter.categorical("material", ["PLA", "ABS"])
    assert isinstance(p, DataCategorical)
    assert p.role == Roles.PARAMETER


def test_parameter_factory_boolean_creates_correct_type():
    p = Parameter.boolean("enabled")
    assert isinstance(p, DataBool)
    assert p.role == Roles.PARAMETER


def test_feature_factory_array_creates_correct_type():
    f = Feature.array("my_feature")
    assert isinstance(f, DataArray)
    assert f.role == Roles.FEATURE


def test_feature_factory_array_with_domain():
    f = Feature.array("my_feature", domain="spatial", depth=1)
    assert isinstance(f, DataArray)
    assert f.domain_code == "spatial"
    assert f.feature_depth == 1


def test_performance_attribute_factory_score_creates_correct_type():
    p = PerformanceAttribute.score("accuracy")
    assert isinstance(p, DataReal)
    assert p.role == Roles.PERFORMANCE


def test_performance_attribute_factory_score_bounds_0_to_1():
    p = PerformanceAttribute.score("accuracy")
    assert p.constraints["min"] == 0
    assert p.constraints["max"] == 1


def test_performance_attribute_score_rejects_value_above_1():
    p = PerformanceAttribute.score("accuracy")
    with pytest.raises(ValueError):
        p.validate(1.001)


def test_performance_attribute_score_rejects_value_below_0():
    p = PerformanceAttribute.score("accuracy")
    with pytest.raises(ValueError):
        p.validate(-0.001)


# ===== Runtime-adjustable parameters (Roles.RUNTIME) =====

def test_roles_enum_contains_only_block_membership_values():
    """Roles is solely a block-membership enum; sub-classifications are attributes on DataObject."""
    role_values = {r.value for r in Roles}
    assert role_values == {"parameter", "performance", "feature"}


def test_datareal_runtime_adjustable_false_by_default():
    obj = DataReal("speed", Roles.PARAMETER, min_val=0.0, max_val=100.0)
    assert obj.runtime_adjustable is False


def test_dataint_runtime_adjustable_false_by_default():
    obj = DataInt("n", Roles.PARAMETER, min_val=1, max_val=10)
    assert obj.runtime_adjustable is False


def test_databool_runtime_adjustable_false():
    obj = DataBool("flag", Roles.PARAMETER)
    assert obj.runtime_adjustable is False


def test_datacategorical_runtime_adjustable_false():
    obj = DataCategorical("material", ["PLA", "ABS"], Roles.PARAMETER)
    assert obj.runtime_adjustable is False


# --- Parameter.real(runtime=True) ---

def test_parameter_real_runtime_true_keeps_parameter_role():
    """runtime=True sets runtime_adjustable; role stays Roles.PARAMETER for DataBlock compatibility."""
    p = Parameter.real("speed", min_val=10.0, max_val=200.0, runtime=True)
    assert p.role == Roles.PARAMETER


def test_parameter_real_runtime_true_sets_runtime_adjustable():
    p = Parameter.real("speed", min_val=10.0, max_val=200.0, runtime=True)
    assert p.runtime_adjustable is True


def test_parameter_real_runtime_false_keeps_parameter_role():
    """Default (runtime=False) must not change existing role assignment."""
    p = Parameter.real("speed", min_val=10.0, max_val=200.0, runtime=False)
    assert p.role == Roles.PARAMETER
    assert p.runtime_adjustable is False


def test_parameter_real_runtime_preserves_type():
    """runtime=True still produces a DataReal, not a different type."""
    p = Parameter.real("speed", min_val=10.0, max_val=200.0, runtime=True)
    assert isinstance(p, DataReal)


def test_parameter_real_runtime_validation_unchanged():
    """runtime flag must not affect validation behavior."""
    p = Parameter.real("speed", min_val=10.0, max_val=200.0, runtime=True)
    assert p.validate(100.0) is True
    with pytest.raises(ValueError):
        p.validate(5.0)   # below min


def test_parameter_real_runtime_coerce_unchanged():
    p = Parameter.real("speed", min_val=0.0, max_val=200.0, round_digits=1, runtime=True)
    assert p.coerce(75.678) == pytest.approx(75.7)


def test_parameter_real_runtime_roundtrip_preserves_role():
    """Serialise → deserialise must keep role=Roles.PARAMETER and restore runtime_adjustable=True."""
    from pred_fab.core.data_objects import DataObject
    p = Parameter.real("speed", min_val=10.0, max_val=200.0, runtime=True)
    restored = DataObject.from_dict(p.to_dict())
    assert restored.role == Roles.PARAMETER
    assert restored.runtime_adjustable is True


def test_parameter_real_runtime_roundtrip_preserves_bounds():
    from pred_fab.core.data_objects import DataObject
    p = Parameter.real("speed", min_val=10.0, max_val=200.0, round_digits=2, runtime=True)
    restored = DataObject.from_dict(p.to_dict())
    assert restored.constraints["min"] == pytest.approx(10.0)
    assert restored.constraints["max"] == pytest.approx(200.0)
    assert restored.round_digits == 2


# --- Parameter.integer(runtime=True) ---

def test_parameter_integer_runtime_true_keeps_parameter_role():
    """runtime=True sets runtime_adjustable; role stays Roles.PARAMETER for DataBlock compatibility."""
    p = Parameter.integer("fan_rpm", min_val=0, max_val=3000, runtime=True)
    assert p.role == Roles.PARAMETER


def test_parameter_integer_runtime_true_sets_runtime_adjustable():
    p = Parameter.integer("fan_rpm", min_val=0, max_val=3000, runtime=True)
    assert p.runtime_adjustable is True


def test_parameter_integer_runtime_false_keeps_parameter_role():
    p = Parameter.integer("fan_rpm", min_val=0, max_val=3000, runtime=False)
    assert p.role == Roles.PARAMETER
    assert p.runtime_adjustable is False


def test_parameter_integer_runtime_preserves_type():
    p = Parameter.integer("fan_rpm", min_val=0, max_val=3000, runtime=True)
    assert isinstance(p, DataInt)


def test_parameter_integer_runtime_validation_unchanged():
    p = Parameter.integer("fan_rpm", min_val=0, max_val=3000, runtime=True)
    assert p.validate(1500) is True
    with pytest.raises(ValueError):
        p.validate(3001)


def test_parameter_integer_runtime_roundtrip_preserves_role():
    """Serialise → deserialise must keep role=Roles.PARAMETER and restore runtime_adjustable=True."""
    from pred_fab.core.data_objects import DataObject
    p = Parameter.integer("fan_rpm", min_val=0, max_val=3000, runtime=True)
    restored = DataObject.from_dict(p.to_dict())
    assert restored.role == Roles.PARAMETER
    assert restored.runtime_adjustable is True
