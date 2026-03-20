"""
Tests for DataBlock operations: role enforcement, populated status,
Parameters dimension stride/combination logic, and Features value_at.
"""
import pytest
import numpy as np

from pred_fab.core.data_blocks import Parameters, Features, PerformanceAttributes, Domains
from pred_fab.core.data_objects import (
    Parameter,
    Feature,
    PerformanceAttribute,
    DataReal,
    DataDomainAxis,
    Domain,
)
from pred_fab.utils.enum import Roles


# ===== Helpers =====

def _make_domain_axis(code: str, iterator_code: str, max_val: int = 10) -> DataDomainAxis:
    """Create a DataDomainAxis with PARAMETER role for test helpers."""
    return DataDomainAxis(code, iterator_code, Roles.PARAMETER, min_val=1, max_val=max_val)


def _make_two_dim_params() -> Parameters:
    """Parameters with dim_1 (size 3) and dim_2 (size 4) as DataDomainAxis objects."""
    params = Parameters()
    params.add("param_1", Parameter.real("param_1", min_val=0.0, max_val=10.0))
    params.add("dim_1", _make_domain_axis("dim_1", "d1"))
    params.add("dim_2", _make_domain_axis("dim_2", "d2"))
    params.set_value("param_1", 5.0)
    params.set_value("dim_1", 3)
    params.set_value("dim_2", 4)
    return params


def _make_three_dim_params() -> Parameters:
    """Parameters with dim_1(2), dim_2(3), dim_3(4) as DataDomainAxis objects."""
    params = Parameters()
    params.add("dim_1", _make_domain_axis("dim_1", "d1"))
    params.add("dim_2", _make_domain_axis("dim_2", "d2"))
    params.add("dim_3", _make_domain_axis("dim_3", "d3"))
    params.set_value("dim_1", 2)
    params.set_value("dim_2", 3)
    params.set_value("dim_3", 4)
    return params


# ===== DataBlock role enforcement =====

def test_datablock_add_raises_for_wrong_role():
    params = Parameters()
    wrong_role_obj = DataReal("x", Roles.FEATURE)
    with pytest.raises(ValueError, match="role"):
        params.add("x", wrong_role_obj)


def test_datablock_add_raises_for_non_dataobject():
    params = Parameters()
    with pytest.raises(TypeError):
        params.add("x", "not_a_dataobject")  # type: ignore


def test_performance_block_rejects_parameter_role():
    perfs = PerformanceAttributes()
    wrong_role = DataReal("score", Roles.PARAMETER)
    with pytest.raises(ValueError):
        perfs.add("score", wrong_role)


def test_features_block_rejects_parameter_role():
    feats = Features()
    wrong_role = DataReal("f", Roles.PARAMETER)
    with pytest.raises(ValueError):
        feats.add("f", wrong_role)


# ===== Populated status =====

def test_populated_status_is_false_after_set_value_with_as_populated_false():
    params = Parameters()
    params.add("p", Parameter.real("p", 0.0, 10.0))
    params.set_value("p", 5.0, as_populated=False)
    assert params.is_populated("p") is False


def test_populated_status_is_true_after_set_value_with_as_populated_true():
    params = Parameters()
    params.add("p", Parameter.real("p", 0.0, 10.0))
    params.set_value("p", 5.0, as_populated=True)
    assert params.is_populated("p") is True


def test_populated_status_defaults_to_true():
    params = Parameters()
    params.add("p", Parameter.real("p", 0.0, 10.0))
    params.set_value("p", 5.0)
    assert params.is_populated("p") is True


def test_is_populated_returns_false_for_never_set():
    params = Parameters()
    params.add("p", Parameter.real("p", 0.0, 10.0))
    assert params.is_populated("p") is False


# ===== DataBlock get/has/values =====

def test_get_value_raises_for_unset_parameter():
    params = Parameters()
    params.add("p", Parameter.real("p", 0.0, 10.0))
    with pytest.raises(KeyError):
        params.get_value("p")


def test_has_value_returns_false_before_set():
    params = Parameters()
    params.add("p", Parameter.real("p", 0.0, 10.0))
    assert params.has_value("p") is False


def test_has_value_returns_true_after_set():
    params = Parameters()
    params.add("p", Parameter.real("p", 0.0, 10.0))
    params.set_value("p", 5.0)
    assert params.has_value("p") is True


def test_validate_value_raises_for_unknown_key():
    params = Parameters()
    with pytest.raises(KeyError):
        params.validate_value("nonexistent", 5.0)


def test_to_numpy_raises_for_categorical_values():
    params = Parameters()
    params.add("cat", Parameter.categorical("cat", ["A", "B"]))
    params.set_value("cat", "A")
    with pytest.raises(ValueError):
        params.to_numpy()


def test_to_numpy_returns_correct_array():
    params = Parameters()
    params.add("p1", Parameter.real("p1", 0.0, 10.0))
    params.add("p2", Parameter.integer("p2", 1, 5))
    params.set_value("p1", 3.0)
    params.set_value("p2", 2)
    arr = params.to_numpy()
    assert arr.shape == (2,)
    assert arr[0] == pytest.approx(3.0)
    assert arr[1] == pytest.approx(2.0)


# ===== DataBlock compatibility =====

def test_is_compatible_returns_false_for_different_keys():
    block_a = Parameters()
    block_a.add("p1", Parameter.real("p1", 0.0, 10.0))

    block_b = Parameters()
    block_b.add("p2", Parameter.real("p2", 0.0, 10.0))

    assert block_a.is_compatible(block_b) is False


def test_is_compatible_returns_true_for_same_keys():
    block_a = Parameters()
    block_a.add("p1", Parameter.real("p1", 0.0, 10.0))

    block_b = Parameters()
    block_b.add("p1", Parameter.real("p1", 5.0, 20.0))  # same key, different constraints

    assert block_a.is_compatible(block_b) is True


def test_is_compatible_returns_false_for_subset_keys():
    block_a = Parameters()
    block_a.add("p1", Parameter.real("p1", 0.0, 10.0))
    block_a.add("p2", Parameter.real("p2", 0.0, 10.0))

    block_b = Parameters()
    block_b.add("p1", Parameter.real("p1", 0.0, 10.0))

    assert block_a.is_compatible(block_b) is False


# ===== DataBlock serialization roundtrip =====

def test_parameters_to_dict_from_dict_roundtrip():
    params = _make_two_dim_params()
    d = params.to_dict()
    restored = Parameters.from_dict(d)

    assert set(restored.data_objects.keys()) == set(params.data_objects.keys())
    assert restored.get_value("dim_1") == 3
    assert restored.get_value("dim_2") == 4
    assert restored.get_value("param_1") == pytest.approx(5.0)


def test_features_from_list_creates_block():
    f1 = Feature.array("f1")
    f2 = Feature.array("f2")
    feats = Features.from_list([f1, f2])
    assert feats.has("f1")
    assert feats.has("f2")


# ===== Parameters domain axis accessors =====

def test_get_domain_axis_objects_returns_only_axis_objects():
    params = _make_two_dim_params()
    axes = params._get_domain_axis_objects()
    assert len(axes) == 2
    assert all(isinstance(ax, DataDomainAxis) for ax in axes)


def test_get_domain_axis_names_returns_axis_codes():
    params = _make_two_dim_params()
    names = params._get_domain_axis_names()
    assert names == ["dim_1", "dim_2"]


def test_get_domain_axis_iterator_codes_returns_iterator_codes():
    params = _make_two_dim_params()
    codes = params._get_domain_axis_iterator_codes()
    assert codes == ["d1", "d2"]


def test_get_domain_axis_values_returns_current_values():
    params = _make_two_dim_params()
    values = params._get_domain_axis_values()
    assert values == [3, 4]


def test_get_sorted_domain_axes_preserves_insertion_order():
    params = _make_three_dim_params()
    axes = params._get_sorted_domain_axes()
    assert [ax.code for ax in axes] == ["dim_1", "dim_2", "dim_3"]


# ===== Parameters dimension stride calculation =====

def test_get_dimension_strides_two_dims():
    """Stride for dim_1(size=3) = 4 (size of inner dim). Stride for dim_2(size=4) = 1."""
    params = _make_two_dim_params()
    strides = params._get_dimension_strides()
    assert strides["dim_1"] == 4
    assert strides["dim_2"] == 1


def test_get_dimension_strides_three_dims():
    """Three dims: dim_1(2), dim_2(3), dim_3(4). Strides: 12, 4, 1."""
    params = _make_three_dim_params()
    strides = params._get_dimension_strides()
    assert strides["dim_3"] == 1
    assert strides["dim_2"] == 4
    assert strides["dim_1"] == 12


def test_get_start_and_end_indices_first_outer_step():
    """First step of dim_1 covers rows 0..3 (stride=4)."""
    params = _make_two_dim_params()
    start, end = params.get_start_and_end_indices("dim_1", 0)
    assert start == 0
    assert end == 4


def test_get_start_and_end_indices_second_outer_step():
    """Second step of dim_1 covers rows 4..7."""
    params = _make_two_dim_params()
    start, end = params.get_start_and_end_indices("dim_1", 1)
    assert start == 4
    assert end == 8


def test_get_start_and_end_indices_innermost_dim():
    """First step of dim_2 (innermost) covers exactly 1 row."""
    params = _make_two_dim_params()
    start, end = params.get_start_and_end_indices("dim_2", 0)
    assert start == 0
    assert end == 1


def test_get_start_and_end_raises_for_unknown_dimension():
    params = _make_two_dim_params()
    with pytest.raises(ValueError, match="not found"):
        params.get_start_and_end_indices("nonexistent_dim", 0)


# ===== Parameters.get_dim_combinations() =====

def test_get_dim_combinations_total_count():
    params = _make_two_dim_params()
    combos = params.get_dim_combinations(["dim_1", "dim_2"])
    assert len(combos) == 12  # 3 * 4


def test_get_dim_combinations_first_element():
    params = _make_two_dim_params()
    combos = params.get_dim_combinations(["dim_1", "dim_2"])
    assert combos[0] == (0, 0)


def test_get_dim_combinations_last_element():
    params = _make_two_dim_params()
    combos = params.get_dim_combinations(["dim_1", "dim_2"])
    assert combos[-1] == (2, 3)  # dim_1 max is 2, dim_2 max is 3


def test_get_dim_combinations_with_evaluate_from():
    params = _make_two_dim_params()
    combos = params.get_dim_combinations(["dim_1", "dim_2"], evaluate_from=4)
    assert len(combos) == 8


def test_get_dim_combinations_with_evaluate_to():
    params = _make_two_dim_params()
    combos = params.get_dim_combinations(["dim_1", "dim_2"], evaluate_from=0, evaluate_to=6)
    assert len(combos) == 6


def test_get_dim_combinations_empty_slice():
    params = _make_two_dim_params()
    combos = params.get_dim_combinations(["dim_1", "dim_2"], evaluate_from=5, evaluate_to=5)
    assert combos == []


def test_get_dim_combinations_single_dim():
    params = _make_two_dim_params()
    combos = params.get_dim_combinations(["dim_1"])
    assert len(combos) == 3
    assert combos[0] == (0,)
    assert combos[2] == (2,)


# ===== Features.value_at() =====

def test_features_value_at_returns_none_when_not_set():
    feats = Features()
    arr = Feature.array("feat_1")
    arr.set_columns(["d1", "feat_1"])
    feats.add("feat_1", arr)

    params = _make_two_dim_params()
    # No value set
    result = feats.value_at("feat_1", params, {"d1": 0})
    assert result is None


def test_features_value_at_returns_correct_value_for_2d_tensor():
    feats = Features()
    arr = Feature.array("feat_1")
    arr.set_columns(["d1", "d2", "feat_1"])
    feats.add("feat_1", arr)

    params = _make_two_dim_params()
    feats.initialize_arrays(params)

    tensor = feats.get_value("feat_1")
    tensor[1, 2] = 42.0
    feats.set_value("feat_1", tensor)

    val = feats.value_at("feat_1", params, {"d1": 1, "d2": 2})
    assert val == pytest.approx(42.0)


def test_features_value_at_scalar_feature():
    feats = Features()
    arr = Feature.array("scalar_feat")
    arr.set_columns(["scalar_feat"])  # no iterators
    feats.add("scalar_feat", arr)

    params = Parameters()
    feats._initialize_array("scalar_feat", (), False)
    feats.set_value("scalar_feat", np.array(7.0))

    val = feats.value_at("scalar_feat", params, {})
    assert val == pytest.approx(7.0)


def test_features_value_at_returns_none_for_missing_iterator_key():
    """Missing iterator coordinate should return None."""
    feats = Features()
    arr = Feature.array("feat_1")
    arr.set_columns(["d1", "d2", "feat_1"])
    feats.add("feat_1", arr)

    params = _make_two_dim_params()
    feats.initialize_arrays(params)
    feats.get_value("feat_1")[0, 0] = 5.0

    # Missing d2 in iterator_values
    result = feats.value_at("feat_1", params, {"d1": 0})
    assert result is None


# ===== Features.initialize_arrays() edge cases =====

def test_features_initialize_arrays_raises_for_unset_columns():
    feats = Features()
    arr = Feature.array("feat_1")
    # No set_columns() call
    feats.add("feat_1", arr)

    params = _make_two_dim_params()
    with pytest.raises(ValueError, match="Columns not set"):
        feats.initialize_arrays(params)


def test_features_initialize_arrays_produces_nan_filled_tensor():
    feats = Features()
    arr = Feature.array("feat_1")
    arr.set_columns(["d1", "d2", "feat_1"])
    feats.add("feat_1", arr)

    params = _make_two_dim_params()
    feats.initialize_arrays(params)
    tensor = feats.get_value("feat_1")

    assert tensor.shape == (3, 4)
    assert np.all(np.isnan(tensor))


def test_features_initialize_arrays_skips_when_already_initialized():
    feats = Features()
    arr = Feature.array("feat_1")
    arr.set_columns(["d1", "feat_1"])
    feats.add("feat_1", arr)

    params = _make_two_dim_params()
    feats.initialize_arrays(params)
    feats.get_value("feat_1")[0] = 99.0

    feats.initialize_arrays(params, recompute_flag=False)
    assert feats.get_value("feat_1")[0] == 99.0


def test_features_initialize_arrays_resets_when_recompute():
    feats = Features()
    arr = Feature.array("feat_1")
    arr.set_columns(["d1", "feat_1"])
    feats.add("feat_1", arr)

    params = _make_two_dim_params()
    feats.initialize_arrays(params)
    feats.get_value("feat_1")[0] = 99.0

    feats.initialize_arrays(params, recompute_flag=True)
    assert np.isnan(feats.get_value("feat_1")[0])


# ===== Parameters.sanitize_values() =====

def test_sanitize_values_raises_for_unknown_code():
    params = _make_two_dim_params()
    with pytest.raises(KeyError):
        params.sanitize_values({"unknown_param": 5.0})


def test_sanitize_values_with_ignore_unknown_passes_through():
    params = _make_two_dim_params()
    result = params.sanitize_values({"unknown_param": 5.0, "param_1": 3.0}, ignore_unknown=True)
    assert result["unknown_param"] == 5.0
    assert result["param_1"] == pytest.approx(3.0)


def test_sanitize_values_coerces_types():
    params = _make_two_dim_params()
    result = params.sanitize_values({"dim_1": 2.7})
    assert result["dim_1"] == 3  # round to nearest int
    assert isinstance(result["dim_1"], int)


# ===== Domains container =====

def test_domains_add_and_get():
    domains = Domains()
    d = Domain("spatial", [("n_layers", "layer_idx", 1, 5), ("n_segments", "seg_idx", 1, 3)])
    domains.add(d)
    assert domains.has("spatial")
    assert domains.get("spatial") is d


def test_domains_get_raises_for_unknown():
    domains = Domains()
    with pytest.raises(KeyError):
        domains.get("nonexistent")


def test_domains_has_returns_false_for_unknown():
    domains = Domains()
    assert domains.has("nonexistent") is False


def test_domains_to_dict_from_dict_roundtrip():
    domains = Domains()
    domains.add(Domain("spatial", [("n_layers", "layer_idx", 1, 5), ("n_segments", "seg_idx", 1, 3)]))
    d = domains.to_dict()
    restored = Domains.from_dict(d)
    assert restored.has("spatial")
    restored_domain = restored.get("spatial")
    assert len(restored_domain.axes) == 2
    assert restored_domain.axes[0].param_code == "n_layers"
