"""Tests for tensor-typed params_to_tensor / tensor_to_params.

Three guarantees this commit promises:
  1. Numerical equivalence with the existing numpy params_to_array /
     array_to_params at ~1e-6 relative tolerance.
  2. Round-trip: tensor_to_params(params_to_tensor(p)) ≈ p.
  3. Gradient flow: when a continuous param value is a torch.Tensor with
     requires_grad=True, the gradient flows through params_to_tensor
     (via the affine normalisation) into the output tensor.
"""

import numpy as np
import pytest
import torch

from pred_fab.core import DataModule, Dataset, DatasetSchema
from pred_fab.core.data_blocks import (
    Domains, Features, Parameters, PerformanceAttributes,
)
from pred_fab.core.data_objects import (
    Dimension, Domain, Feature, Parameter, PerformanceAttribute,
)
from pred_fab.utils import SplitType


def _build_dm(tmp_path) -> DataModule:
    """A minimal DataModule with two real params + one feature, fitted on
    fabricated data. Mirrors the smallest meaningful schema for these tests.
    """
    spatial = Domain("spatial", [
        Dimension("n_layers", "layer_idx", 1, 4),
        Dimension("n_segments", "segment_idx", 1, 3),
    ])
    schema = DatasetSchema(
        root_folder=str(tmp_path),
        name="ptt_schema",
        parameters=Parameters.from_list([
            Parameter.real("speed", min_val=20.0, max_val=60.0),
            Parameter.real("water", min_val=0.3, max_val=0.5),
        ]),
        features=Features.from_list([
            Feature("perf_a", domain=spatial),
        ]),
        performance=PerformanceAttributes.from_list([
            PerformanceAttribute.score("score_a"),
        ]),
        domains=Domains([spatial]),
    )
    dataset = Dataset(schema=schema, debug_flag=True)

    rng = np.random.default_rng(0)
    for i in range(5):
        code = f"exp_{i:02d}"
        dataset.create_experiment(
            code,
            parameters={
                "speed": float(rng.uniform(20, 60)),
                "water": float(rng.uniform(0.3, 0.5)),
                "n_layers": 4,
                "n_segments": 3,
            },
        )
        exp = dataset.get_experiment(code)
        # Populate a tiny perf_a tensor (4 layers × 3 segments)
        perf_tensor = rng.uniform(-1, 1, size=(4, 3))
        exp.features.set_value("perf_a", perf_tensor)

    dm = DataModule(dataset=dataset)
    dm.initialize(
        input_parameters=["speed", "water"],
        input_features=[],
        output_columns=["perf_a"],
    )
    dm._split_codes[SplitType.TRAIN] = list(dataset.get_experiment_codes())
    dm.fit_normalization(SplitType.TRAIN)
    return dm


# ── Numerical equivalence with numpy versions ─────────────────────────────


def test_params_to_tensor_matches_array(tmp_path):
    """params_to_tensor and params_to_array produce numerically identical encodings."""
    dm = _build_dm(tmp_path)
    p = {"speed": 35.0, "water": 0.4}

    arr = dm.params_to_array(p)
    t = dm.params_to_tensor(p)

    np.testing.assert_allclose(t.detach().cpu().numpy(), arr, atol=1e-6, rtol=1e-6)


def test_params_to_tensor_at_bounds(tmp_path):
    """Boundary values (param min and max) round-trip cleanly through normalisation."""
    dm = _build_dm(tmp_path)
    for speed in (20.0, 60.0):
        for water in (0.3, 0.5):
            p = {"speed": speed, "water": water}
            arr = dm.params_to_array(p)
            t = dm.params_to_tensor(p)
            np.testing.assert_allclose(t.detach().cpu().numpy(), arr, atol=1e-6, rtol=1e-6)


# ── Round-trip ────────────────────────────────────────────────────────────


def test_round_trip_continuous(tmp_path):
    """tensor_to_params(params_to_tensor(p)) returns the original continuous values."""
    dm = _build_dm(tmp_path)
    p = {"speed": 42.5, "water": 0.37}

    t = dm.params_to_tensor(p)
    p_decoded = dm.tensor_to_params(t)

    assert p_decoded["speed"] == pytest.approx(42.5, rel=1e-5)
    assert p_decoded["water"] == pytest.approx(0.37, rel=1e-5)


def test_round_trip_via_numpy_path(tmp_path):
    """Cross-check: tensor_to_params(params_to_tensor(p)) matches array_to_params(params_to_array(p))."""
    dm = _build_dm(tmp_path)
    p = {"speed": 33.3, "water": 0.41}

    p_via_numpy = dm.array_to_params(dm.params_to_array(p))
    p_via_tensor = dm.tensor_to_params(dm.params_to_tensor(p))

    for code in p_via_numpy:
        if code in p_via_tensor:
            assert p_via_tensor[code] == pytest.approx(p_via_numpy[code], rel=1e-5)


# ── Gradient flow ─────────────────────────────────────────────────────────


def test_grad_flows_through_continuous_param(tmp_path):
    """When a continuous param is a tensor with requires_grad=True, gradient flows
    through params_to_tensor's affine normalisation into the input."""
    dm = _build_dm(tmp_path)
    speed_t = torch.tensor(35.0, requires_grad=True)
    water_t = torch.tensor(0.4, requires_grad=True)
    p = {"speed": speed_t, "water": water_t}

    out = dm.params_to_tensor(p)
    loss = out.sum()
    loss.backward()

    # Both param leaves should have non-None, non-zero gradients
    assert speed_t.grad is not None
    assert water_t.grad is not None
    assert torch.isfinite(speed_t.grad).all()
    assert torch.isfinite(water_t.grad).all()
    # For STANDARD normalisation the gradient is 1/(std + eps), positive scalar.
    # Without asserting the exact value, just confirm grad ≠ 0.
    assert float(speed_t.grad.abs()) > 1e-12
    assert float(water_t.grad.abs()) > 1e-12


def test_grad_through_normalisation_is_affine(tmp_path):
    """Verify the gradient magnitude equals 1/(std+eps) for STANDARD normalisation —
    confirms the affine assumption holds end-to-end through params_to_tensor."""
    dm = _build_dm(tmp_path)
    # Fetch the std from fitted stats for speed
    speed_stats = dm._parameter_stats["speed"]
    if speed_stats["method"].name != "STANDARD":
        pytest.skip("Test assumes default STANDARD normalisation.")
    std_speed = float(speed_stats["std"])

    speed_t = torch.tensor(35.0, requires_grad=True)
    p = {"speed": speed_t, "water": 0.4}

    out = dm.params_to_tensor(p)
    # Pick out the speed-corresponding output entry; loss = that entry only
    speed_idx = dm.input_columns.index("speed")
    out[speed_idx].backward()

    # d/dspeed [(speed - mean) / (std + 1e-8)] = 1 / (std + 1e-8)
    expected_grad = 1.0 / (std_speed + 1e-8)
    assert float(speed_t.grad) == pytest.approx(expected_grad, rel=1e-5)


# ── Edge cases ────────────────────────────────────────────────────────────


def test_missing_param_defaults_to_zero(tmp_path):
    """Missing continuous params get 0.0 (matches _one_hot_encode's nan_to_num behaviour)."""
    dm = _build_dm(tmp_path)
    p = {"speed": 35.0}  # 'water' missing

    arr = dm.params_to_array(p)
    t = dm.params_to_tensor(p)

    np.testing.assert_allclose(t.detach().cpu().numpy(), arr, atol=1e-6, rtol=1e-6)


def test_nan_value_treated_as_zero(tmp_path):
    """NaN raw values get 0.0 before normalisation (recursive boundary semantics)."""
    dm = _build_dm(tmp_path)
    p = {"speed": float("nan"), "water": 0.4}

    arr = dm.params_to_array(p)
    t = dm.params_to_tensor(p)

    np.testing.assert_allclose(t.detach().cpu().numpy(), arr, atol=1e-6, rtol=1e-6)


def test_wrong_shape_raises(tmp_path):
    """tensor_to_params validates input shape against input_columns."""
    dm = _build_dm(tmp_path)
    bad = torch.zeros(len(dm.input_columns) + 1)
    with pytest.raises(ValueError, match="does not match input columns"):
        dm.tensor_to_params(bad)
