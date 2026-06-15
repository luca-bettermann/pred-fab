"""Value-correctness tests for the acquisition objective and its 2D grids.

``compute_acquisition_grids`` claims to slice the *same* ``evidence_gain`` /
``system_performance`` the optimizer uses — "no separate computation path".
These tests pin that: every grid cell equals the pointwise function at the
swept params, the κ-blend matches ``acquisition``, and the κ short-circuits
(κ=1 drops performance, κ=0 drops evidence) hold on the grid.

The evidence backend is stubbed with a known function of the candidate so
the value path (params → ``params_to_array`` → backend → ΔE → blend) is
checked exactly without standing up a trained PredictionSystem; the estimator
itself is covered in ``test_evidence_integral.py``.
"""

import numpy as np
import torch

from pred_fab.core import Dataset
from pred_fab.orchestration.calibration import EvidenceBackend
from tests.utils.builders import (
    build_calibration_system,
    build_initialized_datamodule,
    build_workflow_schema,
)


def _perf_fn(params):
    return {"performance_1": params["param_1"] / 10.0, "performance_2": params["speed"] / 200.0}


def _build_system(tmp_path):
    """Calibration system with a fitted (identity-norm) datamodule and a
    candidate-dependent evidence stub, ready for acquisition-value checks."""
    dataset = Dataset(schema=build_workflow_schema(tmp_path))
    datamodule = build_initialized_datamodule(
        dataset,
        input_parameters=["param_1", "speed"],
        input_features=[],
        output_columns=["feature_3"],
        fitted=True,
    )
    cal = build_calibration_system(tmp_path, dataset, perf_fn=_perf_fn)
    cal._active_datamodule = datamodule
    # ΔE depends on the candidate position so the grid genuinely varies cell-to-cell.
    cal._evidence_backend = EvidenceBackend(
        batched_tensor=lambda X, w: torch.tensor([float((X.flatten() ** 2).sum())])
    )
    cal.set_performance_weights({"performance_1": 1.0, "performance_2": 1.0})
    return cal, datamodule


def test_evidence_gain_delegates_to_backend(tmp_path):
    """evidence_gain returns the backend value for the candidate (weight 1, no trajectory)."""
    cal, datamodule = _build_system(tmp_path)
    params = {"param_1": 6.0, "speed": 120.0}
    expected = float((datamodule.params_to_array(params) ** 2).sum())
    assert cal.evidence_gain(params) == expected


def test_acquisition_blends_performance_and_evidence(tmp_path):
    """A = (1-κ)·P_sys + κ·ΔE, checked across the κ range."""
    cal, _datamodule = _build_system(tmp_path)
    params = {"param_1": 4.0, "speed": 80.0}
    p = cal.system_performance(params)
    e = cal.evidence_gain(params)
    for kappa in (0.0, 0.25, 0.5, 1.0):
        assert cal.acquisition(params, kappa) == (1.0 - kappa) * p + kappa * e


def test_acquisition_kappa_extremes_short_circuit(tmp_path):
    """κ=1 is pure evidence (perf dropped); κ=0 is pure performance (evidence dropped)."""
    cal, _datamodule = _build_system(tmp_path)
    params = {"param_1": 4.0, "speed": 80.0}
    assert cal.acquisition(params, 1.0) == cal.evidence_gain(params)
    assert cal.acquisition(params, 0.0) == cal.system_performance(params)


def test_grids_equal_pointwise_functions(tmp_path):
    """Each grid cell equals the pointwise function at the swept params, and
    acq_grid is the κ-blend — the "no separate computation path" guarantee."""
    cal, _datamodule = _build_system(tmp_path)
    res, kappa = 5, 0.5
    xs, ys, ev_grid, perf_grid, acq_grid = cal.compute_acquisition_grids(
        x_key="param_1", y_key="speed",
        x_bounds=(0.0, 10.0), y_bounds=(0.0, 200.0),
        fixed_params={}, kappa=kappa, resolution=res,
    )

    assert ev_grid.shape == perf_grid.shape == acq_grid.shape == (res, res)
    np.testing.assert_allclose(xs, np.linspace(0.0, 10.0, res))
    np.testing.assert_allclose(ys, np.linspace(0.0, 200.0, res))

    for i in range(res):
        for j in range(res):
            params = {"param_1": float(xs[i]), "speed": float(ys[j])}
            assert ev_grid[j, i] == cal.evidence_gain(params)
            assert perf_grid[j, i] == cal.system_performance(params)

    np.testing.assert_allclose(acq_grid, (1.0 - kappa) * perf_grid + kappa * ev_grid)


def test_grids_kappa_extremes_zero_the_unused_arm(tmp_path):
    """κ=1 leaves perf_grid all zeros; κ=0 leaves ev_grid all zeros."""
    cal, _datamodule = _build_system(tmp_path)
    _, _, ev1, perf1, acq1 = cal.compute_acquisition_grids(
        "param_1", "speed", (0.0, 10.0), (0.0, 200.0), {}, kappa=1.0, resolution=4,
    )
    assert np.all(perf1 == 0.0)
    np.testing.assert_allclose(acq1, ev1)

    _, _, ev0, perf0, acq0 = cal.compute_acquisition_grids(
        "param_1", "speed", (0.0, 10.0), (0.0, 200.0), {}, kappa=0.0, resolution=4,
    )
    assert np.all(ev0 == 0.0)
    np.testing.assert_allclose(acq0, perf0)
