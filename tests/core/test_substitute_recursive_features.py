"""Tests for DataModule.substitute_recursive_features (Strategy D commit 15b).

Stateless tensor-typed substitution. The legacy stateful
``_perturb_recursive_features`` is replaced by this method; tests exercise
the same semantics but as a pure tensor function.

Five guarantees:
  1. ``p_student=0`` → returns X unchanged (no substitution)
  2. ``p_student=1`` → every interior recursive cell gets the prior cell's prediction
  3. boundary cells (prior < 0) substitute to NaN
  4. non-recursive columns are never touched
  5. partial substitution (0 < p < 1) replaces approximately the right fraction
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pred_fab.core import Dataset, DatasetSchema, DataModule
from pred_fab.core.data_blocks import (
    Domains, Features, Parameters, PerformanceAttributes,
)
from pred_fab.core.data_objects import (
    Dimension, Domain, Feature, Parameter, PerformanceAttribute,
)
from pred_fab.utils import SplitType


def _build_recursive_schema(tmp_path) -> DatasetSchema:
    spatial = Domain("spatial", [
        Dimension("n_layers", "layer_idx", 1, 10),
        Dimension("n_segments", "segment_idx", 1, 10),
    ])
    layer_dim, _ = spatial.axes
    grid_feat = Feature.array("grid", domain=spatial)
    return DatasetSchema(
        root_folder=str(tmp_path),
        name="ss_substitute_schema",
        parameters=Parameters.from_list([
            Parameter.real("p1", min_val=0.0, max_val=1.0),
        ]),
        features=Features.from_list([
            grid_feat,
            *Feature.recursive("prev_grid", source=grid_feat, dimensions=(layer_dim,), max_depth=1),
        ]),
        performance=PerformanceAttributes.from_list([
            PerformanceAttribute.score("perf_1"),
        ]),
        domains=Domains([spatial]),
    )


def _seed_experiment(dataset: Dataset, n_layers: int, n_segments: int) -> None:
    dataset.create_experiment(
        "exp_001",
        parameters={"p1": 0.5, "n_layers": n_layers, "n_segments": n_segments},
    )
    exp = dataset.get_experiment("exp_001")
    grid_rows = []
    for k in range(n_layers):
        for s in range(n_segments):
            grid_rows.append([k, s, k * 100.0 + s])
    grid = np.array(grid_rows, dtype=np.float64)
    grid_tensor = exp.features.table_to_tensor("grid", grid, exp.parameters)
    exp.features.set_value("grid", grid_tensor)
    prev_tensor = np.full_like(grid_tensor, np.nan)
    if n_layers > 1:
        prev_tensor[1:, :] = grid_tensor[:-1, :]
    exp.features.set_value("prev_grid_1", prev_tensor)


def _build_dm(tmp_path, n_layers: int = 4, n_segments: int = 3) -> DataModule:
    schema = _build_recursive_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    _seed_experiment(dataset, n_layers, n_segments)
    dm = DataModule(dataset=dataset)
    dm.initialize(
        input_parameters=["p1", "n_layers", "n_segments"],
        input_features=["prev_grid_1"],
        output_columns=["grid"],
    )
    dm._split_codes[SplitType.TRAIN] = ["exp_001"]
    dm.fit_normalization(SplitType.TRAIN)
    return dm


def _make_synthetic_predictions(n_layers: int, n_segments: int, base: float = 1000.0) -> dict:
    grid = torch.zeros((n_layers, n_segments), dtype=torch.float32)
    for k in range(n_layers):
        for s in range(n_segments):
            grid[k, s] = float(base + k * 10 + s)
    return {"exp_001": {"grid": grid}}


def _build_batch(dm: DataModule):
    """Get one batch + cell_meta via export_to_tensor_dict (commit 16 path)."""
    codes = ["exp_001"]
    exported = dm.dataset.export_to_tensor_dict(
        codes,
        x_columns=dm.input_columns,
        y_columns=dm.output_columns,
        categorical_mappings=dm.categorical_mappings,
    )
    X = dm.prepare_input_from_tensor_dict(exported.X)
    return X, exported.cell_meta, codes


# ── Tests ──


def test_p_zero_returns_unchanged(tmp_path):
    """p_student=0 → no substitution; returns X."""
    dm = _build_dm(tmp_path, n_layers=4, n_segments=3)
    X, cell_meta, codes = _build_batch(dm)
    preds = _make_synthetic_predictions(4, 3)

    X_out = dm.substitute_recursive_features(
        X, cell_meta, codes, preds, p_student=0.0,
    )
    torch.testing.assert_close(X_out, X)


def test_p_one_substitutes_interior_cells(tmp_path):
    """p=1 → every interior recursive cell == prior cell's prediction."""
    n_layers, n_segments = 4, 3
    dm = _build_dm(tmp_path, n_layers, n_segments)
    X, cell_meta, codes = _build_batch(dm)
    preds = _make_synthetic_predictions(n_layers, n_segments, base=1000.0)

    rng = torch.Generator().manual_seed(0)
    X_out = dm.substitute_recursive_features(
        X, cell_meta, codes, preds, p_student=1.0, rng=rng,
    )

    prev_idx = dm.input_columns.index("prev_grid_1")
    # Layout: cell_idx_flat = layer * n_segments + segment
    # Row 3 = layer 1, segment 0 → prior (0, 0) = 1000.0
    # Row 7 = layer 2, segment 1 → prior (1, 1) = 1011.0
    # Row 11 = layer 3, segment 2 → prior (2, 2) = 1022.0
    # (Normalisation may apply; check unscaled value comparisons against the prediction.)
    # If prev_grid_1 has stats fitted, X_out[r, prev_idx] = stats.forward(prediction)
    stats = dm._parameter_stats.get("prev_grid_1")
    expected_3 = stats(torch.tensor(1000.0, dtype=X_out.dtype)).item() if stats else 1000.0  # type: ignore[union-attr]
    expected_7 = stats(torch.tensor(1011.0, dtype=X_out.dtype)).item() if stats else 1011.0  # type: ignore[union-attr]
    assert X_out[3, prev_idx].item() == pytest.approx(expected_3, abs=1e-3)
    assert X_out[7, prev_idx].item() == pytest.approx(expected_7, abs=1e-3)


def test_p_one_boundary_cells_get_nan(tmp_path):
    """First-layer cells (prior < 0 along layer axis) substitute to NaN."""
    n_layers, n_segments = 4, 3
    dm = _build_dm(tmp_path, n_layers, n_segments)
    X, cell_meta, codes = _build_batch(dm)
    preds = _make_synthetic_predictions(n_layers, n_segments)

    rng = torch.Generator().manual_seed(0)
    X_out = dm.substitute_recursive_features(
        X, cell_meta, codes, preds, p_student=1.0, rng=rng,
    )

    prev_idx = dm.input_columns.index("prev_grid_1")
    # Rows 0, 1, 2 are layer 0 (segments 0, 1, 2) — prior is layer -1 → NaN
    for j in range(n_segments):
        assert torch.isnan(X_out[j, prev_idx]), f"row {j} (layer 0, segment {j}) should be NaN"


def test_non_recursive_columns_untouched(tmp_path):
    """Static parameter columns (p1, n_layers, n_segments) must remain unchanged."""
    n_layers, n_segments = 4, 3
    dm = _build_dm(tmp_path, n_layers, n_segments)
    X, cell_meta, codes = _build_batch(dm)
    preds = _make_synthetic_predictions(n_layers, n_segments)

    rng = torch.Generator().manual_seed(0)
    X_out = dm.substitute_recursive_features(
        X, cell_meta, codes, preds, p_student=1.0, rng=rng,
    )

    for col in dm.input_columns:
        if col == "prev_grid_1":
            continue
        col_idx = dm.input_columns.index(col)
        torch.testing.assert_close(X_out[:, col_idx], X[:, col_idx])


def test_p_intermediate_partial_substitution(tmp_path):
    """p=0.5 → roughly half of the eligible (interior) cells get substituted."""
    n_layers, n_segments = 8, 4
    dm = _build_dm(tmp_path, n_layers, n_segments)
    X, cell_meta, codes = _build_batch(dm)
    preds = _make_synthetic_predictions(n_layers, n_segments, base=1000.0)

    rng = torch.Generator().manual_seed(123)
    X_out = dm.substitute_recursive_features(
        X, cell_meta, codes, preds, p_student=0.5, rng=rng,
    )

    prev_idx = dm.input_columns.index("prev_grid_1")
    # Rows 0..n_segments-1 are boundary (NaN under p=1; some random subset under p=0.5).
    # Interior rows: n_segments and beyond. Original ground-truth prev_grid_1 values
    # come from grid[layer-1, seg] = (layer-1) * 100 + seg, normalised.
    # Substituted predictions are >= 1000, normalised.
    # After normalisation, both values map to some scaled range — distinguish by
    # checking which rows changed vs the un-substituted X.
    interior = X_out[n_segments:, prev_idx]
    interior_orig = X[n_segments:, prev_idx]
    n_changed = int((interior != interior_orig).sum().item())
    n_total = int(interior.numel())
    # With p=0.5 and 28 interior cells, ~14 ± 6 should be changed.
    assert 6 < n_changed < 24, f"Expected ~50% changed, got {n_changed}/{n_total}"
