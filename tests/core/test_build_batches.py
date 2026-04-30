"""Tests for DataModule.build_flat_batch and build_sequence_batch.

These helpers replace the cell-loop input construction in
``_predict_autoregressive_batched_tensor``: framework-side shape building
that concrete model classes (``MLPModel`` flat, ``TransformerModel`` sequence)
will call from their ``predict`` method.
"""

from __future__ import annotations

from types import SimpleNamespace

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


def _build_2d_schema(tmp_path) -> DatasetSchema:
    spatial = Domain("spatial", [
        Dimension("n_layers", "layer_idx", 1, 10),
        Dimension("n_segments", "segment_idx", 1, 10),
    ])
    layer_dim, segment_dim = spatial.axes
    grid_feat = Feature.array("grid", domain=spatial)
    layer_iter = Feature.iterator(spatial, layer_dim)  # layer_idx_pos
    segment_iter = Feature.iterator(spatial, segment_dim)  # segment_idx_pos
    return DatasetSchema(
        root_folder=str(tmp_path),
        name="build_batch_schema",
        parameters=Parameters.from_list([
            Parameter.real("p1", min_val=0.0, max_val=1.0),
        ]),
        features=Features.from_list([
            grid_feat,
            layer_iter,
            segment_iter,
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
            grid_rows.append([k, s, float(k * 10 + s)])
    grid = np.array(grid_rows, dtype=np.float64)
    grid_tensor = exp.features.table_to_tensor("grid", grid, exp.parameters)
    exp.features.set_value("grid", grid_tensor)


def _build_dm(tmp_path, n_layers: int = 4, n_segments: int = 3) -> DataModule:
    schema = _build_2d_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    _seed_experiment(dataset, n_layers, n_segments)
    dm = DataModule(dataset=dataset)
    dm.initialize(
        input_parameters=["p1", "n_layers", "n_segments"],
        input_features=["layer_idx_pos", "segment_idx_pos"],
        output_columns=["grid"],
    )
    dm._split_codes[SplitType.TRAIN] = ["exp_001"]
    dm.fit_normalization(SplitType.TRAIN)
    return dm


def _dim_info(shape: tuple, iterator_feats: list[tuple[str, int, int]],
              dim_codes: list[str], dim_iters: list[str]) -> dict:
    return {
        'shape': shape,
        'dim_iterators': dim_iters,
        'dim_codes_ordered': dim_codes,
        'param_base': {},
        'iterator_feats': iterator_feats,
        'total_positions': int(np.prod(shape)) if shape else 1,
    }


# === flat batch ===========================================================


def test_flat_batch_shape_and_row_map(tmp_path):
    """build_flat_batch returns (S × n_cells, n_input) with a per-row (s, cell) map."""
    dm = _build_dm(tmp_path, n_layers=4, n_segments=3)

    # Two candidates, same shape (4, 3) → 12 cells each → 24 rows total.
    params_list = [
        {"p1": 0.2, "n_layers": 4, "n_segments": 3},
        {"p1": 0.7, "n_layers": 4, "n_segments": 3},
    ]
    di = _dim_info(
        shape=(4, 3),
        iterator_feats=[("layer_idx_pos", 0, 4), ("segment_idx_pos", 1, 3)],
        dim_codes=["n_layers", "n_segments"],
        dim_iters=["layer_idx", "segment_idx"],
    )
    dim_info_list = [di, di]

    X_flat, row_map = dm.build_flat_batch(params_list, dim_info_list)

    n_cells = 4 * 3
    assert X_flat.shape == (2 * n_cells, len(dm.input_columns))
    assert len(row_map) == 2 * n_cells
    # Within a shape group, row order is [cell_0_s0, cell_0_s1, cell_1_s0, ...].
    assert row_map[0] == (0, 0)
    assert row_map[1] == (1, 0)
    assert row_map[2] == (0, 1)


def test_flat_batch_iterator_overrides_filled_per_cell(tmp_path):
    """Iterator-feature columns carry pos / (size - 1) per cell, normalised."""
    dm = _build_dm(tmp_path, n_layers=4, n_segments=3)
    params_list = [{"p1": 0.5, "n_layers": 4, "n_segments": 3}]
    di = _dim_info(
        shape=(4, 3),
        iterator_feats=[("layer_idx_pos", 0, 4), ("segment_idx_pos", 1, 3)],
        dim_codes=["n_layers", "n_segments"],
        dim_iters=["layer_idx", "segment_idx"],
    )
    X_flat, row_map = dm.build_flat_batch(params_list, [di])

    layer_col = dm.input_columns.index("layer_idx_pos")
    segment_col = dm.input_columns.index("segment_idx_pos")
    layer_stats = dm._parameter_stats.get("layer_idx_pos")
    segment_stats = dm._parameter_stats.get("segment_idx_pos")

    # Cell (k=2, s=1) → layer = 2/3, segment = 1/2; flat index = 2*3 + 1 = 7.
    flat_idx = 7
    expected_layer_raw = 2.0 / 3.0
    expected_segment_raw = 1.0 / 2.0
    if layer_stats is not None:
        expected_layer = float(dm._apply_normalization_tensor(
            torch.tensor(expected_layer_raw, dtype=torch.float32), layer_stats,
        ).item())
    else:
        expected_layer = expected_layer_raw
    if segment_stats is not None:
        expected_segment = float(dm._apply_normalization_tensor(
            torch.tensor(expected_segment_raw, dtype=torch.float32), segment_stats,
        ).item())
    else:
        expected_segment = expected_segment_raw

    assert row_map[flat_idx] == (0, flat_idx)
    assert X_flat[flat_idx, layer_col].item() == pytest.approx(expected_layer, abs=1e-6)
    assert X_flat[flat_idx, segment_col].item() == pytest.approx(expected_segment, abs=1e-6)


def test_flat_batch_groups_by_shape(tmp_path):
    """Different shapes per candidate → separate shape groups, both included."""
    dm = _build_dm(tmp_path, n_layers=4, n_segments=3)
    params_list = [
        {"p1": 0.1, "n_layers": 2, "n_segments": 2},  # shape (2, 2) → 4 cells
        {"p1": 0.9, "n_layers": 3, "n_segments": 2},  # shape (3, 2) → 6 cells
    ]
    di_a = _dim_info(
        shape=(2, 2),
        iterator_feats=[("layer_idx_pos", 0, 2), ("segment_idx_pos", 1, 2)],
        dim_codes=["n_layers", "n_segments"],
        dim_iters=["layer_idx", "segment_idx"],
    )
    di_b = _dim_info(
        shape=(3, 2),
        iterator_feats=[("layer_idx_pos", 0, 3), ("segment_idx_pos", 1, 2)],
        dim_codes=["n_layers", "n_segments"],
        dim_iters=["layer_idx", "segment_idx"],
    )

    X_flat, row_map = dm.build_flat_batch(params_list, [di_a, di_b])

    assert X_flat.shape[0] == 4 + 6
    s_indices = {s for s, _ in row_map}
    assert s_indices == {0, 1}
    rows_for_s0 = [r for r in row_map if r[0] == 0]
    rows_for_s1 = [r for r in row_map if r[0] == 1]
    assert len(rows_for_s0) == 4
    assert len(rows_for_s1) == 6


def test_flat_batch_empty_inputs(tmp_path):
    """Empty params_list returns (0, n_input) tensor + empty row map."""
    dm = _build_dm(tmp_path)
    X_flat, row_map = dm.build_flat_batch([], [])
    assert X_flat.shape == (0, len(dm.input_columns))
    assert row_map == []


def test_flat_batch_gradient_flow(tmp_path):
    """Continuous-tensor params propagate gradient through build_flat_batch."""
    dm = _build_dm(tmp_path, n_layers=4, n_segments=3)
    p1_t = torch.tensor(0.5, requires_grad=True)
    params_list = [{"p1": p1_t, "n_layers": 4, "n_segments": 3}]
    di = _dim_info(
        shape=(4, 3),
        iterator_feats=[("layer_idx_pos", 0, 4), ("segment_idx_pos", 1, 3)],
        dim_codes=["n_layers", "n_segments"],
        dim_iters=["layer_idx", "segment_idx"],
    )
    X_flat, _ = dm.build_flat_batch(params_list, [di])
    p1_col = dm.input_columns.index("p1")

    X_flat.sum().backward()
    assert p1_t.grad is not None and p1_t.grad.item() != 0.0


# === sequence batch =======================================================


def test_sequence_batch_shape(tmp_path):
    """build_sequence_batch returns (S, L, n_input) along the sequence axis."""
    # Use n_segments=1 so the multi-axis guard doesn't trip.
    dm = _build_dm(tmp_path, n_layers=4, n_segments=1)
    params_list = [
        {"p1": 0.2, "n_layers": 4, "n_segments": 1},
        {"p1": 0.7, "n_layers": 4, "n_segments": 1},
    ]
    di = _dim_info(
        shape=(4, 1),
        iterator_feats=[("layer_idx_pos", 0, 4), ("segment_idx_pos", 1, 1)],
        dim_codes=["n_layers", "n_segments"],
        dim_iters=["layer_idx", "segment_idx"],
    )

    model = SimpleNamespace(sequence_axis_code="n_layers")
    X_seq = dm.build_sequence_batch(model, params_list, [di, di])

    assert X_seq.shape == (2, 4, len(dm.input_columns))


def test_sequence_batch_position_encoding(tmp_path):
    """The sequence-axis iterator column carries pos / (L - 1) per position."""
    dm = _build_dm(tmp_path, n_layers=5, n_segments=1)
    params_list = [{"p1": 0.5, "n_layers": 5, "n_segments": 1}]
    di = _dim_info(
        shape=(5, 1),
        iterator_feats=[("layer_idx_pos", 0, 5), ("segment_idx_pos", 1, 1)],
        dim_codes=["n_layers", "n_segments"],
        dim_iters=["layer_idx", "segment_idx"],
    )
    model = SimpleNamespace(sequence_axis_code="n_layers")
    X_seq = dm.build_sequence_batch(model, params_list, [di])

    layer_col = dm.input_columns.index("layer_idx_pos")
    layer_stats = dm._parameter_stats.get("layer_idx_pos")
    for pos in range(5):
        raw = float(pos) / 4.0
        if layer_stats is not None:
            expected = float(dm._apply_normalization_tensor(
                torch.tensor(raw, dtype=torch.float32), layer_stats,
            ).item())
        else:
            expected = raw
        assert X_seq[0, pos, layer_col].item() == pytest.approx(expected, abs=1e-6)


def test_sequence_batch_rejects_missing_axis(tmp_path):
    """Sequence axis must resolve to a real domain axis; unknown axis raises."""
    dm = _build_dm(tmp_path, n_layers=4, n_segments=1)
    params_list = [{"p1": 0.5, "n_layers": 4, "n_segments": 1}]
    di = _dim_info(
        shape=(4, 1),
        iterator_feats=[],
        dim_codes=["n_layers", "n_segments"],
        dim_iters=["layer_idx", "segment_idx"],
    )
    bad_model = SimpleNamespace(sequence_axis_code="not_a_real_axis")
    with pytest.raises(ValueError, match="not in this model's domain axes"):
        dm.build_sequence_batch(bad_model, params_list, [di])


def test_sequence_batch_rejects_multi_axis(tmp_path):
    """Multi-axis grids (segment > 1 alongside a layer sequence axis) raise."""
    dm = _build_dm(tmp_path, n_layers=4, n_segments=3)
    params_list = [{"p1": 0.5, "n_layers": 4, "n_segments": 3}]
    di = _dim_info(
        shape=(4, 3),
        iterator_feats=[("layer_idx_pos", 0, 4), ("segment_idx_pos", 1, 3)],
        dim_codes=["n_layers", "n_segments"],
        dim_iters=["layer_idx", "segment_idx"],
    )
    model = SimpleNamespace(sequence_axis_code="n_layers")
    with pytest.raises(NotImplementedError, match="multi-axis grids"):
        dm.build_sequence_batch(model, params_list, [di])


def test_sequence_batch_empty_inputs(tmp_path):
    """Empty params_list returns a (0, 0, n_input) tensor."""
    dm = _build_dm(tmp_path)
    model = SimpleNamespace(sequence_axis_code="n_layers")
    X_seq = dm.build_sequence_batch(model, [], [])
    assert X_seq.shape == (0, 0, len(dm.input_columns))
