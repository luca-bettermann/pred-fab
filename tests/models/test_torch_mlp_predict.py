"""TorchMLPModel.predict — flat-batched per-candidate dispatch.

These tests exercise the new ``predict`` contract that replaces the
framework-side cell-loop autoreg dispatch (commits 6-7 swap and delete).
``TorchMLPModel.predict`` builds a flat batch via DataModule, forwards
once, and de-multiplexes per-(s, cell) into per-feature tensors.
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
from pred_fab.models import TorchMLPModel
from pred_fab.utils import SplitType
from tests.utils.builders import build_test_logger


class _GridMLP(TorchMLPModel):
    HIDDEN = (8,)
    EPOCHS = 50
    COMPILE = False  # Avoid compile probe noise in unit test.

    @property
    def input_parameters(self):
        return ["p1", "n_layers", "n_segments"]

    @property
    def input_features(self):
        return ["layer_idx_pos", "segment_idx_pos"]

    @property
    def outputs(self):
        return ["grid"]


def _build_2d_schema(tmp_path) -> DatasetSchema:
    spatial = Domain("spatial", [
        Dimension("n_layers", "layer_idx", 1, 8),
        Dimension("n_segments", "segment_idx", 1, 8),
    ])
    layer_dim, segment_dim = spatial.axes
    grid_feat = Feature.array("grid", domain=spatial)
    layer_iter = Feature.iterator(spatial, layer_dim)
    segment_iter = Feature.iterator(spatial, segment_dim)
    return DatasetSchema(
        root_folder=str(tmp_path),
        name="mlp_predict_schema",
        parameters=Parameters.from_list([Parameter.real("p1", 0.0, 1.0)]),
        features=Features.from_list([grid_feat, layer_iter, segment_iter]),
        performance=PerformanceAttributes.from_list([PerformanceAttribute.score("perf_1")]),
        domains=Domains([spatial]),
    )


def _seed_experiment(dataset: Dataset, n_layers: int, n_segments: int) -> None:
    dataset.create_experiment(
        "exp_001",
        parameters={"p1": 0.5, "n_layers": n_layers, "n_segments": n_segments},
    )
    exp = dataset.get_experiment("exp_001")
    rows = []
    for k in range(n_layers):
        for s in range(n_segments):
            rows.append([k, s, float(k * 10 + s)])
    grid = np.array(rows, dtype=np.float64)
    grid_tensor = exp.features.table_to_tensor("grid", grid, exp.parameters)
    exp.features.set_value("grid", grid_tensor)


def _build_trained_mlp(tmp_path):
    schema = _build_2d_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    _seed_experiment(dataset, n_layers=4, n_segments=3)

    dm = DataModule(dataset=dataset)
    dm.initialize(
        input_parameters=["p1", "n_layers", "n_segments"],
        input_features=["layer_idx_pos", "segment_idx_pos"],
        output_columns=["grid"],
    )
    dm._split_codes[SplitType.TRAIN] = ["exp_001"]
    dm.fit_normalization(SplitType.TRAIN)

    logger = build_test_logger(tmp_path)
    mlp = _GridMLP(logger)
    mlp.set_ref_features(list(dataset.schema.features.data_objects.values()))  # type: ignore[arg-type]
    # Skip the dimensional-coherence call here — the test fixture isn't a
    # PredictionSystem; we manually train without that wrapper.

    # Provide synthetic training data: (n_train_rows, n_input) and (n_train_rows, n_outputs).
    n_input = len(dm.input_columns)
    X_train = torch.randn(20, n_input)
    y_train = torch.randn(20, 1)
    mlp.train([(X_train, y_train)], [])
    return dm, mlp


def _dim_info(shape: tuple, iterator_feats, dim_codes, dim_iters) -> dict:
    return {
        'shape': shape,
        'dim_iterators': dim_iters,
        'dim_codes_ordered': dim_codes,
        'param_base': {},
        'iterator_feats': iterator_feats,
        'total_positions': int(np.prod(shape)) if shape else 1,
    }


def test_predict_returns_per_candidate_per_feature_tensor(tmp_path):
    """Predict on 2 candidates → list of 2 dicts, each with grid tensor of shape (n_layers, n_segments)."""
    dm, mlp = _build_trained_mlp(tmp_path)

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
    out = mlp.predict(params_list, dm, [di, di], {})

    assert len(out) == 2
    for s in (0, 1):
        assert "grid" in out[s]
        assert out[s]["grid"].shape == (4, 3)


def test_predict_self_consistency(tmp_path):
    """Predict's per-candidate values match a manual forward + reshape on the same flat batch."""
    dm, mlp = _build_trained_mlp(tmp_path)

    params_list = [{"p1": 0.4, "n_layers": 4, "n_segments": 3}]
    di = _dim_info(
        shape=(4, 3),
        iterator_feats=[("layer_idx_pos", 0, 4), ("segment_idx_pos", 1, 3)],
        dim_codes=["n_layers", "n_segments"],
        dim_iters=["layer_idx", "segment_idx"],
    )
    out = mlp.predict(params_list, dm, [di], {})

    # Manual reference: same build_flat_batch + forward + reshape.
    X_flat, row_map = dm.build_flat_batch(params_list, [di])
    input_indices_t = torch.as_tensor(
        dm.get_input_indices(mlp.input_parameters + mlp.input_features), dtype=torch.long,
    )
    X_model = X_flat.index_select(1, input_indices_t)
    y_norm = mlp.forward_pass(X_model)
    y_denorm = dm.denormalize_values(y_norm, mlp.outputs)
    expected = torch.zeros((4, 3), dtype=y_denorm.dtype)
    for row_idx, (s, cell_flat) in enumerate(row_map):
        k, sg = divmod(cell_flat, 3)
        expected[k, sg] = y_denorm[row_idx, 0]

    torch.testing.assert_close(out[0]["grid"], expected, rtol=0, atol=1e-6)


def test_predict_gradient_flow(tmp_path):
    """Continuous-tensor params propagate gradient through predict end-to-end."""
    dm, mlp = _build_trained_mlp(tmp_path)
    p1_t = torch.tensor(0.5, requires_grad=True)
    params_list = [{"p1": p1_t, "n_layers": 4, "n_segments": 3}]
    di = _dim_info(
        shape=(4, 3),
        iterator_feats=[("layer_idx_pos", 0, 4), ("segment_idx_pos", 1, 3)],
        dim_codes=["n_layers", "n_segments"],
        dim_iters=["layer_idx", "segment_idx"],
    )
    out = mlp.predict(params_list, dm, [di], {})
    out[0]["grid"].sum().backward()
    assert p1_t.grad is not None and p1_t.grad.item() != 0.0


def test_predict_empty_candidates_returns_empty(tmp_path):
    dm, mlp = _build_trained_mlp(tmp_path)
    out = mlp.predict([], dm, [], {})
    assert out == []


def test_validate_schema_compatibility_rejects_recursive(tmp_path):
    """An MLP with a recursive input feature on the schema raises ValueError."""
    spatial = Domain("spatial", [Dimension("n_layers", "layer_idx", 1, 8)])
    layer_dim = spatial.axes[0]
    src_feat = Feature.array("src", domain=spatial)
    schema = DatasetSchema(
        root_folder=str(tmp_path),
        name="mlp_reject_schema",
        parameters=Parameters.from_list([Parameter.real("p1", 0.0, 1.0)]),
        features=Features.from_list([
            src_feat,
            *Feature.recursive("prev_src", source=src_feat, dimensions=(layer_dim,), max_depth=1),
        ]),
        performance=PerformanceAttributes.from_list([PerformanceAttribute.score("perf_1")]),
        domains=Domains([spatial]),
    )

    class _RecMLP(TorchMLPModel):
        HIDDEN = (4,)
        @property
        def input_parameters(self): return ["p1"]
        @property
        def input_features(self): return ["prev_src_1"]
        @property
        def outputs(self): return ["src"]

    mlp = _RecMLP(build_test_logger(tmp_path))
    mlp.set_ref_features(list(schema.features.data_objects.values()))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="recursive input feature"):
        mlp._validate_schema_compatibility(schema)


def test_validate_schema_compatibility_accepts_non_recursive(tmp_path):
    """An MLP with no recursive inputs passes the type-specific check."""
    schema = _build_2d_schema(tmp_path)
    mlp = _GridMLP(build_test_logger(tmp_path))
    mlp.set_ref_features(list(schema.features.data_objects.values()))  # type: ignore[arg-type]
    mlp._validate_schema_compatibility(schema)  # must not raise
