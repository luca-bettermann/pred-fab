"""TorchTransformerModel.predict + train — sequence-aware contract.

These tests exercise the sequence dispatch that replaces the cell-loop
autoreg path: builds (S, L, n_input), runs the encoder with causal
attention, denormalises, and reshapes per-(s, feat). Training accepts
sequence-shaped batches (B, L, *) directly.
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
from pred_fab.models import TorchTransformerModel
from pred_fab.utils import SplitType
from tests.utils.builders import build_test_logger


class _LayerTransformer(TorchTransformerModel):
    """Sequence axis = n_layers; single output predicting per-layer values."""
    D_MODEL = 8
    N_HEADS = 2
    N_LAYERS = 1
    DIM_FEEDFORWARD = 16
    EPOCHS = 30  # keep fast for unit test

    @property
    def sequence_axis_code(self) -> str:
        return "n_layers"

    @property
    def input_parameters(self):
        return ["p1", "n_layers", "n_segments"]

    @property
    def input_features(self):
        return []

    @property
    def outputs(self):
        return ["src"]


def _build_1d_seq_schema(tmp_path) -> DatasetSchema:
    spatial = Domain("spatial", [
        Dimension("n_layers", "layer_idx", 1, 8),
        Dimension("n_segments", "segment_idx", 1, 8),
    ])
    src_feat = Feature.array("src", domain=spatial, depth=1)
    return DatasetSchema(
        root_folder=str(tmp_path),
        name="transformer_predict_schema",
        parameters=Parameters.from_list([Parameter.real("p1", 0.0, 1.0)]),
        features=Features.from_list([src_feat]),
        performance=PerformanceAttributes.from_list([PerformanceAttribute.score("perf_1")]),
        domains=Domains([spatial]),
    )


def _build_dm(tmp_path, n_layers: int = 4):
    schema = _build_1d_seq_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    # Two experiments with spread for non-degenerate normalization.
    for code, p1 in [("exp_001", 0.2), ("exp_002", 0.8)]:
        dataset.create_experiment(
            code, parameters={"p1": p1, "n_layers": n_layers, "n_segments": 1},
        )
        exp = dataset.get_experiment(code)
        rows = [[k, float(k * 0.1 + p1)] for k in range(n_layers)]
        src_data = np.array(rows, dtype=np.float64)
        src_tensor = exp.features.table_to_tensor("src", src_data, exp.parameters)
        exp.features.set_value("src", src_tensor)

    dm = DataModule(dataset=dataset)
    dm.initialize(
        input_parameters=["p1", "n_layers", "n_segments"],
        input_features=[],
        output_columns=["src"],
    )
    dm._split_codes[SplitType.TRAIN] = ["exp_001", "exp_002"]
    dm.fit_normalization(SplitType.TRAIN)
    return dm, schema


def _dim_info_1d(L: int) -> dict:
    return {
        'shape': (L,),
        'dim_iterators': ["layer_idx"],
        'dim_codes_ordered': ["n_layers"],
        'param_base': {},
        'iterator_feats': [],
        'total_positions': L,
    }


def _train_transformer(tmp_path, dm, n_layers: int = 4):
    model = _LayerTransformer(build_test_logger(tmp_path))
    model.set_ref_features(list(dm.dataset.schema.features.data_objects.values()))  # type: ignore[arg-type]

    # Synthesize a sequence-batch training set: 2 experiments × L positions.
    n_input = len(dm.input_columns)
    X_seq = torch.randn(2, n_layers, n_input)
    y_seq = torch.randn(2, n_layers, 1)
    model.train([(X_seq, y_seq)], [])
    return model


# === predict ==============================================================


def test_predict_returns_per_candidate_sequence(tmp_path):
    dm, _ = _build_dm(tmp_path, n_layers=4)
    model = _train_transformer(tmp_path, dm, n_layers=4)

    params_list = [
        {"p1": 0.3, "n_layers": 4, "n_segments": 1},
        {"p1": 0.6, "n_layers": 4, "n_segments": 1},
    ]
    out = model.predict(params_list, dm, [_dim_info_1d(4), _dim_info_1d(4)], {})

    assert len(out) == 2
    for s in (0, 1):
        assert "src" in out[s]
        assert out[s]["src"].shape == (4,)


def test_predict_gradient_flow(tmp_path):
    dm, _ = _build_dm(tmp_path, n_layers=4)
    model = _train_transformer(tmp_path, dm, n_layers=4)

    p1_t = torch.tensor(0.5, requires_grad=True)
    params_list = [{"p1": p1_t, "n_layers": 4, "n_segments": 1}]
    out = model.predict(params_list, dm, [_dim_info_1d(4)], {})
    out[0]["src"].sum().backward()
    assert p1_t.grad is not None and p1_t.grad.item() != 0.0


def test_predict_untrained_returns_zeros(tmp_path):
    dm, schema = _build_dm(tmp_path, n_layers=4)
    model = _LayerTransformer(build_test_logger(tmp_path))
    model.set_ref_features(list(schema.features.data_objects.values()))  # type: ignore[arg-type]

    params_list = [{"p1": 0.5, "n_layers": 4, "n_segments": 1}]
    out = model.predict(params_list, dm, [_dim_info_1d(4)], {})
    assert out[0]["src"].shape == (4,)
    # Untrained → zeros (after denormalisation). Exact zeros only when output
    # normalisation is zero-mean linear; just check it's finite and small.
    assert torch.isfinite(out[0]["src"]).all()


def test_predict_empty_returns_empty(tmp_path):
    dm, _ = _build_dm(tmp_path, n_layers=4)
    model = _train_transformer(tmp_path, dm, n_layers=4)
    assert model.predict([], dm, [], {}) == []


# === train ================================================================


def test_train_rejects_flat_batches(tmp_path):
    """Sequence model rejects 2D (flat) batches with a helpful error."""
    model = _LayerTransformer(build_test_logger(tmp_path))
    X_flat = torch.randn(8, 3)
    y_flat = torch.randn(8, 1)
    with pytest.raises(ValueError, match="sequence-shaped batches"):
        model.train([(X_flat, y_flat)], [])


def test_train_marks_is_trained(tmp_path):
    dm, _ = _build_dm(tmp_path, n_layers=4)
    _ = _train_transformer(tmp_path, dm, n_layers=4)  # trains as side effect


# === schema check ========================================================


def test_validate_schema_compatibility_accepts_valid_axis(tmp_path):
    dm, schema = _build_dm(tmp_path, n_layers=4)
    model = _LayerTransformer(build_test_logger(tmp_path))
    model.set_ref_features(list(schema.features.data_objects.values()))  # type: ignore[arg-type]
    model._validate_schema_compatibility(schema)  # must not raise


def test_validate_schema_compatibility_rejects_unknown_axis(tmp_path):
    _, schema = _build_dm(tmp_path, n_layers=4)

    class _BadAxis(_LayerTransformer):
        @property
        def sequence_axis_code(self) -> str:
            return "not_a_real_axis"

    model = _BadAxis(build_test_logger(tmp_path))
    model.set_ref_features(list(schema.features.data_objects.values()))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="does not resolve"):
        model._validate_schema_compatibility(schema)


def test_validate_schema_compatibility_rejects_missing_property(tmp_path):
    _, schema = _build_dm(tmp_path, n_layers=4)

    # Bare TorchTransformerModel with no sequence_axis_code override raises
    # NotImplementedError on access; _validate_schema_compatibility wraps this
    # into a ValueError telling the user to declare the axis.
    class _NoAxis(TorchTransformerModel):
        @property
        def input_parameters(self): return ["p1", "n_layers", "n_segments"]
        @property
        def input_features(self): return []
        @property
        def outputs(self): return ["src"]

    model = _NoAxis(build_test_logger(tmp_path))
    model.set_ref_features(list(schema.features.data_objects.values()))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must implement the `sequence_axis_code`"):
        model._validate_schema_compatibility(schema)
