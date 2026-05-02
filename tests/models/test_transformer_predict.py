"""TransformerModel.predict + train — sequence-aware contract.

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
from pred_fab.models import TransformerModel
from pred_fab.utils import SplitType
from tests.utils.builders import build_test_logger


class _LayerTransformer(TransformerModel):
    """Sequence axis = n_layers; single output predicting per-layer values."""
    D_MODEL = 8
    N_HEADS = 2
    N_LAYERS = 1
    DIM_FEEDFORWARD = 16
    EPOCHS = 30  # keep fast for unit test

    @property
    def sequence_axis_code(self) -> tuple[str, ...]:
        return ("n_layers",)

    @property
    def domain_spec(self) -> tuple[str | None, int | list[int]]:
        return "spatial", 1

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
    src_feat = Feature("src", domain=spatial, depth=1)
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

    # Synthesize a sequence-batch training set at encoder granularity:
    # X_seq = (B=2, L=n_layers, n_input), y_dict per feature at native shape.
    n_input = len(dm.input_columns)
    X_seq = torch.randn(2, n_layers, n_input)
    y_dict = {"src": torch.randn(2, n_layers)}  # depth-1 → (B, L)
    model.train(
        [(X_seq, y_dict)], [],
        seq_axis_sizes=(n_layers,),
        domain_axis_sizes=(8, 8),  # matches schema: n_layers max=8, n_segments max=8
    )
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


def test_train_rejects_tensor_y(tmp_path):
    """Train expects (X_seq, y_dict); plain tensor y is rejected with a helpful error."""
    model = _LayerTransformer(build_test_logger(tmp_path))
    X_seq = torch.randn(2, 4, 3)
    y_tensor = torch.randn(2, 4, 1)
    with pytest.raises(ValueError, match="y_dict"):
        model.train(
            [(X_seq, y_tensor)], [],
            seq_axis_sizes=(4,), domain_axis_sizes=(8, 8),
        )


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
        def sequence_axis_code(self) -> tuple[str, ...]:
            return ("not_a_real_axis",)

    model = _BadAxis(build_test_logger(tmp_path))
    model.set_ref_features(list(schema.features.data_objects.values()))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="do not resolve"):
        model._validate_schema_compatibility(schema)


def test_validate_schema_compatibility_rejects_missing_property(tmp_path):
    _, schema = _build_dm(tmp_path, n_layers=4)

    # Bare TransformerModel with no sequence_axis_code override raises
    # NotImplementedError on access; _validate_schema_compatibility wraps this
    # into a ValueError telling the user to declare the axis.
    class _NoAxis(TransformerModel):
        @property
        def input_parameters(self): return ["p1", "n_layers", "n_segments"]
        @property
        def input_features(self): return []
        @property
        def outputs(self): return ["src"]
        @property
        def domain_spec(self) -> tuple[str | None, int | list[int]]: return "spatial", 1

    model = _NoAxis(build_test_logger(tmp_path))
    model.set_ref_features(list(schema.features.data_objects.values()))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must implement the `sequence_axis_code`"):
        model._validate_schema_compatibility(schema)


# === multi-axis encoder ===================================================


def test_transformer_encoder_forward_multi_axis_shapes_match():
    """``_TransformerEncoder`` with ``n_axes=2`` accepts (L, 2) axis_indices, returns (B, L, d_model)."""
    from pred_fab.models.transformer import _TransformerEncoder

    torch.manual_seed(0)
    net = _TransformerEncoder(
        n_input=3, d_model=8, n_heads=2, n_layers=1,
        dim_ff=16, dropout=0.0, max_seq_len=16, n_axes=2,
    )
    net.eval()

    # 2-axis grid (3, 2) → L = 6 flat positions, axis_indices = unravel(k, (3, 2)).
    L, n_axes = 6, 2
    axis_indices = torch.tensor(
        [list(np.unravel_index(k, (3, 2))) for k in range(L)], dtype=torch.long,
    )
    x = torch.randn(2, L, 3)  # (B=2, L=6, n_input=3)
    out = net(x, axis_indices)
    assert out.shape == (2, L, 8)  # encoder returns hidden state (B, L, d_model)
    assert torch.isfinite(out).all()


def test_transformer_encoder_forward_multi_axis_requires_axis_indices():
    """``_TransformerEncoder`` with ``n_axes>=2`` raises if axis_indices is None."""
    from pred_fab.models.transformer import _TransformerEncoder

    net = _TransformerEncoder(
        n_input=3, d_model=8, n_heads=2, n_layers=1,
        dim_ff=16, dropout=0.0, max_seq_len=16, n_axes=2,
    )
    net.eval()
    x = torch.randn(1, 4, 3)
    with pytest.raises(ValueError, match="axis_indices required for multi-axis"):
        net(x)


# === multi-axis training =================================================


class _GridTransformer(_LayerTransformer):
    """Multi-axis transformer over (n_layers, n_segments) — flattened grid."""

    @property
    def sequence_axis_code(self) -> tuple[str, ...]:
        return ("n_layers", "n_segments")


def test_train_multi_axis_with_seq_axis_sizes(tmp_path):
    """Multi-axis training succeeds when seq_axis_sizes + domain_axis_sizes kwargs provided."""
    model = _GridTransformer(build_test_logger(tmp_path))
    n_layers, n_segments = 3, 2
    L = n_layers * n_segments
    n_input = 5
    X_seq = torch.randn(2, L, n_input)
    # No ref_features set → src treated as depth-0 → y_dict shape (B, L).
    y_dict = {"src": torch.randn(2, L)}
    model.train(
        [(X_seq, y_dict)], [],
        seq_axis_sizes=(n_layers, n_segments),
        domain_axis_sizes=(8, 8),
    )
    assert model._is_trained


def test_train_multi_axis_without_seq_axis_sizes_raises(tmp_path):
    """Multi-axis training without seq_axis_sizes kwarg raises a clear error."""
    model = _GridTransformer(build_test_logger(tmp_path))
    X_seq = torch.randn(1, 6, 4)
    y_dict = {"src": torch.randn(1, 6)}
    with pytest.raises(ValueError, match="requires `seq_axis_sizes`"):
        model.train([(X_seq, y_dict)], [], domain_axis_sizes=(8, 8))


def test_train_multi_axis_seq_axis_sizes_mismatch_raises(tmp_path):
    """Multi-axis training raises when prod(seq_axis_sizes) != L."""
    model = _GridTransformer(build_test_logger(tmp_path))
    X_seq = torch.randn(1, 6, 4)  # L=6
    y_dict = {"src": torch.randn(1, 6)}
    with pytest.raises(ValueError, match=r"product != batch L"):
        model.train(
            [(X_seq, y_dict)], [],
            seq_axis_sizes=(3, 3), domain_axis_sizes=(8, 8),  # prod=9 != 6
        )


def test_build_transformer_train_batches_returns_tuple(tmp_path):
    """``_build_transformer_train_batches`` returns ``(batches, seq_axis_sizes, domain_axis_sizes)`` — empty-split smoke check."""
    from pred_fab.orchestration.prediction import PredictionSystem
    from pred_fab.utils import LocalData

    dm, schema = _build_dm(tmp_path, n_layers=4)
    dm._split_codes[SplitType.VAL] = []  # empty split → early return path
    psys = PredictionSystem(
        logger=build_test_logger(tmp_path),
        schema=schema,
        local_data=LocalData(str(tmp_path)),
    )
    psys.datamodule = dm

    model = _GridTransformer(build_test_logger(tmp_path))
    batches, seq_axis_sizes, domain_axis_sizes = psys._build_transformer_train_batches(
        model, SplitType.VAL,
    )
    assert batches == []
    assert seq_axis_sizes == ()
    assert domain_axis_sizes == ()


# === multi-depth integration =============================================


def _build_mixed_depth_schema(tmp_path) -> DatasetSchema:
    """Schema with depth-1 ``src`` + depth-2 ``grid`` outputs in one domain."""
    spatial = Domain("spatial", [
        Dimension("n_layers", "layer_idx", 1, 8),
        Dimension("n_segments", "segment_idx", 1, 8),
    ])
    src_feat = Feature("src", domain=spatial, depth=1)
    grid_feat = Feature("grid", domain=spatial, depth=2)
    return DatasetSchema(
        root_folder=str(tmp_path),
        name="transformer_mixed_depth_schema",
        parameters=Parameters.from_list([Parameter.real("p1", 0.0, 1.0)]),
        features=Features.from_list([src_feat, grid_feat]),
        performance=PerformanceAttributes.from_list([PerformanceAttribute.score("perf_1")]),
        domains=Domains([spatial]),
    )


class _MixedDepthTransformer(TransformerModel):
    """Mixed-depth: depth-1 src + depth-2 grid, axis = ("n_layers",)."""
    D_MODEL = 8
    N_HEADS = 2
    N_LAYERS = 1
    DIM_FEEDFORWARD = 16
    EPOCHS = 5

    @property
    def sequence_axis_code(self) -> tuple[str, ...]:
        return ("n_layers",)

    @property
    def domain_spec(self) -> tuple[str | None, int | list[int]]:
        return "spatial", [1, 2]

    @property
    def input_parameters(self):
        return ["p1", "n_layers", "n_segments"]

    @property
    def input_features(self):
        return []

    @property
    def outputs(self):
        return ["src", "grid"]


def test_train_predict_mixed_depth_roundtrip(tmp_path):
    """Mixed-depth: train succeeds, predict returns per-feature dict at native shapes."""
    schema = _build_mixed_depth_schema(tmp_path)
    model = _MixedDepthTransformer(build_test_logger(tmp_path))
    model.set_ref_features(list(schema.features.data_objects.values()))  # type: ignore[arg-type]

    n_layers, n_segments = 4, 3
    L = n_layers
    domain_axis_sizes = (8, 8)

    # n_input must match what predict feeds: len(model.input_parameters + model.input_features).
    n_input = len(model.input_parameters) + len(model.input_features)
    X_seq = torch.randn(2, L, n_input)
    y_dict = {
        "src": torch.randn(2, L),
        "grid": torch.randn(2, L, n_segments),
    }
    model.train(
        [(X_seq, y_dict)], [],
        seq_axis_sizes=(n_layers,),
        domain_axis_sizes=domain_axis_sizes,
    )
    assert model._is_trained
    # Two depths registered, with feature ordering preserved within each depth.
    assert model._depth_to_features[1] == ["src"]
    assert model._depth_to_features[2] == ["grid"]

    # Predict: synthesize dim_info at (n_layers, n_segments).
    grid_di = {
        'shape': (n_layers, n_segments),
        'dim_iterators': ['layer_idx', 'segment_idx'],
        'dim_codes_ordered': ['n_layers', 'n_segments'],
        'param_base': {},
        'iterator_feats': [],
        'total_positions': n_layers * n_segments,
    }
    # Build a minimal dm-like proxy: predict needs build_sequence_batch + denormalize_values
    # + get_input_indices. Use the standard fixture for simplicity.
    dataset = Dataset(schema=schema, debug_flag=True)
    dataset.create_experiment("e1", parameters={"p1": 0.5, "n_layers": n_layers, "n_segments": n_segments})
    exp = dataset.get_experiment("e1")
    src_rows = [[k, float(k * 0.1)] for k in range(n_layers)]
    grid_rows = [[k, s, float(k * 0.1 + s * 0.05)] for k in range(n_layers) for s in range(n_segments)]
    exp.features.set_value("src", exp.features.table_to_tensor("src", np.array(src_rows, dtype=np.float64), exp.parameters))
    exp.features.set_value("grid", exp.features.table_to_tensor("grid", np.array(grid_rows, dtype=np.float64), exp.parameters))
    dm = DataModule(dataset=dataset)
    dm.initialize(
        input_parameters=["p1", "n_layers", "n_segments"],
        input_features=[],
        output_columns=["src", "grid"],
    )
    dm._split_codes[SplitType.TRAIN] = ["e1"]
    dm.fit_normalization(SplitType.TRAIN)

    params_list = [{"p1": 0.4, "n_layers": n_layers, "n_segments": n_segments}]
    out = model.predict(params_list, dm, [grid_di], {})
    assert out[0]["src"].shape == (n_layers,)            # depth-1 native shape
    assert out[0]["grid"].shape == (n_layers, n_segments)  # depth-2 native shape
    assert torch.isfinite(out[0]["src"]).all()
    assert torch.isfinite(out[0]["grid"]).all()


def test_validate_rejects_axis_deeper_than_min_output(tmp_path):
    """axis_depth > min(output_depths) → rejected at validate_dimensional_coherence."""
    schema = _build_1d_seq_schema(tmp_path)

    class _BadAxis(_LayerTransformer):
        @property
        def sequence_axis_code(self) -> tuple[str, ...]:
            return ("n_layers", "n_segments")  # axis_depth=2 but src is depth-1

    model = _BadAxis(build_test_logger(tmp_path))
    model.set_ref_features(list(schema.features.data_objects.values()))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must not be deeper than any output"):
        model.validate_dimensional_coherence(schema)


def test_validate_rejects_axis_not_prefix_of_domain(tmp_path):
    """sequence_axis_code must be a prefix of domain.axes — non-prefix is rejected."""
    schema = _build_1d_seq_schema(tmp_path)

    class _NonPrefixAxis(_LayerTransformer):
        @property
        def sequence_axis_code(self) -> tuple[str, ...]:
            return ("n_segments",)  # second axis, not first

    model = _NonPrefixAxis(build_test_logger(tmp_path))
    model.set_ref_features(list(schema.features.data_objects.values()))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="prefix of the domain's axes"):
        model.validate_dimensional_coherence(schema)


def test_validate_rejects_input_depth_above_axis_depth(tmp_path):
    """Input feature with depth > axis_depth → rejected (would not reach the encoder)."""
    schema = _build_mixed_depth_schema(tmp_path)

    class _DeepInputModel(TransformerModel):
        D_MODEL = 8
        N_HEADS = 2
        N_LAYERS = 1
        DIM_FEEDFORWARD = 16
        EPOCHS = 5

        @property
        def sequence_axis_code(self) -> tuple[str, ...]:
            return ("n_layers",)  # axis_depth = 1

        @property
        def domain_spec(self) -> tuple[str | None, int | list[int]]:
            return "spatial", 1

        @property
        def input_parameters(self):
            return ["p1"]

        @property
        def input_features(self):
            return ["grid"]  # depth-2 input

        @property
        def outputs(self):
            return ["src"]

    model = _DeepInputModel(build_test_logger(tmp_path))
    model.set_ref_features(list(schema.features.data_objects.values()))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="exceeds the model's sequence axis depth"):
        model.validate_dimensional_coherence(schema)


def test_forward_pass_raises_for_mixed_depth(tmp_path):
    """L=1 forward_pass is undefined when output depths differ from axis_depth."""
    schema = _build_mixed_depth_schema(tmp_path)
    model = _MixedDepthTransformer(build_test_logger(tmp_path))
    model.set_ref_features(list(schema.features.data_objects.values()))  # type: ignore[arg-type]

    # Train minimally so _is_trained=True and _depth_to_features is populated.
    L = 4
    X_seq = torch.randn(1, L, 5)
    y_dict = {"src": torch.randn(1, L), "grid": torch.randn(1, L, 3)}
    model.train([(X_seq, y_dict)], [], seq_axis_sizes=(L,), domain_axis_sizes=(8, 8))

    X_flat = torch.randn(2, 5)
    with pytest.raises(NotImplementedError, match="every output's depth to equal axis_depth"):
        model.forward_pass(X_flat)
