"""Tests for Dataset.export_to_tensor_dict + DataModule.prepare_input_from_tensor_dict
().

Three guarantees:
  1. ``export_to_tensor_dict`` produces per-column tensors with dtype-aware
     encoding (cats → long, numerics → float) and matches values from the
     pandas path within float-precision tolerance.
  2. ``cell_meta[i] = (exp_idx, cell_idx)`` for row i — the SS substitution
     hook needs this for prior-cell lookup.
  3. ``prepare_input_from_tensor_dict`` produces the same prepared
     ``(n_rows, n_input_cols)`` tensor as the pandas path.
"""

from __future__ import annotations

import numpy as np
import torch

from pred_fab.core.dataset import ExportedTensorDict
from pred_fab.core import DataModule

from tests.utils.builders import (
    build_workflow_stack,
    build_prepared_workflow_datamodule,
    evaluate_loaded_workflow_experiments,
)


def test_export_to_tensor_dict_has_correct_shape(tmp_path):
    """Per-column tensor + cell_meta with right n_rows."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    dm = build_prepared_workflow_datamodule(agent, dataset)

    out = dataset.export_to_tensor_dict(
        codes,
        x_columns=dm.input_columns,
        y_columns=dm.output_columns,
        categorical_mappings=dm.categorical_mappings,
    )

    assert isinstance(out, ExportedTensorDict)
    assert out.cell_meta.shape[1] == 2  # (exp_idx, cell_idx)
    n_rows = out.n_rows
    # Every X column tensor matches n_rows
    for col, t in out.X.items():
        assert t.shape == (n_rows,), f"X[{col}] shape {t.shape} != ({n_rows},)"
    for col, t in out.y.items():
        assert t.shape == (n_rows,), f"y[{col}] shape {t.shape} != ({n_rows},)"


def test_export_to_tensor_dict_categorical_dtypes(tmp_path):
    """Categoricals are long, numerics are float."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)

    # Build a DM that includes the categorical param_3
    dm = DataModule(dataset)
    dm.initialize(
        input_parameters=["param_1", "param_2", "n_layers", "n_segments", "param_3"],
        input_features=[],
        output_columns=["feature_1"],
    )
    dm.prepare(val_size=0.0, test_size=0.0, recompute=True)

    out = dataset.export_to_tensor_dict(
        codes,
        x_columns=dm.input_columns,
        categorical_mappings=dm.categorical_mappings,
    )

    # Categorical should be long (cat-index)
    assert out.X["param_3"].dtype == torch.long
    # Numerics float
    assert out.X["param_1"].dtype == torch.float32

    # Indices should be in valid range
    cats = dm.categorical_mappings["param_3"]
    assert (out.X["param_3"] >= 0).all()
    assert (out.X["param_3"] < len(cats)).all()


def test_export_to_tensor_dict_cell_meta_tracks_rows(tmp_path):
    """cell_meta[i] = (exp_idx, cell_idx) — exp_idx grouped, cell_idx per-exp."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    dm = build_prepared_workflow_datamodule(agent, dataset)

    out = dataset.export_to_tensor_dict(codes, x_columns=dm.input_columns)
    cm = out.cell_meta.numpy()

    # exp_idx values appear in [0, len(codes))
    assert cm[:, 0].min() >= 0
    assert cm[:, 0].max() < len(codes)
    # cell_idx >= 0 always
    assert (cm[:, 1] >= 0).all()
    # First-experiment rows come before second-experiment rows (export order)
    if len(codes) >= 2:
        first_exp_rows = (cm[:, 0] == 0).sum()
        last_first_idx = np.where(cm[:, 0] == 0)[0].max()
        first_second_idx = np.where(cm[:, 0] == 1)[0].min()
        assert last_first_idx < first_second_idx


def test_prepare_input_from_tensor_dict_matches_pandas_path(tmp_path):
    """The tensor-native and pandas-roundtrip paths produce identical output."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent, dataset)
    dm = build_prepared_workflow_datamodule(agent, dataset)

    # Pandas path
    X_df, _ = dataset.export_to_dataframe(codes)
    X_pandas = dm.prepare_input(X_df)

    # Tensor path
    out = dataset.export_to_tensor_dict(
        codes,
        x_columns=dm.input_columns,
        categorical_mappings=dm.categorical_mappings,
    )
    X_tensor = dm.prepare_input_from_tensor_dict(out.X)

    # Same shape
    assert X_tensor.shape == X_pandas.shape
    # Same values (modulo float rounding)
    torch.testing.assert_close(X_tensor, X_pandas, atol=1e-5, rtol=1e-5)


def test_export_to_tensor_dict_empty_codes(tmp_path):
    """Empty experiment list returns empty tensors."""
    agent, dataset, _ = build_workflow_stack(tmp_path)

    out = dataset.export_to_tensor_dict([])
    assert out.is_empty()
    assert out.cell_meta.shape == (0, 2)
    assert out.X == {}
    assert out.y == {}
