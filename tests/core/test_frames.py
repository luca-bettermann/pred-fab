"""Coordinate-frame transforms — round-trips and the categorical/integer codec.

Guards the single-definition transforms in ``pred_fab.core.frames`` that the
calibration decode path and the prediction/evidence path both delegate to.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pred_fab.core.frames import param_value_to_fill, raw_scalar_to_param, to_unit_frame


# ── to_unit_frame (latent → [0,1]) ─────────────────────────────────────────


def test_to_unit_frame_numpy_and_torch_agree():
    db = np.array([[0.0, 10.0]])
    pts = np.array([[2.0], [5.0]])
    expected = [[0.2], [0.5]]
    assert np.allclose(to_unit_frame(pts, db), expected)
    out_t = to_unit_frame(torch.tensor(pts), torch.tensor(db))
    assert isinstance(out_t, torch.Tensor)
    assert torch.allclose(out_t, torch.tensor(expected, dtype=out_t.dtype))


def test_to_unit_frame_none_is_noop():
    pts = np.array([[2.0]])
    assert to_unit_frame(pts, None) is pts


def test_to_unit_frame_guards_degenerate_span():
    # span 0 → guarded to 1.0 (no div-by-zero), value maps to (x - lo)
    out = to_unit_frame(np.array([[5.0]]), np.array([[5.0, 5.0]]))
    assert np.isfinite(out).all()


# ── value codec: param_value_to_fill ↔ raw_scalar_to_param ──────────────────


def test_categorical_value_round_trip():
    cats = ["A", "B", "C"]  # the datamodule stores categories sorted
    for label in cats:
        fill = param_value_to_fill(label, categories=cats)
        back = raw_scalar_to_param(torch.tensor(fill), categories=cats)
        assert back == label


def test_numeric_value_round_trip_through_decode():
    bounds = (10.0, 50.0)
    for v in (10.0, 25.0, 50.0):
        fill = param_value_to_fill(v, bounds=bounds)
        assert 0.0 <= fill <= 1.0
        raw = fill * (bounds[1] - bounds[0]) + bounds[0]  # the _decode_frames step
        back = raw_scalar_to_param(torch.tensor(raw))      # continuous → tensor
        assert float(back) == pytest.approx(v)


def test_integer_decode_rounds_to_int():
    out = raw_scalar_to_param(torch.tensor(2.4), is_integer=True)
    assert out == 2 and isinstance(out, int)


def test_continuous_decode_keeps_gradient():
    raw = torch.tensor(3.0, requires_grad=True)
    out = raw_scalar_to_param(raw)
    assert out is raw and out.requires_grad


def test_categorical_index_is_clamped():
    cats = ["A", "B"]
    # out-of-range round (e.g. optimiser noise) clamps into the category list
    assert raw_scalar_to_param(torch.tensor(9.0), categories=cats) == "B"
    assert raw_scalar_to_param(torch.tensor(-3.0), categories=cats) == "A"


def test_param_value_to_fill_needs_categories_or_bounds():
    with pytest.raises(ValueError):
        param_value_to_fill(1.0)
