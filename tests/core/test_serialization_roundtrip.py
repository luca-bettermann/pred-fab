"""Round-trip serialization harness — guards `from_dict(to_dict(x)) == x`.

The audit found three independent `to_dict`/`from_dict` drops (DataInt
round_digits, DataBlock populated_status, Provenance design/origin), each
hand-maintained and drifted. This property test exercises the round trip for
every DataObject type, the DataBlock value/populated state, and Provenance, so
a future drop fails here instead of silently corrupting persisted state.
"""
import warnings

import numpy as np
import pytest

from pred_fab.core.data_objects import (
    DataObject, DataReal, DataInt, DataBool, DataCategorical, DataDomainAxis,
)
from pred_fab.core.data_blocks import Parameters
from pred_fab.core.provenance import Provenance
from pred_fab.utils.enum import Roles


def _roundtrip_obj(obj: DataObject) -> DataObject:
    return DataObject.from_dict(obj.to_dict())


# Each entry exercises constraints + round_digits where the type supports it.
DATA_OBJECTS = [
    DataReal("water", Roles.PARAMETER, min_val=0.1, max_val=0.9, round_digits=3),
    DataReal("plain", Roles.PARAMETER),
    DataInt("layers", Roles.PARAMETER, min_val=1, max_val=5, round_digits=2),
    DataInt("count", Roles.PARAMETER, min_val=0, max_val=10),
    DataBool("flag", Roles.PARAMETER),
    DataCategorical("material", ["PLA", "ABS"], Roles.PARAMETER),
    DataDomainAxis("n_layers", "layer_idx", Roles.PARAMETER, min_val=1, max_val=8),
]


@pytest.mark.parametrize("obj", DATA_OBJECTS, ids=lambda o: o.code)
def test_dataobject_roundtrip_is_lossless(obj):
    """to_dict is stable through a from_dict round trip (catches dropped fields)."""
    restored = _roundtrip_obj(obj)
    assert restored.to_dict() == obj.to_dict()
    # round_digits specifically (the DataInt drop)
    assert restored.round_digits == obj.round_digits


def test_datablock_preserves_values_and_populated_status():
    block = Parameters([
        DataReal("a", Roles.PARAMETER, round_digits=2),
        DataInt("b", Roles.PARAMETER),
    ])
    block.set_value("a", 1.23, as_populated=True)
    block.set_value("b", 4, as_populated=False)  # set but flagged not-populated

    restored = Parameters.from_dict(block.to_dict())
    assert restored.values == block.values
    assert restored.populated_status == block.populated_status
    assert restored.populated_status["b"] is False  # the dropped flag


def test_provenance_roundtrip_normal():
    snap = {"design": "exploration", "seed": 7, "origin": ["E1", 3],
            "kappa": 0.5, "bounds": {"x": [0, 1]}}
    out = Provenance.from_dict(snap).to_dict()
    assert out == snap


def test_provenance_unknown_design_preserved():
    snap = {"design": "some_future_strategy", "seed": 1}
    with pytest.warns(UserWarning, match="unknown design"):
        prov = Provenance.from_dict(snap)
    assert prov.strategy is None
    assert prov.to_dict() == snap  # lossless — not dropped


def test_provenance_malformed_origin_preserved():
    snap = {"design": "sobol", "origin": "not-a-pair"}
    with pytest.warns(UserWarning, match="malformed origin"):
        prov = Provenance.from_dict(snap)
    assert prov.origin is None
    assert prov.to_dict() == snap  # lossless


def test_provenance_empty():
    assert Provenance.from_dict(None).to_dict() == {}
