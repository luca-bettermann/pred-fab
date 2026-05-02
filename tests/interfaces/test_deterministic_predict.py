"""DeterministicModel.predict — flat-batched dispatch via inherited default.

The deterministic base inherits ``predict`` from ``IPredictionModel`` (the
flat-batched default that builds a flat batch, runs forward_pass, and
de-multiplexes per-(s, cell) into per-feature tensors). The test
exercises that inheritance end-to-end with a simple formula.
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
from pred_fab.interfaces import DeterministicModel
from pred_fab.utils import SplitType
from tests.utils.builders import build_test_logger


class _DoublingModel(DeterministicModel):
    """formula(X) → 2 * p1 (scalar output, no domain)."""

    @property
    def input_parameters(self):
        return ["p1"]

    @property
    def input_features(self):
        return []

    @property
    def outputs(self):
        return ["y"]

    def formula(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {"y": X[:, 0] * 2.0}


def _build_dm(tmp_path) -> tuple[DataModule, _DoublingModel]:
    schema = DatasetSchema(
        root_folder=str(tmp_path),
        name="det_predict_schema",
        parameters=Parameters.from_list([Parameter.real("p1", 0.0, 1.0)]),
        features=Features.from_list([Feature("y")]),
        performance=PerformanceAttributes.from_list([PerformanceAttribute.score("perf_1")]),
        domains=Domains([]),
    )
    dataset = Dataset(schema=schema, debug_flag=True)
    # Need ≥2 experiments with spread so normalization has non-zero std and
    # round-trips correctly. Otherwise denormalisation collapses to the mean.
    dataset.create_experiment("exp_001", parameters={"p1": 0.0})
    dataset.create_experiment("exp_002", parameters={"p1": 1.0})
    dataset.get_experiment("exp_001").features.set_value("y", np.array([0.0]))
    dataset.get_experiment("exp_002").features.set_value("y", np.array([2.0]))

    dm = DataModule(dataset=dataset)
    dm.initialize(
        input_parameters=["p1"],
        input_features=[],
        output_columns=["y"],
    )
    dm._split_codes[SplitType.TRAIN] = ["exp_001", "exp_002"]
    dm.fit_normalization(SplitType.TRAIN)

    model = _DoublingModel(build_test_logger(tmp_path))
    model.set_ref_features(list(schema.features.data_objects.values()))  # type: ignore[arg-type]
    model.set_normalization_context(
        parameter_stats=dm.get_normalization_state().get('parameter_stats', {}),
        feature_stats=dm.get_normalization_state().get('feature_stats', {}),
        categorical_mappings=dm.get_normalization_state().get('categorical_mappings', {}),
    )
    return dm, model


def _dim_info(shape: tuple = ()) -> dict:
    return {
        'shape': shape,
        'dim_iterators': [],
        'dim_codes_ordered': [],
        'param_base': {},
        'iterator_feats': [],
        'total_positions': int(np.prod(shape)) if shape else 1,
    }


def test_predict_inherits_flat_batch_default(tmp_path):
    """DeterministicModel inherits the flat-batched predict from IPredictionModel."""
    dm, model = _build_dm(tmp_path)
    params_list = [{"p1": 0.25}, {"p1": 0.75}]
    out = model.predict(params_list, dm, [_dim_info(), _dim_info()], {})

    assert len(out) == 2
    # formula = 2 * p1 → [0.5, 1.5]
    assert out[0]["y"].item() == pytest.approx(0.5, abs=1e-6)
    assert out[1]["y"].item() == pytest.approx(1.5, abs=1e-6)


def test_predict_empty_returns_empty(tmp_path):
    dm, model = _build_dm(tmp_path)
    out = model.predict([], dm, [], {})
    assert out == []


