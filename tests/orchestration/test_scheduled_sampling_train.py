"""Tests for cross-model dependency ordering in PredictionSystem.

Replaces the previous SS-orchestration tests after the migration deleted
the cell-loop autoreg / per-row Bernoulli scheduled-sampling machinery.
The topo-sort logic that was load-bearing for ordering recursive sources
before their consumers is preserved here under its more general framing:
plain cross-model dependencies (model B's input_features overlap model
A's outputs → A precedes B).
"""

from typing import Any

import numpy as np
import pytest

from pred_fab.core import DataModule, Dataset, DatasetSchema
from pred_fab.core.data_blocks import (
    Domains, Features, Parameters, PerformanceAttributes,
)
from pred_fab.core.data_objects import (
    Dimension, Domain, Feature, Parameter, PerformanceAttribute,
)
from pred_fab.interfaces import IPredictionModel
from pred_fab.orchestration.prediction import PredictionSystem
from pred_fab.utils import LocalData, PfabLogger, SplitType


class _SourceModel(IPredictionModel):
    """Produces 'src'; consumed by _ConsumerModel as a plain cross-model input."""

    @property
    def input_parameters(self) -> list[str]:
        return ["p1", "n_layers", "n_segments"]

    @property
    def input_features(self) -> list[str]:
        return []

    @property
    def outputs(self) -> list[str]:
        return ["src"]

    def train(self, train_batches, val_batches, **kwargs) -> None:
        self._is_trained = True

    def forward_pass(self, X, gradient_pass: bool = False):
        import torch
        zero = torch.zeros((X.shape[0],), dtype=torch.float32)
        return {feat: zero.clone() for feat in self.outputs}


class _ConsumerModel(IPredictionModel):
    """Consumes 'src' as a plain (non-recursive) input feature."""

    @property
    def input_parameters(self) -> list[str]:
        return ["p1", "n_layers", "n_segments"]

    @property
    def input_features(self) -> list[str]:
        return ["src"]

    @property
    def outputs(self) -> list[str]:
        return ["consumer_out"]

    def train(self, train_batches, val_batches, **kwargs) -> None:
        self._is_trained = True

    def forward_pass(self, X, gradient_pass: bool = False):
        import torch
        zero = torch.zeros((X.shape[0],), dtype=torch.float32)
        return {feat: zero.clone() for feat in self.outputs}


class _CycleModelA(IPredictionModel):
    """Consumes 'b'; produces 'a' — half of a cycle with _CycleModelB."""

    @property
    def input_parameters(self) -> list[str]: return []
    @property
    def input_features(self) -> list[str]: return ["b"]
    @property
    def outputs(self) -> list[str]: return ["a"]
    def train(self, *_, **__) -> None: ...
    def forward_pass(self, X, gradient_pass: bool = False):
        import torch
        zero = torch.zeros((X.shape[0],))
        return {feat: zero.clone() for feat in self.outputs}


class _CycleModelB(IPredictionModel):
    """Consumes 'a'; produces 'b' — half of a cycle with _CycleModelA."""

    @property
    def input_parameters(self) -> list[str]: return []
    @property
    def input_features(self) -> list[str]: return ["a"]
    @property
    def outputs(self) -> list[str]: return ["b"]
    def train(self, *_, **__) -> None: ...
    def forward_pass(self, X, gradient_pass: bool = False):
        import torch
        zero = torch.zeros((X.shape[0],))
        return {feat: zero.clone() for feat in self.outputs}


def _build_schema(tmp_path) -> DatasetSchema:
    spatial = Domain("spatial", [
        Dimension("n_layers", "layer_idx", 1, 4),
        Dimension("n_segments", "segment_idx", 1, 3),
    ])
    src = Feature("src", domain=spatial)
    consumer_out = Feature("consumer_out", domain=spatial)
    return DatasetSchema(
        root_folder=str(tmp_path),
        name="cross_model_schema",
        parameters=Parameters.from_list([Parameter.real("p1", 0.0, 1.0)]),
        features=Features.from_list([src, consumer_out]),
        performance=PerformanceAttributes.from_list([PerformanceAttribute.score("perf_1")]),
        domains=Domains([spatial]),
    )


def _build_pred_system(tmp_path, models: list[IPredictionModel]) -> PredictionSystem:
    schema = _build_schema(tmp_path)
    logger = PfabLogger.get_logger(str(tmp_path / "logs"))
    local_data = LocalData(root_folder=str(tmp_path))
    pred = PredictionSystem(logger=logger, schema=schema, local_data=local_data)
    for m in models:
        pred.models.append(m)
    return pred


# === Topological sort over cross-model deps ==============================


def test_topo_sort_no_deps_returns_input_order(tmp_path):
    """A single model with no cross-model inputs has no deps."""
    logger = PfabLogger.get_logger(str(tmp_path / "logs"))
    pred = _build_pred_system(tmp_path, [_SourceModel(logger=logger)])
    sorted_models = pred._topo_sort_models()
    assert len(sorted_models) == 1


def test_topo_sort_orders_source_before_consumer(tmp_path):
    """ConsumerModel reads 'src' produced by SourceModel; topo sort puts source first."""
    logger = PfabLogger.get_logger(str(tmp_path / "logs"))
    src = _SourceModel(logger=logger)
    cons = _ConsumerModel(logger=logger)
    pred = _build_pred_system(tmp_path, [cons, src])  # registered backwards
    sorted_models = pred._topo_sort_models()
    assert sorted_models[0] is src
    assert sorted_models[1] is cons


def test_topo_sort_raises_on_cycle(tmp_path):
    """A reads B's output and B reads A's output → no DAG, raises."""
    schema_with_ab = DatasetSchema(
        root_folder=str(tmp_path),
        name="cycle_schema",
        parameters=Parameters.from_list([Parameter.real("p", 0.0, 1.0)]),
        features=Features.from_list([Feature("a"), Feature("b")]),
        performance=PerformanceAttributes.from_list([PerformanceAttribute.score("perf")]),
    )
    logger = PfabLogger.get_logger(str(tmp_path / "logs"))
    local_data = LocalData(root_folder=str(tmp_path))
    pred = PredictionSystem(logger=logger, schema=schema_with_ab, local_data=local_data)
    pred.models.extend([_CycleModelA(logger=logger), _CycleModelB(logger=logger)])
    with pytest.raises(ValueError, match="cycle"):
        pred._topo_sort_models()
