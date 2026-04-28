"""Tests for scheduled-sampling orchestration in PredictionSystem.train()."""

from typing import Any
from unittest.mock import patch

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
    """Model whose output 'src' feeds another model's recursive input."""

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

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((X.shape[0], 1), dtype=np.float32)

    def encode(self, X: np.ndarray) -> np.ndarray:
        return X


class _RecursiveModel(IPredictionModel):
    """Model that consumes 'prev_src_1' (recursive feature on src)."""

    @property
    def input_parameters(self) -> list[str]:
        return ["p1", "n_layers", "n_segments"]

    @property
    def input_features(self) -> list[str]:
        return ["prev_src_1"]

    @property
    def outputs(self) -> list[str]:
        return ["consumer_out"]

    def train(self, train_batches, val_batches, **kwargs) -> None:
        self._is_trained = True

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((X.shape[0], 1), dtype=np.float32)

    def encode(self, X: np.ndarray) -> np.ndarray:
        return X


class _CycleModelA(IPredictionModel):
    @property
    def input_parameters(self) -> list[str]: return []
    @property
    def input_features(self) -> list[str]: return ["prev_b_1"]
    @property
    def outputs(self) -> list[str]: return ["a"]
    def train(self, *_, **__) -> None: ...
    def forward_pass(self, X): return np.zeros((X.shape[0], 1))
    def encode(self, X): return X


class _CycleModelB(IPredictionModel):
    @property
    def input_parameters(self) -> list[str]: return []
    @property
    def input_features(self) -> list[str]: return ["prev_a_1"]
    @property
    def outputs(self) -> list[str]: return ["b"]
    def train(self, *_, **__) -> None: ...
    def forward_pass(self, X): return np.zeros((X.shape[0], 1))
    def encode(self, X): return X


def _build_schema(tmp_path) -> DatasetSchema:
    spatial = Domain("spatial", [
        Dimension("n_layers", "layer_idx", 1, 4),
        Dimension("n_segments", "segment_idx", 1, 3),
    ])
    layer_dim, _ = spatial.axes
    src = Feature.array("src", domain=spatial)
    consumer_out = Feature.array("consumer_out", domain=spatial)
    return DatasetSchema(
        root_folder=str(tmp_path),
        name="ss_orch_schema",
        parameters=Parameters.from_list([Parameter.real("p1", 0.0, 1.0)]),
        features=Features.from_list([
            src,
            consumer_out,
            *Feature.recursive("prev_src", source=src, dimensions=(layer_dim,), max_depth=1),
        ]),
        performance=PerformanceAttributes.from_list([PerformanceAttribute.score("perf_1")]),
        domains=Domains([spatial]),
    )


def _build_pred_system(tmp_path, models: list[IPredictionModel]) -> tuple[PredictionSystem, DataModule]:
    schema = _build_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    dataset.create_experiment(
        "exp_001", parameters={"p1": 0.5, "n_layers": 4, "n_segments": 3},
    )
    exp = dataset.get_experiment("exp_001")
    grid_rows = []
    for k in range(4):
        for s in range(3):
            grid_rows.append([k, s, k * 10.0 + s])
    grid = np.array(grid_rows, dtype=np.float64)
    src_tensor = exp.features.table_to_tensor("src", grid, exp.parameters)
    exp.features.set_value("src", src_tensor)
    exp.features.set_value("consumer_out", src_tensor)
    prev = np.full_like(src_tensor, np.nan)
    prev[1:, :] = src_tensor[:-1, :]
    exp.features.set_value("prev_src_1", prev)

    logger = PfabLogger.get_logger(str(tmp_path / "logs"))
    local_data = LocalData(root_folder=str(tmp_path))
    pred = PredictionSystem(logger=logger, schema=schema, local_data=local_data)
    for m in models:
        pred.models.append(m)

    dm = DataModule(dataset=dataset)
    dm.initialize(
        input_parameters=["p1", "n_layers", "n_segments"],
        input_features=["prev_src_1"],
        output_columns=["src", "consumer_out"],
    )
    dm._split_codes[SplitType.TRAIN] = ["exp_001"]
    return pred, dm


# ── Topological sort ──

def test_topo_sort_self_recursion_only(tmp_path):
    """A single model that's its own recursive source has no cross-model deps."""
    logger = PfabLogger.get_logger(str(tmp_path / "logs"))
    pred, _ = _build_pred_system(tmp_path, [_RecursiveModel(logger=logger)])
    sorted_models = pred._topo_sort_models()
    assert len(sorted_models) == 1


def test_topo_sort_orders_source_before_consumer(tmp_path):
    """SourceModel produces 'src'; RecursiveModel's prev_src_1 references it."""
    logger = PfabLogger.get_logger(str(tmp_path / "logs"))
    src = _SourceModel(logger=logger)
    cons = _RecursiveModel(logger=logger)
    pred, _ = _build_pred_system(tmp_path, [cons, src])  # registered backwards
    sorted_models = pred._topo_sort_models()
    assert sorted_models[0] is src
    assert sorted_models[1] is cons


def test_topo_sort_raises_on_cycle(tmp_path):
    schema = _build_schema(tmp_path)
    # Manually add a recursive feature pointing the other way to make a cycle
    a_feat = Feature.array("a")
    b_feat = Feature.array("b")
    cycle_schema = DatasetSchema(
        root_folder=str(tmp_path),
        name="cycle_schema",
        parameters=Parameters.from_list([Parameter.real("p", 0.0, 1.0)]),
        features=Features.from_list([
            a_feat, b_feat,
            *Feature.recursive("prev_a", source=a_feat, dimensions=(), max_depth=1),
            *Feature.recursive("prev_b", source=b_feat, dimensions=(), max_depth=1),
        ]),
        performance=PerformanceAttributes.from_list([PerformanceAttribute.score("perf")]),
    )
    logger = PfabLogger.get_logger(str(tmp_path / "logs"))
    local_data = LocalData(root_folder=str(tmp_path))
    pred = PredictionSystem(logger=logger, schema=cycle_schema, local_data=local_data)
    pred.models.extend([_CycleModelA(logger=logger), _CycleModelB(logger=logger)])
    with pytest.raises(ValueError, match="cycle"):
        pred._topo_sort_models()


# ── Recursive-input detection ──

def test_model_has_recursive_inputs_true(tmp_path):
    logger = PfabLogger.get_logger(str(tmp_path / "logs"))
    pred, _ = _build_pred_system(tmp_path, [_RecursiveModel(logger=logger)])
    assert pred._model_has_recursive_inputs(pred.models[0]) is True


def test_model_has_recursive_inputs_false(tmp_path):
    logger = PfabLogger.get_logger(str(tmp_path / "logs"))
    pred, _ = _build_pred_system(tmp_path, [_SourceModel(logger=logger)])
    assert pred._model_has_recursive_inputs(pred.models[0]) is False


# ── Schedule ──

def test_ss_schedule_round_zero_is_teacher_forced(tmp_path):
    logger = PfabLogger.get_logger(str(tmp_path / "logs"))
    pred, _ = _build_pred_system(tmp_path, [_RecursiveModel(logger=logger)])
    pred.n_ss_rounds = 4
    assert pred._ss_p_for_round(0, 4) == 0.0


def test_ss_schedule_last_round_is_full_student(tmp_path):
    logger = PfabLogger.get_logger(str(tmp_path / "logs"))
    pred, _ = _build_pred_system(tmp_path, [_RecursiveModel(logger=logger)])
    pred.n_ss_rounds = 4
    assert pred._ss_p_for_round(3, 4) == 1.0


def test_ss_schedule_floor_lifts_round_one(tmp_path):
    logger = PfabLogger.get_logger(str(tmp_path / "logs"))
    pred, _ = _build_pred_system(tmp_path, [_RecursiveModel(logger=logger)])
    pred.ss_schedule_floor = 0.2
    p1 = pred._ss_p_for_round(1, 4)
    assert p1 > 0.2
