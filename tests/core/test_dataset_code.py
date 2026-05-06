"""Tests for ``dataset_code`` — optional dataset-grouping label on ExperimentData.

Covers:
  - Default-None behaviour on ExperimentData and create_experiment.
  - Tagged creation via ``dataset_code=...``.
  - ``DataModule.set_split_dataset`` filtering by dataset_code.
  - Save/load round-trip through LocalData.
"""
from __future__ import annotations

import pytest

from pred_fab.core import Dataset, DataModule
from pred_fab.utils import SplitType
from tests.utils.builders import (
    build_mixed_feature_schema,
    build_dataset_with_single_experiment,
)


# ===== ExperimentData.dataset_code =====

def test_experiment_data_dataset_code_defaults_to_none(tmp_path):
    """ExperimentData with no dataset_code arg has dataset_code=None."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    assert exp.dataset_code is None


def test_create_experiment_stores_dataset_code(tmp_path):
    """create_experiment(dataset_code=...) attaches the label to the experiment."""
    schema = build_mixed_feature_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    exp = dataset.create_experiment(
        "exp_baseline_001",
        parameters={"param_1": 1.5, "dim_1": 2, "dim_2": 3},
        dataset_code="ADVEI_2026/baseline",
    )
    assert exp.dataset_code == "ADVEI_2026/baseline"
    # Round-trip through Dataset's registry.
    assert dataset.get_experiment("exp_baseline_001").dataset_code == "ADVEI_2026/baseline"


# ===== DataModule.set_split_dataset =====

def _build_dm_with_tagged_experiments(tmp_path) -> tuple[Dataset, DataModule]:
    schema = build_mixed_feature_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    # Two baseline + one test, tagged distinctly.
    for code, ds_code in [
        ("exp_b001", "ADVEI_2026/baseline"),
        ("exp_b002", "ADVEI_2026/baseline"),
        ("exp_t001", "ADVEI_2026/test"),
    ]:
        dataset.create_experiment(
            code,
            parameters={"param_1": 1.0, "dim_1": 2, "dim_2": 3},
            dataset_code=ds_code,
        )
    dm = DataModule(dataset=dataset)
    dm.initialize(
        input_parameters=["param_1"],
        input_features=[],
        output_columns=["feature_grid"],
    )
    return dataset, dm


def test_set_split_dataset_filters_train(tmp_path):
    """set_split_dataset assigns matching experiments to the TRAIN split."""
    _, dm = _build_dm_with_tagged_experiments(tmp_path)
    matched = dm.set_split_dataset("ADVEI_2026/baseline")
    assert sorted(matched) == ["exp_b001", "exp_b002"]
    assert sorted(dm.get_split_codes(SplitType.TRAIN)) == ["exp_b001", "exp_b002"]
    # TEST and VAL splits are untouched (empty).
    assert dm.get_split_codes(SplitType.TEST) == []
    assert dm.get_split_codes(SplitType.VAL) == []


def test_set_split_dataset_to_test_split(tmp_path):
    """set_split_dataset accepts a non-default split kwarg (e.g. TEST)."""
    _, dm = _build_dm_with_tagged_experiments(tmp_path)
    dm.set_split_dataset("ADVEI_2026/baseline")          # TRAIN
    dm.set_split_dataset("ADVEI_2026/test", split=SplitType.TEST)
    assert sorted(dm.get_split_codes(SplitType.TRAIN)) == ["exp_b001", "exp_b002"]
    assert dm.get_split_codes(SplitType.TEST) == ["exp_t001"]


def test_set_split_dataset_raises_when_no_match(tmp_path):
    """set_split_dataset raises ValueError when no experiment carries the dataset_code."""
    _, dm = _build_dm_with_tagged_experiments(tmp_path)
    with pytest.raises(ValueError, match="no experiments tagged"):
        dm.set_split_dataset("ADVEI_2026/nonexistent")


# ===== Save / load round-trip through LocalData =====

def test_dataset_code_round_trips_through_save_load(tmp_path):
    """dataset_code persists via save_experiment + reloads via load_experiment."""
    schema = build_mixed_feature_schema(tmp_path)
    dataset_a = Dataset(schema=schema, debug_flag=True)
    dataset_a.create_experiment(
        "exp_001",
        parameters={"param_1": 1.0, "dim_1": 2, "dim_2": 3},
        dataset_code="ADVEI_2026/baseline",
    )
    dataset_a.save_experiment("exp_001", verbose=False)

    # Fresh Dataset on the same schema folder reloads the experiment.
    dataset_b = Dataset(schema=schema, debug_flag=True)
    dataset_b.load_experiment("exp_001", verbose=False)
    reloaded = dataset_b.get_experiment("exp_001")
    assert reloaded.dataset_code == "ADVEI_2026/baseline"


def test_dataset_code_none_does_not_create_metadata_file(tmp_path):
    """Experiments without dataset_code don't write a metadata file (clean default)."""
    import os

    schema = build_mixed_feature_schema(tmp_path)
    dataset_a = Dataset(schema=schema, debug_flag=True)
    dataset_a.create_experiment(
        "exp_untagged",
        parameters={"param_1": 1.0, "dim_1": 2, "dim_2": 3},
    )
    dataset_a.save_experiment("exp_untagged", verbose=False)

    exp_folder = dataset_a.local_data.get_experiment_folder("exp_untagged")
    metadata_file = os.path.join(exp_folder, "metadata.json")
    assert not os.path.exists(metadata_file), (
        f"metadata.json should not be created for untagged experiments, found at {metadata_file}"
    )

    # Reloading still works and dataset_code stays None.
    dataset_b = Dataset(schema=schema, debug_flag=True)
    dataset_b.load_experiment("exp_untagged", verbose=False)
    assert dataset_b.get_experiment("exp_untagged").dataset_code is None


# ===== Dataset.list_dataset_codes =====

def test_list_dataset_codes_returns_distinct_codes(tmp_path):
    """list_dataset_codes returns unique codes in insertion order."""
    schema = build_mixed_feature_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    for code, ds_code in [
        ("e1", "baseline"), ("e2", "baseline"), ("e3", "exploration"),
        ("e4", None), ("e5", "test"),
    ]:
        dataset.create_experiment(
            code, parameters={"param_1": 1.0, "dim_1": 2, "dim_2": 3},
            dataset_code=ds_code,
        )
    assert dataset.list_dataset_codes() == ["baseline", "exploration", "test"]


def test_list_dataset_codes_empty_when_no_tags(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    assert dataset.list_dataset_codes() == []


# ===== DataModule.set_split_datasets (multi-code) =====

def test_set_split_datasets_unions_multiple_codes(tmp_path):
    """set_split_datasets unions experiments from multiple dataset_codes."""
    dataset, dm = _build_dm_with_tagged_experiments(tmp_path)
    matched = dm.set_split_datasets(["ADVEI_2026/baseline", "ADVEI_2026/test"])
    assert sorted(matched) == ["exp_b001", "exp_b002", "exp_t001"]


def test_set_split_datasets_raises_when_none_match(tmp_path):
    _, dm = _build_dm_with_tagged_experiments(tmp_path)
    with pytest.raises(ValueError, match="no experiments tagged"):
        dm.set_split_datasets(["nonexistent"])
