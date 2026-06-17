"""Tests for ``dataset_code`` — optional dataset-grouping label on ExperimentData.

Covers:
  - Default-None behaviour on ExperimentData and create_experiment.
  - Tagged creation via ``dataset_code=...``.
  - ``DataModule.set_split_dataset`` filtering by dataset_code.
  - Save/load round-trip through LocalData.
"""
from __future__ import annotations

import pytest

from pred_fab.core import Dataset, DataModule, ExperimentSet
from pred_fab.interfaces import IExternalData
from pred_fab.utils import SplitType
from pred_fab.utils.enum import SourceStep
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


# ===== Dataset.select (the selection primitive) =====

def _build_varied_dataset(tmp_path) -> Dataset:
    """Five experiments with distinct param_1 values and dataset_codes."""
    schema = build_mixed_feature_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    for code, p1, ds_code in [
        ("e1", 1.0, "discovery"),
        ("e2", 5.0, "discovery"),
        ("e3", 9.0, "exploration"),
        ("e4", 3.0, None),
        ("e5", 7.0, "test"),
    ]:
        dataset.create_experiment(
            code, parameters={"param_1": p1, "dim_1": 2, "dim_2": 3},
            dataset_code=ds_code,
        )
    return dataset


def test_select_filters_by_dataset_code_predicate(tmp_path):
    """select() with a dataset_code predicate returns matching codes (what set_split delegates to)."""
    dataset = _build_varied_dataset(tmp_path)
    assert dataset.select(lambda e: e.dataset_code == "discovery") == ["e1", "e2"]


def test_select_preserves_dataset_order(tmp_path):
    """select() returns codes in dataset insertion order, not predicate-evaluation order."""
    dataset = _build_varied_dataset(tmp_path)
    assert dataset.select(lambda e: e.dataset_code is not None) == ["e1", "e2", "e3", "e5"]


def test_select_by_param_region(tmp_path):
    """Arbitrary predicate: select over a parameter region — beyond what dataset_code allows."""
    dataset = _build_varied_dataset(tmp_path)
    codes = dataset.select(
        lambda e: e.parameters.has_value("param_1") and e.parameters.get_value("param_1") >= 5.0
    )
    assert codes == ["e2", "e3", "e5"]


def test_select_all_and_none(tmp_path):
    """Trivial predicates: True selects every code, False selects nothing (no raise — caller's policy)."""
    dataset = _build_varied_dataset(tmp_path)
    assert dataset.select(lambda e: True) == ["e1", "e2", "e3", "e4", "e5"]
    assert dataset.select(lambda e: False) == []


# ===== generative provenance round-trip (config_snapshot / design) =====

class _RecordingExternal(IExternalData):
    """Minimal IExternalData that records pushed provenance and serves it on pull."""

    def __init__(self) -> None:
        super().__init__(client=object())
        self.pushed: dict[str, dict] = {}
        self.sets: list[dict] = []

    def pull_parameters(self, exp_codes):
        return list(exp_codes), {}

    def push_provenance(self, exp_codes, provenance, recompute=False):
        self.pushed.update(provenance)
        return True

    def pull_provenance(self, exp_codes):
        found = {c: self.pushed[c] for c in exp_codes if c in self.pushed}
        missing = [c for c in exp_codes if c not in self.pushed]
        return missing, found

    def push_experiment_sets(self, sets, recompute=False):
        self.sets = list(sets)
        return True

    def pull_experiment_sets(self):
        return list(self.sets)


def test_source_and_provenance_properties_default_empty(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_all_experiments()[0]
    assert exp.source is None
    assert exp.kappa is None
    assert exp.provenance == {}


def test_provenance_round_trips_through_local_save_load(tmp_path):
    """config_snapshot (source + settings) persists via save and reloads via load."""
    schema = build_mixed_feature_schema(tmp_path)
    ds_a = Dataset(schema=schema, debug_flag=True)
    ds_a.create_experiment(
        "exp_sobol_001",
        parameters={"param_1": 1.0, "dim_1": 2, "dim_2": 3},
        config_snapshot={"source": "sobol_step", "seed": 7, "kappa": None},
    )
    ds_a.save_experiment("exp_sobol_001", verbose=False)

    ds_b = Dataset(schema=schema, debug_flag=True)
    ds_b.load_experiment("exp_sobol_001", verbose=False)
    exp = ds_b.get_experiment("exp_sobol_001")
    assert exp.source is SourceStep.SOBOL
    assert exp.provenance == {"source": "sobol_step", "seed": 7, "kappa": None}


def test_select_by_source_on_reloaded_experiments(tmp_path):
    """Provenance is queryable via select() after a save/reload — local mirrors NocoDB."""
    schema = build_mixed_feature_schema(tmp_path)
    ds_a = Dataset(schema=schema, debug_flag=True)
    for code, source in [("e_s", "sobol_step"), ("e_d", "discovery_step"), ("e_x", "exploration_step")]:
        ds_a.create_experiment(
            code, parameters={"param_1": 1.0, "dim_1": 2, "dim_2": 3},
            config_snapshot={"source": source},
        )
    ds_a.save_experiments(["e_s", "e_d", "e_x"], verbose=False)

    ds_b = Dataset(schema=schema, debug_flag=True)
    ds_b.load_experiments(["e_s", "e_d", "e_x"], verbose=False)
    assert ds_b.select(lambda e: e.source is SourceStep.SOBOL) == ["e_s"]
    assert sorted(
        ds_b.select(lambda e: e.source in {SourceStep.DISCOVERY, SourceStep.EXPLORATION})
    ) == ["e_d", "e_x"]


def test_save_pushes_provenance_to_external(tmp_path):
    schema = build_mixed_feature_schema(tmp_path)
    ext = _RecordingExternal()
    ds = Dataset(schema=schema, external_data=ext, debug_flag=True)
    ds.create_experiment(
        "e1", parameters={"param_1": 1.0, "dim_1": 2, "dim_2": 3},
        config_snapshot={"source": "sobol_step", "seed": 1},
    )
    ds.save_experiment("e1", verbose=False)
    assert ext.pushed["e1"] == {"source": "sobol_step", "seed": 1}


def test_create_experiment_appends_to_declared_set(tmp_path):
    """The accumulation path: experiment_set= appends each code, in order."""
    schema = build_mixed_feature_schema(tmp_path)
    ds = Dataset(schema=schema, debug_flag=True)
    ds.add_experiment_set(ExperimentSet("run_1", ordered=True))
    for code in ("e1", "e2", "e3"):
        ds.create_experiment(
            code, parameters={"param_1": 1.0, "dim_1": 2, "dim_2": 3}, experiment_set="run_1",
        )
    assert ds.get_experiment_set("run_1").members == ["e1", "e2", "e3"]


def test_create_experiment_unknown_set_raises_and_creates_nothing(tmp_path):
    """An undeclared set is a fail-fast error — no half-built experiment left behind."""
    schema = build_mixed_feature_schema(tmp_path)
    ds = Dataset(schema=schema, debug_flag=True)
    with pytest.raises(KeyError, match="not declared"):
        ds.create_experiment(
            "e1", parameters={"param_1": 1.0, "dim_1": 2, "dim_2": 3}, experiment_set="missing",
        )
    assert "e1" not in [e.code for e in ds.get_all_experiments()]


def test_select_set_builds_registered_set_from_query(tmp_path):
    """The query path: select_set builds + registers an (unordered) set from a predicate."""
    schema = build_mixed_feature_schema(tmp_path)
    ds = Dataset(schema=schema, debug_flag=True)
    for code, src in [("e_s", "sobol_step"), ("e_x", "exploration_step")]:
        ds.create_experiment(
            code, parameters={"param_1": 1.0, "dim_1": 2, "dim_2": 3},
            config_snapshot={"source": src},
        )
    s = ds.select_set("sobols", lambda e: e.source is SourceStep.SOBOL)
    assert s.members == ["e_s"] and s.ordered is False
    assert ds.get_experiment_set("sobols") is s   # registered


def test_experiment_set_registry_round_trips_locally(tmp_path):
    """ExperimentSets save to / load from local storage."""
    schema = build_mixed_feature_schema(tmp_path)
    ds = Dataset(schema=schema, debug_flag=True)
    ds.add_experiment_set(ExperimentSet("D1", members=["e_d1", "e_d2"]))
    ds.add_experiment_set(ExperimentSet("E1", members=["e_e1", "e_e2"], ordered=True))
    ds.save_experiment_sets()

    ds2 = Dataset(schema=schema, debug_flag=True)
    ds2.load_experiment_sets()
    assert {e.code for e in ds2.list_experiment_sets()} == {"D1", "E1"}
    loaded = ds2.get_experiment_set("E1")
    assert loaded.members == ["e_e1", "e_e2"] and loaded.ordered is True


def test_experiment_sets_push_to_external_on_save(tmp_path):
    schema = build_mixed_feature_schema(tmp_path)
    ext = _RecordingExternal()
    ds = Dataset(schema=schema, external_data=ext, debug_flag=True)
    ds.add_experiment_set(ExperimentSet("D1", members=["a", "b"]))
    ds.save_experiment_sets()
    assert {s["code"] for s in ext.sets} == {"D1"}     # pushed externally alongside local


def test_experiment_sets_load_falls_back_to_external(tmp_path):
    """No local experiment_sets.json → load pulls definitions from the external source."""
    schema = build_mixed_feature_schema(tmp_path)
    ext = _RecordingExternal()
    ext.sets = [
        {"code": "E1", "members": ["e1"], "ordered": True},
        {"code": "D1", "members": ["d1"], "ordered": False},
    ]
    ds = Dataset(schema=schema, external_data=ext, debug_flag=True)
    ds.load_experiment_sets()
    assert {e.code for e in ds.list_experiment_sets()} == {"D1", "E1"}
    assert ds.get_experiment_set("E1").ordered is True


def test_load_provenance_falls_back_to_external(tmp_path):
    """When no local config.json exists, provenance is pulled from the external source."""
    schema = build_mixed_feature_schema(tmp_path)
    ext = _RecordingExternal()
    ext.pushed["e1"] = {"source": "exploration_step", "kappa": 0.4}
    ds = Dataset(schema=schema, external_data=ext, debug_flag=True)
    ds.create_experiment("e1", parameters={"param_1": 1.0, "dim_1": 2, "dim_2": 3})  # no snapshot
    assert ds.get_experiment("e1").config_snapshot is None

    ds._load_provenance(["e1"])
    assert ds.get_experiment("e1").source is SourceStep.EXPLORATION
