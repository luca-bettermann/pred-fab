"""Train-only normalisation fit — the split-leakage guarantee.

``DataModule._fit_normalize`` must compute normaliser stats from the TRAIN
split's rows only; validation/test values must never influence them
(otherwise model evaluation leaks information from held-out data). A
domain-less scalar schema gives exactly one row per experiment, so a param's
stats are the plain statistics of its per-experiment values — easy to pin.
"""

import pytest

from pred_fab.core import Dataset, DatasetSchema
from pred_fab.core.data_blocks import Domains, Features, Parameters, PerformanceAttributes
from pred_fab.core.data_objects import Feature, Parameter, PerformanceAttribute
from pred_fab.core.normalisers import NormaliserModule
from pred_fab.utils.enum import SplitType
from tests.utils.builders import build_initialized_datamodule


def _scalar_dataset(tmp_path, param_values: dict[str, float]) -> Dataset:
    """One real param + one scalar feature, no domain → one row per experiment."""
    param = Parameter.real("param_1", min_val=0.0, max_val=1e9)
    feature = Feature("feat_scalar")
    feature.set_columns(["feat_scalar"])
    schema = DatasetSchema(
        root_folder=str(tmp_path),
        name="leak",
        parameters=Parameters.from_list([param]),
        features=Features.from_list([feature]),
        performance=PerformanceAttributes.from_list([PerformanceAttribute.score("perf_1")]),
        domains=Domains([]),
    )
    dataset = Dataset(schema=schema, debug_flag=True)
    for code, value in param_values.items():
        dataset.create_experiment(code, parameters={"param_1": value})
    return dataset


def _fit(dataset, train, val, test) -> NormaliserModule:
    datamodule = build_initialized_datamodule(
        dataset,
        input_parameters=["param_1"],
        input_features=[],
        output_columns=["feat_scalar"],
        split_codes={SplitType.TRAIN: train, SplitType.VAL: val, SplitType.TEST: test},
    )
    datamodule.fit_normalization()
    return datamodule._parameter_stats["param_1"]


def test_stats_computed_from_train_rows_only(tmp_path):
    """Train param_1 ∈ {2, 4} → StandardScaler mean=3, population std=1."""
    dataset = _scalar_dataset(tmp_path, {"A": 2.0, "B": 4.0, "C": 1000.0, "D": 2000.0})
    stats = _fit(dataset, train=["A", "B"], val=["C"], test=["D"])
    assert stats.get("mean") == pytest.approx(3.0)
    assert stats.get("std") == pytest.approx(1.0)


def test_val_test_values_do_not_leak(tmp_path):
    """Identical train split, wildly different val/test contents → identical stats."""
    moderate = _scalar_dataset(tmp_path / "a", {"A": 2.0, "B": 4.0, "C": 5.0, "D": 6.0})
    extreme = _scalar_dataset(tmp_path / "b", {"A": 2.0, "B": 4.0, "C": 1e8, "D": 9e8})

    s_moderate = _fit(moderate, train=["A", "B"], val=["C"], test=["D"])
    s_extreme = _fit(extreme, train=["A", "B"], val=["C"], test=["D"])

    assert s_moderate.get("mean") == s_extreme.get("mean")
    assert s_moderate.get("std") == s_extreme.get("std")


def test_moving_an_experiment_into_train_changes_stats(tmp_path):
    """Positive control: the fit genuinely reads the rows it is given — adding C
    to TRAIN shifts the mean, so the invariance above is real, not vacuous."""
    dataset = _scalar_dataset(tmp_path, {"A": 2.0, "B": 4.0, "C": 1000.0, "D": 2000.0})
    without_c = _fit(dataset, train=["A", "B"], val=["C"], test=["D"])
    with_c = _fit(dataset, train=["A", "B", "C"], val=[], test=["D"])
    assert with_c.get("mean") != without_c.get("mean")
    assert with_c.get("mean") == pytest.approx((2.0 + 4.0 + 1000.0) / 3.0)
