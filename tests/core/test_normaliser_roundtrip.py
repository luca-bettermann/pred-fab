"""``to_dict`` / ``from_dict`` round-trips for the normaliser modules.

The existing ``test_datamodule_normalization`` round-trip only checks the
restored column lists; these pin that a serialised normaliser rebuilds to the
*same stats and the same transform* — per module and through the DataModule's
``get/set_normalization_state`` used by the inference bundle.
"""

import numpy as np
import pytest
import torch

from pred_fab.core import Dataset, DatasetSchema
from pred_fab.core.data_blocks import Domains, Features, Parameters, PerformanceAttributes
from pred_fab.core.data_objects import Feature, Parameter, PerformanceAttribute
from pred_fab.core.normalisers import (
    IdentityNormaliser,
    MinMaxScalerModule,
    NormaliserModule,
    RobustScalerModule,
    StandardScalerModule,
    make_normaliser,
)
from pred_fab.utils.enum import NormMethod, SplitType
from tests.utils.builders import build_initialized_datamodule


_MODULES = [
    StandardScalerModule(mean=3.0, std=2.0),
    MinMaxScalerModule(min_val=-1.0, max_val=4.0),
    RobustScalerModule(median=2.0, q1=1.0, q3=5.0),
    IdentityNormaliser(),
]
_IDS = ["standard", "minmax", "robust", "identity"]


@pytest.mark.parametrize("module", _MODULES, ids=_IDS)
def test_to_from_dict_preserves_stats_and_transform(module):
    """from_dict(to_dict(m)) yields the same dict and the same forward/reverse."""
    rebuilt = NormaliserModule.from_dict(module.to_dict())

    assert rebuilt.to_dict() == module.to_dict()
    assert type(rebuilt) is type(module)

    probe = torch.tensor([-2.0, 0.0, 3.5, 10.0], dtype=torch.float64)
    assert torch.allclose(rebuilt(probe), module(probe))
    assert torch.allclose(rebuilt.reverse(probe), module.reverse(probe))


@pytest.mark.parametrize("module", _MODULES, ids=_IDS)
def test_roundtrip_preserves_input_type(module):
    """numpy in → numpy out; tensor in → tensor out, before and after round-trip."""
    rebuilt = NormaliserModule.from_dict(module.to_dict())
    np_probe = np.array([0.5, 2.0, 4.0])

    np_out = rebuilt(np_probe)
    assert isinstance(np_out, np.ndarray)
    np.testing.assert_allclose(np_out, module(np_probe))

    t_out = rebuilt(torch.as_tensor(np_probe))
    assert isinstance(t_out, torch.Tensor)


def test_make_normaliser_roundtrip_matches_fit():
    """A fitted module survives a dict round-trip identically (the on-disk path)."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fitted = make_normaliser(NormMethod.STANDARD, data)
    rebuilt = NormaliserModule.from_dict(fitted.to_dict())
    assert rebuilt.get("mean") == pytest.approx(3.0)
    assert rebuilt.get("std") == pytest.approx(float(np.std(data)))  # population std


def _fitted_datamodule(tmp_path):
    param = Parameter.real("param_1", min_val=0.0, max_val=100.0)
    feature = Feature("feat_scalar")
    feature.set_columns(["feat_scalar"])
    schema = DatasetSchema(
        root_folder=str(tmp_path),
        name="rt",
        parameters=Parameters.from_list([param]),
        features=Features.from_list([feature]),
        performance=PerformanceAttributes.from_list([PerformanceAttribute.score("perf_1")]),
        domains=Domains([]),
    )
    dataset = Dataset(schema=schema, debug_flag=True)
    for code, val in {"A": 2.0, "B": 4.0, "C": 6.0}.items():
        dataset.create_experiment(code, parameters={"param_1": val})
    dm = build_initialized_datamodule(
        dataset, ["param_1"], [], ["feat_scalar"],
        split_codes={SplitType.TRAIN: ["A", "B", "C"]},
    )
    dm.fit_normalization()
    return dataset, dm


def test_datamodule_state_roundtrip_preserves_transform(tmp_path):
    """get/set_normalization_state restores the exact same params→array transform."""
    dataset, dm = _fitted_datamodule(tmp_path)
    probe = {"param_1": 5.0}
    before = dm.params_to_array(probe)

    state = dm.get_normalization_state()
    fresh = build_initialized_datamodule(dataset, ["param_1"], [], ["feat_scalar"])
    fresh.set_normalization_state(state)

    np.testing.assert_array_equal(fresh.params_to_array(probe), before)


def test_get_normalization_state_requires_fit(tmp_path):
    """Exporting state before fitting is an error (no silent empty state)."""
    dataset, _ = _fitted_datamodule(tmp_path)
    unfitted = build_initialized_datamodule(dataset, ["param_1"], [], ["feat_scalar"])
    with pytest.raises(RuntimeError):
        unfitted.get_normalization_state()
