import pytest
import numpy as np
import torch

from pred_fab.orchestration.inference_bundle import InferenceBundle
from pred_fab.utils import SplitType
from tests.utils.builders import (
    build_calibration_system,
    build_dataset_with_single_experiment,
    build_initialized_datamodule,
    build_shape_checking_prediction_system,
    populate_single_experiment_features,
)


def test_prediction_validate_uses_model_specific_input_slices(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    populate_single_experiment_features(dataset)

    datamodule = build_initialized_datamodule(
        dataset=dataset,
        input_parameters=["param_1", "dim_1", "dim_2"],
        input_features=[],
        output_columns=["feature_grid", "feature_d1", "feature_scalar"],
        fitted=True,
        split_codes={SplitType.TRAIN: ["exp_001"], SplitType.VAL: ["exp_001"], SplitType.TEST: []},
    )
    pred_system, models = build_shape_checking_prediction_system(
        tmp_path=tmp_path,
        dataset=dataset,
        datamodule=datamodule,
        model_specs=[
            (["param_1"], ["feature_grid"]),
            (["dim_1", "dim_2"], ["feature_d1"]),
        ],
    )

    pred_system.validate(use_test=False)
    assert models[0].seen_widths and models[0].seen_widths[0] == 1
    assert models[1].seen_widths and models[1].seen_widths[0] == 2


def test_calibration_acquisition_uses_perf_fn_and_delta_evidence_fn(tmp_path):
    """_blend_objective combines perf with Δ∫E via κ — the single objective path."""
    dataset = build_dataset_with_single_experiment(tmp_path)

    perf_calls = []
    de_calls = []

    def perf_fn(params_dict):
        perf_calls.append(params_dict)
        return {"performance_1": 0.7}

    def de_fn(batch):
        de_calls.append(batch.shape)
        return 0.4

    calibration = build_calibration_system(
        tmp_path=tmp_path,
        dataset=dataset,
        perf_fn=perf_fn,
        delta_integrated_evidence_fn=de_fn,
    )

    datamodule = build_initialized_datamodule(
        dataset=dataset,
        input_parameters=["param_1", "dim_1", "dim_2"],
        input_features=[],
        output_columns=[],
        fitted=True,
        split_codes={SplitType.TRAIN: ["exp_001"], SplitType.VAL: [], SplitType.TEST: []},
    )
    calibration._active_datamodule = datamodule

    # Drive the single objective core directly: one candidate, one point.
    # raw_pts (perf) / zscore_pts (evidence) values are irrelevant to the stubs.
    D = len(datamodule.input_columns)
    raw_pts = torch.zeros((1, 1, D), dtype=torch.float64)
    zscore_pts = torch.zeros((1, 1, D), dtype=torch.float64)
    weights = torch.ones((1, 1), dtype=torch.float64)
    w = 0.5
    result = float(calibration._blend_objective(raw_pts, zscore_pts, weights, kappa=w)[0].item())

    assert len(perf_calls) == 1
    assert len(de_calls) == 1
    # score = (1-w)·0.7 + w·0.4 = 0.55 → objective = -0.55
    assert result == pytest.approx(-((1 - w) * 0.7 + w * 0.4), abs=1e-4)


def test_inference_bundle_handles_degenerate_minmax():
    bundle = InferenceBundle(prediction_models=[], normalization_state={"method": "none"}, schema_dict={})
    # NormMethod.MIN_MAX's value is "min_max" — the bundle now matches the enum.
    stats = {"method": "min_max", "min": 2.0, "max": 2.0}

    x = np.array([1.0, 2.0, 3.0])
    x_norm = bundle._apply_normalization(x, stats)
    x_denorm = bundle._reverse_normalization(np.array([0.1, 0.9]), stats)

    assert np.allclose(x_norm, np.zeros_like(x))
    assert np.allclose(x_denorm, np.array([2.0, 2.0]))


def test_prediction_tune_respects_requested_row_slice(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    populate_single_experiment_features(dataset)
    exp = dataset.get_experiment("exp_001")

    datamodule = build_initialized_datamodule(
        dataset=dataset,
        input_parameters=["param_1", "dim_1", "dim_2"],
        input_features=[],
        output_columns=["feature_grid", "feature_d1", "feature_scalar"],
    )
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)

    pred_system, models = build_shape_checking_prediction_system(
        tmp_path=tmp_path,
        dataset=dataset,
        datamodule=datamodule,
        model_specs=[
            (
                ["param_1", "dim_1", "dim_2"],
                ["feature_grid", "feature_d1", "feature_scalar"],
            )
        ],
    )
    model = models[0]

    pred_system.tune(exp_data=exp, start=3, end=6)

    # The base model should see only the requested slice rows.
    assert len(model.seen_batch_sizes) > 0
    assert 3 in model.seen_batch_sizes
    # Default residual model path should be active and fitted after tune.
    assert getattr(pred_system.residual_model, "_is_fitted", False) is True


def test_inference_bundle_rejects_unknown_categorical():
    """A categorical value outside the trained vocabulary must raise, not
    silently encode to index 0 (wrong predictions for a deployed model)."""
    import pandas as pd
    bundle = InferenceBundle(
        prediction_models=[],
        normalization_state={"categorical_mappings": {"material": ["A", "B"]}},
        schema_dict={"parameters": {"data_objects": {"material": {}}}},
    )
    bundle._validate_inputs(pd.DataFrame({"material": ["A", "B"]}))  # ok
    with pytest.raises(ValueError, match="Unknown categorical value"):
        bundle._validate_inputs(pd.DataFrame({"material": ["A", "Z"]}))


def test_inference_bundle_slices_per_model_input_columns():
    """predict must feed each model only its own input columns, ordered by the
    model's input_parameters + input_features — not the full input matrix."""
    import pandas as pd

    class _StubModel:
        input_parameters = ["p2"]
        input_features = []
        outputs = ["y"]

        def __init__(self):
            self.seen_width = None

        def forward_pass(self, X):
            self.seen_width = X.shape[1]
            return {"y": torch.zeros(X.shape[0])}

    model = _StubModel()
    bundle = InferenceBundle(
        prediction_models=[model],  # type: ignore[list-item]
        normalization_state={"input_columns": ["p1", "p2", "p3"],
                             "categorical_mappings": {}, "parameter_stats": {},
                             "feature_stats": {}},
        schema_dict={"parameters": {"data_objects": {"p1": {}, "p2": {}, "p3": {}}}},
    )
    bundle.predict(pd.DataFrame({"p1": [1.0], "p2": [2.0], "p3": [3.0]}))
    # Model declares one input (p2) — it must receive 1 column, not all 3.
    assert model.seen_width == 1
