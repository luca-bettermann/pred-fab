import numpy as np

from pred_fab.orchestration.inference_bundle import InferenceBundle
from pred_fab.utils import SplitType
from tests.utils.builders import (
    build_calibration_system_with_capturing_surrogate,
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


def test_calibration_surrogate_training_skips_missing_performance_rows(tmp_path):
    dataset = build_dataset_with_single_experiment(tmp_path)
    datamodule = build_initialized_datamodule(
        dataset=dataset,
        input_parameters=["param_1", "dim_1", "dim_2"],
        input_features=[],
        output_columns=[],
        fitted=True,
        split_codes={SplitType.TRAIN: ["exp_001"], SplitType.VAL: [], SplitType.TEST: []},
    )
    calibration, surrogate = build_calibration_system_with_capturing_surrogate(
        tmp_path=tmp_path,
        dataset=dataset,
    )

    calibration.train_surrogate_model(datamodule)
    assert surrogate.fit_calls == 0

    exp = dataset.get_experiment("exp_001")
    exp.performance.set_values_from_dict({"performance_1": 0.5}, logger=dataset.logger)
    calibration.train_surrogate_model(datamodule)
    assert surrogate.fit_calls == 1
    assert surrogate.last_shapes == ((1, 3), (1, 1))


def test_inference_bundle_handles_degenerate_minmax():
    bundle = InferenceBundle(prediction_models=[], normalization_state={"method": "none"}, schema_dict={})
    stats = {"method": "minmax", "min": 2.0, "max": 2.0}

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
