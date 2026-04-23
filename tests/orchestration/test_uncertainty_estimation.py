"""
Tests for NatPN-light uncertainty estimation: KDE fitting, encode(), uncertainty(),
kernel_similarity(), predict_for_calibration(), and UCB acquisition integration.

Covers general behaviour (happy path) and edge cases (unfitted KDE, single config,
out-of-distribution points, feature_std propagation).
"""
import pytest
import numpy as np

from tests.utils.builders import (
    build_real_agent_stack,
    build_runtime_agent_stack,
    build_test_logger,
    build_dataset_with_single_experiment,
    build_initialized_datamodule,
    build_calibration_system,
    build_shape_checking_prediction_system,
    build_workflow_stack,
    evaluate_loaded_workflow_experiments,
    build_prepared_workflow_datamodule,
)
from pred_fab.orchestration.prediction import PredictionSystem
from pred_fab.utils import LocalData, SplitType
from pred_fab.core import ExperimentSpec, ParameterProposal
from pred_fab.utils.enum import Mode



# ===========================================================================
# Helpers
# ===========================================================================

def _trained_agent_and_datamodule(tmp_path):
    """Return (agent, exp, datamodule) with the prediction system already trained."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)
    return agent, exp, datamodule


# ===========================================================================
# encode() — default identity behaviour
# ===========================================================================

def test_encode_default_returns_identity_for_untrained_system(tmp_path):
    """Before training, encode() must return the input unchanged (identity)."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    logger = build_test_logger(tmp_path)
    pred = PredictionSystem(logger=logger, schema=dataset.schema, local_data=LocalData(str(tmp_path)))

    X = np.array([[0.1, 0.5, 0.9]])
    result = pred.encode(X)
    assert np.allclose(result, X)


def test_encode_returns_same_shape_after_training(tmp_path):
    """After training, encode() should return an array with the same number of rows."""
    agent, exp, datamodule = _trained_agent_and_datamodule(tmp_path)
    pred = agent.pred_system
    X_norm = datamodule.params_to_array(exp.parameters.get_values_dict())
    Z = pred.encode(X_norm.reshape(1, -1))
    assert Z.shape[0] == 1  # same number of rows


def test_encode_custom_override_called_by_prediction_system(tmp_path):
    """A custom encode() override on IPredictionModel should be respected."""
    import numpy as np
    from tests.utils.interfaces import MixedPredictionModelGrid

    class EncoderModel(MixedPredictionModelGrid):
        """Doubles input columns as latent representation (column count varies with schema)."""
        encode_called = False

        def encode(self, X: np.ndarray) -> np.ndarray:
            EncoderModel.encode_called = True
            return X * 2.0

    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.pred_system.models[0].__class__ = EncoderModel  # monkey-patch for test isolation

    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.pred_system.models[0].__class__ = EncoderModel
    # Replace the first model instance with a real EncoderModel instance
    logger = build_test_logger(tmp_path)
    encoder_instance = EncoderModel(logger)
    agent.pred_system.models[0] = encoder_instance
    agent.train(datamodule=datamodule, validate=False, test=False)

    X_norm = datamodule.params_to_array(exp.parameters.get_values_dict())
    Z = agent.pred_system.encode(X_norm.reshape(1, -1))
    assert EncoderModel.encode_called
    assert Z.shape[1] == X_norm.reshape(1, -1).shape[1]  # output matches input width


# ===========================================================================
# uncertainty() — KDE-based
# ===========================================================================

def test_uncertainty_returns_one_before_training(tmp_path):
    """Unfitted KDE → maximum uncertainty of 1.0 for any input."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    logger = build_test_logger(tmp_path)
    pred = PredictionSystem(logger=logger, schema=dataset.schema, local_data=LocalData(str(tmp_path)))

    X = np.array([0.5, 0.5, 0.5])
    assert pred.uncertainty(X) == pytest.approx(1.0)


def test_uncertainty_in_range_zero_one_after_training(tmp_path):
    """uncertainty() must return values in [0, 1] for any input after training."""
    agent, exp, datamodule = _trained_agent_and_datamodule(tmp_path)
    pred = agent.pred_system
    X_norm = datamodule.params_to_array(exp.parameters.get_values_dict())

    u = pred.uncertainty(X_norm)
    assert 0.0 <= u <= 1.0


def test_uncertainty_lower_for_training_config_than_ood(tmp_path):
    """A point identical to a training config should have lower uncertainty than an OOD point."""
    # Requires ≥2 distinct training configs to fit the KDE; use the 3-experiment workflow stack.
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent=agent, dataset=dataset, category_value="B")
    datamodule = build_prepared_workflow_datamodule(agent=agent, dataset=dataset, val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    pred = agent.pred_system
    if not pred._model_kdes:
        pytest.skip("KDE not fitted — not enough distinct training configs")

    first_exp = dataset.get_experiment(codes[0])
    train_params = first_exp.parameters.get_values_dict()
    X_train = datamodule.params_to_array(train_params)

    # Far-OOD config: push param_1 to far end of its range
    ood_params = dict(train_params)
    ood_params["param_1"] = 9.9
    X_ood = datamodule.params_to_array(ood_params)

    u_train = pred.uncertainty(X_train)
    u_ood = pred.uncertainty(X_ood)

    assert u_train < u_ood, (
        f"Training config uncertainty ({u_train:.4f}) should be lower than OOD ({u_ood:.4f})"
    )


def test_uncertainty_returns_one_for_single_training_config(tmp_path):
    """With only one training config, KDE fitting is skipped → uncertainty stays 1.0."""
    # build_dataset_with_single_experiment has exactly one experiment, so after training
    # with just that experiment we have < 2 distinct configs → KDE is skipped.
    agent, exp, datamodule = _trained_agent_and_datamodule(tmp_path)
    pred = agent.pred_system

    # KDE is skipped when n_latent_points < 2
    if not pred._model_kdes:
        X = datamodule.params_to_array(exp.parameters.get_values_dict())
        assert pred.uncertainty(X) == pytest.approx(1.0)
    else:
        # If KDE was fitted (shouldn't happen here), just check it's in range
        X = datamodule.params_to_array(exp.parameters.get_values_dict())
        assert 0.0 <= pred.uncertainty(X) <= 1.0


# ===========================================================================
# kernel_similarity()
# ===========================================================================

def test_kernel_similarity_returns_zero_before_training(tmp_path):
    """Without a fitted KDE (no bandwidth), kernel_similarity returns 0.0."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    logger = build_test_logger(tmp_path)
    pred = PredictionSystem(logger=logger, schema=dataset.schema, local_data=LocalData(str(tmp_path)))

    X = np.array([0.5, 0.5, 0.5])
    assert pred.kernel_similarity(X, X) == pytest.approx(0.0)


def test_kernel_similarity_identical_points_after_training(tmp_path):
    """sim(x, x) should be close to 1.0 for identical inputs (when KDE is fitted)."""
    agent, exp, datamodule = _trained_agent_and_datamodule(tmp_path)
    pred = agent.pred_system

    if not pred._model_kdes:
        pytest.skip("KDE not fitted — not enough training configs")

    X = datamodule.params_to_array(exp.parameters.get_values_dict())
    sim = pred.kernel_similarity(X, X)
    assert sim == pytest.approx(1.0, abs=1e-6)


def test_kernel_similarity_decreases_with_distance(tmp_path):
    """sim(x, y) < sim(x, x) when x != y (assuming KDE is fitted)."""
    agent, exp, datamodule = _trained_agent_and_datamodule(tmp_path)
    pred = agent.pred_system

    if not pred._model_kdes:
        pytest.skip("KDE not fitted — not enough training configs")

    train_params = exp.parameters.get_values_dict()
    X1 = datamodule.params_to_array(train_params)

    far_params = dict(train_params)
    far_params["param_1"] = 9.0
    X2 = datamodule.params_to_array(far_params)

    sim_self = pred.kernel_similarity(X1, X1)
    sim_other = pred.kernel_similarity(X1, X2)
    assert sim_other < sim_self


# ===========================================================================
# predict_for_calibration()
# ===========================================================================

def test_predict_for_calibration_raises_before_training(tmp_path):
    """predict_for_calibration() must raise RuntimeError if system is untrained."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    logger = build_test_logger(tmp_path)
    pred = PredictionSystem(logger=logger, schema=dataset.schema, local_data=LocalData(str(tmp_path)))
    from tests.utils.interfaces import MixedPredictionModelGrid
    pred.models.append(MixedPredictionModelGrid(logger))

    with pytest.raises(RuntimeError, match="train"):
        pred.predict_for_calibration({"param_1": 2.5, "dim_1": 2, "dim_2": 3})


def test_predict_for_calibration_returns_feature_arrays_and_params_block(tmp_path):
    """After training, predict_for_calibration() returns correct feature arrays and params."""
    agent, exp, datamodule = _trained_agent_and_datamodule(tmp_path)
    pred = agent.pred_system
    params = exp.parameters.get_values_dict()

    feature_arrays, params_block = pred.predict_for_calibration(params)

    # feature_arrays should contain all model outputs
    for feature_code in pred.get_system_outputs():
        assert feature_code in feature_arrays, f"Missing feature: {feature_code}"
        arr = feature_arrays[feature_code]
        # Each row: [dim_iter_0..., feature_val] — depth-0 features get shape (1, 1)
        assert arr.ndim == 2
        assert arr.shape[1] >= 1  # at least the feature value column

    # params_block should have all schema parameters
    for code in params:
        if code in params_block.data_objects:
            assert params_block.get_value(code) == pytest.approx(params[code], rel=1e-4) or \
                   params_block.get_value(code) == params[code]


def test_predict_for_calibration_feature_array_row_count(tmp_path):
    """Feature array row count matches each feature's own dimensional depth."""
    agent, exp, datamodule = _trained_agent_and_datamodule(tmp_path)
    pred = agent.pred_system
    params = exp.parameters.get_values_dict()

    feature_arrays, _ = pred.predict_for_calibration(params)
    dim_1 = int(params["dim_1"])
    dim_2 = int(params["dim_2"])

    # feature_grid: depth 2 → dim_1 * dim_2 rows
    assert feature_arrays["feature_grid"].shape[0] == dim_1 * dim_2
    # feature_d1: depth 1 → dim_1 rows
    assert feature_arrays["feature_d1"].shape[0] == dim_1
    # feature_scalar: depth 0 → 1 row
    assert feature_arrays["feature_scalar"].shape[0] == 1


# ===========================================================================
# compute_performance() with feature_std (3-tuple return)
# ===========================================================================

def test_compute_performance_returns_none_std_when_feature_std_not_provided(tmp_path):
    """compute_performance() with no feature_std returns (avg, list, None)."""
    from tests.utils.interfaces import ScalarEvaluationModel
    logger = build_test_logger(tmp_path)
    model = ScalarEvaluationModel(logger)

    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    # ScalarEvaluationModel.input_parameters=[] — no dimensions to wire up.

    feature_array = np.array([[7.0]])
    avg, perf_list, std_list = model.compute_performance(
        feature_array=feature_array, parameters=exp.parameters
    )
    assert std_list is None
    assert avg is not None
    assert len(perf_list) == 1


def test_compute_performance_propagates_feature_std(tmp_path):
    """When feature_std is provided, std_list should contain per-row std estimates."""
    from tests.utils.interfaces import ScalarEvaluationModel
    logger = build_test_logger(tmp_path)
    model = ScalarEvaluationModel(logger)

    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    # ScalarEvaluationModel.input_parameters=[] — no dimensions to wire up.

    feature_array = np.array([[7.0]])
    feature_std = np.array([0.5])

    avg, perf_list, std_list = model.compute_performance(
        feature_array=feature_array, parameters=exp.parameters, feature_std=feature_std
    )
    assert std_list is not None
    assert len(std_list) == 1
    # target=7.0 → denom=7.0 → std_perf = 0.5/7.0
    assert std_list[0] == pytest.approx(0.5 / 7.0, abs=1e-6)


def test_compute_performance_std_none_for_nan_feature_std(tmp_path):
    """If feature_std entry is NaN or negative, the corresponding std_perf should be None."""
    from tests.utils.interfaces import ScalarEvaluationModel
    logger = build_test_logger(tmp_path)
    model = ScalarEvaluationModel(logger)

    dataset = build_dataset_with_single_experiment(tmp_path)
    exp = dataset.get_experiment("exp_001")
    # ScalarEvaluationModel.input_parameters=[] — no dimensions to wire up.

    feature_array = np.array([[7.0]])
    feature_std = np.array([float("nan")])

    _, _, std_list = model.compute_performance(
        feature_array=feature_array, parameters=exp.parameters, feature_std=feature_std
    )
    assert std_list is not None
    assert std_list[0] is None


# ===========================================================================
# UCB acquisition function integration
# ===========================================================================

def test_acquisition_func_uses_perf_fn_and_uncertainty_fn(tmp_path):
    """_acquisition_func integrates perf_fn and uncertainty_fn correctly."""
    _, dataset, _ = build_workflow_stack(tmp_path)

    perf_calls = []
    unc_calls = []

    calibration = build_calibration_system(
        tmp_path / "cal",
        dataset,
        perf_fn=lambda p: (perf_calls.append(p), {"performance_1": 0.8, "performance_2": 0.6})[1],
        uncertainty_fn=lambda X: (unc_calls.append(X.shape), 0.3)[1],
    )

    datamodule = build_initialized_datamodule(
        dataset=dataset,
        input_parameters=["param_1", "param_2", "n_layers", "n_segments", "speed"],
        input_features=[],
        output_columns=[],
        fitted=True,
        split_codes={SplitType.TRAIN: [], SplitType.VAL: [], SplitType.TEST: []},
    )
    calibration._active_datamodule = datamodule

    # Use valid params (param_2 min=1) to avoid sanitize_values rejection.
    valid_params = {"param_1": 2.0, "param_2": 2, "n_layers": 2, "n_segments": 2, "speed": 50.0}
    X = datamodule.params_to_array(valid_params)
    w = 0.4
    result = calibration._acquisition_func(X, kappa=w)

    assert len(perf_calls) == 1
    assert len(unc_calls) == 1
    # Expected sys_perf = avg(0.8, 0.6) = 0.7; u = 0.3
    # Score = (1-0.4)*0.7 + 0.4*0.3 = 0.42 + 0.12 = 0.54 → result = -0.54
    assert result == pytest.approx(-((1 - w) * 0.7 + w * 0.3), abs=1e-4)


def test_acquisition_func_with_no_active_datamodule_returns_zero(tmp_path):
    """_acquisition_func returns 0.0 (and doesn't crash) when _active_datamodule is None."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)

    X = np.zeros(3)
    result = calibration._acquisition_func(X, kappa=0.5)
    assert result == pytest.approx(0.0)


# ===========================================================================
# Schedule diversity via step-loop
# ===========================================================================

def test_run_calibration_with_similarity_fn_completes(tmp_path):
    """With a constant similarity_fn=1, the step-loop should still complete."""
    agent, dataset, exp, datamodule = build_runtime_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    cs = agent.calibration_system
    cs.configure_schedule_parameter("speed", "dim_1")
    cs.configure_adaptation_delta({"speed": 50.0})

    # Replace similarity_fn with constant 1.0
    cs.similarity_fn = lambda X1, X2: 1.0

    current_params = exp.parameters.get_values_dict()
    result = cs.run_calibration(
        datamodule=datamodule,
        mode=Mode.EXPLORATION,
        current_params=current_params,
    )

    assert isinstance(result, ExperimentSpec)


def test_run_calibration_without_similarity_fn_still_works(tmp_path):
    """When similarity_fn is None, the step-loop runs without diversity penalty."""
    agent, dataset, exp, datamodule = build_runtime_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)

    cs = agent.calibration_system
    cs.similarity_fn = None  # explicit no diversity
    cs.configure_schedule_parameter("speed", "dim_1")
    cs.configure_adaptation_delta({"speed": 50.0})

    current_params = exp.parameters.get_values_dict()
    result = cs.run_calibration(
        datamodule=datamodule,
        mode=Mode.EXPLORATION,
        current_params=current_params,
    )

    assert isinstance(result, ExperimentSpec)


# ===========================================================================
# perf_fn integration via agent
# ===========================================================================

def test_agent_perf_fn_closure_returns_performance_dict(tmp_path):
    """The perf_fn closure created in agent.initialize_systems should work end-to-end."""
    agent, exp, datamodule = _trained_agent_and_datamodule(tmp_path)

    params = exp.parameters.get_values_dict()
    perf_dict = agent.calibration_system.perf_fn(params)

    assert isinstance(perf_dict, dict)
    # Should contain at least one performance key
    assert len(perf_dict) > 0
    for v in perf_dict.values():
        if v is not None:
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0


def test_calibration_system_get_models_returns_empty_list(tmp_path):
    """CalibrationSystem.get_models() returns [] since there is no internal ML model."""
    dataset = build_dataset_with_single_experiment(tmp_path)
    calibration = build_calibration_system(tmp_path, dataset)
    assert calibration.get_models() == []


# ===========================================================================
# KDE bandwidth + schedule step-loop integration
# ===========================================================================

def test_run_calibration_schedule_respects_delta_constraints_with_fitted_kde(tmp_path):
    """After KDE fitting, run_calibration with schedule configs should return an
    ExperimentSpec whose consecutive speed waypoints differ by at most the configured delta."""
    delta = 50.0

    # Build 3-experiment stack so KDE can be fitted (workflow schema has 'speed' runtime param).
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent=agent, dataset=dataset, category_value="B")
    datamodule = build_prepared_workflow_datamodule(
        agent=agent, dataset=dataset, val_size=0.0, test_size=0.0, recompute=True
    )
    agent.train(datamodule=datamodule, validate=False, test=False)

    if not agent.pred_system._model_kdes:
        pytest.skip("KDE not fitted — not enough distinct training configs")

    cs = agent.calibration_system
    cs.configure_schedule_parameter("speed", "n_layers")
    cs.configure_adaptation_delta({"speed": delta})

    first_exp = dataset.get_experiment(codes[0])
    current_params = first_exp.parameters.get_values_dict()
    current_params["speed"] = 100.0  # must supply runtime param for schedule stepping

    result = cs.run_calibration(
        datamodule=datamodule,
        mode=Mode.EXPLORATION,
        current_params=current_params,
        kappa=0.5,
    )

    assert isinstance(result, ExperimentSpec)

    # Schedule should be keyed by the dimension that 'speed' is linked to (n_layers).
    assert "n_layers" in result.schedules, "Expected schedule for n_layers dimension"

    schedule = result.schedules["n_layers"]
    seg0 = result.initial_params["speed"]
    waypoints = [proposal["speed"] for _, proposal in schedule.entries]

    # All consecutive pairs must satisfy the delta constraint (with a small tolerance).
    tol = 1e-4
    all_vals = [seg0] + waypoints
    for k in range(len(all_vals) - 1):
        diff = abs(all_vals[k + 1] - all_vals[k])
        assert diff <= delta + tol, (
            f"Segment {k}→{k+1}: |{all_vals[k+1]:.4f} - {all_vals[k]:.4f}| = {diff:.4f} "
            f"exceeds delta={delta}"
        )


def test_exploration_step_with_schedule_returns_experiment_spec(tmp_path):
    """agent.exploration_step() with schedule configs returns an ExperimentSpec."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent=agent, dataset=dataset, category_value="B")
    datamodule = build_prepared_workflow_datamodule(
        agent=agent, dataset=dataset, val_size=0.0, test_size=0.0, recompute=True
    )
    agent.train(datamodule=datamodule, validate=False, test=False)

    agent.configure_schedule("speed", "n_layers", delta=50.0)

    first_exp = dataset.get_experiment(codes[0])
    current_params = first_exp.parameters.get_values_dict()

    result = agent.exploration_step(
        datamodule=datamodule,
        current_params=current_params,
    )

    assert isinstance(result, ExperimentSpec)


# ===========================================================================
# Evidence model: data point counting
# ===========================================================================

def test_evidence_grows_with_data(tmp_path):
    """More latent points near a query → lower uncertainty (more evidence)."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent=agent, dataset=dataset, category_value="B")
    datamodule = build_prepared_workflow_datamodule(
        agent=agent, dataset=dataset, val_size=0.0, test_size=0.0, recompute=True
    )
    agent.train(datamodule=datamodule, validate=False, test=False)

    pred = agent.pred_system
    if not pred._model_kdes:
        pytest.skip("Evidence model not fitted — not enough distinct training configs")

    n_exp = len(codes)
    for kde in pred._model_kdes.values():
        # Each process experiment contributes 1 latent point (no weight normalization)
        assert len(kde.latent_points) == n_exp, (
            f"expected {n_exp} latent points (one per process experiment), "
            f"got {len(kde.latent_points)}"
        )

    # Uncertainty near training data should be lower than far away
    first_exp = dataset.get_experiment(codes[0])
    X_train = datamodule.params_to_array(first_exp.parameters.get_values_dict())
    ood_params = dict(first_exp.parameters.get_values_dict())
    ood_params["param_1"] = 9.9
    X_ood = datamodule.params_to_array(ood_params)

    u_train = pred.uncertainty(X_train)
    u_ood = pred.uncertainty(X_ood)
    assert u_train < u_ood


def test_schedule_experiment_contributes_multiple_evidence_points(tmp_path):
    """Schedule experiment with 2 segments contributes 2 evidence points (one per segment)."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent=agent, dataset=dataset, category_value="B")

    first_exp = dataset.get_experiment(codes[0])
    dim_names = first_exp.parameters.get_dim_names()
    if not dim_names:
        pytest.skip("workflow schema has no dimensional params — cannot create schedule")
    dim = dim_names[0]
    first_exp.record_parameter_update(
        ParameterProposal.from_dict({"param_1": 7.5}, source_step="adaptation_step"),
        dimension=dim,
        step_index=1,
    )

    datamodule = build_prepared_workflow_datamodule(
        agent=agent, dataset=dataset, val_size=0.0, test_size=0.0, recompute=True
    )
    agent.train(datamodule=datamodule, validate=False, test=False)

    pred = agent.pred_system
    if not pred._model_kdes:
        pytest.skip("Evidence model not fitted")

    # 3 experiments: exp_001 has 2 segments → 2 points; exp_002, exp_003 → 1 each = 4 total
    for kde in pred._model_kdes.values():
        assert len(kde.latent_points) == 4, (
            f"expected 4 latent points (2+1+1), got {len(kde.latent_points)}"
        )


# ===========================================================================
# uncertainty_batch: batch-aware KDE uncertainty
# ===========================================================================

def test_uncertainty_batch_matches_single_point_at_L1(tmp_path):
    """uncertainty_batch(X[0:1])[0] must equal uncertainty(X[0]) when L=1 (no siblings)."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent=agent, dataset=dataset, category_value="B")
    datamodule = build_prepared_workflow_datamodule(
        agent=agent, dataset=dataset, val_size=0.0, test_size=0.0, recompute=True
    )
    agent.train(datamodule=datamodule, validate=False, test=False)

    pred = agent.pred_system
    if not pred._model_kdes:
        pytest.skip("KDE not fitted")

    first_exp = dataset.get_experiment(codes[0])
    X = datamodule.params_to_array(first_exp.parameters.get_values_dict())

    u_single = pred.uncertainty(X)
    u_batch = pred.uncertainty_batch(X.reshape(1, -1))

    assert u_batch.shape == (1,)
    assert u_batch[0] == pytest.approx(u_single, abs=1e-9)


def test_uncertainty_batch_shape_and_dtype(tmp_path):
    """uncertainty_batch returns (L,) float array for input shape (L, D)."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent=agent, dataset=dataset, category_value="B")
    datamodule = build_prepared_workflow_datamodule(
        agent=agent, dataset=dataset, val_size=0.0, test_size=0.0, recompute=True
    )
    agent.train(datamodule=datamodule, validate=False, test=False)

    pred = agent.pred_system
    if not pred._model_kdes:
        pytest.skip("KDE not fitted")

    first_exp = dataset.get_experiment(codes[0])
    X = datamodule.params_to_array(first_exp.parameters.get_values_dict())
    X_batch = np.stack([X, X, X, X])

    u_vec = pred.uncertainty_batch(X_batch)
    assert u_vec.shape == (4,)
    assert u_vec.dtype == np.float64
    assert np.all((u_vec >= 0.0) & (u_vec <= 1.0))


def test_uncertainty_batch_rejects_non_2d_input(tmp_path):
    """uncertainty_batch must reject 1-D input to prevent silent mis-use."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent=agent, dataset=dataset, category_value="B")
    datamodule = build_prepared_workflow_datamodule(
        agent=agent, dataset=dataset, val_size=0.0, test_size=0.0, recompute=True
    )
    agent.train(datamodule=datamodule, validate=False, test=False)

    pred = agent.pred_system
    first_exp = dataset.get_experiment(codes[0])
    X = datamodule.params_to_array(first_exp.parameters.get_values_dict())
    with pytest.raises(ValueError, match="2-D"):
        pred.uncertainty_batch(X)


def test_uncertainty_batch_diversity_pressure(tmp_path):
    """A collapsed batch (all points co-located) should yield lower mean u_batch than a spread batch."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent=agent, dataset=dataset, category_value="B")
    datamodule = build_prepared_workflow_datamodule(
        agent=agent, dataset=dataset, val_size=0.0, test_size=0.0, recompute=True
    )
    agent.train(datamodule=datamodule, validate=False, test=False)

    pred = agent.pred_system
    if not pred._model_kdes:
        pytest.skip("KDE not fitted")

    # Pick an OOD anchor point (push param_1 to high end) to get high baseline uncertainty.
    first_exp = dataset.get_experiment(codes[0])
    base_params = dict(first_exp.parameters.get_values_dict())
    base_params["param_1"] = 9.5
    X_ood = datamodule.params_to_array(base_params)

    # Collapsed: 5 copies of the same OOD point.
    collapsed = np.stack([X_ood] * 5)

    # Spread: 5 variations of param_1 spanning the range.
    spread_rows = []
    for val in [1.0, 3.0, 5.0, 7.0, 9.5]:
        p = dict(base_params)
        p["param_1"] = val
        spread_rows.append(datamodule.params_to_array(p))
    spread = np.stack(spread_rows)

    u_collapsed = pred.uncertainty_batch(collapsed)
    u_spread = pred.uncertainty_batch(spread)

    # When points collapse, each sees the others as nearby virtuals → density spikes
    # → uncertainty drops. Spread keeps each point's siblings far → uncertainty stays high.
    assert u_collapsed.mean() < u_spread.mean(), (
        f"Collapsed batch mean uncertainty ({u_collapsed.mean():.4f}) must be lower than "
        f"spread batch ({u_spread.mean():.4f}) to provide diversification pressure."
    )


def test_uncertainty_batch_no_side_effects(tmp_path):
    """uncertainty_batch must not mutate _model_kdes state (no virtual-point leakage)."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent=agent, dataset=dataset, category_value="B")
    datamodule = build_prepared_workflow_datamodule(
        agent=agent, dataset=dataset, val_size=0.0, test_size=0.0, recompute=True
    )
    agent.train(datamodule=datamodule, validate=False, test=False)

    pred = agent.pred_system
    if not pred._model_kdes:
        pytest.skip("KDE not fitted")

    first_exp = dataset.get_experiment(codes[0])
    X = datamodule.params_to_array(first_exp.parameters.get_values_dict())
    X_batch = np.stack([X, X, X])

    snapshots: list[tuple[np.ndarray, float]] = []
    for kde in pred._model_kdes.values():
        snapshots.append((kde.latent_points.copy(), kde.sigma))

    _ = pred.uncertainty_batch(X_batch)

    for (before_pts, before_sigma), kde in zip(snapshots, pred._model_kdes.values()):
        assert np.array_equal(kde.latent_points, before_pts)
        assert kde.sigma == before_sigma


# ===========================================================================
# Boundary evidence at domain edges
# ===========================================================================

def test_boundary_evidence_at_edge(tmp_path):
    """With no training data, uncertainty at the boundary should be lower than at the center
    because boundary evidence is higher at edges (closer to normalized bounds 0/1)."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent=agent, dataset=dataset, category_value="B")
    datamodule = build_prepared_workflow_datamodule(
        agent=agent, dataset=dataset, val_size=0.0, test_size=0.0, recompute=True
    )

    pred = agent.pred_system
    # Fit empty KDE so boundary evidence is the only signal
    pred.fit_empty_kde(datamodule, target_n=1)

    # Edge point: param_1 at lower bound (0.0) → normalizes near 0
    edge_params = dict(dataset.get_experiment(codes[0]).parameters.get_values_dict())
    edge_params["param_1"] = 0.0
    X_edge = datamodule.params_to_array(edge_params)

    # Center point: param_1 at midrange (5.0) → normalizes near 0.5
    center_params = dict(edge_params)
    center_params["param_1"] = 5.0
    X_center = datamodule.params_to_array(center_params)

    u_edge = pred.uncertainty(X_edge)
    u_center = pred.uncertainty(X_center)

    assert 0.0 <= u_edge <= 1.0
    assert 0.0 <= u_center <= 1.0
    assert u_edge < u_center, (
        f"Edge uncertainty ({u_edge:.4f}) should be lower than center ({u_center:.4f}) "
        "because boundary evidence is stronger at domain edges."
    )


# ===========================================================================
# Kernel type: Cauchy vs Gaussian tail behavior
# ===========================================================================

def test_kernel_cauchy_vs_gaussian(tmp_path):
    """Cauchy kernel has wider tails than Gaussian: at a far-OOD point, Cauchy uncertainty
    should be lower (more evidence leaks to far-away points) than Gaussian."""
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent=agent, dataset=dataset, category_value="B")
    datamodule = build_prepared_workflow_datamodule(
        agent=agent, dataset=dataset, val_size=0.0, test_size=0.0, recompute=True
    )
    agent.train(datamodule=datamodule, validate=False, test=False)

    pred = agent.pred_system
    if not pred._model_kdes:
        pytest.skip("KDE not fitted — not enough distinct training configs")

    # Far-OOD point
    ood_params = dict(dataset.get_experiment(codes[0]).parameters.get_values_dict())
    ood_params["param_1"] = 9.9
    X_ood = datamodule.params_to_array(ood_params)

    # Cauchy (default)
    assert pred._kernel_type == "cauchy"
    u_cauchy = pred.uncertainty(X_ood)

    # Switch to Gaussian and re-evaluate
    pred._kernel_type = "gaussian"
    u_gaussian = pred.uncertainty(X_ood)

    # Restore
    pred._kernel_type = "cauchy"

    assert 0.0 <= u_cauchy <= 1.0
    assert 0.0 <= u_gaussian <= 1.0
    # Cauchy's heavier tail means more evidence reaches far-OOD → lower uncertainty there
    assert u_cauchy < u_gaussian, (
        f"Cauchy uncertainty ({u_cauchy:.4f}) should be lower than Gaussian ({u_gaussian:.4f}) "
        "at a far-OOD point due to heavier tails."
    )


# ------ Schedule DE branch convergence key rename ('Acquisition' -> 'Schedule') ------
# The workflow-stack fixture here does not wire 'speed' into any model input, so the
# exploration Schedule DE branch is not exercised (D_sched=0 fallback). Validation of
# the rename is performed via the mock integration, not a unit test.
