"""Equivalence tests for predict_for_calibration_batched vs scalar.

The batched autoreg path runs S candidates' trajectories in parallel via one
forward_pass(S, n_cols) per cell-step instead of S × forward_pass(1, n_cols).
The result must match the scalar path within floating-point summation noise
(matmul order can vary across batch dims).
"""

import numpy as np
import pytest

from tests.utils.builders import build_real_agent_stack


def _trained_agent(tmp_path):
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)
    return agent, exp


def test_batched_matches_scalar_single_candidate(tmp_path):
    agent, exp = _trained_agent(tmp_path)
    p = exp.parameters.get_values_dict()

    scalar = agent.pred_system.predict_for_calibration(p)
    batched = agent.pred_system.predict_for_calibration_batched([p])

    assert len(batched) == 1
    feat_s, _ = scalar
    feat_b, _ = batched[0]
    assert set(feat_s.keys()) == set(feat_b.keys())
    for k in feat_s:
        np.testing.assert_allclose(feat_s[k], feat_b[k], atol=1e-5, rtol=1e-5)


def test_batched_matches_scalar_multi_candidate(tmp_path):
    agent, exp = _trained_agent(tmp_path)
    base = exp.parameters.get_values_dict()
    params_list = [
        dict(base),
        {**base, "param_1": float(base.get("param_1", 1.0)) + 0.5},
        {**base, "param_1": float(base.get("param_1", 1.0)) + 1.0},
    ]

    scalars = [agent.pred_system.predict_for_calibration(p) for p in params_list]
    batched = agent.pred_system.predict_for_calibration_batched(params_list)

    assert len(batched) == len(params_list)
    for s_idx, ((feat_s, _), (feat_b, _)) in enumerate(zip(scalars, batched)):
        for k in feat_s:
            np.testing.assert_allclose(
                feat_s[k], feat_b[k], atol=1e-5, rtol=1e-5,
                err_msg=f"candidate {s_idx} feature {k}",
            )


def test_batched_empty_list_returns_empty(tmp_path):
    agent, _exp = _trained_agent(tmp_path)
    assert agent.pred_system.predict_for_calibration_batched([]) == []
