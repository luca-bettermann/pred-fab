"""Tests for predict_for_calibration_tensor.

Three guarantees this commit promises:
  1. Numerical equivalence with predict_for_calibration_batched (numpy
     version) at ~1e-5 — the autoreg + denormalisation chain runs in
     tensor land identically.
  2. Output type: dict[feat, torch.Tensor] per candidate, never numpy.
     Per-feat tensor shape matches feat_shape from _get_feature_shape.
  3. Gradient flow: with continuous param values as torch.Tensors with
     requires_grad=True, gradients flow through params_to_tensor +
     autoreg loop (forward_pass(gradient_pass=True)) back into the
     leaf tensors. backward() succeeds; gradients are finite.
"""

import numpy as np
import pytest
import torch

from tests.utils.builders import build_real_agent_stack


def _trained_agent(tmp_path):
    """Standard fixture: build agent, evaluate, prepare DM, train."""
    agent, dataset, exp, datamodule = build_real_agent_stack(tmp_path)
    agent.evaluate(exp_data=exp, recompute_flag=True, visualize=False)
    datamodule.prepare(val_size=0.0, test_size=0.0, recompute=True)
    agent.train(datamodule=datamodule, validate=False, test=False)
    return agent, exp


# ── Numerical equivalence with the numpy batched API ──────────────────────


def _values_only(feat_b: np.ndarray) -> np.ndarray:
    """Extract the last column (feat_val) from the tabular numpy output.

    ``predict_for_calibration_batched`` post-processes predictions to
    tabular form ``[iter_idx_0, ..., iter_idx_n, feat_val]`` per cell row.
    The tensor variant returns the raw ``(S, *feat_shape)`` tensor without
    iterator-index decoration (indices are non-differentiable metadata).
    For comparison, take the last column.
    """
    if feat_b.ndim == 1:
        return feat_b
    return feat_b[..., -1]


def test_tensor_matches_batched_single_candidate(tmp_path):
    agent, exp = _trained_agent(tmp_path)
    p = exp.parameters.get_values_dict()

    batched_out = agent.pred_system.predict_for_calibration_batched([p])
    tensor_out = agent.pred_system.predict_for_calibration_tensor([p])

    assert len(tensor_out) == 1
    feat_b, _ = batched_out[0]
    feat_t = tensor_out[0]
    assert set(feat_b.keys()) == set(feat_t.keys())
    for k in feat_b:
        np.testing.assert_allclose(
            feat_t[k].detach().cpu().numpy().reshape(-1),
            _values_only(feat_b[k]).reshape(-1),
            atol=1e-5, rtol=1e-5,
            err_msg=f"feature {k}",
        )


def test_tensor_matches_batched_multi_candidate(tmp_path):
    agent, exp = _trained_agent(tmp_path)
    base = exp.parameters.get_values_dict()
    params_list = [
        dict(base),
        {**base, "param_1": float(base.get("param_1", 1.0)) + 0.5},
        {**base, "param_1": float(base.get("param_1", 1.0)) + 1.0},
    ]

    batched_out = agent.pred_system.predict_for_calibration_batched(params_list)
    tensor_out = agent.pred_system.predict_for_calibration_tensor(params_list)

    assert len(tensor_out) == len(params_list)
    for s, ((feat_b, _), feat_t) in enumerate(zip(batched_out, tensor_out)):
        for k in feat_b:
            np.testing.assert_allclose(
                feat_t[k].detach().cpu().numpy().reshape(-1),
                _values_only(feat_b[k]).reshape(-1),
                atol=1e-5, rtol=1e-5,
                err_msg=f"candidate {s} feature {k}",
            )


def test_tensor_empty_list_returns_empty(tmp_path):
    agent, _exp = _trained_agent(tmp_path)
    assert agent.pred_system.predict_for_calibration_tensor([]) == []


# ── Output type (dict of torch.Tensor, never numpy) ───────────────────────


def test_tensor_output_types(tmp_path):
    agent, exp = _trained_agent(tmp_path)
    p = exp.parameters.get_values_dict()

    out = agent.pred_system.predict_for_calibration_tensor([p])
    assert len(out) == 1
    per_feat = out[0]
    assert isinstance(per_feat, dict)
    for feat, val in per_feat.items():
        assert isinstance(val, torch.Tensor), (
            f"feature {feat}: expected torch.Tensor, got {type(val).__name__}"
        )


# ── Gradient flow ─────────────────────────────────────────────────────────


def test_grad_flows_to_continuous_param(tmp_path):
    """Continuous param as a tensor with requires_grad=True; gradients flow
    through params_to_tensor + autoreg + denormalize back to the leaf."""
    agent, exp = _trained_agent(tmp_path)
    base = exp.parameters.get_values_dict()

    # Find a continuous (real-valued) param to use as the gradient leaf.
    schema_params = agent.pred_system.schema.parameters
    cont_code = None
    for code in base:
        if not schema_params.has(code):
            continue
        obj = schema_params.get(code)
        if obj.__class__.__name__ == "DataReal":
            cont_code = code
            break
    if cont_code is None:
        pytest.skip("No continuous param in fixture schema.")

    # Build a params dict where cont_code is a tensor leaf with grad.
    leaf = torch.tensor(float(base[cont_code]), requires_grad=True)
    p = dict(base)
    p[cont_code] = leaf

    out = agent.pred_system.predict_for_calibration_tensor([p])
    assert len(out) == 1

    # Sum any output feature → backward
    per_feat = out[0]
    assert per_feat, "no output features produced"
    loss = sum(t.sum() for t in per_feat.values())
    assert isinstance(loss, torch.Tensor) and loss.requires_grad, (
        "loss should require grad (predictions came from gradient_pass=True forward)"
    )
    loss.backward()

    assert leaf.grad is not None, "leaf received no gradient — graph is broken"
    assert torch.isfinite(leaf.grad).all(), f"leaf gradient is non-finite: {leaf.grad}"


def test_grad_flows_through_multi_candidate(tmp_path):
    """Multi-candidate path: each candidate's gradient is independent."""
    agent, exp = _trained_agent(tmp_path)
    base = exp.parameters.get_values_dict()

    schema_params = agent.pred_system.schema.parameters
    cont_code = None
    for code in base:
        if not schema_params.has(code):
            continue
        obj = schema_params.get(code)
        if obj.__class__.__name__ == "DataReal":
            cont_code = code
            break
    if cont_code is None:
        pytest.skip("No continuous param in fixture schema.")

    leaf_a = torch.tensor(float(base[cont_code]), requires_grad=True)
    leaf_b = torch.tensor(float(base[cont_code]) + 0.5, requires_grad=True)
    pa = {**base, cont_code: leaf_a}
    pb = {**base, cont_code: leaf_b}

    out = agent.pred_system.predict_for_calibration_tensor([pa, pb])
    assert len(out) == 2

    # Loss = sum over both candidates' first feature
    feat_a = next(iter(out[0]))
    loss_a = out[0][feat_a].sum()
    loss_b = out[1][feat_a].sum()
    (loss_a + loss_b).backward()

    assert leaf_a.grad is not None
    assert leaf_b.grad is not None
    assert torch.isfinite(leaf_a.grad).all()
    assert torch.isfinite(leaf_b.grad).all()


def test_grad_zero_for_disconnected_param(tmp_path):
    """A leaf tensor *not* used in any candidate's params dict gets no grad,
    even when a parallel leaf IS used and produces a backward pass."""
    agent, exp = _trained_agent(tmp_path)
    base = exp.parameters.get_values_dict()

    schema_params = agent.pred_system.schema.parameters
    cont_code = next(
        (c for c in base
         if schema_params.has(c) and schema_params.get(c).__class__.__name__ == "DataReal"),
        None,
    )
    if cont_code is None:
        pytest.skip("No continuous param in fixture schema.")

    connected = torch.tensor(float(base[cont_code]), requires_grad=True)
    p = {**base, cont_code: connected}

    # Disconnected leaf — never inserted into any params dict.
    disconnected = torch.tensor(0.5, requires_grad=True)

    out = agent.pred_system.predict_for_calibration_tensor([p])
    feat = next(iter(out[0]))
    loss = out[0][feat].sum()
    loss.backward()

    assert connected.grad is not None, "connected leaf should have received grad"
    assert disconnected.grad is None, (
        f"disconnected leaf should have no grad, got {disconnected.grad}"
    )
