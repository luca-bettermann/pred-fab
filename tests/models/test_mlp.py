"""MLPModel contract tests.

The base provides train / forward_pass / encode on top of IPredictionModel.
Subclasses define HIDDEN + the three abstract properties.
"""

import numpy as np
import pytest
import torch

from pred_fab.models import MLPModel
from tests.utils.builders import build_test_logger


# ─── Concrete subclasses for tests ────────────────────────────────────────

class _SingleOutMLP(MLPModel):
    HIDDEN = (16, 8)
    EPOCHS = 300

    @property
    def input_parameters(self):
        return ["p1", "p2"]

    @property
    def input_features(self):
        return []

    @property
    def outputs(self):
        return ["y"]


class _MultiOutMLP(MLPModel):
    HIDDEN = (12, 6)
    EPOCHS = 300

    @property
    def input_parameters(self):
        return ["p1"]

    @property
    def input_features(self):
        return []

    @property
    def outputs(self):
        return ["y1", "y2", "y3"]


def _make_xy(n: int, n_inputs: int, n_outputs: int = 1, seed: int = 0):
    """Synthetic (X, y) tensors. y depends on X[:, 0]² - X[:, -1] + small noise for multi-output."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_inputs)).astype(np.float32)
    y = (X[:, :1] ** 2 - X[:, -1:]).astype(np.float32)
    if n_outputs > 1:
        y = np.repeat(y, n_outputs, axis=1) + rng.standard_normal((n, n_outputs)).astype(np.float32) * 0.01
    return torch.from_numpy(X), torch.from_numpy(y)


# ─── Contract tests ───────────────────────────────────────────────────────

class TestUntrainedBehaviour:
    """Untrained model returns zeros from forward_pass and identity from encode."""

    def test_forward_pass_returns_zeros(self, tmp_path):
        m = _SingleOutMLP(build_test_logger(tmp_path))
        X = torch.from_numpy(np.random.randn(7, 2).astype(np.float32))
        out = m.forward_pass(X)
        assert out.shape == (7, 1)
        torch.testing.assert_close(out, torch.zeros((7, 1)))

    def test_forward_pass_zeros_shape_matches_n_outputs(self, tmp_path):
        m = _MultiOutMLP(build_test_logger(tmp_path))
        X = torch.from_numpy(np.random.randn(4, 1).astype(np.float32))
        out = m.forward_pass(X)
        assert out.shape == (4, 3)

    def test_encode_returns_identity_when_untrained(self, tmp_path):
        m = _SingleOutMLP(build_test_logger(tmp_path))
        X = torch.from_numpy(np.random.randn(3, 2).astype(np.float32))
        out = m.encode(X)
        torch.testing.assert_close(out, X)


class TestTraining:
    """train() builds the network and converges on simple data."""

    def test_empty_batches_is_noop(self, tmp_path):
        m = _SingleOutMLP(build_test_logger(tmp_path))
        m.train(train_batches=[], val_batches=[])
        assert not m._is_trained

    def test_train_marks_is_trained(self, tmp_path):
        m = _SingleOutMLP(build_test_logger(tmp_path))
        X, y = _make_xy(50, 2)
        m.train([(X, y)], [])
        assert m._is_trained
        assert m._model is not None

    def test_training_reduces_residual(self, tmp_path):
        m = _SingleOutMLP(build_test_logger(tmp_path))
        X, y = _make_xy(200, 2)
        m.train([(X, y)], [])
        pred = m.forward_pass(X)
        residual_std = float((pred.flatten() - y.flatten()).std())
        target_std = float(y.std())
        # Trained MLP should explain meaningful variance vs predict-mean baseline
        assert residual_std < 0.5 * target_std


class TestForwardPassShape:
    """forward_pass returns (batch, n_outputs) for trained models."""

    def test_single_output_shape(self, tmp_path):
        m = _SingleOutMLP(build_test_logger(tmp_path))
        X, y = _make_xy(40, 2)
        m.train([(X, y)], [])
        assert m.forward_pass(X[:5]).shape == (5, 1)

    def test_multi_output_shape(self, tmp_path):
        m = _MultiOutMLP(build_test_logger(tmp_path))
        X, y = _make_xy(40, 1, n_outputs=3)
        m.train([(X, y)], [])
        assert m.forward_pass(X[:6]).shape == (6, 3)


class TestEncodeShape:
    """encode() returns penultimate-layer activations of width = HIDDEN[-1]."""

    def test_encode_width_matches_last_hidden(self, tmp_path):
        m = _SingleOutMLP(build_test_logger(tmp_path))
        X, y = _make_xy(40, 2)
        m.train([(X, y)], [])
        z = m.encode(X[:3])
        assert z.shape == (3, m.HIDDEN[-1])  # (3, 8)

    def test_encode_width_with_different_hidden(self, tmp_path):
        class _DeepMLP(_SingleOutMLP):
            HIDDEN = (10, 5, 3)
            EPOCHS = 100

        m = _DeepMLP(build_test_logger(tmp_path))
        X, y = _make_xy(40, 2)
        m.train([(X, y)], [])
        z = m.encode(X[:7])
        assert z.shape == (7, 3)  # last hidden dim


class TestReproducibility:
    """Same SEED → same outputs after training."""

    def test_same_seed_same_predictions(self, tmp_path):
        X, y = _make_xy(60, 2)

        m1 = _SingleOutMLP(build_test_logger(tmp_path))
        m1.train([(X, y)], [])
        out1 = m1.forward_pass(X[:5])

        m2 = _SingleOutMLP(build_test_logger(tmp_path))
        m2.train([(X, y)], [])
        out2 = m2.forward_pass(X[:5])

        torch.testing.assert_close(out1, out2, atol=1e-6, rtol=1e-6)


class TestBatchTraining:
    """Multiple train batches are concatenated into one training set."""

    def test_two_batches_train_equivalently_to_one(self, tmp_path):
        X, y = _make_xy(40, 2)
        X1, y1 = X[:20], y[:20]
        X2, y2 = X[20:], y[20:]

        m1 = _SingleOutMLP(build_test_logger(tmp_path))
        m1.train([(X, y)], [])

        m2 = _SingleOutMLP(build_test_logger(tmp_path))
        m2.train([(X1, y1), (X2, y2)], [])

        # Same SEED + same final concatenated data → same network
        torch.testing.assert_close(m1.forward_pass(X[:5]), m2.forward_pass(X[:5]), atol=1e-6, rtol=1e-6)
