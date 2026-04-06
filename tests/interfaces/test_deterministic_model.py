"""
Tests for IDeterministicModel contract:
  - train() is a no-op
  - encode() returns identity
  - forward_pass() is final: denormalize → formula → renormalize
  - set_normalization_context() stores stats
  - formula() receives raw (denormalized) values
"""
import pytest
import numpy as np

from pred_fab.interfaces import IDeterministicModel
from tests.utils.builders import build_test_logger
from tests.utils.interfaces import ContractDeterministicModel


# ===========================================================================
# Basic contract
# ===========================================================================

class TestDeterministicModelContract:
    """Core IDeterministicModel interface guarantees."""

    def test_train_is_noop(self, tmp_path):
        model = ContractDeterministicModel(build_test_logger(tmp_path))
        # train() should not raise and should do nothing
        model.train(train_batches=[], val_batches=[])

    def test_encode_returns_identity(self, tmp_path):
        model = ContractDeterministicModel(build_test_logger(tmp_path))
        X = np.array([[1.0, 2.0, 3.0]])
        result = model.encode(X)
        np.testing.assert_array_equal(result, X)

    def test_forward_pass_raises_without_normalization_context(self, tmp_path):
        model = ContractDeterministicModel(build_test_logger(tmp_path))
        X = np.array([[0.5]])
        with pytest.raises(RuntimeError, match="set_normalization_context"):
            model.forward_pass(X)


# ===========================================================================
# Normalization context
# ===========================================================================

class TestNormalizationContext:
    """set_normalization_context() provides denorm/renorm capability."""

    def test_set_normalization_context_marks_ready(self, tmp_path):
        model = ContractDeterministicModel(build_test_logger(tmp_path))
        assert not model._norm_context_set
        model.set_normalization_context(
            parameter_stats={"param_1": {"mean": 5.0, "std": 2.0, "method": "standard"}},
            feature_stats={"feature_scalar": {"mean": 10.0, "std": 4.0, "method": "standard"}},
            categorical_mappings={},
        )
        assert model._norm_context_set

    def test_categorical_mappings_stored(self, tmp_path):
        model = ContractDeterministicModel(build_test_logger(tmp_path))
        model.set_normalization_context(
            parameter_stats={},
            feature_stats={},
            categorical_mappings={"material": ["clay", "concrete"]},
        )
        assert model.categorical_mappings == {"material": ["clay", "concrete"]}


# ===========================================================================
# forward_pass with normalization
# ===========================================================================

class TestForwardPass:
    """forward_pass: denorm inputs → formula → renorm outputs."""

    def test_forward_pass_produces_correct_shape(self, tmp_path):
        model = ContractDeterministicModel(build_test_logger(tmp_path))
        model.set_normalization_context(
            parameter_stats={"param_1": {"mean": 5.0, "std": 2.0, "method": "standard"}},
            feature_stats={"feature_scalar": {"mean": 10.0, "std": 4.0, "method": "standard"}},
            categorical_mappings={},
        )
        # Normalized input: (raw - mean) / std = (5.0 - 5.0) / 2.0 = 0.0
        X_norm = np.array([[0.0]])
        result = model.forward_pass(X_norm)
        assert result.shape == (1, 1)

    def test_forward_pass_denorm_formula_renorm(self, tmp_path):
        """Verify the full pipeline: denorm → formula(raw) → renorm."""
        model = ContractDeterministicModel(build_test_logger(tmp_path))
        model.set_normalization_context(
            parameter_stats={"param_1": {"mean": 5.0, "std": 2.0, "method": "standard"}},
            feature_stats={"feature_scalar": {"mean": 10.0, "std": 4.0, "method": "standard"}},
            categorical_mappings={},
        )
        # Normalized input: (raw - mean) / std
        # raw_param = 5.0 + 0.0 * 2.0 = 5.0
        # formula(5.0) = 5.0 * 2 = 10.0
        # renorm: (10.0 - 10.0) / 4.0 = 0.0
        X_norm = np.array([[0.0]])
        result = model.forward_pass(X_norm)
        assert abs(float(result[0, 0]) - 0.0) < 1e-6

        # raw_param = 5.0 + 1.0 * 2.0 = 7.0
        # formula(7.0) = 7.0 * 2 = 14.0
        # renorm: (14.0 - 10.0) / 4.0 = 1.0
        X_norm = np.array([[1.0]])
        result = model.forward_pass(X_norm)
        assert abs(float(result[0, 0]) - 1.0) < 1e-6

    def test_forward_pass_batch(self, tmp_path):
        """Batch input should produce batch output."""
        model = ContractDeterministicModel(build_test_logger(tmp_path))
        model.set_normalization_context(
            parameter_stats={"param_1": {"mean": 5.0, "std": 2.0, "method": "standard"}},
            feature_stats={"feature_scalar": {"mean": 10.0, "std": 4.0, "method": "standard"}},
            categorical_mappings={},
        )
        X_norm = np.array([[0.0], [1.0], [-1.0]])
        result = model.forward_pass(X_norm)
        assert result.shape == (3, 1)
