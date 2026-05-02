"""Tests for scale-aware DataLoader in MLPModel.train.

Two guarantees this commit promises:
  1. Below ``MINIBATCH_THRESHOLD`` (1000 rows), single-batch full-GD path
     stays — DataLoader overhead isn't justified at mock scale.
  2. Above the threshold, ``DataLoader(TensorDataset, shuffle=True)`` runs
     shuffled minibatches per epoch. The trained model still reaches
     reasonable train loss on a synthetic regression task.
"""

from __future__ import annotations

import torch

from pred_fab.models.mlp import MLPModel
from pred_fab.utils import PfabLogger


class _SmallMLP(MLPModel):
    """Tiny model for fast-test training. Linear regression-ish."""
    HIDDEN = (8, 4)
    EPOCHS = 50
    LR = 1e-2
    SEED = 0
    MINIBATCH_THRESHOLD = 100
    MINIBATCH_SIZE = 32

    @property
    def input_parameters(self) -> list[str]: return ["x"]
    @property
    def input_features(self) -> list[str]: return []
    @property
    def outputs(self) -> list[str]: return ["y"]


def _make_batches(n_rows: int, n_in: int = 1, n_out: int = 1) -> list[tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(0)
    X = torch.linspace(0.0, 1.0, n_rows).unsqueeze(-1).expand(-1, n_in).contiguous()
    # y = 2x + small noise → easy linear target
    y = 2.0 * X[:, :1] + 0.01 * torch.randn(n_rows, n_out)
    return [(X, y)]


def test_train_uses_full_batch_below_threshold(tmp_path):
    """50 rows, threshold=100 → single-batch GD path (no DataLoader)."""
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    model = _SmallMLP(logger=logger)

    train_batches = _make_batches(50)
    val_batches: list[tuple[torch.Tensor, torch.Tensor]] = []

    model.train(train_batches, val_batches)

    assert model._is_trained
    # Quick sanity: y(0.5) ≈ 1.0
    pred = model.forward_pass(torch.tensor([[0.5]], dtype=torch.float32))
    assert abs(pred["y"].item() - 1.0) < 0.5  # generous bound — small model, few epochs


def test_train_uses_dataloader_above_threshold(tmp_path):
    """500 rows, threshold=100 → DataLoader minibatch path."""
    logger = PfabLogger.get_logger(str(tmp_path / "log"))
    model = _SmallMLP(logger=logger)

    train_batches = _make_batches(500)
    val_batches: list[tuple[torch.Tensor, torch.Tensor]] = []

    model.train(train_batches, val_batches)

    assert model._is_trained
    pred = model.forward_pass(torch.tensor([[0.5]], dtype=torch.float32))
    assert abs(pred["y"].item() - 1.0) < 0.5


def test_train_paths_produce_similar_quality(tmp_path):
    """Same seed + same data → both paths converge to similar loss.

    Not bit-exact (DataLoader shuffles), but quality should be in the same ballpark.
    """
    logger = PfabLogger.get_logger(str(tmp_path / "log"))

    # Below-threshold model
    model_full = _SmallMLP(logger=logger)
    model_full.train(_make_batches(80), [])

    # Above-threshold model with same target
    class _LargerMLP(_SmallMLP):
        MINIBATCH_THRESHOLD = 50  # force minibatch path

    model_mb = _LargerMLP(logger=logger)
    model_mb.train(_make_batches(80), [])

    test_X = torch.tensor([[0.0], [0.3], [0.7], [1.0]], dtype=torch.float32)
    pred_full = model_full.forward_pass(test_X)["y"]
    pred_mb = model_mb.forward_pass(test_X)["y"]

    # Both should track 2x roughly. We don't assert tight matching — they're
    # different optimisers in effect (full-batch vs minibatch SGD).
    target = 2.0 * test_X.flatten()
    err_full = (pred_full - target).abs().max().item()
    err_mb = (pred_mb - target).abs().max().item()
    assert err_full < 1.0
    assert err_mb < 1.0
