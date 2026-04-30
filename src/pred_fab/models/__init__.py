"""Convenience prediction-model bases.

Three classes cover the dominant fab modelling architectures:

- ``IDeterministicModel`` (in ``pred_fab.interfaces.prediction``) — closed-form
  formulas. No training. Inherits the flat-batched ``predict`` default.
- ``TorchMLPModel`` — feed-forward MLP for tabular / non-sequential mappings.
  Inherits the flat-batched ``predict`` default.
- ``TorchTransformerModel`` — encoder-only transformer with causal attention
  for sequential / autoregressive mappings. Overrides ``predict`` with
  sequence dispatch.

Users plug one model per domain; ``PredictionSystem`` orchestrates the mix
via polymorphic ``model.predict`` calls in topological order over cross-model
dependencies.
"""

from .torch_mlp import TorchMLPModel
from .torch_transformer import TorchTransformerModel

__all__ = ["TorchMLPModel", "TorchTransformerModel"]
