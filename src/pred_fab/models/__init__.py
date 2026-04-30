"""Convenience prediction-model bases.

Three classes cover the dominant fab modelling architectures:

- ``IDeterministicModel`` (in ``pred_fab.interfaces.prediction``) — closed-form
  formulas. No training, no autoreg.
- ``TorchMLPModel`` — feed-forward MLP for tabular / non-sequential mappings.
  Per-cell forward; ``HAS_NATIVE_SEQUENCE = False``.
- ``TorchTransformerModel`` — encoder-only transformer with causal attention
  for sequential / autoregressive mappings. Sequence-aware forward;
  ``HAS_NATIVE_SEQUENCE = True``.

Users plug one model per domain; ``PredictionSystem`` orchestrates the mix
via topological sort over cross-model dependencies.
"""

from .torch_mlp import TorchMLPModel
from .torch_transformer import TorchTransformerModel

__all__ = ["TorchMLPModel", "TorchTransformerModel"]
