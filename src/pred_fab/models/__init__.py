"""Convenience prediction-model bases.

Three classes cover the dominant fab modelling architectures:

- ``DeterministicModel`` (in ``pred_fab.interfaces.prediction``) — closed-form
  formulas. No training. Inherits the flat-batched ``predict`` default.
- ``MLPModel`` — feed-forward MLP for tabular / non-sequential mappings.
  Inherits the flat-batched ``predict`` default.
- ``TransformerModel`` — encoder-only transformer with causal attention
  for sequential / autoregressive mappings. Overrides ``predict`` with
  sequence dispatch.

Users plug one model per domain; ``PredictionSystem`` orchestrates the mix
via polymorphic ``model.predict`` calls in topological order over cross-model
dependencies.
"""

from .mlp import MLPModel
from .transformer import TransformerModel
from .depth_decoders import IDepthDecoder, PerNodeMLPDepthDecoder

__all__ = ["MLPModel", "TransformerModel", "IDepthDecoder", "PerNodeMLPDepthDecoder"]
