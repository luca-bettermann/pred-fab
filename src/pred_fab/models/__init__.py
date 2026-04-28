"""Convenience prediction-model bases.

These are opinionated implementations that satisfy the ``IPredictionModel``
contract and let user code stop at "set HIDDEN + the three properties".
Each base brings its own optional dependency:

- ``TorchMLPModel`` requires ``torch`` (install via ``pred-fab[torch]``).

Users who need bespoke architectures (RF, GP, transformer, …) should
implement ``IPredictionModel`` directly — this module is a shortcut, not
a contract.
"""

from .torch_mlp import TorchMLPModel

__all__ = ["TorchMLPModel"]
