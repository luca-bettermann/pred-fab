# Models — Context

## Purpose
Convenience prediction-model bases that satisfy the `IPredictionModel` contract.
User code stops at "set HIDDEN + the three properties" rather than reimplementing
the train / forward_pass / encode trio for every project.

The interface (`pred_fab.interfaces.IPredictionModel`) stays the canonical contract.
This subpackage is a shortcut for the common feed-forward MLP shape — bespoke
architectures (RF, GP, transformer, …) should still subclass the interface directly.

## Modules

| File | Class | Description |
|------|-------|-------------|
| `torch_mlp.py` | `TorchMLPModel(IPredictionModel)` | Linear/ReLU stack + Adam/MSE training. Lazy torch import. |

## Optional Dependencies
- **`TorchMLPModel`** requires `torch`. Install via `pred-fab[torch]`. Importing
  `pred_fab.models` without torch installed succeeds; calling `.train()` /
  `.forward_pass()` raises `ImportError` with an actionable message.

## Key Points
- Inputs are assumed pre-normalised by `DataModule`; bases do not include a scaler.
- `encode()` returns penultimate-layer activations (HIDDEN[-1] dims) for KDE.
- Class-level training hyperparameters (`EPOCHS`, `LR`, `WEIGHT_DECAY`, `SEED`) — subclasses override per model.
- `forward_pass` returns shape `(batch, len(self.outputs))`; multi-output supported.
- Untrained models: `forward_pass` returns zeros, `encode` is identity.
