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
- Categoricals are int-index columns; `TorchMLPModel` learns one `nn.Embedding(C, d)` per categorical (FastAI-sized `d = min(50, (C+1)//2)`) wired up via `set_categorical_context(col_to_cardinality)`.
- Class-level training hyperparameters (`EPOCHS`, `LR`, `WEIGHT_DECAY`, `SEED`, `MINIBATCH_THRESHOLD`) — subclasses override per model. Above the minibatch threshold (default 1000 rows) `train()` switches to `DataLoader(TensorDataset, shuffle=True)`; below, single full-batch GD.
- `train(train_batches, val_batches, epoch_callback=None)` invokes `epoch_callback(progress)` at `SS_N_REFRESHES` checkpoints when supplied — used by `PredictionSystem` to refresh scheduled-sampling batches mid-training.
- `forward_pass(X, gradient_pass=False)` returns shape `(batch, len(self.outputs))`; `gradient_pass=True` skips `torch.no_grad()` and the compiled-forward fast path so autograd flows through.
- Untrained models: `forward_pass` returns zeros, `encode` is identity.
