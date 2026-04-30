# Models — Context

## Purpose
Convenience prediction-model bases that satisfy the `IPredictionModel` contract. End-user code stops at "set hyperparams + declare `input_parameters`/`input_features`/`outputs`" rather than reimplementing the train / forward / predict trio for every project.

Three classes cover the dominant fab modelling architectures, each owning its own dispatch via polymorphic `predict`. The framework (`PredictionSystem`) is oblivious to which is used — it just calls `model.predict(...)` in topological order.

## Modules

| File | Class | Dispatch | Use when |
|------|-------|----------|----------|
| `torch_mlp.py` | `TorchMLPModel` | flat-batched (default `predict` from `IPredictionModel`) | tabular / non-sequential per-cell mappings |
| `torch_transformer.py` | `TorchTransformerModel` | sequence-batched (overrides `predict`) | sequential / autoregressive prediction along a domain axis (causal attention) |
| (in `pred_fab.interfaces.prediction`) | `IDeterministicModel` | flat-batched (inherits default) | closed-form formulas (no training) |

## Key Points

- Inputs are pre-normalised by `DataModule`; bases do not include a scaler.
- `encode()` returns penultimate-layer activations for KDE.
  - `TorchMLPModel`: HIDDEN[-1] dims of the trained network.
  - `TorchTransformerModel`: input projection at position 0 — a static-only embedding (the transformer has no per-row latent).
- `TorchMLPModel` categoricals are int-index columns; one `nn.Embedding(C, d)` per categorical (FastAI-sized `d = min(50, (C+1)//2)`) wired up via `set_categorical_context(col_to_cardinality)`.
- Class-level hyperparameters (`EPOCHS`, `LR`, `WEIGHT_DECAY`, `SEED`, plus `HIDDEN` for MLP, `D_MODEL`/`N_HEADS`/`N_LAYERS`/`MAX_SEQ_LEN` for Transformer) — subclasses override per model.
- `forward_pass(X, gradient_pass=False)` is kept on all bases for non-sequence call sites (KDE encode probes); real prediction goes through `model.predict(params_list, dm, dim_info_list, predictions_so_far)`.
- `TorchTransformerModel.train` accepts sequence-shaped batches `(B, L, n_input)` / `(B, L, n_output)` and rejects flat 2D batches with a helpful error.
- `_validate_schema_compatibility(schema)` is a per-class type-check run by `PredictionSystem.train` after the universal `validate_dimensional_coherence`. Default no-op; `TorchTransformerModel` requires `sequence_axis_code` to resolve to a real domain axis.
- Untrained models: `forward_pass` returns zeros, `encode` is identity.
