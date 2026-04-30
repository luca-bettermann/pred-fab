# Interfaces — Context

## Purpose
Abstract contracts that user-implemented models must satisfy. Each system instantiates models via these interfaces.

## Contracts

| Interface | File | Used by |
|-----------|------|---------|
| `IFeatureModel` | `features.py` | FeatureSystem |
| `IEvaluationModel` | `evaluation.py` | EvaluationSystem |
| `IPredictionModel` | `prediction.py` | PredictionSystem |
| `IDeterministicModel` | `prediction.py` | PredictionSystem (analytical formulas) |
| `IResidualModel` | `tuning.py` | PredictionSystem (online tuning) |
| `IExternalData` | `external_data.py` | Dataset / PfabAgent (optional) |
| `BaseInterface` | `base_interface.py` | All of the above |

## Key Points
- `IFeatureModel` and `IPredictionModel` declare `input_domain: str` (the domain code they operate in); `IEvaluationModel` has no domain (aggregates over full feature tensors)
- `IFeatureModel.depth` (optional int) limits how many domain axes are iterated; default `None` = full domain depth
- `IPredictionModel.depth` is computed from output feature column depths; coherence validated at train time
- `IPredictionModel` carries a `gradient_pass: bool` kwarg on `forward_pass(X, ...)` and `encode(X, ...)`: when `True`, the implementation must skip its `torch.no_grad()` context so autograd flows through. Default `False` keeps inference cheap.
- `IPredictionModel.set_categorical_context(col_to_cardinality)` lets `PredictionSystem` hand each model the cardinality of its categorical inputs at training time; `TorchMLPModel` uses this to size its `nn.Embedding` per categorical.
- `IPredictionModel.train(epoch_callback=...)` accepts an optional per-epoch callback used by `PredictionSystem` to refresh scheduled-sampling batches with current-network predictions.
- `IDeterministicModel` extends `IPredictionModel` for known analytical formulas: `train()` is a no-op, `encode()` returns identity, `forward_pass()` is final (denorm→`formula()`→renorm). User implements only `formula(X_raw)`; normalisation is handled automatically via `set_normalization_context()`.
- `IEvaluationModel.compute_performance_tensor(feature_values_S, parameters_list) → (S,)` is the gradient-traversable variant. `compute_performance_batched` is the numpy variant; both have default base-class implementations matching the per-row scalar API.
- Uncertainty estimation is via KDE on the prediction model's latent space (NatPN-light), not a separate GP surrogate
- `IResidualModel` / `MLPResidualModel` (now `nn.Sequential` + Adam) are used for online residual tuning in adaptation steps
- `IExternalData` is optional — omitting it disables remote data sync

## Open Risks
- Serialization contract (save/load artifacts) is interface-specified but implementation-dependent
- Residual model output shape must match prediction model output shape
