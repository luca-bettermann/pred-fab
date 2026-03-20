# Interfaces — Context

## Purpose
Abstract contracts that user-implemented models must satisfy. Each system instantiates models via these interfaces.

## Contracts

| Interface | File | Used by |
|-----------|------|---------|
| `IFeatureModel` | `features.py` | FeatureSystem |
| `IEvaluationModel` | `evaluation.py` | EvaluationSystem |
| `IPredictionModel` | `prediction.py` | PredictionSystem |
| `IResidualModel` | `tuning.py` | PredictionSystem (online tuning) |
| `IExternalData` | `external_data.py` | Dataset / PfabAgent (optional) |
| `BaseInterface` | `base_interface.py` | All of the above |

## Key Points
- `IFeatureModel` and `IPredictionModel` declare `input_domain: str` (the domain code they operate in); `IEvaluationModel` has no domain (aggregates over full feature tensors)
- `IFeatureModel.depth` (optional int) limits how many domain axes are iterated; default `None` = full domain depth
- `IPredictionModel.depth` is computed from output feature column depths; coherence validated at train time
- Uncertainty estimation is via KDE on the prediction model's latent space (NatPN-light), not a separate GP surrogate
- `IResidualModel` / `MLPResidualModel` are used for online residual tuning in adaptation steps
- `IExternalData` is optional — omitting it disables remote data sync

## Open Risks
- Serialization contract (save/load artifacts) is interface-specified but implementation-dependent
- Residual model output shape must match prediction model output shape
