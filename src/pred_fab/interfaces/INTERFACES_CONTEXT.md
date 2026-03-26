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
| `ISurrogateModel` | `calibration.py` | PfabAgent (GP uncertainty estimation) |
| `GaussianProcessSurrogate` | `calibration.py` | Default ISurrogateModel implementation |
| `BaseInterface` | `base_interface.py` | All of the above |

## Key Points
- `IFeatureModel` and `IPredictionModel` declare `input_domain: str` (the domain code they operate in); `IEvaluationModel` has no domain (aggregates over full feature tensors)
- `IFeatureModel.depth` (optional int) limits how many domain axes are iterated; default `None` = full domain depth
- `IPredictionModel.depth` is computed from output feature column depths; coherence validated at train time
- Uncertainty estimation uses `GaussianProcessSurrogate` (Matérn-5/2 kernel) trained on experiment-level performance data; owned by `PfabAgent`
- `IResidualModel` / `MLPResidualModel` are available for online residual tuning (not part of current step API)
- `IExternalData` is optional — omitting it disables remote data sync

## Open Risks
- Serialization contract (save/load artifacts) is interface-specified but implementation-dependent
- Residual model output shape must match prediction model output shape
