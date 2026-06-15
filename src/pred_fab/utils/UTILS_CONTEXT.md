# Utils — Context

## Purpose
Shared infrastructure with no dependencies on other pred_fab packages.

## Modules

| Module | Role |
|--------|------|
| `enum.py` | All shared enumerations: `Mode`, `NormMethod`, `SourceStep`, `SystemName`, … |
| `local_data.py` | Filesystem persistence (experiment snapshots, model artifacts, logs) |
| `logger.py` | `PfabLogger` singleton wrapping Python logging + console helpers |
| `metrics.py` | `combined_score` (weighted perf) + `Metrics` (MAE, R², informed R²/R²_inf) |
| `console.py` | Console output helpers (tables, score colouring, step summaries) |
| `profiler.py` | Lightweight inline profiler for hot paths (`profiler.section(...)`) |
| `wandb_logger.py` | Optional Weights & Biases logger for prediction-model training |

## Key Points
- `Mode` (DISCOVERY / EXPLORATION / INFERENCE) is the primary dispatch enum in CalibrationSystem
- `NormMethod.NONE` is used for discovery DataModule (no stats available without training data)
- `SourceStep` tags each `ParameterProposal` with its origin for traceability
- `LocalData` path conventions must stay stable — PredictionSystem uses them for artifact save/load
