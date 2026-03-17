# PFAB — Project Context

## Purpose
Predictive Fabrication (PFAB) framework: active-learning loop for manufacturing process calibration.
Combines physical simulation, ML-based prediction, and Bayesian-style optimization to propose
experiment parameters that maximise process performance.

## High-Level Flow
1. **Evaluate** — run feature and evaluation models on physical experiment data
2. **Train** — fit prediction models on historical dataset
3. **Calibrate** — propose next experiments via exploration (UCB), inference (perf), or baseline (maximin)
4. **Execute** — apply proposed parameters; record results; repeat

## Repo Structure

| Path | Role |
|------|------|
| `src/pred_fab/core/` | Data model, schema, DataModule |
| `src/pred_fab/interfaces/` | Model contracts (feature, evaluation, prediction) |
| `src/pred_fab/orchestration/` | System coordination (PfabAgent, CalibrationSystem, …) |
| `src/pred_fab/utils/` | Enums, logging, local persistence, metrics |
| `tests/` | Pytest suite (485 passing, 2 skipped) |

## Entry Point
`PfabAgent` in `orchestration/agent.py` is the single integration surface for users.

## Key Concepts
- **Mode** — EXPLORATION (UCB), INFERENCE (perf-max), BASELINE (space-filling)
- **ExperimentSpec** — output of calibration: `initial_params` + optional `schedules` for step-grid dims
- **Step parameters** — runtime params linked to a dimension via `configure_step_parameter`; re-optimised at each dimension transition
- **MPC lookahead** — `mpc_lookahead=N` simulates N future steps per proposal (default 0 = greedy)
- **Online vs Offline** — online uses trust-region bounds around current params; offline uses global bounds

## Knowledge Base
Full project context, decisions, and research notes live in `../knowledge-base/` (Obsidian vault).
