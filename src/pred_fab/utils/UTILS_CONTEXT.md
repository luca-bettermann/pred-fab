# Utils Context

High-level architecture and long-horizon risks for `src/pred_fab/utils`.
Keep this concise and aligned with code.

## Purpose

`utils` contains shared infrastructure used across core/interfaces/orchestration:
- enums/constants (`enum.py`)
- local filesystem persistence (`local_data.py`)
- logging (`logger.py`)
- regression metrics helpers (`metrics.py`)

## Structure

1. `enum.py`
- Shared enums for roles, normalization, split types, workflow modes, block/file types.

2. `local_data.py`
- Local schema/experiment path management.
- Generic JSON/CSV save/load used by dataset persistence workflows.

3. `logger.py`
- Singleton logger with file + formatted console output helpers.

4. `metrics.py`
- Regression metric calculations (`mae`, `rmse`, `r2`, sample count).

## Recent Fixes (small, already applied)

- Removed direct `print(...)` error output from local data loading path to avoid noisy side effects.

## Open Refactor Risks (Large-Scope Only)

1. Local persistence atomicity/concurrency
- Save/load operations are simple file writes without locking/transaction semantics.
- Risk: partial writes or concurrent access races in multi-process runs.

2. Logger singleton coupling
- Global singleton logger is shared across contexts/workflows.
- Risk: cross-run handler/state coupling in multi-agent or multi-dataset scenarios.

3. File format extensibility
- Persistence helpers are tightly oriented to JSON/CSV patterns.
- Risk: adding alternative structured storage backends will require broad touch points.

## Agent Update Rule

When changing persistence/logging infrastructure:
- update this file for behavior changes with cross-module impact.
- fix small bugs immediately; keep only larger infra-refactor risks here.
