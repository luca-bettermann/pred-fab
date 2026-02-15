# Project Context

This file is the quick-start context for new agents working on this repository.
Update it whenever project direction, external references, or active debugging focus changes.

## Paths

- Project root: `/Users/TUM/Documents/repos/pred-fab`
  - Main PFAB package source code.
- Example integration flow: `/Users/TUM/Documents/repos/pred-fab/tests/workflows/manual_workflow.py`
  - Primary step-through script currently used for debugging.
- OneDrive root: `/Users/TUM/OneDrive - TUM`
  - External context outside repo.
- Knowledge base: `/Users/TUM/OneDrive - TUM/02 Knowledge Base`
  - Obsidian vault. Look for PFAB architecture notes, workflow design notes, and implementation decisions.
- Paper folder: `/Users/TUM/OneDrive - TUM/03 Paper`
  - Contains references related to PFAB and the ISARC paper context.
- PFAB paper PDF (current working draft): `/Users/TUM/OneDrive - TUM/03 Paper/ISARC 2026/ISARC_Paper_2026.pdf`
  - Main paper artifact tied to this repository’s architecture/workflow.

## Conventions

- Core orchestrator class is `PfabAgent`.
- Required step methods (target architecture):
  - `exploration_step`
  - `inference_step`
  - `adaptation_step`

- Canonical operation methods (not steps):
  - `evaluate`
  - `train`
  - `predict`
  - `configure_calibration`
  - `sample_baseline_experiments`
  
- Current practical validation path is the integration workflow in `tests/workflows/`, plus focused unit tests in `tests/core/`.
- Prefer fixing underlying systems first (data flow, calibration/prediction interfaces), then finalize step-method API behavior.
- Coding preferences for this repo:
  - first debug/refactor existing code paths, avoid adding parallel new code blocks unless unavoidable,
  - whenever new logic replaces old logic, remove or merge the replaced code in the same change,
  - add concise one-line purpose comments/docstrings before non-trivial code blocks,
  - fix small bugs immediately; only track large refactor risks in `*_CONTEXT.md` files.
- Data representation convention:
  - canonical in-memory feature representation is tensor-first (`np.ndarray` with dimensional shape),
  - tabular representation exists as a transformation boundary only (IO/export/training table views).
  - step outputs that propose new parameters are typed as `ParameterProposal` (not raw dict API by default).
  - online parameter changes are logged as event records on `ExperimentData`; initial parameter block remains initial state.
- Step semantics convention:
  - a "step" advances decision-making in the workflow and should return typed artifacts for the next decision,
  - mutating/evaluation/training/configuration utilities are operation methods and not treated as steps.

## State

- Agent currently has target methods implemented but mixed with older transitional flows.
- Main active debugging focus:
  - underlying functionality and integration behavior beneath step methods,
  - before finalizing the step method contracts.
- Latest resolved core issue:
  - `export_to_dataframe` dimensional indexing mismatch fixed via canonical tensor access (`Features.value_at`) and explicit tensor/tabular transforms.
- Additional known transition issues in step layer:
  - signature mismatches between `PfabAgent` step methods and `CalibrationSystem`,
  - legacy methods (`step_offline`, `step_online`, `adaptation_step`, `prediction_step`) still coexist with target methods.
  - adaptation API migration in progress: keep step-local behavior and proposal logging semantics aligned with tests.
- Test suite status:
  - previous outdated tests replaced with new baseline structure in `tests/` (`core`, `integration`, `workflows`, `utils`).
  - integration flow previously in `examples/` is now under `tests/workflows/manual_workflow.py`.

## Notes

- In new threads, instruct agents to read `AGENTS.md` and `PROJECT_CONTEXT.md` first.
- For folder internals, read local context docs:
  - `src/pred_fab/core/CORE_CONTEXT.md`
  - `src/pred_fab/interfaces/INTERFACES_CONTEXT.md`
  - `src/pred_fab/orchestration/ORCHESTRATION_CONTEXT.md`
  - `src/pred_fab/utils/UTILS_CONTEXT.md`
- When starting a new thread, also confirm access to OneDrive paths above.
- Keep this file short and operational; link to source files instead of duplicating long design docs.
- User-level circular loop:
  - proposal (`ParameterProposal`) -> optional apply/log (`ParameterUpdateEvent`) -> dataset export/datamodule replay -> next step proposal.
