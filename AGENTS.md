# AGENTS.md

This file provides stable, agent-facing instructions for this repository.

## First Read

1. Universal guidance: `~/OneDrive - TUM/02 Knowledge Base/AGENTS.md`
2. Human-facing overview: `README.md` (keep minimal, high-level, and up to date)
3. Project-specific status and conventions: `PROJECT_CONTEXT.md`
4. Folder-level context notes (`*_CONTEXT.md`) in relevant subfolders for local architecture + known vulnerabilities.
5. Then inspect relevant code paths in `src/pred_fab/` and integration flow in `tests/workflows/`.

## Scope

- Primary project package: `pred_fab`
- Main orchestrator: `PfabAgent`
- Current integration/debug entrypoint: `tests/workflows/manual_workflow.py`

## Working Conventions

- Treat repository tests as secondary for now; primary validation is via the example integration flow.
- Prioritize fixing underlying functionality before finalizing step-method APIs.
- Keep docs and context files aligned with implementation changes.
- Keep `README.md` updated as the human-facing entrypoint: high-level and minimal, then link to detailed docs in `docs/`.
- Prefer refactoring existing code paths over adding parallel one-off helpers.
- If new functionality replaces legacy behavior, remove replaced code in the same change.
- Add short single-line purpose comments/docstrings before non-obvious code blocks.
- Fix small bugs immediately when found; reserve context risk lists for larger refactor topics.
- Prefer polymorphism on typed objects over centralized `if/elif` type-switch chains.

## Context Management

- Use `PROJECT_CONTEXT.md` as the evolving status file.
- Repo context concept:
  - root files (`AGENTS.md`, `PROJECT_CONTEXT.md`) define global conventions and current state,
  - subfolder context files (`*_CONTEXT.md`) capture local structure, assumptions, and vulnerabilities,
  - subfolder files should stay concise and point to code instead of duplicating long design docs.
- Keep linked Obsidian context files up to date when cross-project strategy/state changes.
- Update `PROJECT_CONTEXT.md` whenever any of the following changes:
  - external resource locations,
  - project conventions,
  - current debugging focus/state,
  - near-term upcoming tasks.
