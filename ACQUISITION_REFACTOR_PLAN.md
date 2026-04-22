# Acquisition Refactor Plan

Living document tracking the work to fix the broken exploration Schedule phase and, as a follow-up, unify the calibration objective families.

## Context

The 3-phase optimization architecture in `orchestration/calibration/` currently uses two objective families:

- **Energy / Riesz repulsion** — baseline mode. Space-filling, model-free. Used in Phase 2 (Process, flat N-point repulsion) and Phase 3 (Schedule, per-layer offsets).
- **Acquisition / UCB** — exploration and inference modes. `(1-κ)·perf + κ·uncertainty`. Uncertainty is KDE-based (NatPN-light, per-model KDEs in latent space).

### The bug

Exploration Phase 3 (schedule) produces flat trajectories: all L layers converge to the same `(water, speed)`. Root cause: the schedule objective averages single-point UCB values per layer. KDE uncertainty at a point depends only on training data, not on batch siblings — so 7 layers stacked at the same point score identically to 7 diverse layers. No diversification pressure.

The `add_virtual_point` / `clear_virtual_points` infrastructure already exists in `prediction.py` for this exact purpose (see docstring at line 435), but is not exercised by the exploration schedule path.

## Design principles honored

- **Discuss before implementing** — plan approved by user before any code written.
- **Per-mode objective families** — baseline stays on Riesz (this refactor); exploration/inference use UCB.
- **Write once, apply everywhere** — one batch UCB primitive, used by both Process (L=1) and Schedule (L≥1) phases.
- **Reduce code as much as adding it** — objective naming unified; duplicate inner functions collapsed.
- **No hardcoded values** — KDE bandwidth, virtual weights, κ, and all existing config stays schema-/config-driven.

## Phase 1 — Fix exploration (this branch: `feat/batch-aware-exploration-schedule`)

### Scope

Make exploration Phase 3 produce diverse schedules via batch-aware KDE uncertainty. Leave baseline on Riesz.

### Three commits

#### Commit 1 — KDE weighting policy

Change the KDE point-weighting in `_fit_kde` from `√rows`-per-segment to `rows/total_rows`-per-segment, so each experiment contributes exactly 1.0 to the pre-normalization mass (after global normalization each experiment has weight `1/n_exp`, distributed across its segments in proportion to row counts).

**Motivation**: decouples "coverage" (a spatial, row-count-independent concept) from "measurement confidence" (a nonlinear function of row count, already captured by `u = 1/(1+n_post^γ)`). Avoids the implicit "prefer bigger experiments" bias when `n_layers` is or might become a decision variable.

**Files**: `src/pred_fab/orchestration/prediction.py` (lines 220, 229, and surrounding docstring).

**Tests**:
- `test_kde_weighting_process_experiment` — single process experiment → weight 1.0 post-normalization.
- `test_kde_weighting_schedule_experiment` — one schedule experiment, per-segment weights proportional to `seg_rows/total_rows`, summing to 1.0.
- `test_kde_weighting_mixed` — process + schedule, each contributes 0.5 post-normalization.

#### Commit 2 — Objective-function naming refactor

Pure rename + reorganization, behavior-preserving.

**Rationale**: current names (`_acquisition_func`, `_riesz_objective`, `_p2_objective`, two `_sched_objective`s) mix phase-number suffixes, inconsistent `_func` vs `_objective` conventions, and duplicate names across scopes. The rename surfaces the underlying (family × phase) matrix.

**New naming**:

| Math core | Replaces | Role |
|---|---|---|
| `_riesz_energy(X)` → float | `_riesz` @ 840 | Riesz energy of a point set |
| `_ucb_scores(X)` → np.ndarray | `_acquisition_func` @ 272 (generalized) | Per-row UCB, batch-aware uncertainty, L=1 collapses to single-point |

| Phase-level DE objective | Replaces |
|---|---|
| `_process_energy_objective` | `_riesz_objective` @ 978 |
| `_schedule_energy_objective` | `_sched_objective` (#1) @ 1087 |
| `_process_acquisition_objective` | `_p2_objective` @ 1363 |
| `_schedule_acquisition_objective` | `_sched_objective` (#2) @ 1438 |

| Misc | Change |
|---|---|
| `_build_objective` | Removed or reduced to `_build_ucb_callable` (depending on external callers) |
| `_get_acquisition_ranges` | `_get_ucb_normalization_ranges` |
| `_objective` @ engine.py:113 | `_de_fitness` |
| `_tracked_objective` @ engine.py:224 | `_de_fitness_with_tracking` |
| `_wrap_mpc_objective` | Unchanged — MPC is a separate context |

Vocabulary: use "process experiment" and "schedule experiment" consistently (not "flat experiment") in code comments and logs.

**Files**: `src/pred_fab/orchestration/calibration/system.py`, `src/pred_fab/orchestration/calibration/engine.py`, any plotting or test callers that reference the old names.

**Tests**: snapshot tests that verify pre-refactor and post-refactor produce identical results for a pinned baseline and a pinned exploration run. Run before and after the rename; diff must be zero.

#### Commit 3 — Unified batch UCB + batch-aware uncertainty + exploration schedule fix

**Adds**:
- `uncertainty_batch(X_batch: np.ndarray) -> np.ndarray` in `prediction.py`. For each row k, computes uncertainty as if the other L−1 rows of `X_batch` were virtual KDE points representing one future experiment (each virtual carries `1/L` of the "one experiment" mass). No state mutation.
- `_ucb_scores(X_batch)` as the single batch UCB primitive in `CalibrationSystem`. `_process_acquisition_objective` and `_schedule_acquisition_objective` become thin wrappers around it.

**Wires through**:
- `CalibrationSystem.__init__` accepts `uncertainty_batch_fn` alongside `uncertainty_fn`.
- `agent.py` passes both from the predictor.

**Fixes**:
- Exploration Phase 3 objective coupling across layers → diverse trajectories.
- Renames convergence-history key from `"Acquisition"` to `"Schedule"` for consistency with baseline.

**Files**: `src/pred_fab/orchestration/prediction.py`, `src/pred_fab/orchestration/calibration/system.py`, `src/pred_fab/orchestration/agent.py`, `src/pred_fab/orchestration/ORCHESTRATION_CONTEXT.md`.

**Tests**:
- `test_uncertainty_batch_L1_consistency` — `uncertainty_batch(X[0:1])[0] == uncertainty(X[0])` at float tolerance.
- `test_uncertainty_batch_shape` — output shape `(L,)`, float dtype.
- `test_uncertainty_batch_diversity_pressure` — collapsed batch → lower mean u_batch than spread batch.
- `test_uncertainty_batch_no_side_effects` — `_model_kdes` unchanged after call.
- `test_ucb_scores_L1_equals_single_point_ucb` — pre- and post-refactor single-point value matches.
- `test_exploration_schedule_diverges_from_flat` — end-to-end, scheduled dims have non-zero spread.
- `test_convergence_history_key_renamed` — `"Schedule" in convergence_history`, `"Acquisition"` not present.

Visual validation via the mock (not automated): verify schedule trajectory plots show diversified layers and parameter-space plots show spread points.

### Commit strategy

Three commits on this branch, each self-contained and reversible. Commit 2 (pure rename) has zero behavior change; regressions in Commit 3 can be bisected clean.

### Open design choices locked with user

- **Virtual weight** under per-experiment-1.0 scheme: one virtual batch represents one future experiment (total mass 1.0); each virtual gets `1/L` (uniform across layers in absence of expected row counts). "γ" in earlier discussion.
- **`uncertainty_batch_fn`** is optional in `CalibrationSystem.__init__` (defaults to `None`); a clear error is raised if exploration Phase 3 is reached without it being wired.
- **Convergence key rename** `"Acquisition"` → `"Schedule"`.
- **Boundary handling**: `_boundary_factor` applied per-layer (consistent with single-point UCB).
- **Bandwidth for batch call**: `n_for_bandwidth = max(_n_exp + 1, actual_count)` — the virtual batch counts as one additional experiment.

## Phase 2 — Unification follow-up (future branch, not this one)

Once Phase 1 is validated (visually and in tests), a separate branch will explore collapsing Riesz into the batch-UCB family.

### Idea

Empty-KDE + batch-aware uncertainty (κ=1) produces a Gaussian-kernel space-filling objective. Baseline's Riesz behavior becomes a special case of the unified batch-UCB system: `fit_empty_kde(target_n=N)` at the start of baseline, then the same `_ucb_scores` primitive drives both process and schedule phases. One objective family across all modes.

### Storyline

| Mode | κ | KDE state | Effective behavior |
|---|---|---|---|
| Baseline | 1 | Empty (pre-fit for target N) | Pure space-filling, Gaussian pair repulsion |
| Exploration | ∈ (0, 1) | Populated | UCB with batch-aware diversification |
| Inference | 0 | Populated | Greedy perf exploitation |

### Design questions for Phase 2

- **Boundary handling**: existing `_boundary_factor` is already inward-pushing and could fully replace the Riesz half-step mirror terms. Alternative: mirror-image virtual points at domain edges (more elegant, more invasive).
- **Bandwidth vs Riesz `p`**: empirical comparison. The current baseline produces a validated point distribution — Phase 2's acceptance criterion is "reproduces the same quality of spread."
- **Integer/Domain axes**: unchanged — integer repulsion stays in the Domain phase.

### Acceptance criteria

- Baseline process-phase points look visually equivalent to current Riesz output (same spread pattern, corner access, etc.).
- All tests green (Phase 1 + new Phase 2 tests for empty-KDE baseline).
- Single objective core: `_ucb_scores`. `_riesz_energy` deleted.

### Risks

- **Behavior drift** in baseline: Gaussian kernel curvature differs from inverse-power; DE convergence may differ.
- **Boundary strength tuning** may need iteration.
- **Regression potential** is higher than Phase 1 because baseline is currently working.

## Status

- [x] Plan approved (this document)
- [x] Branch created (`feat/batch-aware-exploration-schedule`)
- [ ] Commit 1: KDE weighting policy
- [ ] Commit 2: objective-function naming refactor
- [ ] Commit 3: batch-aware exploration schedule
- [ ] Validation via mock
- [ ] Phase 2 branch (future)
