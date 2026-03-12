# Orchestration Context

High-level architecture and long-horizon risks for `src/pred_fab/orchestration`.
Keep this concise and synchronized with implementation.

## Purpose

`orchestration` coordinates models and data flow across feature extraction, evaluation,
training, inference, and calibration:
- top-level workflow API (`agent.py`)
- system wrappers (`features.py`, `evaluation.py`, `prediction.py`, `calibration.py`)
- exportable runtime inference wrapper (`inference_bundle.py`)

## Structure

1. `agent.py`
- Main integration surface (`PfabAgent`).
- Owns model registration, system initialization, workflow steps, and calibration config.
- Step methods return `ExperimentSpec` — unified return type for all step methods.
- API taxonomy:
  - step methods: `exploration_step`, `inference_step`, `adaptation_step`
  - operation methods: `evaluate`, `train`, `predict`, `configure_calibration`, `sample_baseline_experiments`

2. `base_system.py`
- Shared helpers for model specs and schema reference wiring.

3. `features.py`
- Runs feature models and writes canonical tensors into experiment data.

4. `evaluation.py`
- Runs evaluation models from computed features to performance attributes.

5. `prediction.py`
- Handles prediction model training/tuning/validation/inference and inference bundle export.
- Online tuning now respects requested slice boundaries (`start:end`) for step-local adaptation.
- After training, fits a weighted KDE on unique latent training configs (NatPN-light).
- Exposes `encode()`, `uncertainty()`, `kernel_similarity()`, `predict_for_calibration()`.
- Degenerate-dimension guard: zero-variance latent dims are dropped before KDE fitting
  (`_kde_active_mask` applied consistently in `uncertainty()` and `kernel_similarity()`).

6. `calibration.py`
- Unified `run_calibration()` entry point for all optimization use cases.
- **No longer owns a surrogate model.** Instead receives three injected callables:
  - `perf_fn(params_dict) → {perf_code: value_or_None}` — encapsulates predict + evaluate
  - `uncertainty_fn(X_norm) → float` — epistemic uncertainty from `PredictionSystem`
  - `similarity_fn(X1, X2) → float` — optional; drives trajectory diversity discounting
- `_active_datamodule` is set before each optimization run to enable `array_to_params`.
- Adaptation uses current effective parameters (initial + recorded updates) as optimization center.

7. `inference_bundle.py`
- Lightweight deployed inference wrapper (prediction + normalization + schema validation).

## Calibration and Tuning Semantics

### Unified `run_calibration()` Architecture

All calibration use cases flow through a single `run_calibration()` method. Two orthogonal
axes configure its behavior:

**Mode** (what to optimize):
- `Mode.EXPLORATION`: UCB acquisition — `(1-κ)·perf_fn(x) + κ·uncertainty_fn(x)`.
- `Mode.INFERENCE`: direct `perf_fn(x)` objective (no uncertainty term).

**Domain** (bounds strategy):
- `Domain.OFFLINE` (default): step 0 uses global bounds + random restarts; subsequent steps
  use trust-region bounds around the previous step's result.
- `Domain.ONLINE`: always trust-region bounds, no restarts, single step. The fabrication
  process IS the outer loop; the method does not loop internally.

### Step-Loop Architecture

When trajectory parameters are configured via `configure_trajectory(code, dimension_code)`,
`run_calibration` iterates over a **flattened Cartesian product of dimensions**, ordered
coarsest-first (by `DataDimension.level`):

```
trajectory_configs = {"layer_height": "dim_1", "speed": "dim_2"}  # dim_1 level 1, dim_2 level 2
current_params["dim_1"] = 2 (layers), current_params["dim_2"] = 3 (segments)

Flat grid:
  step 0: {dim_1: 0, dim_2: 0}  → optimize(layer_height, speed)  — all dims transition
  step 1: {dim_1: 0, dim_2: 1}  → fix(layer_height), optimize(speed)
  step 2: {dim_1: 0, dim_2: 2}  → fix(layer_height), optimize(speed)
  step 3: {dim_1: 1, dim_2: 0}  → optimize(layer_height, speed)  — both dims transition
  step 4: {dim_1: 1, dim_2: 1}  → fix(layer_height), optimize(speed)
  step 5: {dim_1: 1, dim_2: 2}  → fix(layer_height), optimize(speed)
```

**Parameter eligibility** at each step is controlled by which dimensions transition:
- Params mapped to *transitioning* dimensions → eligible for optimization
- Params mapped to *unchanged* dimensions → added to `fixed_param_values`
- Non-trajectory params (always-fixed, schema-fixed) → always fixed

**Bounds logic** (unified for offline and online):
- Step 0 in OFFLINE domain: global bounds (`_get_offline_bounds`), random restarts
- Step 0 in ONLINE domain: trust-region bounds around `current_params`, no restarts
- All subsequent steps: trust-region bounds around previous step's result, no restarts

Without trajectory configs (`trajectory_configs` empty), the step grid has exactly one
step → single `ExperimentSpec` with empty `schedules` dict. This covers the experiment-level
exploration and inference use cases.

**`target_indices` — single targeted step:**
Pass `target_indices={"dim_1": k}` to collapse the full Cartesian grid to a single step,
optimising only the trajectory params mapped to the specified dimensions and fixing all others.
Useful when you want to plan parameters for one specific layer or segment in isolation rather
than running the full trajectory sequence:

```
target_indices={"dim_1": 0}   → optimize(layer_height), fix(speed)   → empty schedules
target_indices={"dim_2": 1}   → fix(layer_height), optimize(speed)   → empty schedules
target_indices={"dim_1": 1, "dim_2": 0}  → optimize(layer_height, speed)
```

Since `target_indices` always produces a single-step grid, no dimension transitions occur
→ `schedules` is always empty. The proposal is returned in `initial_params`.
Step 0 in `OFFLINE` domain still uses global bounds + restarts. Use `ONLINE` domain if you
want trust-region bounds around `current_params` for a targeted step.

**MPC lookahead — reducing greedy myopia:**
Pass `mpc_lookahead_depth=d` (default 0) to augment each step's objective with a d-step
forward rollout.  For each candidate X the optimizer evaluates:

```
MPC(X) = score(X) + γ¹·score(X₁) + γ²·score(X₂) + … + γᵈ·score(Xᵈ)
```

where each Xⱼ₊₁ is obtained by a quick L-BFGS-B step from Xⱼ using trust-region bounds.
`mpc_discount` (default 0.9) controls γ.  Depth 0 is pure greedy.  The step-loop
structure, bounds logic, and return type are all unchanged — only the objective landscape
seen by the optimizer is enriched.  Parameters without a trust region are held fixed in
the lookahead (zero-width bounds), so only runtime-adjustable params participate.

Implementation: `_wrap_mpc_objective(base_obj, datamodule, depth, discount) → Callable`.
Applied unconditionally after `_build_objective` when depth > 0; works in both OFFLINE
and ONLINE domains, with or without trajectory configs.

### Return Type: `ExperimentSpec`

All step methods (`exploration_step`, `inference_step`, `adaptation_step`) return `ExperimentSpec`:
- `initial_params`: `ParameterProposal` for step 0 (all params, tagged with `source_step`)
- `schedules`: `Dict[str, ParameterSchedule]` — one entry per configured dimension,
  keyed by dimension code. Empty when no trajectory configs are set.
- `apply_schedules(exp)`: records all schedule entries as `ParameterUpdateEvent`s.
- Dict-like delegation (`__getitem__`, `__contains__`, `keys()`) forwards to `initial_params`.

`adaptation_step` tags `initial_params.source_step = SourceStep.ADAPTATION` after
delegating to `run_calibration(domain=ONLINE)`, preserving semantic distinction.

### Deleted Infrastructure

The following were removed during the unification refactoring (Phase 5):
- `run_adaptation()` — replaced by `run_calibration(domain=ONLINE)`
- `run_trajectory_exploration()` and all SLSQP sub-methods — replaced by the step-loop
- `SourceStep.TRAJECTORY_EXPLORATION` — no longer needed

### Online Residual Adaptation Lifecycle

- `PredictionSystem.tune(...)` trains residual correction on selected row slices only.
- Residual model uses base inputs + base predictions as tuning input.
- Applied online parameter changes are persisted as `ParameterUpdateEvent` records, then
  replayed in export/datamodule flows.

## Open Refactor Risks (Large-Scope Only)

1. Use of private DataModule internals across systems
- Orchestration systems still rely on internal DataModule methods/fields (normalization internals).
- Risk: tight coupling makes independent evolution/testing of modules harder.

2. Inference bundle schema/input validation depth
- Bundle validates unknown columns but does not yet enforce full schema constraint checks.
- Risk: invalid but schema-shaped requests can pass until deeper model/runtime stages.

3. MPC lookahead depth vs. computation cost
- `mpc_lookahead_depth > 0` runs one inner `minimize` call per lookahead step per outer
  objective evaluation, which can multiply wall-clock time by (1 + depth × optimizer_iters).
  Depth 0 (default) has zero overhead.  Recommended: depth 1–2 for offline planning;
  depth 0 for latency-sensitive online adaptation.

## Agent Update Rule

When changing orchestration APIs or workflow semantics:
- update this file for cross-system behavior changes.
- fix small bugs immediately; keep only larger migration/refactor risks here.
