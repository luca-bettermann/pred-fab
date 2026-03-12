# Exploration Scope: From Experiment-Level to Dimensional-Level Exploration

> **Status:** Living document — Unified step-loop (`run_calibration` Phase 5) implemented; Option A backlogged
> **Related:** [01-parameter-levels.md](01-parameter-levels.md), [03-uncertainty-estimation.md](03-uncertainty-estimation.md)

---

## Current Exploration Paradigm

PFAB's exploration step operates at **experiment level**: it proposes a single parameter configuration for an entire experiment, executes that experiment, observes the aggregate performance, and uses the result to inform the next proposal.

```
Exploration loop (current):
  1. GP-UCB acquisition function → propose x_new (initial parameters)
  2. Run experiment with x_new fixed throughout
  3. Observe scalar performance a = E(y(x_new))
  4. Add (x_new, a) to surrogate training set
  5. Repeat
```

The information yield per experiment is limited: one parameter configuration → one performance scalar. Characterizing the full fabrication parameter space requires many such experiments.

---

## The Opportunity and the Architectural Necessity

For parameters that are physically adjustable mid-experiment — **runtime-adjustable parameters** (see [Document 1](01-parameter-levels.md)) — a single fabrication run can cover *multiple operating points simultaneously*. Instead of one `(parameters, performance)` pair per experiment, the run can produce multiple `(parameter_state_i, feature_value_i)` pairs, one per dimensional step or segment.

This is not only an opportunity for richer data collection. It is an **architectural necessity** once the GP surrogate model is replaced by a PM-based evidence head (see [Document 3](03-uncertainty-estimation.md)):

- The GP surrogate operates at experiment level: it receives a single parameter vector `x` and returns a predicted performance scalar `μ(x)` and uncertainty `σ(x)`. It is structurally decoupled from the PM.
- The evidence head computes uncertainty directly from the PM's latent space. The PM is trained on dimensional-level data — one row per measurement position. Its uncertainty is therefore dimensional in nature.
- To use this dimensional uncertainty for experiment-level calibration decisions, the two scales must be bridged. The question of how parameters change *within* an experiment becomes load-bearing, not optional.

This forces a careful look at the parameter-level architecture and makes the distinction between experiment-level and runtime-adjustable parameters structurally important, not just an optimization opportunity.

Two distinct approaches to exploiting this are discussed below.

---

## Option A — Fabrication-Time Online Exploration (dynamic)

### Concept

Analogous to online adaptation in inference mode, but driven by an *explorative* objective: during execution, the system computes uncertainty in real time from measurements being collected, and dynamically proposes new runtime parameter values at each step to maximize the exploration acquisition function.

```
Fabrication-time exploration loop (Option A):
  For each dimensional step i during experiment execution:
    1. Observe feature measurement y_i at current runtime params x_runtime_i
    2. Update residual model using y_i (same mechanism as online adaptation)
    3. Compute updated uncertainty from PM + residual
    4. Optimize acquisition function over runtime parameter space
       → propose x_runtime_{i+1}
    5. Apply x_runtime_{i+1} before executing step i+1
```

The runtime parameter value at each step is *not predefined* — it is determined on-the-fly by the exploration objective, conditioned on data collected so far in the current run. This is exploration literally "onboarded to fabrication time."

### What This Enables

- Maximum information per fabrication run: the system actively seeks high-uncertainty regions *within* the run
- Discovery of sharp transitions or nonlinear effects along runtime parameter axes (e.g., how does filament width respond as speed ramps up during a single print?)
- Synergy with online adaptation: the same computational infrastructure (PM + residual tuning, online parameter update, `ParameterUpdateEvent`) supports both modes — Option A is structurally the adaptation loop with an explorative rather than a performance-maximizing acquisition function

### Mechanism: PM + Residual as the Incremental Learning Layer

The mechanism for within-experiment learning is the **residual model**: a lightweight model (currently `MLPResidualModel`) trained to correct the PM's predictions using the current experiment's observations. In online adaptation mode, the residual model is already fine-tuned incrementally as measurements arrive. Option A uses this same mechanism:

- **PM (frozen during experiment):** provides base feature predictions for any parameter configuration
- **Residual model (updated per step):** corrects PM predictions using current-experiment observations, progressively reducing the discrepancy between predicted and observed features
- **PM + residual (combined):** provides both predictions and — via the uncertainty head — uncertainty estimates that improve as the experiment progresses

This means Option A does not require retraining the PM during fabrication. The residual model serves as the incremental correction layer. The approach is consistent with how online adaptation already works, which is an architectural advantage: the same training and inference infrastructure applies to both exploration and adaptation.

**Caveat:** Residual model updates at each fabrication step impose latency. The step cycle budget (100ms–1s range depending on hardware) must accommodate both residual update and acquisition optimization. This is the primary feasibility constraint.

### Requirements

| Requirement | Implication |
|-------------|-------------|
| Real-time residual model update | Residual update must complete within the fabrication step cycle time |
| Real-time acquisition optimization | Optimization of `α(x_runtime)` must complete within the same step budget |
| Live data access | Feature measurements must be accessible to the software stack at each step |
| PM inference (forward pass only) | Already available; PM itself is not retrained |

### Risks and Challenges

- **Hardware latency:** Fabrication systems may not pause between steps for software computation. The control loop budget can be tight (100ms–1s range). Residual update + acquisition optimization must fit this window.
- **PM predictions for novel configs may be unreliable:** The PM is frozen; its feature predictions for runtime configs it has never seen may be inaccurate. Uncertainty correctly identifies OOD regions, but the system cannot know *what* it will observe there — only that it doesn't know.
- **Safety constraints:** Rapid parameter changes mid-experiment can cause material failures (thermal shock, clogging, delamination). Safety bounds on change rate and range are essential.
- **Exploratory instability:** Without careful acquisition function design, the system could propose extreme runtime values to maximize uncertainty. The acquisition function's κ parameter provides user control over this: at κ=0 the function is purely performance-driven; at κ=1 purely uncertainty-driven. In practice a mixed setting (e.g., κ=0.5) constrains the system from seeking uncertainty at the expense of fabrication quality.
- **Interaction with PM training:** Data from fabrication-time exploration introduces sequential within-experiment correlations that standard IID PM training assumptions don't account for.
- **Reproducibility:** Step-by-step adaptive decisions are hard to reproduce exactly, making experiment analysis and debugging more complex.

### Feasibility Assessment

| Layer | Readiness | Notes |
|-------|-----------|-------|
| Data | Ready | `ParameterUpdateEvent` + `get_effective_parameters_for_row()` fully support it |
| PM inference | Ready | Forward pass available; PM frozen during experiment |
| Residual model | Ready (mechanism exists) | Same fine-tuning loop as online adaptation; latency within step budget is the open question |
| Uncertainty | Requires design | Must be updatable incrementally; see [Document 3](03-uncertainty-estimation.md) |
| Acquisition | Requires design | Must be optimizable in real time within step budget |
| Schema | Implemented | `runtime_adjustable` flag in place; see [Document 1](01-parameter-levels.md) |
| Safety | Not addressed | Needs change-rate constraints and bounds on runtime proposals |

**Overall: high potential, significant implementation complexity. Backlogged for a subsequent phase.**

---

## Option B — Trajectory-Based Exploration (predefined schedule)

### Concept

Before the experiment begins, the exploration system proposes a *parameter schedule*: a structured assignment of runtime parameter values across dimensional steps. The schedule is optimized offline using the existing exploration acquisition function, then executed deterministically during the fabrication run.

```
Trajectory exploration (Option B):
  Offline phase (before experiment):
    1. Score candidate schedules using acquisition function
       (piecewise-constant runtime parameter values over K segments)
    2. Propose schedule π: {step_index → ParameterProposal}
  Execution phase:
    3. Execute experiment, applying π at each trigger step
       → recorded as ParameterUpdateEvents
    4. Collect feature measurements {y_0, ..., y_M}
  Post-experiment:
    5. Each (x_static, x_runtime_i, y_i) row is an independent PM training sample
       (via get_effective_parameters_for_row() — already supported)
    6. Aggregate dimensional features → experiment-level performance for calibration update
```

The simplest parameterization is **piecewise-constant segments**: for each runtime parameter, divide the dimensional iteration into K contiguous blocks and assign one value per block. A schedule with 3 runtime parameters and K=5 segments produces a 15-dimensional joint proposal space — manageable with the existing LHS-based candidate generation.

### How Trajectory Generation Works in Practice

**Configuration (CalibrationSystem, not schema):**

For each runtime parameter eligible for trajectory exploration, the user configures which dimensional level it adjusts at:

```python
calibration.configure_trajectory(
    code="speed",
    dimension_code="dim_1"   # adjust at each layer (level-1 dimension)
)
# Trust region (max value change per step) from:
calibration.configure_adaptation_delta({"speed": 50.0})
```

Note: the API changed in Phase 5 from `dimension_level: int` to `dimension_code: str`.
The dimension code maps directly to a `DataDimension` in the schema, removing the
intermediate level→dimension lookup.

This is CalibrationSystem configuration — analogous to `configure_adaptation_delta()` for online adaptation. It does not modify the schema. The schema declares that `speed` is runtime-adjustable; the CalibrationSystem config declares *where* it is adjusted (at which dimensional level). The number of actual trigger steps is an internal optimization hyperparameter, not a user-facing config field: the optimizer decides how many proposals to place and where, and the resulting schedule is sparse — it only records steps where values actually change, exactly like `ParameterUpdateEvent`s. This follows the schema/config boundary established in [Document 1](01-parameter-levels.md).

**Trajectory optimization (two-phase, same approach as existing exploration):**

The existing `exploration_step()` already uses a two-phase approach: (1) LHS for global warm-start candidates, (2) `scipy.optimize.minimize` to refine from the best LHS starting point. Trajectory exploration uses the same structure, with the trajectory proposal vector as the decision variable:

```
Phase 1 — LHS warm start:
  Generate N_candidates random schedules by sampling runtime parameter values
  independently for each potential trigger step via LHS.
  Score each candidate using α(π) (see below).
  Select the best LHS candidate as the optimizer starting point.

Phase 2 — Gradient-based optimization:
  Treat the trigger-step values as a continuous decision variable v ∈ [min, max]^K_eff.
  Maximize α(π(v)) using scipy.optimize.minimize (negated objective).
  The optimizer focuses on the region the acquisition function finds promising
  given the current surrogate state — not forced to cover the full space.
```

The LHS warm start ensures reasonable global coverage early on; the optimizer converges to the best schedule given what the surrogate has learned. Later in data collection, as the surrogate identifies uninteresting regions, the optimizer naturally avoids them without any explicit constraint.

**`generate_baseline_experiments()` as the unified LHS entry point:**

`generate_baseline_experiments()` is the entry point for LHS-based sampling at both levels simultaneously. When no trajectories are configured, all parameters are at level 0 (experiment level) — each parameter holds a constant value throughout the run, sampled once from its schema bounds. When trajectories are configured for one or more runtime parameters via `configure_trajectory()`, `generate_baseline_experiments()` additionally samples K independent segment values for each configured parameter, creating a `ParameterSchedule` alongside the initial proposal. The initial segment value (segment 0) is carried in `initial_params`; the subsequent K−1 trigger values are in the schedule. The default level for all parameters is 0; specifying `configure_trajectory(code, dimension_level=N)` elevates a parameter to generate a schedule at that dimensional level.

**Trust region constraints in the gradient-based phase:**

The trust region delta from `configure_adaptation_delta()` — which bounds how much a runtime parameter may move per online adaptation step — is reused in the trajectory optimizer as a **step-change constraint**: consecutive segment values may differ by at most `trust_region[code]`. These are enforced as linear inequality constraints in the SLSQP solver:

```
|v[k+1] - v[k]| ≤ trust_region[p]   for all consecutive segment pairs k, k+1
→ v[k+1] - v[k] ≤ trust_region[p]
→ v[k] - v[k+1] ≤ trust_region[p]
```

These constraints are **not** applied during the LHS warm-start phase: baseline samples are drawn freely from `[min, max]` schema bounds to maximise coverage. The trust region constraint is enforced during SLSQP refinement only, where it prevents the optimiser from proposing physically unrealistic step changes between consecutive segments.

**Trajectory scoring:**

The existing `_acquisition_func()` scores a single parameter vector `x`. For a trajectory, each trigger step has its own runtime parameter values `x_runtime_k` combined with the shared static parameters `x_static`:

```
α(π) = (1/K_eff) Σ_k α(x_static, x_runtime_k)
     = (1/K_eff) Σ_k [(1-κ) · S(μ(x_static, x_runtime_k)) + κ · S(σ(x_static, x_runtime_k))]
```

This is a direct reuse of the existing acquisition function — called once per trigger step, results averaged. No new acquisition function design is required. The step with the highest individual acquisition value indicates the most informative region of the runtime parameter axis.

**Selecting and returning the best schedule:**

The optimizer returns the trajectory vector that maximises `α(π)`. This is decoded into a sparse `ParameterSchedule` — only recording trigger steps where values actually change.

### Representing the Schedule: ParameterSchedule

`ParameterProposal` is a flat `{code: value}` value object — sparse by nature, containing only the parameters whose values are being set. It is already the right unit for a single trigger point.

A trajectory over one dimensional level is a sequence of `(step_index, ParameterProposal)` pairs. Different runtime parameters may adjust at different dimensional levels (e.g., `speed` adjusts per-layer at level 1, `temperature` adjusts per-zone at level 2). One `ParameterSchedule` covers exactly one dimension, containing only the parameters that adjust at that level:

```python
@dataclass(frozen=True)
class ParameterSchedule:
    """Sparse, ordered schedule of parameter proposals for one dimensional level.

    Each entry records a trigger step and a proposal containing only the
    parameters that change at that step. Mirrors ParameterUpdateEvent semantics:
    absent entries mean the parameter holds its previous value.
    """
    entries: List[Tuple[int, ParameterProposal]]   # sparse: (step_index, {changed params only})
    dimension: str                                   # the dimension this schedule steps through

    def apply(self, experiment: ExperimentData) -> None:
        """Record all schedule entries as ParameterUpdateEvents."""
        for step_index, proposal in self.entries:
            experiment.record_parameter_update(
                proposal, dimension=self.dimension, step_index=step_index
            )
```

An experiment with runtime parameters adjusting at multiple dimensional levels receives one `ParameterSchedule` per dimension. The full experiment schedule is `Dict[str, ParameterSchedule]` keyed by dimension name:

```python
# Example: speed adjusts per layer (level 1), temperature adjusts per zone (level 2)
experiment_schedule = {
    "layer_id":   ParameterSchedule(dimension="layer_id",   entries=[(2, proposal_speed_1), ...]),
    "zone_id":    ParameterSchedule(dimension="zone_id",    entries=[(0, proposal_temp_1), ...]),
}
for schedule in experiment_schedule.values():
    schedule.apply(experiment)
```

`ParameterUpdateEvent` (already existing) is the unit of storage: each entry maps to one event, and `get_effective_parameters_for_row()` reconstructs the full effective parameter state at any measurement position. No changes to the data infrastructure are required.

### What This Enables

- Richer datasets per experiment without real-time control requirements
- Discovery of systematic effects: how does filament width vary as speed increases along a pre-planned ramp?
- Compatible with batch fabrication workflows where experiments are submitted as jobs
- Interpretable and reproducible: the schedule is a first-class artifact of the experiment, fixed before execution and inspectable after

### Requirements

| Requirement | Implication |
|-------------|-------------|
| Trajectory configuration | `configure_trajectory(code, dimension_level)` on CalibrationSystem; trust region from existing `configure_adaptation_delta()` |
| Two-phase trajectory optimizer | LHS warm start + `scipy.optimize.minimize` over trigger-step values — mirrors existing exploration logic |
| `ParameterSchedule` type | Sparse, per-dimension wrapper around `(step_index, ParameterProposal)` entries |
| PM training | Each trigger step becomes independent rows via `get_effective_parameters_for_row()` — already supported |

### Risks and Challenges

- **Not truly adaptive:** The schedule is committed before execution; the system cannot react to unexpected outcomes mid-run (unlike Option A). A segment that produces unexpected behavior cannot trigger a course correction. MPC lookahead (`mpc_lookahead_depth > 0`) partially mitigates this by anticipating future trust-region steps offline, but the schedule is still fixed before execution begins.
- **Larger proposal space:** A K=5 segment schedule with 3 runtime parameters gives a 15D proposal space. Standard LHS handles this, but acquisition landscape becomes harder to optimize compared to the single-point case.
- **Correlated training data:** Multiple steps within one trajectory share the same static parameters and are spatially correlated along the dimensional axis. This introduces dependencies that IID PM training assumptions don't account for. The PM's ability to correctly interpret trajectory data depends on whether it has access to sufficient dimensional context (prior feature values within the experiment). If the PM treats each (params, position) row independently, the correlation is absorbed as noise; if it needs sequential context to produce accurate predictions, the PM architecture becomes more demanding.
- **Interpretability:** If a trajectory produces poor performance, attributing responsibility to a specific segment requires analysis. The PM's dimensional-level predictions serve this purpose: per-segment feature estimates enable post-hoc decomposition of experiment outcomes.

### Feasibility Assessment

| Layer | Readiness | Notes |
|-------|-----------|-------|
| Data | Ready | Same infrastructure as Option A; `ParameterUpdateEvent` + `get_effective_parameters_for_row()` |
| PM inference | Ready | Each segment = independent rows; effective params reconstructed correctly |
| Uncertainty | Compatible | NatPN-light works offline; no real-time requirements |
| Acquisition | Minor extension | Existing UCB called per segment; average aggregation — no redesign |
| Schema | Implemented | `runtime_adjustable` flag in place; see [Document 1](01-parameter-levels.md) |
| Safety | More manageable | Schedule can be validated and safety-checked offline before execution |
| New type | `ParameterSchedule` | Lightweight; wraps existing `ParameterProposal` + `record_parameter_update()` |

**Overall: moderate complexity, high compatibility with current architecture. Selected as near-term target.**

---

## Comparison Summary

| Dimension | Current (experiment-level) | Option A (fabrication-time) | Option B (trajectory) |
|-----------|--------------------------|----------------------------|----------------------|
| Info per experiment | 1 config × 1 performance | K segments × M measurements (M≤N, user-defined) | K segments × M measurements (M≤N, user-defined) |
| Adaptivity | None | Maximum (step-by-step) | None (committed schedule) |
| Real-time requirements | None | Hard (step cycle time) | None |
| Incremental learning | Not applicable | Residual model updated per step | Not applicable |
| Uncertainty method | Separate GP surrogate | Must be real-time updatable | Offline sufficient |
| Implementation complexity | Baseline | High | Moderate |
| Safety considerations | Low (single config) | High (rapid online changes) | Medium (offline validation possible) |
| Current infrastructure support | Full | Mostly ready (latency is open question) | Ready (+ ParameterSchedule) |

---

## Relationship to Uncertainty Method

The feasibility of Option A is directly coupled to the uncertainty method chosen in [Document 3](03-uncertainty-estimation.md). Key question: can the uncertainty be computed and updated incrementally, within the fabrication step cycle time?

NatPN-light (the converging candidate from Document 3) can update its KDE incrementally — O(n) per step. However, the PM itself is not updated during the experiment. This means:

- **Option A with NatPN-light:** uncertainty signal is available in real time, and it reflects "how unfamiliar is this parameter config to the PM" — not "what will the PM predict here." The residual model partially addresses this by progressively correcting predictions, but feature estimates in deeply OOD regions remain unreliable.
- **Option B with NatPN-light:** fully compatible. Trajectory proposals are scored offline using KDE uncertainty + PM predictions. No real-time constraints.

This asymmetry is an additional argument for pursuing Option B before Option A.

---

## Recommended Phasing

| Phase | Scope | Complexity | Status |
|-------|-------|-----------|--------|
| **Phase 1** | `runtime_adjustable` flag + schema boundary design | Low | ✅ Done |
| **Phase 2** | CalibrationSystem validation (runtime-only trust regions, completeness check) | Low | ✅ Done |
| **Phase 3** | Option B: `configure_trajectory()` + `ParameterSchedule` + trajectory scoring (SLSQP) | Medium | ✅ Done (superseded) |
| **Phase 4** | NatPN-light: KDE-based uncertainty estimation | Medium | ✅ Done |
| **Phase 5** | Unified step-loop + MPC lookahead: `run_calibration()` with optional `mpc_lookahead_depth` | Medium | ✅ Done |
| **Phase 6** | Option A: fabrication-time residual-driven exploration | High | 🔲 Backlogged |

**Phase 5 — Unified Step-Loop + MPC Lookahead** replaced the Phase 3 SLSQP trajectory
infrastructure with a greedy step-loop that re-uses the same L-BFGS-B optimizer from the
experiment-level path.  The loop iterates over a flattened Cartesian product of configured
dimensions (coarsest-first), fixing parameters mapped to unchanged dimensions at each step.
Online adaptation (previously `run_adaptation`) is now `run_calibration(domain=ONLINE)` —
a degenerate single step.  See `ORCHESTRATION_CONTEXT.md` for the full step-loop design.

**MPC lookahead** (Phase 5 extension) reduces the greedy myopia of the step-loop by
augmenting each step's objective with a short d-step forward rollout:

```
MPC(X) = score(X) + γ¹·score(X₁) + γ²·score(X₂) + … + γᵈ·score(Xᵈ)
```

Each Xⱼ₊₁ is obtained by a quick trust-region L-BFGS-B minimization from Xⱼ.
Usage: `run_calibration(..., mpc_lookahead_depth=d, mpc_discount=γ)`.
Default `mpc_lookahead_depth=0` is pure greedy (no overhead).  The step-loop structure,
return type, and schedule assembly are unchanged — MPC is purely an objective-function
enrichment.  This is the MPC-style approach described earlier in this document as a
future-compatible extension requiring no structural changes.

Key benefits of Phase 5 vs Phase 3:
- One optimizer path (L-BFGS-B) for all use cases — no SLSQP/L-BFGS-B split
- Multi-dimensional parameter mixing (e.g., layer_height→dim_1, speed→dim_2) handled automatically
- Online and offline cases structurally identical — domain selects bounds strategy only
- `target_indices`: collapse full Cartesian grid to a single targeted step for step-specific planning
- MPC lookahead: reduce greedy myopia via `mpc_lookahead_depth` parameter, no structural changes

Phase 6 should remain in the backlog until: (a) hardware latency constraints are quantified for
target fabrication systems, (b) residual model update speed is benchmarked against step cycle
times, and (c) safety bounds are formally specified.
