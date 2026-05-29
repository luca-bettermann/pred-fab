# Exploration Workflow Deep Audit

**Goal**: identify every issue that could cause the symptom "baseline schedule trajectories not contributing to KDE / uncertainty plot, exploration trajectories flat despite batch-aware objective."

**Severity scale**:
- **CRITICAL** — directly causes the reported symptom.
- **HIGH** — bug producing incorrect results in exploration or related code paths.
- **MEDIUM** — logical inconsistency, potential incorrect result under edge cases.
- **LOW** — style, polish, or future maintenance concern.

**Legend for scope**: `[FAB]` = pred-fab library, `[MOCK]` = pred-fab-mock.

---

## Plan

Walk the end-to-end path of a single baseline experiment with a schedule, plus a subsequent exploration round. For each stage, examine the code and identify risks.

Stages:
1. Baseline spec generation (Riesz Phase 2 + 3)
2. Mock baseline step file: spec → exp_data → dataset persistence
3. `run_and_evaluate` → `run_and_record` chain (save-apply-save ordering)
4. `apply_schedules` / `record_parameter_update` semantics
5. Dataset save/load roundtrip of `parameter_updates`
6. Next-session rebuild: dataset reload from disk
7. Datamodule preparation
8. `_fit_kde` (segment weighting, iteration, latent encoding)
9. Uncertainty computation (`uncertainty`, `uncertainty_batch`, `_compute_q_max`)
10. Acquisition / `_ucb_scores` / `_schedule_acquisition_objective`
11. Optimizer behavior (DE, convergence, init population)
12. Plotting path (`plot_acquisition`, uncertainty surface)
13. State consistency (`_n_exp`, bandwidth `h`, `γ`)
14. Virtual KDE pressure / batch weight scales

---

## Findings

### Baseline confirmation (from user diagnostic)

**KDE materializes correctly**:
- baseline_01 → 6 points (speeds 34.9, 37.5, 40.0, 42.5, 44.9, 47.3 at water=0.351), weights 1/18 each.
- baseline_02 → 8 points (speeds 31.2…25.9 at water=0.445), weights 1/24 each.
- baseline_03 → 4 points (speeds 49.8, 52.7, 55.6, 58.3 at water=0.420), weights 1/12 each.
- Total 18 latent points. Per-experiment mass = 1/3. ✓

So `apply_schedules` + `save_experiment` ordering IS working. The KDE sees the full trajectory.

This shifts the investigation away from persistence/_fit_kde and toward:
1. Why does the uncertainty plot look like only initial points contribute? (Visualization / bandwidth)
2. Why does exploration still propose flat schedules despite 18 KDE points + batch-aware UCB?

The second is the critical functional problem.

### FINDING 2 — Boundary factor penalizes schedule spread to range edges (MEDIUM-HIGH) [FAB] — RESOLVED

**Resolution**: Replaced multiplicative `_boundary_factor` with additive `_boundary_evidence` in the evidence model. Boundary evidence is now part of the evidence sum (not a UCB multiplier), so it pushes proposals inward softly without penalizing schedule diversity. Applied consistently across all modes.

**Location**: `system.py:365-398` (`_boundary_factor`) applied per-layer in `_ucb_scores`.

**Mechanism**: multiplicative factor in `(0, 1]` that reduces UCB as a layer approaches a parameter bound (extent = `exploration_radius · _n_decay(n_exp)`, with decay factor `strength=0.5` at the boundary).

For `radius=0.15`, `n=3`: extent ≈ 0.087 (normalized). For a layer at 5% from a bound: boundary factor ≈ 0.67 → **30% UCB reduction for that layer**.

**Impact on Schedule phase**: a diversified schedule spanning the full parameter range necessarily has layers near bounds. Those layers' UCB gets discounted, dragging down the batch's mean UCB. Schedules that stay interior look better by mean UCB. Combined with Finding 1, this creates additional bias toward compact (flat) schedules.

**For single-point acquisition (Process)**: desirable — we want to down-weight near-boundary proposals because extrapolation is unreliable.

**For Schedule**: counterproductive — trajectories legitimately need to sweep the range.

**Severity MEDIUM-HIGH**.

**Fix options**:
- (a) Skip `_boundary_factor` for layers k ≥ 1 in Schedule phase (apply only to step0).
- (b) Use a softer/narrower extent for intra-trajectory layers.
- (c) Move boundary penalty to a SEPARATE additive term that doesn't scale with the UCB value.

### FINDING 3 — KDE bandwidth > typical layer spacing: intra-trajectory layers are KDE-indistinguishable (HIGH) [FAB] — RESOLVED

**Resolution**: Replaced bandwidth-based KDE with fixed-σ evidence model. σ = `exploration_radius · √d` is fixed (does not shrink with data size). Data size is captured by the evidence sum growing, not by bandwidth shrinking. Eliminates the bandwidth-vs-layer-spacing problem.

**Location**: `prediction.py:_compute_model_uncertainty` (and `_compute_model_uncertainty_batch`), bandwidth formula `h = c · √d · _n_decay(n_exp)`.

**Numerical example** (user's latest setup, `radius=0.15`, `n_exp=3`, `d≈1-2` active latent dims):
- **h ≈ 0.086** in normalized [0,1] parameter space.
- Baseline_03 has 4 layer speeds evenly spaced 2.9 mm/s apart → normalized gap ≈ **0.072**.
- **Gap < bandwidth** ⇒ adjacent layers' Gaussian bubbles overlap at `K(gap/h) ≈ 0.7` → they blur into one blob.

**Consequences**:
1. **Plot**: trajectory bars look like single bubbles at the initial-point "centroid of the trajectory." Matches user's observation.
2. **Optimization**: moving layer k within ±h of its neighbors barely changes the KDE density signal. The diversity pressure has no usable gradient over small moves; it only activates when layers jump by >2h (~0.17 normalized = 7 mm/s) apart.
3. **Small trust-region deltas silently disable diversity pressure**. If `delta < h`, the optimizer can't push layers apart enough to feel the pressure.

**Severity HIGH** — this interacts with Findings 1 and 2 to consistently collapse schedules in practice.

**Fix options**:
- (a) Separate, narrower bandwidth for intra-batch virtual interactions (e.g., `h_virt = h / √L` or a fixed small fraction).
- (b) Auto-tune the KDE bandwidth from observed intra-trajectory step distances in training data.
- (c) Scale bandwidth by the LATENT POINT count (18 here) rather than experiment count (3). If density scales with data volume, bubbles should shrink as more data arrives.

Option (c) is principled: bandwidth should reflect information content, not experiment-count-as-unit-of-measurement.

### FINDING 4 — `q` vs `q_max` self-inclusion asymmetry (LOW-MEDIUM) [FAB] — RESOLVED

**Resolution**: Replaced `q/q_max` ratio-based uncertainty with `u = 1/(1+E)` evidence model. No q_max normalization needed; the self-inclusion asymmetry no longer exists. Evidence is a simple sum of kernel values from data points + boundary terms.

**Location**: `prediction.py` — `_compute_q_max` (line ~354) includes self (K(0)=1 diagonal) in the kernel density matrix; `_compute_model_uncertainty` and `_compute_model_uncertainty_batch` exclude self when computing per-query density.

**Effect**: ratio q/q_max is systematically BIASED LOW at query points that would otherwise match a training-data peak. At collapsed batches, this makes u_batch slightly HIGHER (more attractive) than a symmetric convention would. Weakens diversity pressure a small amount, but not the primary driver.

**Severity LOW-MEDIUM** — pre-existing asymmetry that I inherited. Not the root cause of the reported symptoms, but worth fixing for consistency. Fix: `np.fill_diagonal(kernel_matrix, 0.0)` in `_compute_q_max` when treating KDE points.

**Numerical impact** at current parameters: maybe ±0.02 on u_batch values. Negligible compared to Findings 1-3.

### FINDING — Redundant schedule tracking (MEDIUM) [MOCK]

The mock maintains TWO parallel records of schedule state:

1. **`exp.parameter_updates`** on disk (pred-fab canonical source, persisted via `save_experiment`).
2. **`state.schedules`** in journey JSON (mock's session state, passed to plot functions).

This is precisely where the `apply_schedules`-save-ordering bug came from: the mock's plot code looked at `state.schedules` (populated at recording time), but the KDE read `exp.parameter_updates` (populated only in-memory after the disk save). The two representations diverged silently.

**Risk**: future bugs where the two sources of truth desynchronize, leading to plots that show schedules that aren't in the model, or vice versa.

**Proposed fix**: the mock should derive `state.schedules` from `exp.parameter_updates` when rendering plots, or eliminate `state.schedules` entirely and let plotting read from the dataset. All persistence should funnel through pred-fab's save API. Every `state.record(...)` call that takes `schedule=` could become a view over the experiment record.

**Severity MEDIUM** not CRITICAL because the current fix (save-after-apply) closes the immediate hole; but the architectural redundancy is a footgun that will bite again.

### FINDING 5 — `update_perf_range` ignores schedule segments (MEDIUM) [FAB]

**Location**: `system.py:413-463` (`CalibrationSystem.update_perf_range`).

**Bug**: iterates `train_codes` and queries `exp.parameters.get_values_dict()` — the **initial** params only. Schedule segments' effective parameters are never evaluated for the perf range.

**Consequences**:
- `_perf_range_min/max` is computed from initials only. If initials happen to sit near the perf peak, range is narrow (say `[0.4, 0.8]`).
- When `_ucb_scores` normalizes a candidate's predicted perf via this range, candidates at extreme speeds (covered by trajectories but not initials) can normalize to **negative** values or above 1.
- Diverse schedules with layers at extremes drag mean normalized-perf down (possibly below 0). This adds ANOTHER pressure toward collapse: diverse schedules look worse not because they are worse, but because they're normalized against an unfair baseline.

**Fix**: in `update_perf_range`, iterate over all effective configurations per experiment:
```python
for code in train_codes:
    exp = datamodule.dataset.get_experiment(code)
    configs = [exp.parameters.get_values_dict()]
    for event in exp.parameter_updates:
        row = exp._event_start_index(event)
        configs.append(exp.get_effective_parameters_for_row(row))
    for cfg in configs:
        # ... accumulate min/max as before
```

**Severity MEDIUM** — contributes to collapse under steep perf gradients, but not the primary driver.

### FINDING 6 — Silent exception-swallow in `_ucb_scores` perf loop (LOW-MEDIUM) [FAB]

**Location**: `system.py:288-291` and my new `_ucb_scores:305-310`.

Both functions wrap `perf_fn(params_dict)` in `try/except Exception` and fall back to an empty dict, producing `sys_perf = 0`. Silent failures produce a mis-normalized perf of `(-pmin) / span` (often negative). The optimizer then sees a layer as very low-perf and avoids that candidate — incorrect behavior if the true cause is a transient/loggable error.

**Fix**: log exceptions (even if recovery is silent); surface failures during development with a configurable strict mode.

**Severity LOW-MEDIUM** — correctness risk hidden by swallowed errors; unlikely the current issue but a testing liability.

### FINDING 7 — Duplicate smoothing-penalty logic in engine.run vs SolutionSpace (LOW) [FAB]

**Location**: `engine.py:140-152` (inside `engine.run`'s `_objective`) AND `space.py:162-184` (`SolutionSpace.smoothing_penalty`).

Both compute essentially the same formula (`factor = Π (1 − lam · frac)` then `penalty = |energy| · (1 − factor)`). `engine.run` is used by Process phases; `SolutionSpace.smoothing_penalty` is used directly by baseline Schedule Phase 3 and exploration Schedule. Net effect: one smoothing penalty applied, but the logic is duplicated.

**Severity LOW**. Fix: extract a single shared helper to avoid drift.

### FINDING 8 — `calibration._buffer` vs `prediction._base_buffer` name drift (LOW) [FAB] — RESOLVED

**Resolution**: The buffer concept was removed during the evidence model refactor. Boundary handling is now done via `_boundary_evidence` in the evidence model, eliminating the naming inconsistency.

**Severity LOW**.

### FINDING 9 — `integrality_static` parameter typo'd in `engine.run` return path (LOW) [FAB]

`engine.run` accepts `integrality_static` and uses it only for bounds and DE's `integrality` kwarg. No-issue but naming is inconsistent with other `integrality` uses.

**Severity LOW**, cosmetic.

### FINDING 10 — `_n_decay` fallback inconsistency (LOW) [FAB]

`prediction._n_decay` default exponent = 0.5 (so `1/√n`). `calibration._n_decay` fallback uses `1/n^(1/3)` (line 235). In practice `_n_decay_fn` is always set from the prediction system, but if someone ever instantiates `CalibrationSystem` without one, they'll get different decay behavior silently.

**Severity LOW**.

### FINDING 11 — Default popsize can be inadequate for Schedule phase (LOW-MEDIUM) [FAB]

`engine.py` uses `self.de_popsize = 15`. For L=4 and D_sched=1 (total_vars=3 after step0 in vector), scipy DE actual population = 15 × 3 = 45. That's fine. But for larger L or more sched dims, search space blows up quadratically while popsize is fixed — could under-sample.

**Severity LOW**. Mention for future tuning.

### FINDING 12 — Exploration round experiments often end up with EMPTY parameter_updates (MEDIUM) [FAB]

**Cause**: when exploration produces a flat schedule (which the current bugs induce), `record_parameter_update` returns `None` for each "entry" because the delta is zero (same value as previous effective). So the experiment, despite having a schedule in its spec, ends up with `exp.parameter_updates = []`.

In `_fit_kde` that's the process-experiment branch: 1 KDE point at the initial config with weight 1.0.

**Not a bug in isolation** — accurately reflects "flat schedule means no parameter change happened." But it DOES mean the KDE sees exploration rounds as single points, not trajectories. If exploration later produces non-flat schedules, this resolves itself.

**Severity MEDIUM** — feedback loop: flat schedules → fewer KDE points → weaker diversity pressure next round → more flat schedules.

---

## Final prioritized fix list

By combined severity + impact on the reported symptom:

| # | Severity | Finding | Fix complexity |
|---|---|---|---|
| 1 | **CRITICAL** | Smoothing penalty dominates batch-aware diversity | Low — make `schedule_smoothing=0` for exploration, or reformulate as additive (not `abs(energy)·(...)`) |
| 3 | **HIGH** | ~~KDE bandwidth >> intra-trajectory spacing~~ | RESOLVED — fixed-σ evidence model |
| 2 | **MEDIUM-HIGH** | ~~Boundary factor shrinks spread-to-edge UCB~~ | RESOLVED — additive boundary evidence |
| 0 | **MEDIUM** | Step0 not fixed from Process in exploration | Low — pass `step0_values=` to sched_space |
| 5 | **MEDIUM** | `update_perf_range` misses schedule segments | Low — iterate effective params per segment |
| 4 | **MEDIUM** | ~~q vs q_max self-inclusion asymmetry~~ | RESOLVED — u=1/(1+E) evidence model, no q_max |
| 12 | **MEDIUM** | Flat schedules erase parameter_updates entries (feedback loop) | N/A — resolves once 1-3 are fixed |
| [mock] | **MEDIUM** | Redundant `state.schedules` vs `exp.parameter_updates` | Medium — refactor mock's plot data path |
| 6 | **LOW-MEDIUM** | Silent exception swallow in perf loop | Low — log exceptions |
| 7-11 | **LOW** | Cosmetic / hygiene (8 resolved) | Low — cleanups |

### Recommended immediate sequence to restore diverse exploration schedules

1. **Fix 1** (disable or weaken smoothing for exploration Schedule).
2. **Fix 2** (skip boundary factor for non-step0 layers in Schedule).
3. **Fix 4** (exclude self in q_max).
4. **Fix 3** (narrower virtual bandwidth).
5. **Fix 0** (properly fix step0 from Process).
6. **Fix 5** (include schedule segments in perf range).

After 1-3, exploration should produce diverse schedules even against strong perf peaks. Fixes 4-5 are secondary refinements that improve the signal quality.

If you want me to implement these, say the word and I'll plan each one carefully (per the discuss-before-implementing rule).


### FINDING 0 — Exploration Schedule does NOT fix step0 from Process (MEDIUM) [FAB]

**Location**: `system.py:1475-1486` (exploration `sched_space` construction).

**Issue**: `SolutionSpace(..., step0_values=None)` — step0 is NOT passed, so it ends up in the decision vector and gets re-optimized in the Schedule DE. Contrast with baseline `_phase3_schedule:1070-1082` which correctly passes `step0_values=step0_norms`.

**Consequences**:
1. Process-phase optimum for step0 is **discarded** — Schedule re-finds it under a different (mean-over-L, smoothed) objective.
2. Decision vector is L variables instead of L−1 offsets (larger search space).
3. Returned spec's `initial_params[sched_code]` = Schedule's step0, not Process's step0. So the "Phase 2 result" the mock logs and plots is silently overwritten.
4. Breaks the documented architecture ("Initial positions fixed from Process. Delta constraints between adjacent layers enforced via offset bounds.").

**Severity MEDIUM** — functional but wrong, inflates compute, breaks the design contract.

**Fix**: extract step0 from `flat_x`/`opt_p2.best_x` → normalize → pass as `step0_values` argument.

### FINDING 1 — Smoothing penalty actively punishes schedule diversity (CRITICAL) [FAB]

**Location**: `calibration/space.py:162-184` (`SolutionSpace.smoothing_penalty`) and `calibration/engine.py:319-336` (`_schedule_smoothing_factor`).

**Mechanism**:
```python
# _schedule_smoothing_factor: factor is maximized at flat trajectory
factor = 1.0
for k in 1..L:
    for d in sched_dims:
        change = abs(sv[k, d] - sv[k-1, d])
        frac = min(change / deltas[d], 1.0)
        factor *= (1.0 - lam * frac)
# penalty = abs(base_energy) * (1 - factor)  (added to minimization objective)
```

At **flat trajectory** (change=0 everywhere): factor = 1.0 → penalty = 0. At **maximum-diversity trajectory** (change=delta each step): factor shrinks multiplicatively; penalty scales to `abs(base_energy) * (1 − (1-lam)^(L-1))`.

**Numerical example** (default `schedule_smoothing=0.05`, L=4, kappa=0.5, strong perf peak):
- Flat trajectory UCB avg ≈ 0.565 → obj = −0.565 + 0 = **−0.565**.
- Fully diverse UCB avg ≈ 0.6 → factor ≈ 0.95³ = 0.857 → penalty ≈ 0.6 × 0.143 ≈ 0.086 → obj = −0.6 + 0.086 = **−0.514**.

DE minimizes → flat wins by 0.05. The mild batch-aware uncertainty bump doesn't compensate for the ~10% base_energy-scaled penalty that kicks in whenever layers move apart.

**Why baseline Riesz didn't suffer from this**: Riesz pairwise energy ~1/d^p explodes at small d, drowning out smoothing. Acquisition UCB values are O(1); penalty at ~10% of |UCB| is competitive with diversity benefit.

**Severity CRITICAL**: this is likely the primary cause of flat exploration schedules. The batch-aware uncertainty IS creating diversity pressure, but the smoothing penalty is stronger.

**Fix options**:
- (a) Disable smoothing for exploration mode (simple, loses smooth-trajectory preference for adaptation).
- (b) Scale smoothing by a fixed constant rather than `abs(base_energy)`, making the penalty independent of objective magnitude. E.g., `penalty = lam * sum_of_normalized_jumps`.
- (c) Reduce default `schedule_smoothing` to something like 0.005 (10× smaller), so penalty < diversity bump.
- (d) Move smoothing from "additive penalty" to "soft bound": allow large jumps but with elastic cost, not multiplicative-factor shrinkage.

Recommendation: (b) or (c). Keep for adaptation mode (where physical smoothness matters), tune down or reformulate for baseline/exploration.


- `_schedule_acquisition_objective` — is the math right?
- `SolutionSpace.decode` for Schedule phase (offsets, trust regions, bounds)
- `SolutionSpace.smoothing_penalty` — is this PUNISHING diversity?
- `_ucb_scores` — aggregation, sign convention
- `uncertainty_batch` — density math, q_max, ratio
- Batch-aware virtual weight: is `1/(L*(N+1))` actually right or tiny?
- DE init population for the Schedule phase — is it biased to flat?
- `_boundary_factor` — does it over-penalize schedules near perf peaks?


