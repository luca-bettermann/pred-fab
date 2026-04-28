# PFAB Implementation Plan — Open Topics

**Living document.** Updated 2026-04-28. Single source of truth for the open work after the perf-optimisation + architectural-refactor sprint. Per-topic detail inlined below; standalone topic plans (`OPTION_A_*.md`, `PHASE_C_PLAN.md`) merged in and deleted.

## Chosen path (2026-04-28)

**Strategy A immediately, Strategy D as the long-term destination.** Get the perf win from topic 1 (KDE vectorisation in numpy) first to address the 20s/schedule pain. Re-profile to validate. Then proceed through the architectural cleanups (topics 2, 3) and ultimately the full torch migration (Strategy D — including replacing scipy DE with `torch.optim`, dropping pandas everywhere, and the `nn.Module` normalisers / `nn.Embedding` categoricals).

Topic 1's numpy vectorisation is *not thrown away* by the eventual torch migration — the algorithmic structure (batched dense distance matrix, broadcast density, no `cKDTree`) is the same numpy↔torch. The torch port (Option A commit 4) will be a mechanical rewrite of `np.exp` → `torch.exp` etc.

Sequencing committed:
1. **Topic 1: KDE vectorisation (numpy)** — next.
2. Re-profile, validate schedule iteration tractability.
3. **Topics 2 + 3** — calibration unification + Domain phase commit 1.5.
4. **Merge `feat/integrated-evidence` → `main`.** Strategy A complete.
5. **Branch off `main` (`feat/full-torch` or similar).** Strategy D execution begins on the new branch — Option A commits 1-5 + remaining torch-native data layer + scipy.qmc → torch.quasirandom + plotting boundary tightening.

---

## Status snapshot

### What's landed this session

| Commit | Repo | What |
|---|---|---|
| `1363793` | pred-fab | A.2 layers 1-3: batched autoreg + vectorised DE |
| `3d6030b` | pred-fab | Layer 4 lock-in: gather + tensor predictions stack |
| `9bb69bb`, `3869a13` | pred-fab | `torch.compile` default-on with probe-fallback + `smart_maxiter` consolidation |
| `3a2cb96` | pred-fab | Skip recursive features in `get_populated_experiment_codes` (fixed empty-train bug) |
| `5940169` | pred-fab | Eval batching (`compute_performance_batched` + `TARGETS_CONSTANT` fast path) |
| `c57703d` | pred-fab | Profiler + hot-path instrumentation (`PFAB_PROFILE=1`) |
| `a1eaeb4` | pred-fab | KDE batched: `delta_integrated_evidence_batched` (cached `E_old` + batched encoder) — **1.7×** on KDE |
| `424388c` | pred-fab | `_run_phase` helper + Domain → Process split for single-point acq (commit 1 of unification) |
| `39d3c86`, `11adc4e` | pred-fab-mock | Shrink DevMLP `(48,24,12) → (24,12)`; mock evaluators opt into `TARGETS_CONSTANT` |

### Latest profile (smoke test, post Domain split)

```
exploration step (κ=0.5):  0.07s   (down from 0.13s baseline — ~50% cumulative)

engine._run_de                                        68.54ms   100%
acq._acquisition_objective_vectorized                 66.63ms    97%
  acq.delta_integrated_evidence_batched [KDE batch S] 65.63ms    96%   ← still dominant
  calibration.array_to_params [decode S]               0.76ms     1%
  calibration.perf_fn_batched [predict + eval]         0.11ms     0.2%
    predict.predict_for_calibration_batched            0.07ms     0.1%
```

**Bottom line:** prediction + eval is now ~0.2ms/gen (negligible). KDE is **95.7%** of acquisition wall time. The Domain split of commit 1 didn't engage in exploration because `DataModule.input_columns` doesn't include domain axes (they're not declared as `IPredictionModel.input_parameters`).

---

## Open topics — priority-ordered

### 1. True KDE vectorisation 🔴 highest perf leverage

**Status:** ready to implement. No prerequisites.

**Effort:** ~3-4 hours, ~250 LOC.

**Risk:** medium — `KernelIndex.density_at` semantics (truncation, exclude-self) need preservation through the vectorised path. Equivalence tests at multiple σ + n_kernels combinations are mandatory.

#### Problem

`integrated_evidence` still loops over S candidates inside `delta_integrated_evidence_batched`. The cache fix (commit `a1eaeb4`) saved the redundant `E_old` work and the encoder forward; the inner `self_integral` over `n_kernels × n_probes` still runs S times.

#### Fix

Build the `(S, n_kernels, n_probes)` density tensor in one numpy reduction. The `KernelFieldEstimator.self_integral` becomes a vectorised tensor op across the candidate batch dim.

#### Touches

- `pred_fab/orchestration/evidence.py`:
  - `KernelIndex.density_at` → batched query support: `(M, D)` query × `(N, D)` centres → `(M,)` densities now; need to support a `centres_batch: (S, N+1, D)` input where each `s` slice is a *different* "old + new candidate" set, returning `(S, M)` densities.
  - `KernelFieldEstimator.self_integral` → batched variant accepting `(S, n_kernels, ...)` perturbation per candidate, returning `(S,)` integrals.
- `pred_fab/orchestration/prediction.py`:
  - `delta_integrated_evidence_batched` inner loop replaced by single call to the new batched `integrated_evidence`.

#### Mechanics sketch

For each KDE:
1. `index_old` and `E_old` already cached (from `a1eaeb4`).
2. Per old centre `j`: probes_j = `offsets + center_j` (M probes, M independent of S).
3. Old density at probes_j (sum of all old kernels excluding j) — **same across all S candidates, compute once**.
4. New density contribution per candidate: `density_s_j[m] = exp(-||probes_j[m] - new_center_s||² / 2σ²)` — `(S, n_old, M)` tensor in one broadcast.
5. Total density `(S, n_old, M)`: `old_density_at_probes_j[m] + density_s_j[m] * new_weight`.
6. Per-old-centre self-integral `(S, n_old)`: `Σ_m weights[m] * 1/(1+total_density) * in_domain[m]`.
7. New centre's own self-integral per candidate: probes_s = `offsets + new_center_s` (S, M, D); old density at probes_s — `(S, M)` from broadcast against all old centres; new self-density × new_weight added; same reduction.
8. `E_new[s] = Σ_old_centres weights * self_integral_old[s, j] + new_weight * self_integral_new_center[s]`.

#### Expected impact

- 5-10× on KDE path → ~5-10× on exploration step (KDE is 95.7%).
- For schedule iterations (higher D, more kernels, larger n_probes due to `S^{D-1}` shell directions), the multiplier is larger — could reach 20-30× on the schedule case.
- Combined with everything already shipped: estimated 3-5s exploration step time on real (non-smoke) workloads, **schedule iteration ~1-3s instead of 20s**.

#### Tests

- Equivalence at `atol=rtol=1e-4` (looser than `1e-5` because matmul order across batch dim changes float-summation order) against the existing `delta_integrated_evidence_batched` for 3 fixture cases: smoke-scale (3 kernels, D=4), schedule-scale (10 kernels, D=10), edge-case (1 kernel, D=2).
- Smoke test profile post-fix: confirm KDE drops to non-dominant.

---

### 2. Calibration flow unification (commits 2-3) 🟡 structural cleanup

**Status:** commit 1 landed (`424388c`). Commits 2-3 pending.

**Effort:** ~3-4 hours total across the two commits.

**Risk:** medium-high — `run_baseline` is the most-tested part of the calibration system. Behavioural drift would be a real regression; equivalence tests must run before/after each commit.

#### Problem

Two duplicated phase-orchestration mechanisms: `run_baseline` (joint-batch, schema-only DM) and `run_calibration` single-step (trained DM). Commit 1 introduced `_run_phase` for the single-step path; commits 2 and 3 absorb the duplication.

#### Commit 2 — refactor `run_baseline` to use `_run_phase`

- Generalise `_run_phase` to accept an `N` parameter:
  - `N == 1`: existing single-point path (uses `_acquisition_objective_vectorized`).
  - `N > 1`: joint-batch path (uses `_acquisition_batch_objective` already in `_run_acquisition_phase`).
- Replace `_run_acquisition_phase`'s body with a thin wrapper around `_run_phase` (or delete entirely).
- `run_baseline`'s Domain → Process orchestration shrinks to two `_run_phase` calls — same shape as the single-point path.
- **Behavioural equivalence test:** `run_baseline(N=5)` pre-commit produces matching proposals within DE-noise tolerance (~1e-3 on parameter values).

#### Commit 3 — unify under `run_calibration(mode, ...)`

- Single entry point: `run_calibration(mode=BASELINE|EXPLORATION|INFERENCE|ADAPT, n_proposals=N, kappa=κ, ...)`.
- `agent.baseline_step` becomes a thin shell: `run_calibration(mode=BASELINE, n_proposals=N, kappa=1.0)`.
- Public API stays backward-compatible (`baseline_step`, `exploration_step`, `inference_step` still callable).

#### Net LOC

| | Delta |
|---|---|
| Commit 1 (already shipped) | +196 |
| Commit 2 (`run_baseline` → `_run_phase`) | ~−175 |
| Commit 3 (unified entry point) | ~−80 |
| **Cumulative** | **~−60** (smaller than my original "~−150" estimate; main value is structural, not LOC) |

#### Expected impact

- Structural: one phase-orchestration mechanism instead of two; bug fixes propagate everywhere; future per-phase optimisations touch one place.
- No direct perf impact.

---

### 3. Domain phase for exploration (commit 1.5) 🟡 architectural consistency

**Status:** specified, not implemented.

**Effort:** ~1-2 hours, ~150 LOC.

**Risk:** low-medium — outer DE has a small search space (D=1-2), so even O(D²) DE budget is ~80-160 evals. Each outer eval is one `predict_for_calibration_batched` call (≤0.2ms).

#### Problem

Commit 1's Domain split only engages when domain axes are in `datamodule.input_columns`. They aren't — `IPredictionModel.input_parameters` doesn't declare `n_layers`/`n_segments` (they determine *tensor shape*, not per-cell input). So the split is structurally correct but inert in current schemas.

#### Fix

Implement an outer optimisation over schema-level `DataDomainAxis` params not in `input_columns`. Each outer candidate triggers a `predict_for_calibration` call at that structural choice; per-call cost scales with `n_layers × n_segments`. Inner Process phase (existing `_run_phase`) optimises continuous params with the chosen structural values fixed.

#### Mechanism

1. Identify `DataDomainAxis` codes not in `datamodule.input_columns` (and not in `fixed_params`).
2. Outer DE on those (small D, typically 1-2):
   - Each candidate calls `predict_for_calibration_batched` with an instantiated sample at the candidate's structural choice + default process params.
   - Evaluate κ-weighted acquisition (perf + evidence).
3. Pass winning structural values as `fixed_for_step` into the existing `_run_phase` Process call.

#### Expected impact

- Engages the Domain → Process split for real-world exploration runs.
- Reduces total acquisition cost when domain axes have non-trivial bounds (e.g. `n_layers ∈ [2, 12]`).
- Mirrors baseline's structure properly.

---

### 4. Differentiable acquisition (Option A) 🟢 biggest structural lever

**Status:** plan complete, ready to start.

**Effort:** ~6 days, ~1,060 LOC across 6 commits.

**Risk:** medium overall, distributed across commits. Multi-start mitigates local-minima risk. KDE rewrite (commit 4 below) is the hardest; finite-difference fallback exists.

#### Problem

DE evaluates O(D²) candidates per acquisition under `smart_maxiter`. At D=10 schedule that's ~12,000 evals. Gradient-based optimisation needs ~50-200 evals total.

#### Goal

Replace differential evolution with gradient-based acquisition optimisation (Adam or `torch.optim.LBFGS`). Expected: **30-100× fewer acquisition evaluations** at the optimisation level.

#### Expected outcome

| Lever | Before | After |
|---|---|---|
| Acquisition evals (D=4) | 1,920 | **~50-100** (Adam ~50 epochs, L-BFGS ~10-20 line searches) |
| Acquisition evals (D=10) | ~12,000 | **~100-200** (gradient scales linearly with D, not quadratically) |
| Per-eval cost | unchanged | unchanged |
| Total per-acq wall time | 6-20s | **0.2-0.6s** |

Combined with what landed (A.1, A.2, Layer 4, eval batching, KDE batching): **~50-100× total exploration speedup** vs pre-A.2 baseline. 20s/iteration → 0.2-0.4s/iteration.

#### What needs to be true for gradients to flow

```
params (tensor, leaf, requires_grad=True)
   │
   │  params_to_array_tensor(params_dict)   ← Commit 1
   ▼
x_norm: torch.Tensor, shape (D,)
   │
   │  predict_for_calibration_tensor(x_norm)   ← Commit 2
   ▼
predictions: dict[str, torch.Tensor]
   │
   │  evaluate_features_batched_tensor(predictions)   ← Commit 3
   ▼
perf: torch.Tensor, shape ()
   │
   │  delta_integrated_evidence_tensor(x_norm)   ← Commit 4
   ▼
evidence: torch.Tensor, shape ()
   │
   ▼
score: torch.Tensor, shape ()  → backward() through entire chain
```

DE remains as fallback throughout the migration (parallel tensor-native path).

#### 6 commits

**Commit 1: Tensor-native `params_to_tensor` / `tensor_to_params`** (~150 LOC, 0.5 day)
- Mirror of `params_to_array` but tensor-typed. Round-trip + gradient-flow tests.
- Categorical handling: keep one-hot internally (embedding migration is additive in topic 5).

**Commit 2: Tensor-native `predict_for_calibration_tensor`** (~200 LOC, 1 day)
- Returns tensors at the API boundary (no numpy conversion). The autoreg loop already runs in tensor land post-Layer 4 — only change is dropping the final `stack_np = stack.detach().cpu().numpy()`.
- Conditional `gradient_pass: bool` kwarg on `model.forward_pass` to skip the no_grad context for the gradient path; default stays no-grad for inference.

**Commit 3: Tensor-native eval models** (~150 LOC + ~30 LOC mock overrides, 0.5 day)
- `IEvaluationModel.compute_performance_tensor` — tensor variant returning `(S,)`.
- Mock overrides: ~10 LOC each using `torch.clamp` (differentiable except at boundaries; flat gradients only when score is saturated, which is correct).

**Commit 4: Tensor-native KDE evidence** (~250 LOC, 2 days) — **the hard one**
- Path 4A (preferred): rewrite KernelField + KernelIndex in torch. Sum of Gaussian probes evaluated at QMC points → `evidence(x) = -0.5 * ((points - x_centres) / sigma)**2 - log(...)`. **Side benefit: composes with topic 1 (KDE vectorisation) — the same restructure unlocks both.**
- Path 4B (fallback): leave KDE numpy/scipy, compute its gradient via finite differences `dE/dx ≈ (E(x+ε) - E(x-ε)) / (2ε)` per dim. ~30 LOC. Adds 2D extra evidence evals per gradient step; less accurate; doesn't compose with second-order optimisers.
- `cKDTree` truncation: replaced by dense distance broadcast `(K, D)` vs `(M, D)` → `(M, K)` distance matrix masked at 5σ. Faster than tree-build at our typical N≤50.

**Commit 5: Gradient optimiser + multi-start** (~300 LOC, 1.5 days)
- `OptimizationEngine.run_acquisition_gradient(objective_tensor, bounds, n_starts=4)` — multi-start Adam or `torch.optim.LBFGS`. ~50 epochs Adam or ~10-20 line searches L-BFGS per start.
- `Optimizer.GRADIENT` enum value (alongside `LBFGSB`, `DE`).
- Bounds via sigmoid reparameterisation: `x = sigmoid(z) * (hi - lo) + lo`. Smooth gradient; box constraints respected.
- Routed through `_run_phase` (commit 1 of unification) when `chosen_opt == Optimizer.GRADIENT` and tensor APIs are available.

**Commit 6 (optional): Flip default to gradient** (~10 LOC, 0.5 day)
- Default optimiser becomes `Optimizer.GRADIENT`. DE remains via `agent.configure_optimizer(backend=Optimizer.DE)`.

#### Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Gradient gets stuck in local minima where DE escaped | Medium | `n_starts=4-8` multi-start; DE remains fallback |
| `torch.compile` × autograd surfaces a bug | Low | `COMPILE = False` for gradient path until validated |
| KDE tensor rewrite drifts numerically at higher D | Low-Med | Equivalence tests; finite-diff fallback (Path 4B) |
| Eval clamp saturation produces flat gradients | Medium | Soft-clamp via sigmoid if pathologies appear |
| Categoricals don't differentiate well through argmax | Low | At our scale, mostly continuous params. Defer to topic 5 if needed |

#### What stays unchanged

- `PfabAgent.exploration_step`, `acquisition_step`, `inference_step` — internal swap, invisible to callers.
- Schema layer (`DataObject`, `DataCategorical`, `DataFeature`).
- Dataset persistence layer.
- Phase C is *not* a prerequisite — can land in parallel.
- `agent.predict_performance(params)` keeps numpy contract; tensor entry point is internal.

---

### 5. Torch-native data layer 🟢 incremental simplification track

**Status:** documented in `knowledge-base/PFAB - Design Decisions.md`. Most of it absorbs into Option A commits 1-3 if we go that route.

**Effort:** ~5 days standalone; mostly absorbed by Option A.

#### Sub-topics (priority order, per the DD entry)

**5a. One-hot → categorical-index tensors** (long-standing pain point)
- Today: `_col_extraction` plan built at `initialize()`, applied via DataFrame ops in `_one_hot_encode`, reversed via `_decode_one_hot`, threaded through `params_to_array` / `array_to_params`. Three layers of mapping.
- Replacement: keep categorical columns as integer tensors at the framework boundary; let models that need one-hot apply `F.one_hot` themselves; let models that prefer learnable representations use `nn.Embedding`.
- Calibration array becomes flat in original units, no stride math.
- ~80 LOC of `_col_extraction` / `_decode_one_hot` / one-hot-aware calibration logic → ~15 LOC.

**5b. Normalisation as `nn.Module`**
- `StandardScalerModule`, `MinMaxModule`, `QuantileScalerModule` (one per `normalize_strategy`).
- Stats live in `state_dict()`, serialise for free via `torch.save`, `forward()` is the apply path, `inverse()` is denormalise.
- Replaces `_apply_normalization` / `_reverse_normalization` / `_compute_normalization_stats` / `get_normalization_state` / `set_normalization_state` (~150 LOC) with ~30 LOC.
- GPU-correctness for free.

**5c. Tensor-native `prepare_input`**
- Drop the DataFrame intermediate. Today: `list[dict] → pd.DataFrame → _one_hot_encode → numpy → torch`.
- Replacement: caller passes `dict[code, torch.Tensor]`; `prepare_input` is a tensor stack + normalisation apply. Predicates on 5a having landed.
- Per-acquisition `pd.DataFrame(all_rows)` round-trip vanishes.

#### Sequencing

Strict order: **5a → 5b → 5c**. Each refactor depends on the prior loosely (the contract changes accumulate). Each lands with its own equivalence tests on the existing 586+ tests. No bundle commits.

---

### 6. Phase C: per-minibatch scheduled sampling 🟢 training-time cleanup

**Status:** detailed plan complete, paused. Resume when speedup work has landed.

**Effort:** ~1-2 days.

**Risk:** medium — SS contract preserved at per-epoch granularity instead of per-K-round.

#### Goal

Replace the K-refit SS loop in `PredictionSystem.train()` and the DataModule SS-state machinery with a per-epoch SS hook inside `TorchMLPModel.train()`. Net structural cleanup ~120 LOC; no behavioural regression on the SS contract.

**No exploration-time speedup** — training-time only (~4× faster training of recursive models, typically a few seconds per `agent.train()` call).

#### Files

##### `src/pred_fab/core/datamodule.py`

Delete:
- `__init__` lines 60-64: `self._ss_predictions_by_exp`, `self._ss_p_student`, `self._ss_rng` (3 attrs).
- `set_scheduled_sampling_state(...)` (~22 LOC).
- `_perturb_recursive_features(...)` (~71 LOC).
- The perturbation call site in `get_batches` (~6 LOC).

Add:
- `get_train_row_metadata(split=SplitType.TRAIN) -> list[tuple[str, tuple[int, ...]]]`. Per-row `(exp_code, cell_idx)`; row order matches `get_batches(split)` exactly.

##### `src/pred_fab/orchestration/prediction.py`

Delete:
- `n_ss_rounds: int = 4` attribute.
- The K-refit branch in `train()` (lines ~281-299).
- `_ss_p_for_round(round_idx, n_rounds)` (~6 LOC).

Keep:
- `_autoreg_predict_training_data()` — repurposed for cross-model student value pre-computation only (called once per consumer-model from `_build_ss_config`, not per round).
- `_topo_sort_models()`, `_model_has_recursive_inputs()`, `_filter_batches_for_model()`.

Replace the K-refit branch with single fit + `ss_config` build:

```python
for model in ordered_models:
    has_recursive = self._model_has_recursive_inputs(model)
    train_batches = self.datamodule.get_batches(SplitType.TRAIN)
    ss_config = self._build_ss_config(model) if has_recursive else None
    self._fit_single_round(model, train_batches, val_batches, ss_config=ss_config, **kwargs)
```

Add `_build_ss_config(model) -> dict | None` (~80 LOC). Returns:
- `rec_input_cols: list[int]` — column indices in X_train per recursive input.
- `rec_source_outputs: list[int | None]` — same-model output col index, or `None` for cross-model.
- `prior_row_indices: list[torch.Tensor]` — per-spec int64 tensor of length n_rows; value = prior-cell row index, `-1` for boundary.
- `cross_model_values: list[torch.Tensor | None]` — pre-computed cross-model student values in input-normalised space.
- `affine_a`, `affine_b`: per-spec affine transform `student_input_norm = a * y_teacher_norm + b` (composes denorm-output with norm-input). Pre-computed once.
- `boundary_input_norm: list[float]` — per-spec normalised 0 (matches NaN→0→normalize semantics).
- `p_floor: float`, `seed: int`.

##### `src/pred_fab/models/torch_mlp.py`

`train()` accepts `ss_config: dict | None = None`. Per-epoch logic:

```python
for epoch in range(EPOCHS):
    if ss_config is not None:
        p_student = p_floor + (1 - p_floor) * (epoch / max(EPOCHS - 1, 1))
        X_used = X.clone()
        if p_student > 0:
            with torch.no_grad():
                y_teacher = net(X)  # (n_rows, n_outputs)
            for r_idx, rec_col in enumerate(rec_cols):
                # Per recursive input: gather + affine + Bernoulli mask + scatter
                ...
    else:
        X_used = X
    optimizer.zero_grad()
    loss = loss_fn(net(X_used), y)
    loss.backward()
    optimizer.step()
```

##### Tests

- Delete `tests/core/test_datamodule_scheduled_sampling.py` (190 LOC, exercises removed code).
- Update `tests/orchestration/test_scheduled_sampling_train.py`: drop `_ss_p_for_round` tests, replace with per-epoch schedule tests.
- Add `tests/models/test_torch_mlp_ss.py` (~120 LOC): no-config equivalence, p_floor=0 epoch-0 no-perturb, p_floor=1 full-substitute, boundary handling, cross-model path.

##### `src/pred_fab/orchestration/agent.py`

`configure_scheduled_sampling(n_rounds, schedule_floor)` — drop `n_rounds` argument (no more rounds).

#### Net diff

| | LOC delta |
|---|---|
| `datamodule.py` | **−75** |
| `prediction.py` | **+55** |
| `torch_mlp.py` | **+50** |
| `agent.py` | **−3** |
| Tests delete + add | **−90** |
| **Net** | **~−63 LOC** |

#### Risk register

- **Cross-model SS semantics drift:** today's K-refit refreshes cross-model predictions every round. Phase C makes them static (computed once before consumer training starts). Why fine: source model is already topo-frozen; refresh adds no information.
- **Affine transform precision:** `(a, b)` in float64 at construction; `student = a * y_teacher + b` in float32. Within 1e-5 tolerance.
- **Per-epoch refresh cost:** EPOCHS=1500 × `with torch.no_grad(): net(X)` on small batch ≈ ms-scale.
- **`configure_scheduled_sampling(n_rounds=...)` callers:** grep + drop or keep as deprecated no-op.

#### Validation plan

1. Existing 586 tests + 3 batched-equivalence tests stay green.
2. Add 5 new TorchMLPModel SS tests; confirm green.
3. Smoke test exploration step time stays at post-Layer 4 baseline (Phase C should not move it).

---

### 7. DE budget tuning ⚪ tactical, deferred

**Status:** deferred — eclipsed by topic 1.

**Effort:** ~1 day.

#### Description

`smart_maxiter = min(max(40, 15D), de_maxiter)` is a round-number heuristic. Could tighten `de_no_improve_window` (currently 10 generations) and reduce `de_popsize` (8) based on convergence diagnostics.

#### Approach

Profile-driven tuning: collect `convergence_history` across N runs, find the point where best-so-far stalls, reduce `no_improve_window` accordingly. Reduce `de_popsize` if convergence is consistent at smaller pop.

#### Expected impact

1.3-1.8× on acquisition wall time. Small absolute win once topic 1 lands and per-call cost is no longer KDE-bound.

---

### 8. Cell-loop vectorisation ⚪ deprioritised by profile

**Status:** deferred indefinitely.

**Original concern:** the `for cell_row in range(n_cells)` Python loop in `_predict_autoregressive_batched` was estimated at ~35-40% of step time.

**Profile reality:** cell-loop is **<0.2%** of acquisition time post-Layer 4 + KDE batching. Dwarfed by KDE.

Reconsider only if KDE vectorisation lands and a new bottleneck emerges in this area.

---

## Sequencing — three coherent strategies to pick from

### Strategy A — Perf-first (recommended for hitting 5× target)

```
1. KDE vectorisation (topic 1)        — 4 hours, biggest single win
2. Re-profile + validate on schedule  — 30 min
3. If 5× target met:
     → Calibration unification (topic 2) commits 2-3
     → Domain phase commit 1.5 (topic 3)
   If not met:
     → Differentiable acquisition (topic 4) — ~6 days
4. Phase C (topic 6) — orthogonal, when convenient
```

**Expected outcome:** topic 1 alone likely brings exploration into "tractable" range. The next steps depend on whether the profile shows additional bottlenecks after KDE is fixed.

### Strategy B — Architecture-first (recommended if perf is "good enough" already)

```
1. Calibration unification (topic 2) commits 2-3       — 4 hours
2. Domain phase commit 1.5 (topic 3)                   — 2 hours
3. Torch-native data layer (topic 5) — 5a → 5b → 5c    — 5 days
4. Phase C (topic 6)                                   — 1-2 days
5. Then perf passes: KDE (topic 1) + topic 4 if needed
```

**Expected outcome:** clean codebase, smaller diffs in future perf work, but the user's perceived 20s/iteration pain isn't addressed first.

### Strategy C — Big bet (skip incremental, go to differentiable)

```
1. Differentiable acquisition (topic 4) — 6 days
   • Commits 1-3 absorb ~80% of topic 5 (torch-native data layer)
   • Commit 4 absorbs ~80% of topic 1 (KDE vectorisation in torch)
2. KDE numpy vectorisation (topic 1) is no longer needed if commit 4 lands
3. Calibration unification + Domain phase as a follow-up cleanup
```

**Expected outcome:** 50-100× exploration speedup, GPU-ready codebase, all the structural simplifications. Highest leverage but largest single bet.

### Strategy D — Full torch endgame (the maximalist version)

Strategy C taken to its endpoint: eliminate **all** pandas DataFrames and **all** numpy intermediate arrays inside the framework. After migration, only two thin numpy "leaf" sites remain — both at I/O / display edges.

```
1. Strategy C as written (6 days for Option A)             — covers ~80%
2. Topic 5a/5b/5c remaining bits not covered by Option A   — 1 day
3. Eliminate pandas in dataset.export_to_dataframe         — 0.5 day
   → tensor dict export, no DataFrame intermediate
4. Eliminate pandas in data_blocks.set_values_from_df      — 0.5 day
   → tensor-typed setter
5. Replace scipy.stats.qmc.Sobol → torch.quasirandom       — 0.5 day
6. Migrate residual numpy intermediate arrays in           — 1 day
   calibration / evidence math
7. Tighten plotting boundary (numpy at matplotlib edge)    — 0.5 day
8. Phase C (topic 6)                                       — 1-2 days
9. Calibration unification (topic 2) + Domain phase (3)    — 1 day
─────────────────────────────────────────────────────────────────────
Total: ~14 days
```

**End-state architecture:**

| Layer | Representation |
|---|---|
| Schema (`DatasetSchema`, `DataObject`) | Python objects (unchanged) |
| Persistence (JSON/CSV files) | Files (unchanged) |
| Persistence ↔ memory boundary | numpy → tensor at load time, tensor → list/dict at save time (~10 LOC) |
| Dataset / DataModule / Prediction / Evaluation / Calibration / Evidence | **torch.Tensor everywhere** |
| Optimisation | `torch.optim.LBFGS` / Adam (no scipy) |
| Plotting | tensor → numpy at matplotlib edge (~5 callsites) |

**What's gained:**
- Single mental model (tensors everywhere; no "is this numpy or torch?" lookups).
- GPU support: `agent.to('cuda')` works end-to-end.
- `torch.compile` over the whole inference path (not just the model).
- End-to-end autograd → differentiable acquisition is free.
- No DataFrame round-trips anywhere; faster prepare_input.
- Simpler serialisation (`state_dict` for normalisers, models, KDE).

**What's given up:**
- Some debugging convenience (numpy is easier to inspect in pdb than torch tensors — but `t.numpy()` is one call away).
- Dependency footprint stays the same (torch is already primary); pandas could be dropped from `pyproject.toml` after migration.

**When this makes sense:** if the perf target requires the full migration (5× exploration not enough; want 50-100×), AND we want to commit to torch-as-the-backend strategically (already documented in the umbrella DD entry).

**When NOT:** if perf is "good enough" after topic 1 alone, the structural migration's value is taste-based, not necessity-based. Strategy A would close the loop faster.

---

## Cross-cutting infrastructure

Already in place:
- **Profiler** (`PFAB_PROFILE=1` or `profiler.enable()`). Sections instrumented across the hot path. Use to validate every perf change.
- **Equivalence test scaffolding** (`tests/orchestration/test_batched_predict.py`) — pattern at `atol=rtol=1e-5` for batched APIs.
- **Smoke test** (`pred-fab-mock/dev/_smoke_layer3.py --profile`) — fast end-to-end benchmark with profiling.

---

## Historical references (superseded plans)

- **`ACQUISITION_REFACTOR_PLAN.md`** — Phase 1 of that plan (batch-aware exploration schedule) shipped on `feat/batch-aware-exploration-schedule`. Phase 2 of that plan (objective unification) is now the calibration unification track (topic 2). Document remains for historical context.
- **`EXPLORATION_AUDIT.md`** — root-cause analysis preceding `ACQUISITION_REFACTOR_PLAN.md`'s Phase 1. Resolved by that branch.

## Knowledge-base cross-references

- [`PFAB - Design Decisions.md`](../knowledge-base/PFAB%20-%20Design%20Decisions.md) — umbrella commitment to torch + the post-Phase-C data-layer track (topic 5).
- [`PFAB - Calibration.md`](../knowledge-base/PFAB%20-%20Calibration.md) — calibration system mental model.
- [`PFAB - Prediction Model.md`](../knowledge-base/PFAB%20-%20Prediction%20Model.md) — prediction system mental model.
- [`PFAB - Evidence Computation.md`](../knowledge-base/PFAB%20-%20Evidence%20Computation.md) — KDE / KernelField details (topic 1).
- [`PFAB - Scheduled Sampling.md`](../knowledge-base/PFAB%20-%20Scheduled%20Sampling.md) — SS background (topic 6).
