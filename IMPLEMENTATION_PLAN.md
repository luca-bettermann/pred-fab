# PFAB Implementation Plan — Strategy D (Full Torch Migration)

**Living document.** Updated 2026-04-29. Strategy A complete and merged to `main`; Strategy D in progress on `feat/full-torch`.

## Status

### Strategy D — Phase 1 complete (zero scipy in `pred_fab/`); Phase 2-5 planned

**Foundations (commits 1-8) shipped:**

| Commit | What | Tests | Status |
|---|---|---|---|
| 1 | `params_to_tensor` / `tensor_to_params` (gradient-traversable affine norm) | 9 | shipped |
| 2 | `predict_for_calibration_tensor` (autoreg with `gradient_pass=True`) | 7 | shipped |
| 3 | `IEvaluationModel.compute_performance_tensor` + `_evaluate_feature_dict_tensor` | 8 | shipped |
| 4 | KDE in torch (dense regime + σ/D-aware regime dispatcher) | 14 | shipped |
| 5 | `Optimizer.GRADIENT` + `run_acquisition_gradient` (sigmoid bound reparam) + tensor acquisition objective wired into `_run_acquisition_phase` | 12 | shipped |
| 6 | `DataLoader(TensorDataset)` scale-aware in `TorchMLPModel.train` | 3 | shipped |
| 7 | End-to-end GRADIENT-vs-DE validation smoke + empirical findings recorded | (smoke) | shipped |
| 8 | Status block + roadmap in plan | — | shipped |

**Phase 1 (scipy removal, foundations) shipped:**

| Commit | What | LOC | Status |
|---|---|---|---|
| 9 step A | Torch-native vectorised DE replaces `scipy.optimize.differential_evolution` (rand/1/bin mutation, F~U[0.5, 1.0], CR=0.7, integer rounding, no-improvement-window callback) | +151 / −74 | shipped |
| 9 step B | `torch.optim.LBFGS` replaces `scipy.optimize.minimize`; MPC inner loop migrated from scipy.minimize to torch DE; full `scipy.optimize` removal | +97 / −22 | shipped |
| 9b | Smart initial conditions (BoTorch `gen_batch_initial_conditions` pattern: Sobol raw_samples + Boltzmann selection with temperature `eta`) | +46 | shipped |
| 10 | `scipy.spatial.cKDTree` → `torch.cdist + torch.where` 5σ mask in `KernelIndex.density_at`; full `scipy.spatial` removal | +26 / −19 | shipped |
| 11 | `scipy.stats.qmc.Sobol` → `torch.quasirandom.SobolEngine` (had shipped earlier as commit 12 in old numbering) | +5 / −2 | shipped |

**End of Phase 1 status: zero `scipy` imports anywhere in `pred_fab/`.** 652 tests pass. Acquisition optimisation, QMC, and spatial neighbour lookup are all torch-native.

**Phase 5 commit 18 (partial) shipped:**

| Commit | What | LOC | Status |
|---|---|---|---|
| 18a | Drop `Optimizer.LBFGSB` enum + `_run_lbfgsb` + `online_optimizer` + `lbfgsb_*` config knobs (the LBFGSB path was made redundant by `run_acquisition_gradient` using `torch.optim.LBFGS` internally). 4 LBFGSB tests removed, 2 gradient-config tests added. | −208 / +55 | shipped |

**Phase 3 commit 13 shipped:**

| Commit | What | LOC | Status |
|---|---|---|---|
| 13 | `nn.Module` normalisers replace stat dicts. Adds `pred_fab/core/normalisers.py` with `StandardScalerModule`, `MinMaxScalerModule`, `RobustScalerModule`, `IdentityNormaliser`. `DataModule._parameter_stats` / `_feature_stats` hold module instances now; existing dict-style access (`stats[col]["mean"]`) preserved via `NormaliserModule.__getitem__`. `_apply_normalization` / `_reverse_normalization` collapse to `module(data)` / `module.reverse(data)` — ~80 LOC method-dispatch deleted from DataModule. Stats serialise via `state_dict` / `torch.save`; `get_normalization_state` still emits legacy dict format for on-disk bundle compat. | +343 / −117 | shipped |

**Phase 2-3-4-5 still planned, not yet shipped:**

| Phase | Commits | Net LOC | Gist | Risk |
|---|---|---|---|---|
| 3 | 14 | −80 net | One-hot → categorical-index tensors + `nn.Embedding` opt-in | medium |
| 3 | 15 | −330 net | Phase C SS absorption (per-minibatch SS in `train()`); deletes K-refit + DM SS state machinery; rewrites 14 SS tests | high |
| 3 | 16 | −80 net | Tensor-native `prepare_input` + `export_to_tensor_dict`; drops pandas roundtrip from autoreg | medium |
| 2 | 12 | −80 net | Schedule path gradient migration; deletes offset encoding, soft bound penalty, smoothing factor | high |
| 5 | 18b | −20 net | Drop `estimator` knob from `configure_exploration` (KernelField is the only path); drop `smoothing` from `configure_schedule` (after commit 12); drop `configure_scheduled_sampling` (after commit 15) | low (gated on 12, 15) |
| 5 | 19 | −40 net | `run_baseline` / `run_calibration` unification | medium |
| 5 | 20 | +30 net | `agent.to('cuda')` + `torch.compile` over full acquisition graph | medium |
| 4 | 17 | +120 (conditional) | KDE 4c cluster regime if real workloads need it | low |

**Net so far: −388 LOC delivered, −860 LOC delete remaining target.**

**End-to-end works**: `agent.configure_optimizer(backend=Optimizer.GRADIENT)` followed by `agent.baseline_step(n)` runs the gradient path through the Process phase (continuous params; integer / domain phase still DE). 654 tests pass.

**Findings from commit 7 mock smoke (D=2 Process phase, ~10 KDE kernels)**:
- Both paths converge to similar acquisition scores (DE obj=−0.069, GRADIENT obj=−0.061; GRADIENT slightly better).
- DE wins wall-clock at D=2 (3.4s vs 4.7s) — small-D regime favours DE's tight loop over GRADIENT's 4-start × 40-iter Adam over the full encoder + tensor KDE graph.
- The DE → GRADIENT crossover should empirically land somewhere around D ≥ 5-10. Validation on a larger problem before flipping the default in commit 8.

**Deferred** (each needs its own dedicated effort):
- Phase C absorption (delete K-refit + DM SS state machinery) — touches 14 SS-specific tests with detailed per-cell semantics.
- Data-layer migrations 7a/7b/7c (one-hot → categorical-index, nn.Module normalisers, tensor-native `prepare_input` + tensor-dict export).
- KDE 4b (KNN regime) and 4c (cluster regime) — gated on dispatcher INFO logs from real workloads.

---

**Strategy A — done, merged.** ~5.2× cumulative on smoke (130ms → 25ms), ~20× on schedule iteration (20s → 1s). Architecturally:
- Tensor-native `IPredictionModel` contract (A.1).
- Batched autoregressive prediction + vectorised DE acquisition (A.2 + Layer 4).
- KDE vectorised (`integrated_evidence_perturbed_batched` + joint-`_batched_joint` for schedule).
- `torch.compile` default-on with probe-fallback.
- Schedule mode routed through vectorised DE path (the perf fix).
- Soft bound penalty for schedule trajectories (no hard clipping).
- Profiler infrastructure for ongoing validation.
- Domain → Process phase split (commit 1 of unification — `_run_phase` generalised for N≥1, L≥1).

**Strategy D — active.** Full torch migration. Differentiable acquisition, drop scipy DE, drop all pandas DataFrames inside the framework, `nn.Module` normalisers, `nn.Embedding` categoricals, `torch.compile` over the entire acquisition graph. End state: numpy only at JSON I/O and matplotlib edges; everything else tensors.

`run_baseline` / `run_calibration` unification (Stage 3 of A) deferred to Strategy D — falls out naturally from the torch rewrite.

---

## Strategy D — what we're unlocking

The full migration delivers four classes of wins. **Per-class deletion** noted where it shrinks the codebase.

### 1. Optimisation: O(D²) DE → O(D) gradient-based

**~30-100× fewer acquisition evaluations** at typical D, more at higher D. DE's smart_maxiter scales `15D × popsize·D ≈ 12,000 evals at D=10`; gradient methods (`torch.optim.LBFGS` / Adam with multi-start) converge in ~50-200 evals.

For schedule iteration (currently 1s post-A): expect **~50-200ms**. Schedule with L=20 layers (currently impractical) becomes feasible.

**Deletion:** scipy.optimize.differential_evolution + L-BFGS-B → torch.optim. ~50 LOC of engine.run / engine._run_de scaffolding shrinks since torch.optim is simpler.

### 2. Hard bounds via reparameterisation

`x = sigmoid(z) · (hi - lo) + lo`. **`x ∈ [hi, lo]` strictly.** No drift, no clipping, no penalty tuning. The boundary issue we patched in `451cdd8` (soft bound penalty for schedule trajectories) **vanishes structurally** — it was a workaround for DE's offset-encoding drift; gradient + sigmoid eliminates the root cause.

**Deletion:** soft bound penalty (`451cdd8`), SolutionSpace's offset-encoding logic, trust-region delta bookkeeping. ~80 LOC.

### 3. Differentiable autoreg — the biggest structural win for recursive features

Today's recursive trajectory is opaque to the optimiser:
```
DE varies params → autoreg loop runs → final score
                   (opaque internal coupling)
```

With autograd, the trajectory becomes one composed graph:
```
params[step_0] ──┐
                 ├──> cell[0] prediction ──┐
                 │                          ├──> cell[1] (uses prev_layer = cell[0])
params[step_1] ──┘                          ├──> cell[2] (uses prev_layer = cell[1])
                                            ...
                                            └──> joint score
                                                  ↑
                                            backward() flows gradients back through everything
```

**Why this matters most for schedule**: the optimiser sees how changing step `k`'s parameter affects step `k+1`'s prediction *via the recursive feature*. Smoothness emerges naturally — adjacent steps with coherent trajectories are favoured without an explicit smoothing penalty. Trust-region offsets become semantic (small offset → small downstream effect), not bound-bookkeeping.

**Plus structural cleanups specifically for recursive features:**

| Current workaround | Strategy D replacement |
|---|---|
| K-refit (4 separate fits w/ annealed `p_student`) in `PredictionSystem.train` | Per-minibatch SS in `model.train()` with autograd-friendly mixing — ~5-line hook |
| `set_scheduled_sampling_state` + `_ss_predictions_by_exp` + `_perturb_recursive_features` (~120 LOC of stateful cache machinery in `DataModule`) | Single per-epoch hook in `train()` with `requires_grad=True` flowing through substitution |
| `_predict_autoregressive_batched` Python cell loop (Layer 4 collapsed inner numpy boundary, but loop itself remains) | `torch.jit.script`-able recursive function; multi-axis recursion exposes parallel inner dim (e.g. n_segments parallel within a layer step) |
| `_autoreg_predict_training_data` for cross-model SS pre-computation (~20 LOC) | Cross-model deps become graph edges; autograd tracks them automatically |
| NaN→0 conversion at `_one_hot_encode` for recursive boundary cells (implicit, fragile) | Explicit `boundary_value` tensor; `torch.where(mask, predicted, boundary_value)` blocks gradient cleanly |
| `_get_recursive_input_specs` + `recursive_plan` runtime walk + flat-index precompute (~40 LOC) | Dependency expressed as `torch.roll` in the prediction graph; no schema-walking at acquisition time |

**Total deletion enabled by recursive-features simplification: ~330 LOC of stateful workaround machinery**, plus the SS contract becomes more accurate (per-step gradient-traversable, not per-K-round).

**Future capability**: implicit differentiation through deep recursion. If schedules ever scale to L=50+ steps, naive backprop is memory-heavy; `torch.func.jvp` / fixed-point methods give gradients without storing intermediate states. Numpy can't do this without manually coding adjoint methods.

### 4. Tensor-native end-to-end + GPU + `torch.compile` over everything

- `agent.to('cuda')` works end-to-end after migration.
- `torch.compile` JIT-compiles the full acquisition graph (params → predict → eval → evidence → score). **2-5× more on top of everything else.** Today we can only compile the model's forward.
- All current "soft penalty" hacks become first-class differentiable terms (clamp saturation → smooth sigmoid, NaN handling → masked tensor ops, `_in_unit_cube` → smooth indicator).
- Cross-model dependency cycles become torch graph errors at construction, not silent silent runtime breakage.
- Data layer simplifies dramatically:

| Sub-topic | Deletion |
|---|---|
| **5a. One-hot → categorical-index tensors** (long-standing pain point) | ~80 LOC: `_col_extraction`, `_decode_one_hot`, one-hot-aware calibration array math → ~15 LOC. Models that need one-hot use `F.one_hot` themselves; learnable categoricals via `nn.Embedding`. |
| **5b. Normalisation as `nn.Module`** (`StandardScalerModule` etc.) | ~150 LOC: `_apply_normalization` / `_reverse_normalization` / `_compute_normalization_stats` / `get/set_normalization_state` → ~30 LOC. Stats live in `state_dict()`; serialise via `torch.save` for free. |
| **5c. Tensor-native `prepare_input`** | Drop `pd.DataFrame(all_rows)` round-trip in autoreg loop. `prepare_input` becomes tensor stack + normalisation apply. ~30 LOC saved per call site, with measurable inference-step savings. |
| **5d. `dataset.export_to_dataframe` → tensor dict** | ~20 LOC saved at the boundary. |

### Total Strategy D LOC delta

| Component | LOC change |
|---|---|
| Optimisation (DE → torch.optim) | −50 |
| Bounds (offset encoding → sigmoid reparam) | −80 |
| Recursive features (SS state + cell loop + cross-model cache) | −330 |
| Data layer (one-hot, normalisers, prepare_input, export) | −280 |
| KDE in torch (subsumes Topic 1's numpy implementation) | net 0 (replaces existing) |
| Plotting boundary tightening | ~+20 |
| Differentiable acquisition objective + multi-start | ~+200 |
| Sigmoid reparameterisation utilities | ~+40 |
| **Net** | **~−480 LOC, end-state simpler** |

---

## Strategy D — execution plan

8 commits, each independently revertable, gated by tests. Approximate order:

### Commit 1 — `params_to_tensor` / `tensor_to_params` (~150 LOC, 0.5 day)
Tensor-typed mirror of `params_to_array` / `array_to_params`. Round-trip + gradient-flow tests. Categoricals stay one-hot internally; the embedding migration (commit 5a) is additive.

### Commit 2 — `predict_for_calibration_tensor` (~200 LOC, 1 day)
Returns tensors at the API boundary (no numpy round-trip). The autoreg loop is already tensor-internal post-Layer 4; the change is dropping the final `stack_np = stack.detach().cpu().numpy()`. Adds optional `gradient_pass=True` kwarg on `model.forward_pass` to skip the no_grad context for the gradient path; default stays no-grad.

### Commit 3 — Tensor-native eval models (~180 LOC, 0.5 day)
`IEvaluationModel.compute_performance_tensor` returning `(S,)`. Mock overrides for the 3 evaluators (~10 LOC each using `torch.clamp` — differentiable except at clamp saturation; soft sigmoid fallback if pathologies appear). `EvaluationSystem._evaluate_feature_dict_tensor` parallel to the existing batched dispatch.

### Commit 4 — Tensor-native KDE (~250 LOC, 2 days) — *the hardest*
Rewrite `KernelField` + `KernelIndex` in torch. Sum of Gaussian probes evaluated at QMC points.

**Scale-aware regime dispatch.** KDE cost depends not just on `n_kernels` but on the **effective number of contributing kernels** — which is a function of `(n_kernels, σ, D)`:

```
n_active ≈ n_kernels × V(5σ-ball)         # within unit cube
        = n_kernels × π^(D/2) × (5σ)^D / Γ(D/2 + 1)
```

**Examples:**
| `D` | `σ` | `n_kernels` | `n_active` | Optimal regime |
|---|---|---|---|---|
| 4 | 0.075 | 100 | ~10 | dense |
| 4 | 0.075 | 10,000 | ~1,000 | KNN K=20-50 saves 200× |
| 10 | 0.075 | 10,000 | **~1.4** | KNN K=10 saves ~1000× *(curse of dimensionality helps)* |
| 1 | 0.3 | 100 | ~100 | dense (kernels overlap heavily) |
| 2 | 0.05 | 1,000 | ~24 | KNN K=30 fine |

In high D with typical σ, most kernels contribute negligibly even at large `n_kernels`. Conversely in low D with broad σ, all kernels matter regardless of count.

**Three structurally distinct algorithms** (not parameter knobs):

1. **Dense — exact, no summary.** Sum over all N kernels. Correct at any scale; optimal when `n_active ≈ n_kernels`. **Critical for baseline space-filling**: small contributions from far kernels matter for finding good gaps in the parameter space. Single tensor op `(M, N, D) → (M, N) → (M,)`.

2. **Sparse KNN — near-exact, top-K filtering.** `torch.topk` selects the K=20-50 closest kernels per probe; sum only those. The dropped kernels' contributions are below float-32 noise (5σ tail). Same big-O as dense but ~constant memory (`(M, K)` instead of `(M, N)`). Optimal when `n_active ≪ n_kernels`.

3. **Cluster-summarised — approximate.** Replace N kernels with `c ≪ N` cluster centres (mini-batch K-means / GMM). Lossy but bounded; appropriate when even KNN's full distance matrix is wasteful. Mathematically: a c-component mixture summarises the spatial density better than a noisy N-kernel cloud at very high N anyway.

**Selection rule** (σ/D-aware, not n_kernels-only):
```python
def _choose_kde_regime(n_kernels: int, sigma: float, D: int) -> str:
    v_5sigma = (pi ** (D / 2)) * ((5.0 * sigma) ** D) / gamma(D / 2 + 1)
    n_active = n_kernels * min(v_5sigma, 1.0)
    if n_kernels < 100:                  return "dense"        # tiny: overhead dominates KNN
    if n_active < 50:                    return "knn"          # most kernels contribute ~0
    if n_active < n_kernels * 0.5:       return "knn"          # >50% of kernels zero-contribution
    return "dense"                                              # most kernels matter
# Cluster regime triggers separately at n_active > 10K or n_kernels > 100K.
```

**Shipping plan:**

- **Commit 4 (now):** Implement dense regime + the dispatcher pattern. KNN/cluster regimes are scaffolded but stubbed — when selected, the dispatcher logs at INFO and falls back to dense (correct math, slower at scale). This gives empirical signal on when 4b/4c become necessary without hard-failing real workloads that happen to land in a non-dense regime.

- **Commit 4b (later, when 4-dispatcher logs show non-trivial KNN-eligible calls):** Implement KNN regime. Same dispatcher, just one more arm.

- **Commit 4c (conditional, only if real workloads hit cluster regime):** Implement cluster regime.

`scipy.stats.qmc.Sobol` → `torch.quasirandom.SobolEngine` (built-in to torch). ~10 LOC change, gives QMC sequences in tensor land.

Side benefit: subsumes Topic 1 (numpy KDE vectorisation) — same algorithmic structure as the dense regime, just torch ops. The numpy version was a stepping stone, not thrown-away work.

### Commit 5 — Gradient optimiser + multi-start (~300 LOC, 1.5 days)
`OptimizationEngine.run_acquisition_gradient(objective_tensor, bounds, n_starts=4)` — multi-start `torch.optim.LBFGS` or Adam. ~50 epochs Adam or ~10-20 line searches L-BFGS per start.

Bounds via sigmoid reparameterisation: `x = sigmoid(z) · (hi - lo) + lo`. Smooth gradient, strict bounds.

`Optimizer.GRADIENT` enum value alongside `LBFGSB`, `DE`. Routed through `_run_phase` when `chosen_opt == GRADIENT` and tensor APIs available.

### Commit 6 — Scale-aware training loop (~50 LOC) + Phase C deferred (~200 LOC future)
**Shipped now (commit 6):** `DataLoader(TensorDataset, shuffle=True)` scale-aware path inside `TorchMLPModel.train()`. Threshold = `MINIBATCH_THRESHOLD` (1000 rows by default, class-level so subclasses can override). Below threshold: single full-batch GD (mock-scale path). Above: shuffled minibatches per epoch.

**Deferred (commit 6b, when ready):** Per-minibatch SS hook inside `TorchMLPModel.train()` to delete K-refit, `set_scheduled_sampling_state`, `_perturb_recursive_features`, `_ss_predictions_by_exp`. The legacy machinery has detailed semantics (per-cell prior lookup, NaN-on-boundary, prediction caching across rounds) covered by 14 tests; absorbing it cleanly into a per-minibatch hook needs a careful redesign of how source-feature predictions are produced inside the training loop. Going alongside one of the data-layer migrations in commit 7 makes most sense — the categorical-index migration (7a) already touches batch construction.

**Scale-aware training loop**: at `N_train > 1000`, switch to `DataLoader(TensorDataset, batch_size=...)` for shuffled minibatching. The per-minibatch SS substitution maps onto each minibatch (whether it's the whole dataset for small N or one of many for large N). At `N_train < 1000`, single-batch path stays — no DataLoader overhead.

```python
def train(self, train_batches, val_batches, ss_config=None, **kwargs):
    X = torch.cat([b[0] for b in train_batches], dim=0)
    y = torch.cat([b[1] for b in train_batches], dim=0)
    n_rows = X.shape[0]
    if n_rows > 1000:
        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=min(256, n_rows // 4),
            shuffle=True,
        )
    else:
        loader = [(X, y)]  # single-batch fallback for mock-scale
    for epoch in range(EPOCHS):
        p_student = self._ss_p_for_epoch(epoch, ss_config)
        for X_batch, y_batch in loader:
            X_used = self._apply_ss_substitution(X_batch, p_student, ss_config) \
                     if ss_config else X_batch
            optimizer.zero_grad()
            loss = loss_fn(net(X_used), y_batch)
            loss.backward()
            optimizer.step()
```

Becomes natural with autograd-traversable substitution. Substitution gradient flows through the *current* network state (not a cached prediction from K rounds ago).

### Commit 7 — End-to-end gradient validation (committed)
End-to-end validation script `pred-fab-mock/dev/_smoke_gradient.py` runs the same `baseline_step(n=3)` twice on the mock — DE then GRADIENT — and prints wall time + obj. Findings recorded above in the status block.

### Commit 8 — Status block in plan (committed)
Top-of-plan status table summarising shipped/deferred so the next session has a clean picture.

---

## Outstanding work — full commit roadmap (user-approved 2026-04-29)

User direction: **fully commit to torch-native infrastructure; no fallbacks, no parallel paths, no escape hatches**. Single optimiser, single normaliser implementation, single data path. Order: small / low-risk first, then large architectural changes. Each row independently committable; dependencies noted in **deps**.

### Phase 1 — Acquisition: one optimiser, end of story

**Delete at the end of Phase 1:** `Optimizer` enum entirely, `Optimizer.DE`, `Optimizer.LBFGSB`, `_run_de`, `_run_lbfgsb`, all `de_*` / `lbfgsb_*` config knobs, `online_optimizer`, `scipy.optimize` imports, `scipy.spatial.cKDTree`. After: zero scipy in `pred_fab/`.

**Replace with:** one `engine.run_acquisition(objective_tensor, bounds, integer_mask)` method. Internally — pure-continuous → gradient + multi-restart; integer/categorical present → enumerate low-cardinality integer combos × gradient (BoTorch `optimize_acqf_mixed` pattern); for very-high-cardinality integers, vectorised torch-native DE — automatic dispatch by cardinality, no user choice.

| # | Commit | Size | Deps | What lands |
|---|---|---|---|---|
| **11** ✅ | scipy.qmc.Sobol → torch.quasirandom.SobolEngine | −5 / +5 | — | shipped (`980c062`) |
| **9** | Single `run_acquisition` + delete scipy.optimize + delete `Optimizer` enum + integer enumeration + simplify `configure_optimizer` to `(n_starts, n_iters)` only | −250 / +120 | — | torch-native vectorised DE for high-cardinality integer; integer enumeration for low-cardinality; default & only path. Removes `scipy.optimize.differential_evolution` and `scipy.optimize.minimize`. |
| **9b** | **Smart initial conditions** in `run_acquisition` (BoTorch pattern): `raw_samples` Sobol → batched no-grad scoring → Boltzmann-select top-K with temperature `eta` → optional perturb-around-best | +50 | 9 | Replaces uniform-random multi-start. Better global coverage at same `n_starts` budget. |
| **10** | cKDTree → torch KNN (`torch.cdist + topk`); also lands KDE 4b regime | −30 / +60 | — | Removes `from scipy.spatial import cKDTree`. **End of Phase 1: zero scipy in `pred_fab/`.** |

### Phase 2 — Schedule: sigmoid reparam, no offset encoding, no smoothing

**Delete:** `SolutionSpace`'s offset encoding for sched params, `sched_deltas_norm`, `_sched_delta_arr`, `_sched_bounds_list` for relative offsets, `_schedule_acquisition_objective_vectorized`'s soft bound penalty, per-step trust-region delta bookkeeping, `_STATIC_DRIFT_FRAC`, `cal.schedule_smoothing` and `_schedule_smoothing_factor` (autograd-traversable schedule makes smoothness emerge naturally), `cal._schedule_joint_var_limit`.

**Replace with:** absolute-value step encoding in `[0, 1]` z-space, sigmoid-reparameterised. Schedule trajectory = `L × D_sched` continuous variables, gradient-optimised jointly via tensor objective from commit 5.

| # | Commit | Size | Deps | What lands |
|---|---|---|---|---|
| **12** | Schedule path gradient migration + sigmoid reparam + delete all schedule workarounds (smoothing, offset encoding, soft bound penalty, joint-var limit) | −80 / +150 | 9 | **Win #2 fully realised.** Schedule iteration target 1s → 50-200ms. |

### Phase 3 — Data layer: nothing pandas-internal, nothing numpy-stat-dict

**Delete:** `_apply_normalization`, `_reverse_normalization`, `_compute_normalization_stats`, `get_normalization_state`, `set_normalization_state`, numpy stat dicts. `_col_extraction`, `_decode_one_hot`, `_one_hot_encode`, internal one-hot representation. `pd.DataFrame(all_rows)` roundtrip in autoreg, `_inject_context_features`'s pandas path, `dataset.export_to_dataframe`, `prepare_input`'s pandas → numpy → tensor sequence. `set_scheduled_sampling_state`, `_perturb_recursive_features`, `_ss_predictions_by_exp`, `_ss_p_student`, `_ss_rng`, `_autoreg_predict_training_data`, K-refit loop, `agent.configure_scheduled_sampling`, `cal.n_ss_rounds`, `cal.ss_schedule_floor`. The 14 SS-specific tests that test legacy semantics — rewritten or deleted.

**Replace with:** `nn.Module` normalisers (`StandardScalerModule`, `MinMaxModule`, `RobustScalerModule`). Categorical-index long tensors throughout (models use `F.one_hot` / `nn.Embedding` themselves). Tensor-dict export. Tensor-stack `prepare_input`. SS substitution via per-minibatch hook callable inside `TorchMLPModel.train()` using *current* network state.

| # | Commit | Size | Deps | What lands |
|---|---|---|---|---|
| **13** | `nn.Module` normalisers; delete numpy stat-dict normalisers | −150 / +30 | — | Stats live in `state_dict()`; serialise via `torch.save`. |
| **14** | One-hot → categorical-index tensors; delete `_col_extraction`, `_decode_one_hot`, `_one_hot_encode` | −80 / +20 | 13 | Models that want one-hot use `F.one_hot` themselves; learnable cats via `nn.Embedding`. |
| **15** | Phase C SS absorption: per-minibatch SS hook in `train()`; delete K-refit + DM SS state machinery | −330 / +50 | 14 | **Win #3 fully realised.** Touches 14 SS tests — rewritten. |
| **16** | Tensor-native `prepare_input` + `export_to_tensor_dict`; drop pandas roundtrip from autoreg | −80 / +30 | 14, 15 | **End of Phase 3: zero pandas inside `pred_fab/orchestration` and `datamodule.py`.** |

### Phase 4 — KDE depth (cluster regime, conditional)

| # | Commit | Size | Deps | What lands |
|---|---|---|---|---|
| **17** (= 4c) | KDE cluster regime via mini-batch K-means; only if real workloads with >100K kernels appear | +120 | 10 | Conditional — likely deferred until empirical signal from dispatcher logs. |

### Phase 2 — Schedule path gradient migration

| # | Commit | Size | Deps | What lands |
|---|---|---|---|---|
| **14** | **Schedule acquisition objective in tensors** | ~+150 LOC | 5 (commit done) | Tensor-typed `_make_schedule_objective_tensor` returning `(S,)`-shape negated scores. Routes through `delta_integrated_evidence_joint_batched_tensor_fn` (commit 5 done) for joint-L addition. Wires gradient path into `_phase3_schedule`. |
| **15** | **Drop offset encoding + soft bound penalty** | ~−80 LOC | 14 | Sigmoid reparam in z-space → strict `x ∈ [0, 1]` for absolute-value step encoding. Delete `_schedule_acquisition_objective_vectorized`'s soft penalty, the offset encoding inside `SolutionSpace`, the per-step trust-region delta bookkeeping. **Win #2 fully realised.** |

**Phase 2 net:** +150 / −80 = +70 LOC; schedule iteration target 1s → 50-200ms.

### Phase 3 — Data layer (the big numpy / pandas removal)

| # | Commit | Size | Deps | What lands |
|---|---|---|---|---|
| **16** | **`nn.Module` normalisers** | ~−120 LOC | — | `StandardScalerModule`, `MinMaxModule`, `RobustScalerModule`. Delete `_apply_normalization` / `_reverse_normalization` / `_compute_normalization_stats` / `get/set_normalization_state`. Stats live in `state_dict()`; serialise via `torch.save` for free. |
| **17** | **One-hot → categorical-index tensors** | ~−80 LOC | 16 | Drop internal one-hot; categoricals stay as long-tensor indices. Models that want one-hot use `F.one_hot` themselves; learnable cats via `nn.Embedding`. Touches `_col_extraction`, `_decode_one_hot`, calibration array math. |
| **18** | **Phase C — per-minibatch SS absorption** | ~−330 LOC | 17 | Per-minibatch SS hook inside `TorchMLPModel.train()` using categorical-index batches from commit 17. Delete `set_scheduled_sampling_state`, `_perturb_recursive_features`, `_ss_predictions_by_exp`, `_autoreg_predict_training_data`, K-refit loop. **Win #3 fully realised.** Touches 14 SS-specific tests — most rewritten to test the new path, the rest deleted as obsolete. |
| **19** | **Tensor-native `prepare_input` + `export_to_tensor_dict`** | ~−80 LOC | 17, 18 | Drop `pd.DataFrame(all_rows)` round-trip from autoreg loop. `prepare_input` becomes tensor stack + normalisation apply. `dataset.export_to_dataframe` → tensor dict. Removes pandas from the framework hot path entirely. |

**Phase 3 net:** −610 LOC. End state: pandas only at user-facing dataset I/O; framework interior is tensor-only.

### Phase 4 — KDE regime depth

| # | Commit | Size | Deps | What lands |
|---|---|---|---|---|
| **20** (= 4b) | **KDE KNN regime** | ~+80 LOC | — | Sparse top-K gather via `torch.cdist + topk`. Activates from commit 4 dispatcher when `n_active < 50` or `n_active < n_kernels * 0.5`. |
| **21** (= 4c) | **KDE cluster regime** | ~+120 LOC | 20 | Mini-batch K-means / GMM summarisation for `n_active > 10K` or `n_kernels > 100K`. Conditional on real workloads triggering it. |

**Phase 4 net:** +200 LOC, but unlocks scaling to 100K+ kernel workloads.

### Phase 5 — Architecture finalisation + API simplification

**API simplification — drop 5 configure methods × ~25 knobs to 4 × ~10:**
- `configure_performance(weights)` — keep as-is.
- `configure_exploration(sigma)` — drop `estimator` knob (KernelField is the only path).
- `configure_optimizer(n_starts, n_iters)` — drop `backend`, `online_backend`, `de_*`, `lbfgsb_*`, `gradient_lr`, `gradient_method`. Single optimiser, two real quality/speed knobs.
- `configure_scheduled_sampling(...)` — **delete entirely** (Phase 3 commit 15 absorbs SS into `train()`).
- `configure_schedule(parameter, dimension, *, delta, force)` — drop `smoothing` (autograd makes it redundant).

| # | Commit | Size | Deps | What lands |
|---|---|---|---|---|
| **18** | API simplification — collapse the surface; drop `Optimizer` enum imports from public API; delete `configure_scheduled_sampling`; drop `smoothing` from `configure_schedule`; drop `estimator` from `configure_exploration` | −80 | 9, 12, 15 | User-facing surface goes from 5×25 to 4×10. |
| **19** | `run_baseline` / `run_calibration` unification | −40 | 12 | Single entry point N≥1, L≥1. Stage 3 of original Strategy A. |
| **20** | GPU support + `torch.compile` over the full acquisition graph | +30 | 16 | `agent.to('cuda')` works end-to-end. JIT-traces params → predict → eval → KDE → score. Expected 2-5× on top of everything else. |

### Estimated effort

| Phase | Days |
|---|---|
| 1 (foundations + scipy removal) | 1.5 |
| 2 (schedule) | 1.5 |
| 3 (data layer) | 4 |
| 4 (KDE depth, conditional) | 2 |
| 5 (architecture + API simplification) | 2 |
| **Total** | **~11 days** |

Each phase independently revertable. Phase 1 unblocks Phase 2. Phase 3 is the big-ticket cleanup; commits 13 + 14 are prerequisites for 15 (Phase C) and 16 (prepare_input).

---

## Total estimate (commits 9-20)

**~11 days** focused work for full torch-native migration. Net LOC delta: **~−870 LOC, end-state simpler**.

After commit 16 ships: zero scipy / pandas / numpy-as-primary-type inside `pred_fab/`. Numpy stays only at user-facing JSON / matplotlib edges.

After commit 18 ships: user-facing API simplified to 4 configure methods × ~10 knobs (down from 5 × 25).

---

## Reconsidered: PyTorch Lightning + `torch.utils.data.DataLoader`

Both still skip — the reasoning hardens with Strategy D context, doesn't soften.

### PyTorch Lightning — still no

Lightning's value props (Trainer, LightningModule, callbacks, distributed strategies) are centred on **training workflow boilerplate**. Strategy D's heavy lifting is OUTSIDE training:
- The biggest wins are in **acquisition** (DE → gradient), **KDE** (numpy → tensor), **data layer** (pandas → tensor). Lightning has nothing to offer for any of these.
- `TorchMLPModel.train()` remains tiny (~25 lines plus the per-minibatch SS hook from commit 6). Lightning would refactor this into `LightningModule.training_step` with ~60 lines of boilerplate scaffolding — net +40 LOC for no functional gain.
- Lightning's `Trainer` wants to **own** the training loop. PFAB's `PredictionSystem.train(model)` orchestrates training, manages topo-sort, handles cross-model SS pre-compute. Inverting control creates abstraction churn — `Trainer.fit(LightningModule, datamodule)` doesn't fit the orchestration shape.
- ~50 MB dependency footprint + transitive deps (jsonargparse, torchmetrics, lightning-utilities, fsspec). Strategy D adds none of these.

**Where Lightning would genuinely help (not now, possibly future):**
- Multi-GPU / distributed training. Not in scope.
- Mixed precision. Models too small to benefit.
- TensorBoard / WandB logger integration. PFAB has its own logger; minor benefit.
- Checkpointing standardisation. Currently nothing to checkpoint between sessions. `state_dict()` is one call away if needed.

**Verdict:** still skip. Re-evaluate if PFAB ever scales to multi-GPU or large hyperparameter sweeps.

### `torch.utils.data.DataLoader` — scale-dependent

Three sub-topics, three different verdicts depending on scale.

**A. Replace `Dataset` with `torch.utils.data.Dataset`** — reject at any scale. Different abstraction levels:
- PFAB `Dataset`: experiment-run container with schema validation, 3-tier persistence (memory↔local↔external), `create_experiment` / `save_experiment` / `load_experiment` lifecycle.
- `torch.utils.data.Dataset`: sample-level indexable iterable with `__getitem__(idx) → (X, y)` and `__len__`.

There's no overlap. Subclassing the latter would obscure the schema contract.

**B. Use `DataLoader` for minibatched training** — **conditionally yes, depending on training-set size**. PFAB is a general framework, not just the mock. At larger scales DataLoader is the right primitive:

| Training rows | DataLoader usage |
|---|---|
| **<1,000** (mock-scale) | Skip. Full-batch GD is simpler and faster — no batch overhead. Current behaviour. |
| **1,000-100,000** | **Use it.** `DataLoader(TensorDataset(X, y), batch_size=256, shuffle=True)` for training. Standard SGD/Adam pattern at this scale. |
| **>100,000** | Use + `num_workers > 0` + `pin_memory=True`. I/O parallelism starts paying. |

**Encapsulation:** the choice lives inside `TorchMLPModel.train()` based on `N_train`; callers don't think about it. Strategy D commit 6 (per-minibatch SS) maps naturally onto the minibatch loop — SS substitution happens per minibatch whether the minibatch is the whole dataset (small) or one of many (large):

```python
def train(self, train_batches, val_batches, ss_config=None, **kwargs):
    X = torch.cat([b[0] for b in train_batches], dim=0)
    y = torch.cat([b[1] for b in train_batches], dim=0)
    n_rows = X.shape[0]
    if n_rows > 1000:
        loader = DataLoader(
            TensorDataset(X, y), batch_size=min(256, n_rows // 4), shuffle=True,
        )
    else:
        loader = [(X, y)]  # single full-batch "loader"
    for epoch in range(EPOCHS):
        for X_batch, y_batch in loader:
            X_used = self._apply_ss(X_batch, ss_config, epoch) if ss_config else X_batch
            optimizer.zero_grad()
            loss = loss_fn(net(X_used), y_batch)
            loss.backward()
            optimizer.step()
```

**Where in Strategy D:** part of commit 6 (Phase C absorbed). The per-minibatch SS plumbing already needs to handle multiple minibatches per epoch — DataLoader is the natural way to produce them at scale.

**C. Use `LightningDataModule` interface** — reject. PFAB `DataModule` already exposes `get_batches(SplitType.TRAIN/VAL/TEST)`. Renaming to `train_dataloader()` / `val_dataloader()` / `test_dataloader()` is a cosmetic change without functional benefit. The `DataLoader` itself is the useful primitive; the wrapping abstraction isn't.

**Updated verdict:** scale-aware adoption of `DataLoader` for training; `Dataset` and `LightningDataModule` still skip.

---

## Cross-cutting infrastructure

Already in place from Strategy A:
- **Profiler** (`PFAB_PROFILE=1` or `profiler.enable()`). Hot-path sections instrumented. Use for every Strategy D commit's perf validation.
- **Equivalence test scaffolding** (`tests/orchestration/test_evidence_batched.py`, `test_batched_predict.py`) — pattern at `atol=rtol=1e-5` for batched APIs. Strategy D adds gradient-flow unit tests on top.
- **Smoke test** (`pred-fab-mock/dev/_smoke_layer3.py --profile`) — fast end-to-end benchmark with profiling.

---

## Knowledge-base cross-references

- [`PFAB - Design Decisions.md`](../knowledge-base/PFAB%20-%20Design%20Decisions.md) — umbrella commitment to torch + the post-Phase-C data-layer track + ground-up redesign DD entry.
- [`PFAB - Calibration.md`](../knowledge-base/PFAB%20-%20Calibration.md) — calibration system mental model.
- [`PFAB - Prediction Model.md`](../knowledge-base/PFAB%20-%20Prediction%20Model.md) — prediction system mental model.
- [`PFAB - Evidence Computation.md`](../knowledge-base/PFAB%20-%20Evidence%20Computation.md) — KDE / KernelField details (Strategy D commit 4).
- [`PFAB - Scheduled Sampling.md`](../knowledge-base/PFAB%20-%20Scheduled%20Sampling.md) — SS background (Strategy D commit 6).

---

## Historical (Strategy A — done, merged to main)

Strategy A's progress is preserved in commit history on `main` (PR #6 in `pred-fab`, PR #3 in `pred-fab-mock`). Headline commits:

| Commit | What |
|---|---|
| `eadd198` (pred-fab merge) | All Strategy A work merged to main |
| `7f11e75` (mock merge) | Mock alignment merged to main |
| `db7ae75` | True KDE vectorisation (Topic 1) |
| `cbc6051` | Generalised `_run_phase` for N≥1, L≥1 |
| `6adf19b` | Schedule mode through vectorised DE path (perf fix) |
| `451cdd8` | Soft bound penalty for schedule trajectories |

Cumulative perf timeline (smoke):
- Pre-A.2: 130.93ms (1.0×)
- + KDE batching (cache only): 77.30ms (1.7×)
- + Domain split: 68.54ms (1.9×)
- + KDE vectorisation: 35.29ms (3.7×)
- + Joint estimator + schedule vectorised: **25.08ms (5.2×)**

Schedule iteration: **20s → 1s (20×)**.

Old `ACQUISITION_REFACTOR_PLAN.md` (Phase 1 of that plan shipped on `feat/batch-aware-exploration-schedule`; Phase 2 now obsoleted by Strategy D). `EXPLORATION_AUDIT.md` similarly historical.
