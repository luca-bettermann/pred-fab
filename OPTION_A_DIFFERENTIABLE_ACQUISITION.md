# Option A — Differentiable Acquisition (Implementation Plan)

**Status:** proposed (2026-04-28). Awaiting profile validation before kickoff.
**Branch base:** `feat/integrated-evidence` at the most recent green commit
(profile instrumentation landed). Phase C and the post-Phase C data-layer
refactor track are *not* prerequisites — this plan supersedes both, since
the data-layer torch refactor needed for differentiable acquisition is the
*same* refactor planned in the post-Phase C track.

## Goal

Replace differential evolution (DE) with gradient-based acquisition optimisation
(Adam or `torch.optim.LBFGS`) for the calibration single-point and per-experiment
schedule paths. Expected: **30–100× fewer acquisition evaluations** at the
optimisation level, with naturally vectorised per-candidate evaluation already
available from A.2.

## Expected outcome

| Lever | Before | After |
|---|---|---|
| Acquisition evals (D=4, smart_maxiter) | 1,920 (= 32 popsize × 60 maxiter) | **~50–100** (Adam ~50 epochs, L-BFGS ~10–20 line searches × few evals each) |
| Acquisition evals (D=10) | ~12,000 | **~100–200** (gradient scales with D linearly, not D²) |
| Per-eval cost | unchanged (same predict + eval pipeline) | unchanged |
| **Total per-acq wall time** | 6–20s (current bottleneck) | **0.2–0.6s** |

Combined with what already landed (A.1, A.2, Layer 4, eval batching):
**~50–100× total exploration speedup vs pre-A.2 baseline**. 20s/iteration → ~0.2–0.4s/iteration. Schedule passes that take 100s+ today drop to ~5–10s.

## What needs to be true for gradients to flow

The acquisition objective is `score = (1−κ) · perf(params) + κ · evidence(params)`.
For autograd to compute `dscore/dparams`, **every step** of that chain must run
in tensor-land with `requires_grad=True` propagating from `params` to `score`:

```
params (tensor, leaf)
   │
   │  params_to_array_tensor(params_dict)   ← MISSING (currently numpy)
   ▼
x_norm: torch.Tensor, shape (D,)
   │
   │  predict_for_calibration_batched_tensor(x_norm)   ← TENSOR-INTERIOR (Layer 4)
   ▼                                                     but converts to numpy at API
predictions: dict[str, torch.Tensor]   ← MISSING (currently numpy at API boundary)
   │
   │  evaluate_features_batched_tensor(predictions)   ← MISSING (currently pandas/numpy)
   ▼
perf: torch.Tensor, shape ()
   │
   │  delta_integrated_evidence_tensor(x_norm)   ← MISSING (currently scipy/numpy)
   ▼
evidence: torch.Tensor, shape ()
   │
   │  scalar combination
   ▼
score: torch.Tensor, shape ()
```

Each `MISSING` is a layer to convert. Each is independent and individually testable.

## Migration: 5 commits, gated by tests

The migration is **non-disruptive** — DE remains as a fallback throughout. We
introduce a parallel tensor-native path next to the existing numpy path, then
flip the default once proven.

### Commit 1: Tensor-native `params_to_tensor` / `tensor_to_params` (~150 LOC)

**Where:** `src/pred_fab/core/datamodule.py`.

**Adds:**
- `params_to_tensor(params_dict: dict) -> torch.Tensor` — encode parameter dict
  into a normalised `(D,)` tensor with `requires_grad=False` (caller can flip).
  Mirror of `params_to_array` but tensor-typed.
- `tensor_to_params(x: torch.Tensor) -> dict` — inverse. Categorical decoding via
  argmax over one-hot blocks (or, in a follow-on, learnable embeddings).

**Categorical handling decision:** keep one-hot internally but route through
tensor ops. The post-Phase C track plan to migrate categoricals to integer-index
+ `nn.Embedding` is *additive* on top of this commit — not gated.

**Tests:**
- Round-trip: `tensor_to_params(params_to_tensor(p)) ≈ p` to 1e-6 for both
  continuous and categorical parameters.
- Gradient flow: a unit test that constructs a small loss `L = sum(params_to_tensor(p))`,
  calls `L.backward()`, confirms a tensor with `requires_grad=True` produces non-zero
  gradients on its leaf.
- Equivalence with existing `params_to_array`: for the same params, both produce the
  same numerical encoding to 1e-6.

### Commit 2: Tensor-native `predict_for_calibration_tensor` (~200 LOC)

**Where:** `src/pred_fab/orchestration/prediction.py`.

**Adds:**
- `predict_for_calibration_tensor(params_tensors: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]`
  — like `predict_for_calibration_batched` but takes tensors and **returns
  tensors at the API boundary** (no numpy conversion). The autoreg loop
  already runs in tensor land post-Layer 4 — the only change is dropping the
  final `stack_np = stack.detach().cpu().numpy()` round-trip.

**Key constraint:** the returned tensors must have a graph back to the input
`params_tensors` so autograd can compute `dpredictions/dparams`. This means
the autoreg path must avoid `.detach()` calls anywhere along the gradient
chain. We currently `.detach()` in two places we'll need to remove (or
gate behind a `gradient_pass: bool` kwarg):
1. `model.forward_pass` wraps with `torch.no_grad()` — needs a `gradient: bool`
   kwarg that conditionally skips the no_grad context.
2. `_predict_autoregressive_batched` calls `.detach().cpu().numpy()` at the
   API boundary — this gets replaced with tensor return when called via the
   new tensor entry point.

**Why this isn't expensive:** we're keeping the existing `predict_for_calibration_batched`
(the numpy-returning, no-grad version) for the DE fallback path and for the
`agent.predict_performance(...)` user-facing API. The new tensor path is
opt-in via a separate function name.

**Tests:**
- Equivalence: `predict_for_calibration_tensor` and `predict_for_calibration_batched`
  produce identical numerical predictions to 1e-5 on the same inputs.
- Gradient flow: a 2-row training example, params_tensor with `requires_grad=True`,
  full forward pass, `predictions['feat'].sum().backward()` produces non-NaN
  finite gradients on params_tensor.

### Commit 3: Tensor-native eval models (~150 LOC + per-mock-model overrides)

**Where:** `src/pred_fab/interfaces/evaluation.py`, `src/pred_fab/orchestration/evaluation.py`,
plus `pred-fab-mock/models/evaluation_models.py` for the 3 mock evaluators.

**Adds:**
- `IEvaluationModel.compute_performance_tensor(feature_arrays_S: list[torch.Tensor], parameters_list)`
  — tensor variant returning `(S,)` tensor of per-candidate avg performance.
  Default impl raises `NotImplementedError` with a pointer to override.
- `EvaluationSystem._evaluate_feature_dict_tensor(features_dicts_S, parameters_list)`
  — tensor dispatch parallel to `_evaluate_feature_dict_batched`.

**Mock overrides:** the 3 mock evaluators (PathAccuracy, EnergyEfficiency,
ProductionRate) all use the affine `1 - |feat - target| / scaling` clamped
to `[0, 1]` formula — trivially differentiable. Each gets a ~10-line
`compute_performance_tensor` override using `torch.clamp` (which is
differentiable except at the clamp boundaries; the gradient is 0 outside
`[0, 1]`, which is correct behaviour for a clamped score).

**The clamp gradient question:** `torch.clamp` produces zero gradient outside
the `[0, 1]` range, so candidates whose performance is already saturated at
1.0 contribute no gradient signal. This is correct (we shouldn't push toward
those candidates) but means very-good candidates can produce flat gradients.
Standard solution: a small slope outside `[0, 1]` via a soft-clamp (sigmoid).
We start with hard clamp; if it produces optimisation pathologies, switch.

**Tests:**
- Tensor / numpy equivalence to 1e-5 for each mock evaluator on a fixed input.
- Gradient flow: small synthetic feature tensor, params_tensor with
  `requires_grad=True`, end-to-end predict → eval → loss, confirm gradients
  reach params_tensor.

### Commit 4: Tensor-native KDE evidence (~250 LOC)

**Where:** `src/pred_fab/orchestration/prediction.py` (`delta_integrated_evidence_*`)
plus `src/pred_fab/orchestration/calibration/kde.py` (or wherever the
`KernelField` estimator lives — find via grep).

**The hard one.** Two paths:
- **Path 4A** (preferred): rewrite the KDE evidence integral using torch ops.
  KernelField is a sum of Gaussian probes evaluated at QMC quadrature points.
  Translates to torch as: `evidence(x) = -0.5 * ((points - x_centres) / sigma)**2 - log(...)`
  evaluated at probe locations + weighted sum. ~200 LOC, fully differentiable.
- **Path 4B** (fallback): leave KDE numpy/scipy, compute its gradient via
  finite differences `dE/dx ≈ (E(x+ε) - E(x-ε)) / (2ε)` per dim. Cheap to
  implement (~30 LOC) but adds 2D extra evidence evals per gradient step;
  less accurate; doesn't compose with second-order optimisers like L-BFGS.

**Recommendation: Path 4A.** The work is mechanical (translate scipy `cKDTree`
+ Gaussian sums to torch), and it unlocks GPU support for the evidence path
as a free side benefit.

**One subtlety:** KernelField uses a `cKDTree` for truncated kernel evaluation
(5σ cutoff). Tensor land has no `cKDTree` equivalent at our scale; instead,
compute distances against all kernel centres via broadcasting `(K, D)` vs
`(M, D)` → `(M, K)` distance matrix, mask via `dist < 5σ`, sum the kept
kernels. At our typical N≤50 kernels, the dense computation is faster than
the tree-build anyway. The post-Phase C "Truncation activates only at N ≥ 10"
DD entry already documented this trade-off.

**Tests:**
- Numerical equivalence to existing `delta_integrated_evidence_aggregated`
  to 1e-4 (looser than other equivalence tests because scipy QMC and torch
  reductions have different summation orders).
- Gradient flow at a non-saturated probe location.
- Smoke test: `evidence(x)` with `x.requires_grad=True`, `evidence.backward()`
  produces finite gradients.

### Commit 5: Gradient-based acquisition optimiser + DE fallback (~300 LOC)

**Where:** `src/pred_fab/orchestration/calibration/engine.py`.

**Adds:**
- `OptimizationEngine.run_acquisition_gradient(objective_tensor, bounds, n_starts=4)`
  — multi-start Adam (or torch.optim.LBFGS). Each start runs N_EPOCHS=50–100
  Adam steps or N_LBFGS_STEPS=10–20 line searches, takes the best.
- `Optimizer.GRADIENT` enum value (alongside existing `LBFGSB`, `DE`).
- A new wired path in `CalibrationSystem._run_optimization_pass` and
  `_optimise_schedule_for_experiment` that routes to gradient-based when
  the optimiser is set to `GRADIENT` (gated by a class-level
  `prefer_gradient: bool = False` until validated).

**Multi-start strategy:** sample `n_starts` initial points (latin hypercube
or sobol within bounds), run gradient descent from each, return the best.
At our D≤10, n_starts=4 typically catches the global optimum and costs
4 × 50 = 200 acq evals — still 10× cheaper than DE's 1,920.

**Bounds handling:** `torch.optim` doesn't natively support box constraints.
Two options:
- **Sigmoid reparameterisation**: optimise unconstrained `z`, transform via
  `x = sigmoid(z) * (hi - lo) + lo`. Smooth, but introduces a non-affine
  warp.
- **Project after each step**: `x = x.clamp(lo, hi)`. Simpler, but breaks
  gradient flow at the boundary.

**Recommendation:** sigmoid reparameterisation — keeps the optimisation
landscape clean.

**Tests:**
- Convergence test on a synthetic quadratic + Gaussian surface (known optimum,
  multi-modal): gradient finds the right optimum within tolerance.
- Equivalence test with DE on the smoke test: gradient and DE both find an
  acquisition score within 1% of each other on the same problem.
- Speed test on the smoke test: gradient path runs in <1s vs DE in 5–10s.

### Commit 6 (optional): Flip default to gradient

**Where:** `src/pred_fab/orchestration/calibration/system.py` and `engine.py`.

After Commit 5 has shipped and we've validated convergence quality on the
mock smoke + 1–2 real exploration runs, flip the default optimiser from
`Optimizer.DE` to `Optimizer.GRADIENT`. DE remains available as a fallback
via `agent.configure_optimizer(backend=Optimizer.DE)`.

## What stays unchanged

- The agent orchestration layer (`PfabAgent.exploration_step`,
  `acquisition_step`, `inference_step`) — internal optimiser swap is
  invisible to callers.
- The schema layer (`DataObject`, `DataCategorical`, `DataFeature`).
- The Dataset persistence layer.
- The K-refit / scheduled sampling path — Phase C is *not* a prerequisite
  for this plan; the per-minibatch SS migration can land in parallel.
- Public APIs: `agent.predict_performance(params)` keeps its numpy contract;
  the tensor entry point is internal to the calibration hot path.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Gradient-based optimiser gets stuck in local minima where DE escaped | Medium | Multi-start (n_starts=4–8) catches most cases; DE remains fallback |
| `torch.compile` interaction with autograd surfaces a bug | Low | Disable compile (set `COMPILE = False`) for the gradient path until validated; re-enable later |
| KDE tensor rewrite produces numerical drift at higher D | Low-Medium | Equivalence tests at multiple σ; finite-difference fallback (Path 4B) if Path 4A fails |
| Eval clamp saturation produces flat gradients | Medium | Soft-clamp (sigmoid) if hard clamp pathologies appear; this is a 1-line config flip |
| First gradient run is slow due to torch graph build | Low | Warm-up call after training (similar to torch.compile probe); amortised over many DE-replacement evals |
| Categorical handling via one-hot doesn't differentiate well through argmax | Low | At our scale, categoricals are rare (mostly continuous params). For the mock, no categoricals at all. Defer to embedding migration if needed |

## Estimated effort

| Commit | Effort | Lines |
|---|---|---|
| 1. params_to_tensor | 0.5 day | ~150 |
| 2. predict_for_calibration_tensor | 1 day | ~200 |
| 3. eval tensor variants | 0.5 day | ~150 |
| 4. KDE tensor (Path 4A) | 2 days | ~250 |
| 5. gradient optimiser + multi-start | 1.5 days | ~300 |
| 6. flip default + validate | 0.5 day | ~10 |
| **Total** | **~6 days** | **~1,060** |

Compares to the original "1–2 weeks" estimate I gave; the data-layer refactor
the post-Phase C track originally scoped is mostly absorbed into commits 1–3
of this plan, which is why the total is shorter. Two refactors merge into one.

## Validation plan (per-commit)

Each commit lands behind `git push` only when:
1. All existing 586 tests still pass.
2. The commit's own equivalence tests pass.
3. (Commits 2, 3, 5) Gradient-flow unit tests confirm autograd propagates
   end-to-end through that commit's path.
4. (Commit 5+) Mock smoke test runs with `--profile` showing the new
   gradient path's wall time and confirming the expected speedup.

Each commit is independently revertable; if any commit produces a regression
that can't be diagnosed in <1 day, revert and refactor.

## Open questions for validation post-profile

1. **Is N_evals or per_eval_cost the bigger contributor to the 20s?** If
   per-eval is dominated by the cell loop's Python overhead (autoreg.recursive_substitute,
   autoreg.predictions_scatter), gradient acquisition gives less win than estimated
   because we'd still pay the per-eval cost. In that case, **collapse the
   cell-loop into a vectorised tensor op first** (separate plan, ~1 day) and
   *then* do gradient acquisition.
2. **Which evaluator gradient flow gives most leverage?** PathAccuracy, EnergyEfficiency,
   ProductionRate all use the same affine formula — implementing one validates
   the pattern for the others.
3. **What's the expected user-facing API change?** None — this is a backend
   swap. Existing `agent.exploration_step(...)` semantics are preserved.

## Sequencing decision (for after profile review)

Three orderings to choose from after seeing the profile:

- **(A1) Plan as-written** (commits 1→2→3→4→5): if profile confirms N_evals
  is the dominant factor.
- **(A2) Cell-loop vectorisation first, then commits 1→5**: if profile shows
  per-eval Python overhead in the cell loop dominates. Adds ~1 day for
  cell-loop refactor, then resumes plan.
- **(A3) Skip commit 4 (KDE) initially, use Path 4B finite-difference
  fallback** + commits 1, 2, 3, 5: if KDE evidence is a small fraction of
  per-eval time (per profile), we can defer the rewrite and capture 80%
  of the win in 4 days instead of 6.

User picks A1/A2/A3 after profile review.
