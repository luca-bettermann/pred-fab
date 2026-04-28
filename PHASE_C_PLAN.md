# Phase C Resume Plan — Per-Minibatch SS in TorchMLPModel

**Status:** paused (2026-04-28). Pivoted to `torch.compile` + DE tuning to hit the 5× exploration-speedup target. Phase C is training-time cleanup with no exploration impact; resume when speedup work has landed.

**Branch state at pause:** `feat/integrated-evidence` at commit `3d6030b` (Layer 4 lock-in landed). Tree clean. The in-flight Phase C edits (DataModule deletions, minor `prediction.py` adjustments) were reverted — they had left the codebase in a non-compiling state.

## Goal

Replace the K-refit SS loop in `PredictionSystem.train()` and the DataModule SS-state machinery with a per-epoch SS hook inside `TorchMLPModel.train()`. Net structural cleanup ~120 LOC; no behavioural regression on the SS contract.

## Files to change

### 1. `src/pred_fab/core/datamodule.py`

**Delete:**
- `__init__` lines 60-64: `self._ss_predictions_by_exp`, `self._ss_p_student`, `self._ss_rng` (3 attrs).
- `set_scheduled_sampling_state(...)` (~22 LOC).
- `_perturb_recursive_features(...)` (~71 LOC).
- The perturbation call site in `get_batches` (~6 LOC: the `if split == TRAIN and self._ss_p_student > 0 and …:` block).

**Add:**
- `get_train_row_metadata(split=SplitType.TRAIN) -> list[tuple[str, tuple[int, ...]]]`. Returns per-row `(exp_code, cell_idx)`; row order matches `get_batches(split)` exactly. Used by orchestration to build the per-row prior-cell index map.

### 2. `src/pred_fab/orchestration/prediction.py`

**Delete:**
- `n_ss_rounds: int = 4` attribute (line ~96).
- The K-refit branch in `train()` (lines ~281-299): the entire `if has_recursive and self.n_ss_rounds > 1:` block including `for round_idx in range(self.n_ss_rounds): … set_scheduled_sampling_state … _fit_single_round(…)`.
- `_ss_p_for_round(round_idx, n_rounds)` (~6 LOC).

**Keep:**
- `_autoreg_predict_training_data()` — repurposed for cross-model student value pre-computation only (called once per consumer-model from `_build_ss_config`, not per round).
- `_topo_sort_models()`, `_model_has_recursive_inputs()`, `_filter_batches_for_model()`.

**Replace** the K-refit branch with a single fit + ss_config build:

```python
for model in ordered_models:
    has_recursive = self._model_has_recursive_inputs(model)
    label = f"Training {model.__class__.__name__}"
    _bar = ProgressBar(label, max_iter=1) if console else None

    train_batches = self.datamodule.get_batches(SplitType.TRAIN)
    ss_config = self._build_ss_config(model) if has_recursive else None

    if _bar is not None:
        _bar.step()
    self._fit_single_round(model, train_batches, val_batches, ss_config=ss_config, **kwargs)
    if _bar is not None:
        _bar.finish(suffix="done  (autoregressive)" if has_recursive else "done")
    trained_count += 1
```

**Add** `_build_ss_config(model) -> dict | None` (~80 LOC). Returns a dict with:
- `rec_input_cols: list[int]` — column indices in X_train per recursive input.
- `rec_source_outputs: list[int | None]` — same-model output col index, or `None` for cross-model.
- `prior_row_indices: list[torch.Tensor]` — per-spec int64 tensor of length n_rows, value = prior-cell row index, `-1` for boundary.
- `cross_model_values: list[torch.Tensor | None]` — pre-computed cross-model student values in input-normalized space; `None` for same-model specs.
- `affine_a: list[float]`, `affine_b: list[float]` — per-spec affine transform `student_input_norm = a * y_teacher_norm + b` (composes denorm-output with norm-input). Pre-computed from `dm._feature_stats[source]` and `dm._parameter_stats[input]`. Boundary value = `b` (since `y_norm = 0` raw → `a*0 + b`).
- `boundary_input_norm: list[float]` — per-spec normalized 0 (matches NaN→0→normalize semantics from current code).
- `p_floor: float` (= `self.ss_schedule_floor`).
- `seed: int`.

Construction:
1. `train_row_metadata = dm.get_train_row_metadata(TRAIN)`.
2. Build a `(exp_code, cell_idx) → row_idx` lookup from the metadata.
3. For each recursive input feature on the model, for each `iter_code` in its `recursive_dimensions`:
   - Determine same-model (`source_code in model.outputs`) vs cross-model.
   - For each row, compute prior cell = `cell_idx` with `axis_idx -= depth`. If invalid (axis < 0): `prior_row = -1`. Else: lookup `(exp_code, prior_cell) → row_idx`.
   - Same-model: `rec_source_outputs.append(model.outputs.index(source_code))`, compute affine `(a, b)` from output stats + input stats.
   - Cross-model: pre-compute `cross_model_values` tensor by iterating `_autoreg_predict_training_data()` (once per consumer model) and per-row normalizing with input stats.

The affine transform avoids needing a DataModule reference inside the model. For all four normalization methods (NONE, STANDARD, MIN_MAX, ROBUST) the denorm-then-renorm composition is affine — pre-compute once.

### 3. `src/pred_fab/models/torch_mlp.py`

**Modify** `train()` to accept `ss_config: dict | None = None`. Per-epoch logic:

```python
def train(self, train_batches, val_batches, *, ss_config=None, **kwargs):
    if not train_batches:
        return
    n_outputs = len(self.outputs)

    X = torch.cat([b[0] for b in train_batches], dim=0)
    y = torch.cat([b[1] for b in train_batches], dim=0)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    torch.manual_seed(self.SEED)
    net = self._build_network(X.shape[1], n_outputs)
    optimizer = torch.optim.Adam(net.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    net.train()

    if ss_config is not None:
        n_rows = X.shape[0]
        rec_cols = ss_config["rec_input_cols"]
        rec_outs = ss_config["rec_source_outputs"]
        priors = ss_config["prior_row_indices"]
        cross_vals = ss_config["cross_model_values"]
        a_list = ss_config["affine_a"]
        b_list = ss_config["affine_b"]
        boundary_vals = ss_config["boundary_input_norm"]
        p_floor = float(ss_config["p_floor"])
        ss_gen = torch.Generator()
        ss_gen.manual_seed(int(ss_config["seed"]))

    for epoch in range(self.EPOCHS):
        if ss_config is not None:
            progress = epoch / max(self.EPOCHS - 1, 1)
            p_student = p_floor + (1.0 - p_floor) * progress

            X_used = X.clone()
            if p_student > 0:
                with torch.no_grad():
                    y_teacher = net(X)  # (n_rows, n_outputs), normalised output space

                for r_idx, rec_col in enumerate(rec_cols):
                    prior_rows = priors[r_idx]
                    valid = prior_rows >= 0
                    src_out = rec_outs[r_idx]
                    if src_out is not None:
                        gathered = y_teacher[prior_rows.clamp(min=0), src_out]
                        student = a_list[r_idx] * gathered + b_list[r_idx]
                    else:
                        student = cross_vals[r_idx]  # already in input-norm space
                    flip = torch.rand(n_rows, generator=ss_gen) < p_student
                    apply_student = flip & valid
                    apply_boundary = flip & (~valid)
                    X_used[apply_student, rec_col] = student[apply_student].to(X_used.dtype)
                    X_used[apply_boundary, rec_col] = boundary_vals[r_idx]
        else:
            X_used = X

        optimizer.zero_grad()
        loss = loss_fn(net(X_used), y)
        loss.backward()
        optimizer.step()

    net.eval()
    self._model = net
    self._is_trained = True
```

### 4. Tests

**Delete:** `tests/core/test_datamodule_scheduled_sampling.py` (190 LOC, all 7 tests exercise the removed `set_scheduled_sampling_state` / `_perturb_recursive_features` path).

**Update:** `tests/orchestration/test_scheduled_sampling_train.py`:
- Delete the three `_ss_p_for_round` tests (lines 213-232) — that helper is gone. Replace with tests over the per-epoch schedule (compute `p_student = p_floor + (1-p_floor) * epoch/(EPOCHS-1)` at sampled epochs).
- Topo-sort tests (3) and `_model_has_recursive_inputs` tests (2) stay unchanged.

**Add:** `tests/models/test_torch_mlp_ss.py` (~120 LOC):
- `test_train_no_ss_config_unchanged_path`: verify ss_config=None gives same loss trajectory as today.
- `test_train_with_ss_config_p_floor_zero_round_one_no_perturb`: at epoch=0, p_student=0 → no perturbation → identical to no-SS path for first epoch.
- `test_train_with_ss_config_substitutes_at_p_one`: with p_floor=1.0 → all rows substituted from epoch 0.
- `test_boundary_rows_use_boundary_value`: rows where `prior_row=-1` use `boundary_vals[r_idx]`, not `a*0 + b`.
- `test_cross_model_uses_cross_vals_directly`: `rec_outs[r_idx] is None` path uses `cross_vals[r_idx]` and not the model's own forward.

### 5. `src/pred_fab/orchestration/agent.py`

Line ~631: `self.pred_system.n_ss_rounds = int(n_rounds)` — delete (no more rounds). The `configure_scheduled_sampling(n_rounds, schedule_floor)` agent method should drop the `n_rounds` argument:

```python
def configure_scheduled_sampling(self, *, schedule_floor: float = 0.0) -> None:
    """Configure scheduled-sampling schedule floor for recursive models."""
    self._assert_initialized()
    self.pred_system.ss_schedule_floor = float(schedule_floor)
```

Or, for backwards compatibility during transition, keep `n_rounds` as a no-op kwarg with a deprecation warning. Cleaner to just delete since the API surface is internal.

## Risk register

- **Cross-model SS semantics drift.** Today's K-refit refreshes cross-model predictions every round (using the consumer's evolving state via topo-sort, then re-predicting all). Phase C makes cross-model values *static* (computed once before the consumer's training starts). Why this is fine: the source model is already trained and frozen by topo-sort; refreshing during the consumer's training adds no information. Verify on the smoke test that cross-model SS-using models (if any in the mock) don't regress.
- **Affine transform precision.** The `(a, b)` composition is in float64 at construction time; `student = a * y_teacher + b` runs in float32. Should be fine within the 1e-5 tolerance the equivalence tests use.
- **Per-epoch refresh cost.** With EPOCHS=1500 and n_rows ~100, each epoch's `with torch.no_grad(): net(X)` is a single forward pass on a small batch — negligible. Total SS overhead per training: 1500 forward passes × a few microseconds = ms-scale.
- **Backward compat for `configure_scheduled_sampling(n_rounds=...)`.** Mock and pred-fab callers may pass `n_rounds`. Grep for callers before deleting; either drop the kwarg cleanly or keep it as a deprecated no-op for one release.

## Validation plan

1. Pre-existing 586 tests + 3 batched-equivalence tests should stay green after Phase C — the SS contract is preserved, only its implementation moves.
2. Add the 5 new TorchMLPModel SS tests; confirm green.
3. Smoke test the mock end-to-end (`pred-fab-mock/dev/_smoke_layer3.py`) — verify exploration step time stays at the post-Layer 4 baseline (Phase C shouldn't move exploration; if it does, investigate).
4. Verify `MODELS_CONTEXT.md` and `PROJECT_CONTEXT.md` references to SS state / K-refit are updated.

## Estimated diff

| | LOC delta |
|---|---|
| `datamodule.py` | -100 (delete state + setter + perturb + call site), +25 (`get_train_row_metadata`) → **-75** |
| `prediction.py` | -25 (K-refit + `_ss_p_for_round` + `n_ss_rounds`), +80 (`_build_ss_config`) → **+55** |
| `torch_mlp.py` | +50 (per-epoch SS branch) |
| `agent.py` | -3 |
| `tests/core/test_datamodule_scheduled_sampling.py` | -190 (deleted) |
| `tests/orchestration/test_scheduled_sampling_train.py` | -20 (drop `_ss_p_for_round` tests) |
| `tests/models/test_torch_mlp_ss.py` | +120 (new) |
| **Net** | **-63 LOC** with structural simplification (single SS path, model-owned schedule). |
