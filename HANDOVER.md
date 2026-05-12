# Handover — feat/slope-trajectory

## What was done

Unified the calibration optimization pipeline into a single Sobol→LBFGS path. Major refactor across pred-fab and learning-by-printing.

### New architecture

```
Step method (run_baseline, etc.)
  → composes list[Variable] (StaticVariable / TrajectoryVariable)
  → calls _optimize(n, variables, kappa, ...)
    → SolutionSpace: builds bounds, prior_fill, single decode()
    → Engine: Sobol → top-N → 16 independent LBFGS → pick best
    → SolutionSpace: decode_to_specs(best_x)
```

### Key files

| File | Role |
|------|------|
| `calibration/space.py` | Variable types + SolutionSpace (bounds, decode, specs) + tanh slope decode |
| `calibration/engine.py` | OptimizationEngine: Sobol → top-N → independent LBFGS |
| `calibration/system.py` | `run_baseline` composes Variables, `_optimize` is the single path (~30 lines) |
| `agent.py` | `configure_optimizer(n_starts, n_sobol, lr, acquisition_scale)` |

### What was deleted

- Differential Evolution (entirely)
- `_run_acquisition_phase`, `_optimize_experiments`, `_recursive_split`, `_filter_phase_params`
- `slope.py`, `variables.py` (merged into `space.py`)
- Old `SolutionSpace` (static-only, rewritten)
- Coordinate descent trajectory, smoothness penalties
- CLI args: `--gradient-method`, `--local-maxiter`, `--raw-samples`, `--init-eta`, `--smoothness-weight`, `--max-trajectory-rounds`

### Net change

-395 lines, 683 tests passing, 0 pyright errors.

## Uncommitted fix in working tree

**`space.py` — `_derive_L_per_exp` tensor shape bug in `decode_to_specs` and `get_trajectory_plot_data`.**
The old code did `x_t.reshape(1, N, D)[0:1, :, :D_static]` which kept the batch dim, causing `static_vals[i, si]` to return a tensor instead of a scalar. Fixed by reshaping to `(N, D)` directly without the batch dim. This fix is staged but not committed.

## Known issues to address next

### 1. D vs V labeling (user-reported)
The progress bar shows `D=60` but the actual parameter dimensionality is `D=5` (with `V=60` total decision variables for 12 experiments × 5 params). This matters because KernelField evidence probe count scales with D. Need to pass both D_per_exp and total_vars to the engine/progress bar.

### 2. Sobol scaling uses V instead of D
`n_sobol_scaled = max(512, 32 * D)` uses the decision vector dimension (60) instead of per-experiment parameter count (5). With the real D=5: `max(512, 160) = 512` candidates — much fewer and faster. Need to pass a `d_param` argument to `engine.optimize()`.

### 3. Exploration/inference still use old code path
`exploration_step` and `inference_step` flow through `run_calibration` → `_run_phase` → `_build_objective` — a completely separate optimization path that duplicates objective construction, trajectory handling, etc. These should be migrated to compose Variables and call `_optimize` with the appropriate κ, just like `run_baseline` does. The only real difference is κ value and n=1.

The adaptation path (`adaptation_step`) has trust-region bounds + MPC lookahead and can stay separate for now.

### 4. Decode if/else cleanup (partially done)
`decode()` in space.py was cleaned up to remove trajectory vs non-trajectory branching. The remaining `if self._D_traj > 0:` guards are not alternate code paths — they skip work that doesn't apply when there are no trajectory variables (the scatter loop is empty). These are clean.

## Branch state

- **pred-fab**: `feat/slope-trajectory`, pushed to remote. One uncommitted fix (tensor shape bug).
- **learning-by-printing**: `feat/slope-trajectory`, CLI args and session.py updated. Pushed.
- **knowledge-base**: `main`, pushed. Updated: PFAB - Calibration.md, PFAB - Baseline.md, PFAB - Design Decisions.md, ORCHESTRATION_CONTEXT.md, task note.

## Kanban

Task: `Sigmoid slope trajectory in Global optimization` — in progress column.
