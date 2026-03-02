# Parameter Levels: Experiment-Level vs. Runtime-Adjustable Parameters

> **Status:** Living document — `runtime_adjustable` attribute + `runtime` flag on `Parameter.real()` / `Parameter.integer()` implemented
> **Related:** [02-exploration-scope.md](02-exploration-scope.md), [03-uncertainty-estimation.md](03-uncertainty-estimation.md)

---

## Problem Statement

PFAB's schema system currently treats all parameters as equivalent: they are defined once, set before an experiment begins, and held constant throughout execution. In practice, fabrication processes contain parameters that operate at fundamentally different temporal levels:

- **Experiment-level parameters**: fixed before execution begins (e.g., layer height, material, nozzle diameter). These define the *configuration* of an experiment.
- **Runtime-adjustable parameters**: physically and computationally changeable *during* execution at each dimensional step (e.g., nozzle speed, extrusion rate, temperature setpoint). These can be adapted mid-experiment by the adaptation system, and in future by the exploration system.

This distinction existed implicitly in the codebase (`ParameterUpdateEvent`, `get_effective_parameters_for_row()`), but was not formalized at the schema level. The `runtime_adjustable` attribute on `DataObject` now closes this gap.

---

## Current Architecture

### Parameter Types in `DatasetSchema`

All parameters are defined using the type hierarchy with role `PARAMETER`:

| Type | Use case | `runtime_adjustable` | `role` |
|------|----------|----------------------|--------|
| `DataReal` / `Parameter.real()` | Continuous experiment-level param | `False` | `Roles.PARAMETER` |
| `DataInt` / `Parameter.integer()` | Integer experiment-level param | `False` | `Roles.PARAMETER` |
| `DataBool` / `Parameter.boolean()` | Boolean flag | `False` | `Roles.PARAMETER` |
| `DataCategorical` / `Parameter.categorical()` | Discrete choices | `False` | `Roles.PARAMETER` |
| `DataDimension` / `Parameter.dimension()` | Iteration structure (n_layers etc.) | `False` | `Roles.PARAMETER` |
| `DataReal` / `Parameter.real(..., runtime=True)` | Continuous runtime-adjustable param | **`True`** | `Roles.PARAMETER` |
| `DataInt` / `Parameter.integer(..., runtime=True)` | Integer runtime-adjustable param | **`True`** | `Roles.PARAMETER` |

`runtime_adjustable` is a plain boolean instance attribute on `DataObject` (default `False`), set by the `Parameter` factory after construction when `runtime=True`. The `role` stays `Roles.PARAMETER` in all cases — `Roles` is used exclusively for block membership (`DataBlock.add()` enforcement). This enables the rest of the system to query any parameter's adjustability without type-specific branching and without breaking the `Parameters` block.

### Runtime Parameter Semantics

A `RuntimeParameter` does **not** follow a predefined trajectory or change schedule. Its value at any dimensional step is determined by the optimization objective (adaptation or exploration) within the configured trust region. The defining property is *permission to be re-proposed at runtime*, not a specific change pattern.

This mirrors how online adaptation already works: `CalibrationSystem.run_adaptation()` proposes new values for parameters within their trust regions, and the result is recorded as a `ParameterUpdateEvent`. RuntimeParameter types formalize which parameters are eligible for this process.

### Effective Parameter Reconstruction (already correct)

The infrastructure for recording and replaying mid-experiment parameter changes is already in place and working correctly:

```python
# ParameterUpdateEvent records step-level changes
exp.record_parameter_update(
    proposal=ParameterProposal({"speed": 150.0}),
    dimension="layer_id",
    step_index=3
)

# Reconstructs the full effective parameter state at any flattened row index
eff_params = exp.get_effective_parameters_for_row(row_index=350)
```

Critically, `Dataset.export_to_dataframe()` already calls `get_effective_parameters_for_row(row_idx)` for every row when building PM training data. This means PM training already correctly sees runtime parameter changes when they have been recorded. No changes are needed here.

---

## Implementation: `runtime_adjustable` attribute and the `runtime` flag

### Design

`Roles` serves a single purpose: block membership (`PARAMETER`, `FEATURE`, `PERFORMANCE`). Sub-classifications within a block — such as runtime-adjustability — are expressed as plain attributes on `DataObject` directly.

`DataObject.runtime_adjustable` is a boolean instance attribute (default `False`). The `Parameter` factory sets it to `True` after construction when `runtime=True`. The `role` remains `Roles.PARAMETER` in all cases, preserving full compatibility with `DataBlock.add()`.

### Location
- `src/pred_fab/utils/enum.py` — `Roles` contains only `PARAMETER`, `FEATURE`, `PERFORMANCE`
- `src/pred_fab/core/data_objects.py` — `runtime_adjustable: bool = False` attribute on `DataObject.__init__()`, serialised in `to_dict()` / restored in `from_dict()`, `runtime` flag on `Parameter.real()` and `Parameter.integer()`

### API

```python
# Schema definition — runtime=True sets runtime_adjustable=True; role stays Roles.PARAMETER
schema.parameters.add(Parameter.real("speed",   min_val=10.0, max_val=200.0, runtime=True))
schema.parameters.add(Parameter.integer("fan_rpm", min_val=0, max_val=3000,  runtime=True))

# Querying adjustability — works on any DataObject
for code, param in schema.parameters.data_objects.items():
    if param.runtime_adjustable:
        print(f"{code} can be adjusted at runtime")

# Validation/coercion is unchanged — same DataReal/DataInt behavior
p = Parameter.real("speed", 10.0, 200.0, runtime=True)
p.validate(75.0)      # → True
p.coerce(75.678)      # → 75.678 (rounds per round_digits)
p.role                # → Roles.PARAMETER  (block membership unchanged)
p.runtime_adjustable  # → True

# Serialization roundtrip preserves runtime_adjustable
restored = DataObject.from_dict(p.to_dict())
restored.role              # → Roles.PARAMETER
restored.runtime_adjustable  # → True
```

### Design Notes

- No new classes. `DataReal` and `DataInt` are unchanged; `runtime_adjustable` is an additional instance attribute.
- `Roles` is exclusively for block membership — `DataBlock.add()` enforces `data_obj.role == self.role`. Runtime parameters pass this check because `role == Roles.PARAMETER`.
- Serialization: `to_dict()` emits `"runtime_adjustable": true` only when set; `from_dict()` restores it via `data.get("runtime_adjustable", False)`. The role field is unchanged.
- No migration needed — pre-production software.

---

## PM Handling of Runtime Parameters

### Dimensional PM (operates at feature-measurement level)

If the PM is trained on dimensional data (one row per measurement position), each row already carries effective parameter values via `get_effective_parameters_for_row()`. Runtime parameter changes propagate naturally — each position in the training data reflects the actual parameter state at that position. No additional handling needed.

### Non-Dimensional PM (operates at experiment level)

If a PM aggregates data per experiment (e.g., predicts a single summary feature rather than a spatial array), the runtime parameter varies within the experiment but the PM receives only one input vector. A helper function on `ExperimentData` can extract statistical representations:

```python
# Proposed API — not yet implemented
stats = exp.get_runtime_parameter_stats("speed")
# Returns: {"initial": 50.0, "final": 150.0, "min": 50.0, "max": 150.0, "mean": 95.3, "std": 32.1}
```

The PM implementer chooses which statistics to use as input features (e.g., mean speed, speed range). This is a PM-design concern and can be handled in the concrete `IPredictionModel` implementation without changes to the core schema. When this pattern becomes common, `get_runtime_parameter_stats()` will be added to `ExperimentData`.

---

## Schema Boundaries and Configuration Design

### The Governing Principle

The schema answers one question: *what is a dataset?* Specifically, what parameters exist, what are their types, what are their physical limits, and which ones can be re-proposed during execution. Everything else is optimizer configuration: how to explore, how far to move per step, what to hold fixed this run.

This produces a clean two-layer boundary:

| Concern | Where it lives | Changes require new schema? |
|---------|---------------|----------------------------|
| Parameter identity (code, type) | Schema structural identity | Yes — changes hash |
| Physical bounds (min / max) | Schema structural identity | Yes — changes hash |
| `runtime_adjustable` flag | Schema operational metadata | **No — stored but not hashed** |
| Trust region delta | `CalibrationSystem` config | No |
| Performance weights | `CalibrationSystem` config | No |
| Exploration sub-bounds (narrower than schema) | `CalibrationSystem` config | No |
| Fixed-param overrides | `CalibrationSystem` config | No |

The criterion for what enters the hash is: does this change alter what data *can be recorded*? Codes, types, and physical bounds define the recording structure — changing any of them means old and new data are structurally incompatible. The `runtime_adjustable` flag does not affect what can be recorded: `ParameterUpdateEvent`s can be recorded for any parameter regardless of this flag (the infrastructure does not gate on it). The flag only changes what the CalibrationSystem *targets* — which is optimizer behaviour, not recording structure.

This is implemented via a separate `to_hash_dict()` method on `DataObject` that includes only structural fields, and is used by `_block_to_hash_structure()` in `DatasetSchema._compute_schema_hash()`. The full `to_dict()` (which includes `runtime_adjustable`) is still used for storage and serialization, so the flag is persisted correctly and restored on deserialization. The hash simply does not see it.

### Do Trust Regions Belong in the Schema?

No, for three reasons:

1. **They are optimizer configuration, not parameter physics.** `min`/`max` express the physical envelope of the parameter (the nozzle speed physically cannot exceed 200 mm/s). A trust region delta expresses the optimizer's conservatism (we limit changes to ±50 mm/s per step to avoid thermal shock). The first property belongs to the parameter; the second belongs to the strategy applied to it.

2. **They are expected to change as the process is understood.** Early in dataset collection, trust regions are conservative. After observing that the process tolerates large changes, they are widened. This should not force a schema change and a new dataset.

3. **They are machine- and context-dependent.** The same schema might be used on two printers with different thermal dynamics, each requiring different deltas. The schema should be portable; the calibration configuration should not be.

### What Changes When runtime_adjustable Changes?

Changing a parameter from `runtime=False` to `runtime=True` is an **operational** change, not a structural one:

- **Dataset recording**: `ParameterUpdateEvent`s *can* be recorded for any parameter regardless of the flag — the infrastructure does not gate on `runtime_adjustable`. In practice, old experiments for a parameter marked `runtime=False` simply have no such events (the parameter was held constant). After changing the flag, new experiments may have events; old experiments still have none. Both are consistent: `get_effective_parameters_for_row()` handles absent events by returning the initial value.
- **PM training**: `export_to_dataframe()` already calls `get_effective_parameters_for_row()` for every parameter unconditionally. Adding or removing the flag changes nothing about what the dataframe contains — the effective parameter value for a static experiment is just the constant initial value.
- **Hash and registry**: `_compute_schema_hash()` uses `to_hash_dict()`, which excludes `runtime_adjustable`. The hash is unchanged. The same schema name continues to work. No new dataset is required.
- **CalibrationSystem behavior**: After changing the flag, `configure_trajectory()` will now include (or exclude) the parameter from trajectory proposals. This is a pure optimizer behavior change, applied from the next call forward.

### Trust Region Enforcement in CalibrationSystem

Even though trust regions are not in the schema, they are required for `run_adaptation()` to work correctly for any runtime parameter. Two validation rules follow from the boundary above:

1. **`configure_adaptation_delta()` must reject non-runtime parameters.** If a parameter has `runtime_adjustable == False`, configuring a trust region for it is a configuration error. The intent of trust regions is precisely to constrain how much a runtime parameter can move per step; applying one to a static parameter is semantically nonsensical and should surface as an error at configuration time, not silently succeed and produce confusing behavior during adaptation.

2. **`run_adaptation()` should fail fast if a runtime parameter has no configured trust region.** A schema that declares `runtime=True` but has no delta configured in `CalibrationSystem` is internally inconsistent for the purpose of adaptation. Rather than silently fixing the parameter to its current value (which is what happens today via the `if k not in self.trust_regions: fix to current` path in `run_adaptation()`), the system should raise at adaptation time with a clear message identifying the unconfigured parameters.

Both validations are implemented in `CalibrationSystem`, not in the schema. This keeps the schema clean and enforces the contract at the level where it can be checked: when the optimizer is actually running.

### Is the Boolean Flag Sufficient?

Yes, for all currently identified needs. The boolean answers the structural question ("is this parameter eligible for runtime re-proposal?") and that is all the schema needs to express.

The concerns that might seem to argue for a richer representation are all either CalibrationSystem concerns (trust regions, adaptation frequency) or deferred architectural concerns (per-step rate limits if trajectory-based exploration is implemented per [Document 2](02-exploration-scope.md)).

One pattern that might arise: a concrete `IPredictionModel` implementation may want to treat runtime parameters differently during inference — e.g., constructing separate input heads for static vs. runtime parameters. This is handled by querying `schema.parameters.data_objects[code].runtime_adjustable` at model construction time. No richer representation is needed for this.

If dimensional-level exploration (Document 2, Option A) eventually requires expressing *how* a parameter can change mid-experiment (e.g., a rate limit as a physical property distinct from the optimizer's trust region), a `max_rate_per_step: Optional[float]` field could be added to `DataObject`. That is a future concern and should not drive the current design.

---

## CalibrationSystem and Runtime Parameters

The existing `CalibrationSystem.configure_adaptation_delta()` accepts parameter codes to set trust regions for online adaptation. Two changes follow from the boundary analysis above:

1. **`configure_adaptation_delta()` must validate runtime-adjustability.** If a parameter has `runtime_adjustable == False`, the method should raise `ValueError`, not silently apply the trust region. This enforces the principle that trust regions are exclusively a runtime-parameter concept.

2. **`run_adaptation()` should validate completeness.** Before running, it should check that every runtime-adjustable parameter in the schema has a configured trust region, and raise with a clear message if any are missing.

These are the next implementation items for CalibrationSystem — they are follow-on to the schema design work here and are independent of the exploration redesign in [Document 2](02-exploration-scope.md).

---

## Triggered Step-Level Proposals (replacing trajectory concept)

Rather than a predefined trajectory where runtime parameter changes are scheduled before fabrication, the preferred model is **triggered proposals**: at predefined dimensional steps during execution, the system calls the adaptation or exploration step for runtime parameters only, and applies the result as a `ParameterUpdateEvent`.

```
During experiment execution:
  For each dimensional step i:
    if i in trigger_steps:
      proposal = calibration_system.run_adaptation(runtime_params_only=True, ...)
      exp.record_parameter_update(proposal, dimension="layer_id", step_index=i)
    execute step i with effective parameters
```

This requires no changes to `ParameterProposal` (it already carries a flat dict) — just a scheduling mechanism that determines when mid-experiment proposals are triggered. The adaptation or exploration logic itself is unchanged.

---

## Open Questions

1. **Non-dimensional PM helper:** When should `get_runtime_parameter_stats()` be added to `ExperimentData`? Once a concrete non-dimensional PM implementation requires it.

2. **Exploration trigger schedule:** Who defines `trigger_steps`? Should this be part of the experiment configuration, the exploration system, or both? (See [Document 2](02-exploration-scope.md) for discussion.)

3. **CalibrationSystem subspace filtering:** When running mid-experiment proposals, the CalibrationSystem must operate only over the runtime parameter subspace. The bounds, trust regions, and acquisition function must be restricted accordingly. This is straightforward given the `runtime_adjustable` property is now queryable.

4. **Rate-of-change limits as schema field:** If trajectory-based exploration (Document 2, Option B) requires expressing physical constraints on how fast a runtime parameter can change (independent of the optimizer's trust region), a `max_rate_per_step: Optional[float]` field on `DataObject` would be appropriate. This would be a schema-level physical constraint, not optimizer configuration, and would require a new schema when changed. Deferred until a concrete use case drives it.

---

## Status

| Item | Status |
|------|--------|
| `Roles` enum — block membership only (PARAMETER / FEATURE / PERFORMANCE) | ✅ Implemented |
| `DataObject.runtime_adjustable` plain boolean attribute (default `False`) | ✅ Implemented |
| `Parameter.real(..., runtime=True)` flag | ✅ Implemented |
| `Parameter.integer(..., runtime=True)` flag | ✅ Implemented |
| `to_dict()` / `from_dict()` roundtrip preserves `runtime_adjustable` | ✅ Implemented |
| `to_hash_dict()` excludes `runtime_adjustable` — flag does not change schema hash | ✅ Implemented |
| Tests for all runtime-adjustable behavior | ✅ Implemented (372 passing) |
| `export_to_dataframe()` uses effective params | ✅ Already correct |
| Schema boundary documented (schema vs CalibrationSystem config) | ✅ Documented |
| `configure_adaptation_delta()` rejects non-runtime parameters | ✅ Implemented (Phase 2) |
| `run_adaptation()` validates all runtime params have trust regions | ✅ Implemented (Phase 2) |
| `ParameterSchedule` + `ExperimentSpec` dataclasses | ✅ Implemented (Phase 3) |
| `configure_trajectory()` on `CalibrationSystem` | ✅ Implemented (Phase 3) |
| `generate_baseline_experiments()` returns `List[ExperimentSpec]` with optional schedules | ✅ Implemented (Phase 3) |
| `run_trajectory_exploration()` with LHS warm-start + SLSQP step constraints | ✅ Implemented (Phase 3) |
| `get_runtime_parameter_stats()` helper | 🔲 Deferred (PM-design concern) |
