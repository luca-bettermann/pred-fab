# Coding Standards

**Quick Reference:** Self-documenting code, type safety, validation at boundaries, integration-first testing.

---

## Code Style

### Principles
- **Modularity** + **Type Safety** (Pylance): Small modules, explicit type hints everywhere
- **Encapsulation**: Private `_methods`, public API via `@final`
- **Self-Documenting**: Code structure explains intent, not comments

### Code Blocks

Compact blocks (5-15 lines) with descriptive comments:

```python
# Create evaluation function  
def evaluate_fn(exp_code: str, predicted_features: Dict[str, float]) -> Dict[str, float]:
    """Evaluate predicted features using existing evaluation logic."""
    performances = {}
    for code, eval_model in self.eval_system.get_active_eval_models().items():
        if eval_model.weight is None:
            continue
        if code not in predicted_features:
            raise ValueError(f"Predicted features missing code '{code}'")
        performances[code] = float(eval_model._compute_performance(...))
    return performances

# Run calibration
optimal_params = self.calibration_model.calibrate(...)
```

### Docstrings

**Abstract methods** (no code): Comprehensive with context
```python
@property
def feature_model_types(self) -> Dict[str, Type[IFeatureModel]]:
    """
    Feature models this prediction model depends on.
    
    Returns:
        Dict mapping feature codes to IFeatureModel types
        (e.g., {'path_dev': PathDeviationFeature})
    """
```

**Concrete methods** (code explains itself): One-line summary
```python
def _infer_data_object(self, param_name: str, field) -> Any:
    """Factory for creating parameter DataObjects."""
```

**`__init__` methods**: Always document arguments with Args section
```python
def __init__(self, dataset: Dataset, logger: LBPLogger, threshold: float = 0.5):
    """
    Initialize evaluation model.
    
    Args:
        dataset: Dataset instance for feature memoization
        logger: Logger for debugging and progress tracking
        threshold: Performance threshold for pass/fail (default: 0.5)
    """
```

**Classes**: 3-5 bullet points stating purpose/responsibilities (no design rationale)

---

## Documentation

### Files in `docs/`
- `DESIGN_DECISIONS.md` - Architectural choices
- `SEPARATION_OF_CONCERNS.md` - Module boundaries
- `CORE_DATA_STRUCTURES.md` - Key data models
- `IMPLEMENTATION_SUMMARY.md` - Major refactorings

### Rules
1. **Remove deprecated content** when updating
2. **Document final state** (A → D), not journey (A → B → C)
3. **Review design docs** before refactoring to avoid circular changes

---

## Testing

**Hierarchy:** Integration tests (primary) > Unit tests (critical logic) > E2E (smoke tests)

**Workflow:**
1. Run `pytest tests/`
2. Add tests for new functionality
3. Clean up deprecated tests

**Focus:**
- Edge cases (null, empty, boundaries)
- Failure paths
- Public API methods
- Validation logic

---

## Validation

### Always validate abstract method outputs:
```python
target_value = self._compute_target_value(**params)
if not isinstance(target_value, (int, float)):
    raise TypeError(f"Expected numeric, got {type(target_value).__name__}")
```

### Error messages: state problem + expected vs actual + action
```python
if missing:
    raise ValueError(f"Missing params: {missing}. Expected: {required}")
```

### Patterns:
- **Pre-condition**: Check inputs before processing
- **Post-condition**: Validate outputs before use
- **Type safety**: `float(v) if v is not None else 0.0`

### Code Annotations

Use standard comment tags for searchability:

- `# TODO:` - Missing functionality to implement later
- `# FIXME:` - Known bugs or incorrect behavior that needs fixing
- `# NOTE:` - Important context or non-obvious design decisions
- `# HACK:` - Temporary workaround that should be refactored
- `# OPTIMIZE:` - Performance improvement opportunities

Example:
```python
# NOTE: Schema hash excludes values to ensure stability across experiments
schema_hash = self._compute_schema_hash()

# TODO: Add batch processing for large datasets
for exp_code in exp_codes:
    process_single(exp_code)
```

---

## Quick Patterns

```python
# Nested functions for closures
def outer():
    def inner(x):
        return process(x)
    return inner(data)

# Logging hierarchy
self.logger.info("Internal progress")
self.logger.console_warning("User-facing issue")
self.logger.console_summary(formatted_output)

# Safe type coercion
formatted = {k: round(float(v), digits) for k, v in params.items()}
```

---

## Agent Checklist

Before implementation:
- [ ] Read `DESIGN_DECISIONS.md` and `SEPARATION_OF_CONCERNS.md`
- [ ] Type hints on all signatures
- [ ] Validate user-implemented methods
- [ ] Integration tests for new features
- [ ] Remove deprecated content from docs
- [ ] Run full test suite
