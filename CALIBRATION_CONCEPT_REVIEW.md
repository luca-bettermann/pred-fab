# Calibration Concept Review & Architecture Strategy

**Date**: November 28, 2025
**Topic**: Calibration (Optimization, Exploration, Mixed)

---

## 1. Concept Validation
Your concept aligns perfectly with **Active Learning** and **Bayesian Optimization (BO)** principles.
- **Optimization** $\rightarrow$ **Exploitation**: Trust the model and go to the peak.
- **Exploration** $\rightarrow$ **Exploration**: Go where uncertainty is high to learn more.
- **Mixed** $\rightarrow$ **Acquisition Functions**: Mathematical formulas (like UCB) that balance both.

This is the correct approach for "Automated DoE". By systematically reducing uncertainty in high-performing regions, you converge to the optimal process parameters significantly faster than Grid Search or Random Search.

---

## 2. Architecture Decisions (Q&A)

### Q1: Tradeoff Flexibility vs. Predefined Methods?
**Recommendation**: **Predefine the Algorithm, User defines the Context.**
Implementing a robust Active Learning loop is mathematically complex (Gaussian Processes, Acquisition Functions, Optimization loops).
- **Don't** ask the user to implement the optimization algorithm (e.g., "write your own gradient descent").
- **Do** ask the user to define:
    1.  **Search Space**: Which parameters can vary? (Already in Schema)
    2.  **Objective**: What are we optimizing? (Which performance attributes?)
    3.  **Constraints**: Are there unsafe regions?

**Strategy**: Provide a robust `BayesianCalibration` class as part of the framework. Users can instantiate this with their specific configuration. Advanced users can still subclass `ICalibrationModel` if they want to implement something custom (e.g., Genetic Algorithms).

### Q2: Do we need separate algorithms for the two modes?
**No.**
You can use a single **Bayesian Optimization** engine for all three modes by changing the **Acquisition Function**:
- **Optimization**: Maximize `Predicted_Mean` (or Expected Improvement).
- **Exploration**: Maximize `Predicted_StdDev` (Uncertainty).
- **Mixed**: Maximize `Predicted_Mean + k * Predicted_StdDev` (Upper Confidence Bound).

The `exploration` parameter $[0, 1]$ you proposed maps directly to the $k$ (kappa) parameter in UCB.
- $k=0$: Pure Optimization.
- $k=\infty$: Pure Exploration.
- $k \approx 1.96$: Balanced (95% CI).

### Q3: How to fix some parameters and explore others?
**Implementation**: Add a `fixed_params` argument to the `calibrate` method.
The optimizer will treat these dimensions as constants (collapsing the search space) and only optimize the remaining free parameters. This is essential for "slicing" the solution space.

### Q4: Quantifying Uncertainty?
**Challenge**: Most user-defined `IPredictionModel`s (e.g., Linear Regression, simple Neural Nets) do **not** output uncertainty.
**Solution**: The Calibration System should maintain its own **Surrogate Model** (e.g., a Gaussian Process).
1.  The Surrogate trains on the same `(X, y)` data as the Prediction Model.
2.  The Surrogate *estimates* the landscape and uncertainty.
3.  The Surrogate proposes the next point.
4.  The User's Prediction Model is then used to *validate* or *predict* the value at that point, but the *proposal* comes from the Surrogate.

This decouples the complex math of Exploration from the user's domain-specific Prediction Model.

---

## 3. Implementation Plan

### Step 1: Refactor `ICalibrationModel`
Update the interface to support:
- `fixed_params`: For partial optimization.
- `exploration_weight`: The $[0, 1]$ parameter.
- Access to `Dataset`: To train the surrogate model.

### Step 2: Implement `BayesianCalibration`
Create a concrete implementation using `scikit-learn` (GaussianProcessRegressor) and `scipy.optimize`.
- **Inputs**: Dataset, Target Performance.
- **Logic**:
    1.  Fit GP to current data.
    2.  Define Acquisition Function based on `exploration_weight`.
    3.  Optimize Acquisition Function to find next `X`.
- **Outputs**: Proposed `X`.

### Step 3: Integration
Update `LBPAgent` to expose these capabilities.

---

## 4. Proposed Interface Changes

```python
class ICalibrationModel(ABC):
    @abstractmethod
    def propose_next_experiment(
        self,
        dataset: Dataset,
        target_performance: str,
        param_ranges: Dict[str, Tuple[float, float]],
        fixed_params: Dict[str, Any],
        exploration_weight: float = 0.0
    ) -> Dict[str, Any]:
        """
        Propose parameters for the next experiment.
        
        Args:
            dataset: Current dataset (for training surrogate)
            target_performance: Name of performance attribute to optimize
            param_ranges: Allowed ranges for free parameters
            fixed_params: Values for fixed parameters
            exploration_weight: 0.0 (Optimize) to 1.0 (Explore)
        """
        ...
```
