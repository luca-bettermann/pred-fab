# Calibration Strategy: System + Strategy Pattern

## Overview
The calibration module in the LBP framework is designed to find optimal process parameters that maximize performance metrics. This process involves exploring the parameter space and exploiting high-performing regions.

We have adopted a **System + Strategy** architecture where the orchestration logic is separated from the optimization algorithm.

## Rationale

### 1. Decoupling Uncertainty from Prediction Models
Most state-of-the-art Deep Learning models (Transformers, GNNs, etc.) are deterministic and do not output calibrated uncertainty estimates by default. Implementing uncertainty quantification (e.g., via Ensembles or Monte Carlo Dropout) for every user-defined model adds significant complexity and computational overhead.

By using a Gaussian Process (GP) as a surrogate model within the calibration strategy, we decouple the uncertainty requirement from the prediction model. The GP models the objective function's surface and provides its own uncertainty estimates based on the density of sampled points.

### 2. Computational Efficiency (Online Learning)
In an online learning context (e.g., during fabrication), we need to optimize parameters quickly.
- **Direct Optimization**: Running an optimizer (like L-BFGS-B) directly on a heavy prediction model (e.g., a large Transformer) requires thousands of forward passes, which is computationally prohibitive for real-time applications.
- **Surrogate Optimization**: The surrogate model is trained on a small number of samples (e.g., 20-50) from the heavy prediction model. The optimizer then runs on the *surrogate* model, which is extremely fast (milliseconds). This allows for efficient exploration of the search space without bottlenecking the system.

### 3. Simplified User Interface
The "Direct Mode" (where users implement `predict_uncertainty`) was removed to simplify the interface. Users only need to provide a standard prediction model (inputs -> outputs). The calibration system handles all exploration/exploitation logic internally.

## Architecture

### CalibrationSystem
The `CalibrationSystem` is the orchestrator that manages the optimization loop. It:
- Owns the **Surrogate Model** (GP).
- Defines **System Performance** as a weighted sum of individual performance metrics.
- Generates baseline experiments using **Latin Hypercube Sampling (LHS)**.
- Delegates the decision of "where to sample next" to a **Strategy**.

### BayesianCalibrationStrategy
The default strategy (`BayesianCalibrationStrategy`) implements `ICalibrationStrategy` and uses:
- **Surrogate Model**: `sklearn.gaussian_process.GaussianProcessRegressor` with Matern kernel.
- **Acquisition Function**: Upper Confidence Bound (UCB).
- **Modes**:
    - `exploration`: Prioritizes high uncertainty (Active Learning).
    - `optimization`: Prioritizes high predicted performance (Exploitation).

### Workflow
1.  **Initialize**: `agent.configure_calibration(performance_weights=...)` sets up the system goals.
2.  **Baseline**: `agent.calibration_system.generate_baseline_experiments(...)` creates initial random points.
3.  **Loop**:
    - `agent.propose_next_experiments(...)` asks the strategy for the next best point(s).
    - User/System runs the experiment.
    - Data is added to the dataset.
    - `CalibrationSystem` updates the surrogate model with the new history.
    - Repeat.

## Future Potential
The surrogate model artifact itself has value beyond calibration:
- **Fast Inference**: It can be exported as a lightweight proxy for the heavy prediction model for design-stage applications requiring instant feedback.
- **Transparency**: The GP provides an explicit uncertainty map of the design space, showing where the model is confident and where it is guessing.
