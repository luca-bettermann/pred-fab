# Calibration Strategy: Surrogate-First Approach

## Overview
The calibration module in the LBP framework is designed to find optimal process parameters that maximize performance metrics. This process involves exploring the parameter space and exploiting high-performing regions.

We have adopted a **Surrogate-First** strategy using Bayesian Optimization (Gaussian Processes) as the standard implementation.

## Rationale

### 1. Decoupling Uncertainty from Prediction Models
Most state-of-the-art Deep Learning models (Transformers, GNNs, etc.) are deterministic and do not output calibrated uncertainty estimates by default. Implementing uncertainty quantification (e.g., via Ensembles or Monte Carlo Dropout) for every user-defined model adds significant complexity and computational overhead.

By using a Gaussian Process (GP) as a surrogate model, we decouple the uncertainty requirement from the prediction model. The GP models the objective function's surface and provides its own uncertainty estimates based on the density of sampled points.

### 2. Computational Efficiency (Online Learning)
In an online learning context (e.g., during fabrication), we need to optimize parameters quickly.
- **Direct Optimization**: Running an optimizer (like L-BFGS-B) directly on a heavy prediction model (e.g., a large Transformer) requires thousands of forward passes, which is computationally prohibitive for real-time applications.
- **Surrogate Optimization**: The surrogate model is trained on a small number of samples (e.g., 20-50) from the heavy prediction model. The optimizer then runs on the *surrogate* model, which is extremely fast (milliseconds). This allows for efficient exploration of the search space without bottlenecking the system.

### 3. Simplified User Interface
The "Direct Mode" (where users implement `predict_uncertainty`) was removed to simplify the interface. Users only need to provide a standard prediction model (inputs -> outputs). The calibration system handles all exploration/exploitation logic internally.

## Architecture

### BayesianCalibration
The core implementation is `BayesianCalibration`, which uses:
- **Surrogate Model**: `sklearn.gaussian_process.GaussianProcessRegressor` with Matern kernel.
- **Acquisition Function**: Upper Confidence Bound (UCB).
- **Exploration Weight**: A parameter (0.0 - 1.0) that scales the UCB kappa, allowing users to shift between pure optimization (exploitation) and active learning (exploration).

### Workflow
1.  **Initialize**: `agent.configure_calibration(...)` sets up the GP.
2.  **Define Objectives**: Users specify which performance metrics to optimize and their weights.
3.  **Calibrate**: `agent.calibrate(...)` runs the loop:
    - Sample initial random points.
    - Train GP on observed (param -> performance) pairs.
    - Optimize Acquisition Function to find next best point.
    - Evaluate next point using the Prediction Model.
    - Repeat.

## Future Potential
The surrogate model artifact itself has value beyond calibration:
- **Fast Inference**: It can be exported as a lightweight proxy for the heavy prediction model for design-stage applications requiring instant feedback.
- **Transparency**: The GP provides an explicit uncertainty map of the design space, showing where the model is confident and where it is guessing.
