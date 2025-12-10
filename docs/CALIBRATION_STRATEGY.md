# Calibration Strategy: Active Learning and Inference

## 1. Overview
The calibration module in the LBP framework implements a dual-phase optimization architecture designed to address the distinct challenges of **Active Learning** (data generation) and **Inference** (process control). By decoupling the exploration of the parameter space from the exploitation of learned relationships, the framework enables both efficient dataset curation and precise process optimization.

## 2. Theoretical Foundation

### 2.1 Bayesian Optimization and Active Learning
Active Learning is a subfield of machine learning where the algorithm interactively queries the information source (the fabrication process) to label new data points with the desired outputs. In the context of manufacturing, this corresponds to selecting process parameters $x$ to conduct an experiment and observe the performance $y$.

The framework employs **Bayesian Optimization (BO)**, a sequential design strategy for global optimization of black-box functions. BO constructs a probabilistic surrogate model $f(x)$ that approximates the true objective function and uses an **acquisition function** $\alpha(x)$ to determine the next sampling point.

The acquisition function balances two competing objectives:
1.  **Exploitation**: Sampling regions where the surrogate model predicts high performance (maximizing the mean $\mu(x)$).
2.  **Exploration**: Sampling regions where the prediction uncertainty is high (maximizing the standard deviation $\sigma(x)$).

### 2.2 The Surrogate Model
To enable uncertainty quantification without imposing constraints on the user's primary prediction model (which may be a deterministic Deep Neural Network), the framework utilizes a dedicated **Gaussian Process (GP)** as a surrogate.

A Gaussian Process defines a distribution over functions, specified by a mean function $m(x)$ and a covariance kernel $k(x, x')$. For any set of points, the function values follow a multivariate normal distribution. This provides a closed-form expression for the posterior mean and variance at any candidate point $x_*$:

$$ P(f(x_*) | X, y) \sim \mathcal{N}(\mu_*(x_*), \sigma_*^2(x_*)) $$

This separation of concerns allows the framework to perform rigorous active learning regardless of the architecture of the primary prediction model.

## 3. Methodology

The framework distinguishes between two operational phases, each utilizing a distinct optimization strategy tailored to its objective.

### 3.1 Phase I: Active Learning (Exploration)
**Objective**: To efficiently explore the parameter space and reduce epistemic uncertainty in the dataset.

In this phase, the system utilizes the **Surrogate Model** to guide the data generation process. The optimization objective is defined by a weighted acquisition function that blends the predicted performance with the predictive uncertainty:

$$ \alpha(x; w_{explore}) = (1 - w_{explore}) \cdot \mu(x) + w_{explore} \cdot \sigma(x) $$

*   **$\mu(x)$**: The posterior mean of the surrogate model (predicted performance).
*   **$\sigma(x)$**: The posterior standard deviation (uncertainty).
*   **$w_{explore} \in [0, 1]$**: A user-defined hyperparameter controlling the trade-off.
    *   $w_{explore} \to 1$: Pure exploration (Maximal Uncertainty).
    *   $w_{explore} \to 0$: Pure exploitation (Maximal Performance).

This formulation allows the user to seamlessly transition from a "discovery mode" (generating diverse data) to an "optimization mode" (refining high-performing regions) using a single control parameter.

### 3.2 Phase II: Inference (Exploitation)
**Objective**: To determine the optimal process parameters for a specific target geometry or performance criteria using the fully trained model.

In this phase, the uncertainty component is discarded. The system utilizes the high-fidelity **Prediction Model** (e.g., a Transformer or MLP) directly. The optimization problem becomes a deterministic maximization of the predicted system performance:

$$ x^* = \arg\max_{x} \text{Evaluate}(\text{Predict}(x)) $$

Here, `Predict(x)` maps parameters to high-dimensional features (e.g., layer geometry), and `Evaluate(features)` maps those features to a scalar performance score. This two-step inference allows for **Outcome-Informed Design**, where the process is optimized not just for abstract stability, but for specific physical attributes of the final part.

## 4. Implementation Architecture

### 4.1 The Calibration System
The `CalibrationSystem` acts as the central orchestrator, managing the transition between phases and the lifecycle of the models.

*   **Unified Optimization Interface**: Both phases utilize a unified `propose_params` interface, which internally switches the objective function based on the active `Phase` (Learning vs. Inference).
*   **Trust Region Management**: To prevent model divergence during online adaptation, the system implements trust regions that constrain the optimization search space to a local neighborhood around the current operating point.
*   **Multi-Objective Scalarization**: The system automatically aggregates multiple performance attributes (e.g., geometric accuracy, print time, energy) into a single scalar utility score using a weighted sum, enabling standard single-objective optimization algorithms to solve multi-objective problems.

### 4.2 Computational Efficiency in Online Control
A critical advantage of the surrogate-based approach is its suitability for **Online Learning**.
*   **Direct Optimization** of a heavy Deep Learning model (Phase II) requires thousands of forward passes, which may be too slow for real-time closed-loop control (e.g., layer-to-layer adaptation).
*   **Surrogate Optimization** (Phase I) is computationally lightweight. The GP can be updated with a single new data point in milliseconds, and the acquisition function is cheap to evaluate. This enables the framework to perform "Online Tuning" — adjusting process parameters in real-time to compensate for drift — with minimal latency.
