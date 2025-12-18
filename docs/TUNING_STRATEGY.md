# Tuning Strategy: Online Adaptation via Residual Learning

## 1. Overview
The tuning module in the LBP framework addresses the "Sim-to-Real" gap—the discrepancy between a model trained on historical data (General Model) and the specific conditions of the current print job. By implementing an **Online Adaptation** strategy, the framework allows the system to compensate for temporal drift (e.g., nozzle wear, filter clogging, ambient temperature shifts) in real-time without compromising the stability of the base model.

## 2. Theoretical Foundation

### 2.1 The Stability-Plasticity Dilemma
In continuous learning systems, a core challenge is balancing **stability** (retaining previously learned knowledge) with **plasticity** (adapting to new information). Naively fine-tuning a deep neural network on a small stream of incoming data often leads to **Catastrophic Forgetting**, where the model overfits to the recent noise and loses its generalization capabilities.

### 2.2 Residual Learning (Boosting)
To resolve this, the framework employs an additive **Residual Correction** strategy. Instead of modifying the weights of the robust General Model ($f_{\theta_{base}}$), we introduce a lightweight, dynamic residual model ($g_\phi$) that learns to predict the *error* of the base model.

This approach guarantees a "graceful degradation": if the residual model predicts zero (due to lack of evidence or regularization), the system output reverts to the stable prediction of the General Model.

## 3. Methodology

### 3.1 Residual Formulation
The adaptation process treats the discrepancy between the base model's prediction and the observed reality as a learnable signal.

For an input vector $x$ (process parameters) and observed ground truth $y$, the residual error $r$ is defined as:

$$ r = y - f_{\theta_{base}}(x) $$

The residual model $g_\phi$, parameterized by $\phi$, is trained to predict this error term. To capture state-dependent biases, $g_\phi$ is conditioned on both the input parameters and the base model's initial guess:

$$ \hat{r} = g_\phi(x, f_{\theta_{base}}(x)) $$

The final corrected prediction $\tilde{y}$ is the sum of the robust baseline and the local correction:

$$ \tilde{y} = f_{\theta_{base}}(x) + \eta \cdot g_\phi(x, f_{\theta_{base}}(x)) $$

Where $\eta \in [0, 1]$ is a scaling factor that controls the influence of the online correction.

### 3.2 Optimization Objective
During the fabrication loop, the system accumulates a local tuning batch $D_{new}$. The adaptation step consists of optimizing the parameters $\phi$ of the residual model to minimize the reconstruction error of the residual:

$$ \phi^* = \arg\min_{\phi} \sum_{(x,y) \in D_{new}} \mathcal{L}\left( g_\phi(x, f_{\theta_{base}}(x)), \; y - f_{\theta_{base}}(x) \right) $$

## 4. Implementation Architecture

### 4.1 Orchestration Mechanisms
The `PredictionSystem` orchestrates the critical data engineering required for stability during closed-loop control.

*   **Frozen Normalization**: To prevent covariate shift, the system ensures that the incoming data stream $D_{new}$ is normalized using the exact statistics ($\mu_{train}, \sigma_{train}$) of the offline dataset. This ensures the model's internal feature representation remains consistent.
*   **Temporal Batching**: Rather than updating on single noisy data points, the framework maintains a sliding window buffer. Updates are triggered only when a sufficient batch of recent observations is accumulated, smoothing out high-frequency sensor noise.

### 4.2 The Residual Model
The residual model is designed to be computationally lightweight to support rapid retraining.
*   **Architecture**: Typically a shallow Multi-Layer Perceptron (MLP) or a linear regressor.
*   **Lifecycle**: Unlike the base model, the residual model can be reset or re-initialized between distinct print jobs or layers, ensuring that adaptation to transient conditions does not permanently pollute the system's knowledge base.
