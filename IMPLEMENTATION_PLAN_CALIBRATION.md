# Implementation Plan: Calibration & Active Learning Refinement

## 1. Overview
This plan addresses the refinement of the Calibration and Active Learning workflow to align with the "Surrogate-First" strategy and the specific user requirements for offline/online learning, non-determinism, and deployment.

## 2. Key Architectural Decisions

### A. Surrogate Model Location
*   **Decision:** The Surrogate Model will be managed by the `PredictionSystem` but orchestrated by the `CalibrationModel`.
*   **Rationale:** The Surrogate is a "model of the data" (specifically, a proxy for the prediction model's performance landscape). Storing it in `PredictionSystem` allows it to leverage the existing training/data infrastructure. However, its primary consumer is the `CalibrationModel` for optimization.
*   **Implementation:** `PredictionSystem` will have a `surrogate_model` attribute (likely a `GaussianProcessRegressor` wrapped in a helper class).

comment: what is your plan for architectural configurations of the surrogate model? how can a GP regressor be modified? what would be other options to GP, and why is GP currently the choice? please document these decisions.

### B. Performance Mapping (Surrogate vs. Prediction)
*   **Decision:** The Calibration step will use the **Surrogate Model** for *both* uncertainty (exploration) and performance (exploitation) during the optimization search.
*   **Rationale:**
    1.  **Efficiency:** The primary goal of using a surrogate is to avoid expensive calls to the Prediction Model during the optimization loop (which may require thousands of evaluations).
    2.  **Consistency:** Optimizing on the surrogate's surface ensures that the acquisition function is smooth and differentiable (if using GP), facilitating efficient optimization.
    3.  **Active Learning:** The goal is to find the next best point to *label* (i.e., run a physical experiment). The surrogate's uncertainty guides us to unexplored regions, and its mean guides us to promising regions.
*   **Refinement:** The "Prediction Model" is used to *label* the points for the Surrogate in the offline phase (simulating experiments), but the Surrogate drives the search.

comment: the main reason for the surrogate in the first place was the capability of uncertainty quantification, which is not available in all ML models. Since the uncertainty is a core requirement of the architecture, that is why we went with the idea of surrogate model in the first place. efficiency is only a conditional requirement; it depends on the complexity of the prediction model. so, the choice between using the surrogate model or the prediction model should be user-specified, unless we require uncertainty quantification. the optimization loop itself can also be adjusted, so less calls are required (i.e. tighter bounds, etc.)

here is my interetation of the acquisition function: generally, we can see the acquisition function as w_exploration * f_uncertainty(X) + (1-w_exploration) * f_performance(X), f_performance(X) is evaluation(prediction(X)), i.e. the evaluation of the predicted features for X. Higher uncertainty should give higher scores, higher evaluation should get higher scores. We want to find the next sample that maximises the acqusition function. Surrogate model should be used for uncertainty by default and for feature prediction only if specifically requested by the user. The inference time of the prediction model might be efficient enough. 

If we take a step back, the goal of this acquisition function is to find the next data sample that maximises the value of the prediction model. that should be a combination of high performing areas and under-explored areas. do you think the proposed acquisition function in combination with the exploration weight satisfies this goal?

How do you think your approach to the active learning problem compares? If we replace the prediction model with the surrogate model, why do we have the prediction model in the first place? This is its main use case in Learning by Printing. 

### C. Non-Determinism & Noise
*   **Decision:** The Gaussian Process will explicitly model noise using a `WhiteKernel`.
*   **Rationale:** Fabrication is non-deterministic ($y = f(x) + \epsilon$). The GP must account for this aleatoric uncertainty to avoid overfitting to noisy observations.

### D. Initial Dataset Generation
*   **Decision:** Implement `generate_initial_design` in `CalibrationModel`.
*   **Mechanism:** Accepts `n_samples` and `bounds`. Uses Latin Hypercube Sampling (LHS) or Grid Search to generate a space-filling design. Returns parameters to be saved as experiments.

## 3. Detailed Workflow & Implementation Steps

### Phase 1: Initial Dataset (No-Learning)
1.  **`CalibrationModel.generate_initial_design(n_samples, bounds)`**:
    *   Implement LHS (using `scipy.stats.qmc` or similar) to generate `n_samples` points.
    *   Return list of parameter dictionaries.
2.  **`LBPAgent.create_initial_experiments(n_samples, bounds)`**:
    *   Call `generate_initial_design`.
    *   Create `ExperimentData` objects.
    *   Save them using `dataset.save_experiment(...)`.

comment: do we have options to use sklearn for this? I dont want to introduce additional dependencies if not a necessity.

### Phase 2: Offline Learning (Active Learning)
1.  **`PredictionSystem.train_surrogate(X, y)`**:
    *   New method to train the internal GP surrogate.
    *   Input: `X` (parameters), `y` (performance metric).
    *   **Storage:** Save surrogate training data (or the model itself) to `local_data/surrogates/`.
2.  **`LBPAgent.run_active_learning_loop(n_steps)`**:
    *   Loop `n_steps` times:
        a.  **Train Prediction Model:** `agent.train_prediction_model()`.
        b.  **Update Surrogate:**
            *   Extract (X, Y) from *all* available experiments (labeled data).
            *   *Crucial:* Use real experimental results (Y), not PM predictions, to capture true process noise.
            *   Train `PredictionSystem.surrogate`.
        c.  **Calibrate (Propose Next Point):**
            *   Call `CalibrationModel.propose_next_point(surrogate)`.
            *   Optimizer maximizes `Acquisition(Mean, Std)` using Surrogate.
        d.  **Fabricate (Simulated/Real):**
            *   In offline mode, this might just be "add to dataset and wait" or "simulate if simulator exists".
            *   For now, output the proposed parameters.

comment: in my opinion, the surrogate model should be trained on the predictions of the prediction model, not the real data. it is a surrogate of the model itself, not of what the model is trying to predict. if the model is bad, the surrogate should model the bad predictions, not the actual features.

again, the calibration should be done using the prediction model (and only the surrogate for the uncertainty part.)
there would be an argument that we use your approach to the acquisition function with mu and sigma for faster exploration, and then use the prediction model only in the optimization mode, when we want to find the best performing sample. this would give a clear boundary between for what the surrogate is used and for what the actual prediction model is used.

### Phase 3: Deployment
1.  **Export:**
    *   `PredictionSystem.export_surrogate(path)`: Pickle the GP model.
    *   `PredictionSystem.export_model(path)`: Existing functionality.
2.  **Inference:**
    *   `CalibrationModel.optimize_for_deployment(surrogate, fixed_params)`:
        *   Uses the exported surrogate to find optimal parameters given constraints.

comment: optimize should definitely use the actual prediction model, otherwise there is no point in the prediction model at all.

### Phase 4: Online Learning
1.  **Tuning:**
    *   `PredictionSystem.tune_model(new_data)`: Existing functionality.
2.  **Online Optimization:**
    *   `CalibrationModel.optimize_online(surrogate, current_params, bounds)`:
        *   Update surrogate with new `(X, Y)` point.
        *   Run local optimization (narrow bounds) to adjust parameters for the *next* print.

comment: since we are in optimization stage here, I think we probably should only use the prediction model for calibrations again. surrorgates should be for active learning only, not for optimization and parameter calibration (offline or online).

## 4. File Structure Changes

*   **`src/lbp_package/interfaces/calibration.py`**:
    *   Add `generate_initial_design`.
    *   Add `propose_next_point`.
*   **`src/lbp_package/orchestration/calibration.py`**:
    *   Implement LHS in `generate_initial_design`.
    *   Refactor `optimize` to use the passed `surrogate_model` explicitly.
*   **`src/lbp_package/orchestration/prediction.py`**:
    *   Add `surrogate_model` attribute.
    *   Add `train_surrogate`.
    *   Add `export_surrogate`.
*   **`src/lbp_package/utils/local_data.py`**:
    *   Add `get_surrogate_folder`.
    *   Add `save_surrogate_model` / `load_surrogate_model`.

## 5. Addressing User Questions

*   **"Should surrogate be part of PredictionSystem?"**: Yes, as a managed resource, but used by Calibration.
*   **"Use surrogate for performance mapping?"**: Yes, for the optimization search. It's the only way to be efficient.
*   **"Non-determinism"**: Handled via `WhiteKernel` in GP.
*   **"Online Surrogate Tuning"**: Not a gimmick. It's standard "Online Bayesian Optimization". We update the GP with the single new data point and re-optimize.

## 6. Next Steps
1.  Review this plan.
2.  Execute changes in `local_data.py` (storage).
3.  Execute changes in `prediction.py` (surrogate management).
4.  Execute changes in `calibration.py` (initial design, surrogate-based optimization).
5.  Update `agent.py` to orchestrate the new workflow.


OVERALL:
I think we need to find clear boundaries between when to use the surrogate and when to use the prediction model. I think active learning -> surrogate models vs optimization -> prediction model would be a good approach. let me know your thoughts.