# Uncertainty Estimation in Exploration

> **Status:** Living document — converging on a recommended direction
> **Related:** [01-parameter-levels.md](01-parameter-levels.md), [02-exploration-scope.md](02-exploration-scope.md)

---

## Motivation

PFAB's exploration step uses a UCB acquisition function:

```
α(x) = (1 - κ) · S(μ(x)) + κ · S(σ(x))
```

where `μ(x)` is predicted performance (exploitation), `σ(x)` is uncertainty (exploration), and `S(·)` is the weighted performance aggregation. Currently, both `μ` and `σ` are produced by the `GPSurrogateModel` — a Gaussian Process trained separately from the Prediction Model (PM) on experiment-level `(x_params, a_performance)` pairs.

**The core architectural problem:** the GP surrogate is an independent model with no connection to the PM. They can diverge after incremental retraining, they are trained on different data (the GP bypasses feature-level measurements entirely), and maintaining two parallel models adds complexity without principled justification.

**Goal:** replace the GP surrogate with an uncertainty mechanism that is directly connected to the PM — using the PM for both the exploitation term (as inference mode already does) and the exploration term.

---

## Requirements

| # | Requirement | Why it matters |
|---|-------------|----------------|
| 1 | **General** | Applicable to any `IPredictionModel` implementation |
| 2 | **Lightweight** | Complexity grows with PM complexity, not independently |
| 3 | **PM-connected** | Uncertainty derived from the PM's learned representation |
| 4 | **Small-dataset-compatible** | Must work from as few as 3 experiments |

---

## Data Structure Insight

PFAB has two data levels that are critical to understanding cold-start behavior:

| Level | Data | Size at 3 experiments |
|-------|------|-----------------------|
| Experiment level | `(x_params, a_performance)` per experiment | **3 points** — what the GP sees |
| Feature-measurement level | `(x_params, position_idx, y_features)` per dimensional step | **3 × N_positions** — what the PM sees |

At 3 experiments with ~1000 positions each, the GP is trained on 3 points while the PM trains on ~3000. Any uncertainty method connected to the PM has access to orders of magnitude more data than the GP surrogate, even at the cold-start.

However, there is a critical constraint: **uncertainty for exploration must be expressed at experiment level** (one value per candidate parameter configuration), not at feature-measurement level. The propagation chain is:

```
σ_y_features  →  σ_a_performance  →  S(σ_a)  →  exploration term in α(x)
```

The middle step — `σ_y → σ_a` — goes through `IEvaluationModel.compute_performance()`. Since the evaluation function is `p_i = 1 - |y_i - t_i| / s_i`, analytical uncertainty propagation gives:

```
σ_p_i ≈ σ_y_i / s_i(x)        (first-order, exact away from y = t)
σ_S   = (1/W) · Σ_i w_i · σ_p_i
```

This is tractable and can be added to `compute_performance()` as an optional argument.

---

## Options Review

### Baseline: GP Surrogate

| Req 1 General | Req 2 Lightweight | Req 3 PM-connected | Req 4 Small data |
|:---:|:---:|:---:|:---:|
| ✓ | ✓ | ✗ | ✓ |

Matern-2.5 + WhiteKernel on `(x_params, a_performance)`. Works well with few points; O(n³) scaling. Completely disconnected from PM. No path to unifying the architecture.

---

### MC Dropout

| Req 1 | Req 2 | Req 3 | Req 4 |
|:---:|:---:|:---:|:---:|
| ~ (NN only) | ✓ | ✓ | ✗ |

Multiple stochastic forward passes through a PM with dropout; variance = uncertainty. **Fails requirement 4**: at 3 experiments, model weights are near-random, so variance reflects initialization noise rather than data coverage. Requires dropout layers in the PM architecture (not truly general). **Rejected.**

---

### Mini-Ensemble (K=3)

| Req 1 | Req 2 | Req 3 | Req 4 |
|:---:|:---:|:---:|:---:|
| ✓ | ~ | ✓ | ✓ |

Train K independent PM instances; variance across predictions = uncertainty. Works at 3 experiments (K models trained on same data with different seeds disagree meaningfully in unexplored regions). **3× training cost** partially violates the lightweight requirement, but serves as a strong reference baseline. Not the target architecture, but the most practical fallback.

---

### Last-Layer Laplace Approximation

| Req 1 | Req 2 | Req 3 | Req 4 |
|:---:|:---:|:---:|:---:|
| ~ (differentiable only) | ✓ | ✓ | ~ |

Post-hoc Gaussian approximation over last-layer weights using Fisher information. No architecture changes; fitted after training. **Weakness:** captures local curvature (aleatoric, "how sensitive is output to weight perturbations"), but has no OOD detection — can be confidently wrong far from training data. For the exploitation term this is valuable; for the exploration term (pure uncertainty) it is insufficient on its own. **Not a standalone solution for exploration.**

---

### Evidential Regression Head (EDL, Amini et al.)

| Req 1 | Req 2 | Req 3 | Req 4 |
|:---:|:---:|:---:|:---:|
| ~ (NN only) | ✓ | ✓ | ~ |

Add a NIG output head `(γ, ν, α, β)` to the PM; evidence `ν` = uncertainty signal. Same encoder, lightweight head, modified loss. **Known calibration issue:** without OOD detection, the network can assign high evidence to smoothly extrapolated regions far from training data. Calibration can be improved via automated λ-scheduling (balance prediction loss vs. KL loss), but OOD robustness requires a density gate on the latent space — bringing it architecturally close to NatPN-light (see below).

---

### Full NatPN (Charpentier et al., ICLR 2022)

| Req 1 | Req 2 | Req 3 | Req 4 |
|:---:|:---:|:---:|:---:|
| ✓ | ✗ | ✓ | ~ |

NIG prior + normalizing flow in latent space for evidence estimation. Theoretically optimal: Theorem 1 guarantees `n^post → n^prior` (maximum uncertainty) for any input far from training data in latent space. **Fails requirement 2:** the normalizing flow is an intrinsic complexity overhead independent of PM size — a large flow is needed even if the PM is small. Cold-start at 3 experiments is partially resolved by PFAB's feature-level data abundance, but flow training is still nontrivial.

---

## Converging Candidate: NatPN-light

### Concept

Preserve NatPN's full theoretical framework — NIG conjugate prior, evidence posterior update, and the Bayesian uncertainty decomposition — but replace the normalizing flow with a lightweight latent-space density estimator (KDE or Mahalanobis distance). This recovers the same theoretical properties at a fraction of the implementation cost, and complexity scales only with the PM.

```
Training:
  1. Train PM normally on feature-level data
  2. For each experiment: aggregate latent vectors across positions
     z_exp = mean_j { PM.encode(x_params, pos_j) }
  3. Fit KDE on { z_exp_1, ..., z_exp_N } (experiment-level latent points)

Inference (exploration):
  For query x_new:
  1. z_new = PM.encode(x_new)   (or mean encode over positions)
  2. evidence  n^post = n_prior · KDE_density(z_new)
  3. uncertainty = 1 / (1 + n^post)    (normalized to [0, 1])
  4. Propagate through evaluation: σ_perf = σ_features / s_i(x_new)
  5. α(x_new) = (1-κ)·S(μ_PM(x_new)) + κ·S(σ_perf(x_new))
```

| Req 1 | Req 2 | Req 3 | Req 4 |
|:---:|:---:|:---:|:---:|
| ✓ | ✓ | ✓ | ✓ |

- **General:** KDE on latent vectors works for any PM that exposes an encoder. Falls back to input space if no encoder is available.
- **Lightweight:** KDE is O(n·d_latent) — negligible vs. forward pass. No normalizing flow overhead.
- **PM-connected:** uncertainty is derived from the PM's own learned representation of the parameter space.
- **Small data:** KDE with 3 experiment-level latent points works. At 3 points, every novel config is distant from all clusters → maximum uncertainty everywhere except near seen experiments. This is correct behavior.

---

## NatPN-light: Behavior Analysis

### Offline Exploration (current step — initial parameters only)

At N experiments × M positions of feature-level training data:

- PM encoder produces M latent vectors per experiment, which are averaged to one `z_exp` per experiment
- KDE is fit on N experiment-level latent points
- At exploration time: `x_new → PM.encode(x_new) → z_new → KDE density → evidence → uncertainty`

**At N=3 (cold start):** Three tight latent clusters; everything outside → near-zero KDE density → maximum exploration drive. Correct behavior: the system strongly prefers unexplored regions.

**At N=30:** KDE begins to form a meaningful density landscape. Uncertainty decreases in explored regions and remains high in the gaps. The exploration term correctly guides the UCB toward informative new configurations.

**At N=300:** KDE is well-estimated; uncertainty landscape has fine-grained structure that reflects the PM's actual coverage. The system transitions smoothly from exploration-dominated to exploitation-dominated behavior as κ decays or as the dataset grows dense.

### Trajectory-Based Exploration (Option B from [Document 2](02-exploration-scope.md))

NatPN-light is **fully compatible** with trajectory-based exploration. Trajectory proposals are scored offline: for each candidate schedule, the PM predicts features at each step using effective parameters, uncertainty is estimated from KDE density, and the acquisition value is integrated over the trajectory. No real-time requirements; the architecture is a natural extension of the current `exploration_step`.

### Fabrication-Time Online Exploration (Option A from [Document 2](02-exploration-scope.md))

This is where NatPN-light faces a structural limitation:

- **KDE update:** incremental — O(n) per step. Adding a new latent point and updating the bandwidth is fast enough for real-time use.
- **PM weights:** frozen during the experiment. The PM cannot be retrained in real time.

This creates a **"frozen PM + live uncertainty" mismatch**: the KDE correctly signals "this runtime parameter value is unfamiliar," but the PM's feature predictions for that unfamiliar config may be extrapolations of unknown quality. The exploration signal is valid (go here, you haven't seen it), but the exploitation signal (expected performance) may be unreliable.

For Option A to work safely, one of the following would be needed:
1. **Accept the mismatch**: use KDE uncertainty alone to drive runtime decisions; treat PM predictions in novel regions as exploratory estimates with acknowledged risk
2. **Incremental PM learning**: warm-start fine-tuning during the experiment (computationally expensive; requires online learning infrastructure)
3. **Conservative safety bounds**: restrict runtime parameter proposals to a trust region around the PM's training distribution, limiting exploration to regions where PM extrapolation is reliable

**Conclusion for Option A:** NatPN-light is a necessary but not sufficient component. The frozen-PM problem is an independent architectural challenge that must be addressed before fabrication-time exploration is safe and reliable. **This supports keeping Option A in the backlog** and pursuing offline + trajectory-based exploration first.

---

## Required Interface Changes

### `IPredictionModel`

```python
# New optional method — default implementation uses input space
def encode(self, X: np.ndarray) -> np.ndarray:
    """
    Map normalized parameter vectors to latent representations.
    Default: returns X (identity — falls back to input-space density).
    Override for richer latent representations.
    """
    return X

# New method — default uses KDE on encode() output
def uncertainty(self, X: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """
    Estimate epistemic uncertainty for query points X given training data X_train.
    Returns uncertainty values in [0, 1]; higher = more uncertain.
    Default: KDE-based evidence from encode() latent space.
    """
    ...
```

### `IEvaluationModel`

```python
# Extended signature — feature_std is optional
def compute_performance(
    self,
    feature_array: np.ndarray,
    parameters: Parameters,
    feature_std: Optional[np.ndarray] = None   # NEW
) -> Tuple[Optional[float], List[Optional[float]], Optional[float]]:
    # Returns (mean_perf, per-position_perf, std_perf)
    # std_perf ≈ mean(feature_std / scaling_factor) if feature_std provided
    ...
```

### `CalibrationSystem`

Replace surrogate training and acquisition function:

```python
# Current acquisition function (GP-based):
def _acquisition_func(self, X, w_explore):
    mean, std = self.model.predict(X.reshape(1, -1))   # GP
    ...

# New acquisition function (PM-based):
def _acquisition_func(self, X, w_explore):
    y_pred = self.prediction_system.predict(X)          # PM forward pass
    sigma_y = self.prediction_system.uncertainty(X)    # NatPN-light
    perf_mean = self.evaluation_system.compute_performance(y_pred)
    perf_std = self.evaluation_system.compute_performance(y_pred, sigma_y)
    score = (1 - w_explore) * S(perf_mean) + w_explore * S(perf_std)
    return -score
```

---

## Mini-Ensemble as Reference Baseline

Before implementing NatPN-light, implement a K=3 ensemble as a reference:

- Train K=3 independent PM instances (different random seeds)
- Uncertainty = variance across K predictions: `σ(x) = std({ f_θ_k(x) }_{k=1}^{K})`
- No additional interface changes required; works with current `IPredictionModel`
- Serves as ground truth: if NatPN-light produces similar exploration behavior to the ensemble, it is well-calibrated

The ensemble can be deprecated once NatPN-light is validated across multiple fabrication tasks.

---

## Summary: Option Comparison

| Method | Req 1 | Req 2 | Req 3 | Req 4 | OOD detection | Online exploration (Opt A) |
|--------|:-----:|:-----:|:-----:|:-----:|:-------------:|:------------------------:|
| GP Surrogate (baseline) | ✓ | ✓ | ✗ | ✓ | ✓ (by design) | ✗ (frozen, offline only) |
| MC Dropout | ~ | ✓ | ✓ | ✗ | ✗ | ~ |
| Mini-Ensemble (K=3) | ✓ | ~ | ✓ | ✓ | ✓ (implicit) | ~ |
| Last-Layer Laplace | ~ | ✓ | ✓ | ~ | ✗ | ✗ |
| EDL | ~ | ✓ | ✓ | ~ | ✗ (w/o gate) | ~ |
| Full NatPN | ✓ | ✗ | ✓ | ~ | ✓ (Thm 1) | ~ (frozen PM) |
| **NatPN-light** | **✓** | **✓** | **✓** | **✓** | **✓** | **partial** |

---

## Open Questions

1. **Encoder requirement:** Making `encode()` optional (defaulting to input space) keeps the design general, but input-space KDE is less informative than latent-space KDE. Is requiring `encode()` an acceptable PM constraint?

2. **Aggregation strategy:** Averaging latent vectors across positions per experiment reduces each experiment to one latent point. Is this the right aggregation, or should the KDE be fit on all measurement-level latent vectors?

3. **KDE bandwidth:** How should the bandwidth be selected? Standard rules-of-thumb (Silverman's rule) work for small n but may not reflect the geometry of fabrication parameter space. A learned or cross-validated bandwidth may be needed.

4. **Evidence scaling:** The mapping from KDE density to NatPN evidence `n^post` requires a normalization factor. Setting `n_prior = 1` (one virtual observation) is a principled default, but the scale of the KDE density must match. This deserves careful treatment.

5. **Interaction with Document 2:** If trajectory-based exploration is implemented, uncertainty must be defined over trajectory proposals, not single parameter vectors. Does the KDE generalize straightforwardly to trajectory space, or does it require a separate design?
