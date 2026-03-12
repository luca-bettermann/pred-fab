# Uncertainty Estimation in Exploration

> **Status:** Implemented — NatPN-light is the active uncertainty mechanism
> **Related:** [01-parameter-levels.md](01-parameter-levels.md), [02-exploration-scope.md](02-exploration-scope.md)

---

## Implementation Status

NatPN-light has been implemented and replaces the former `GPSurrogateModel`. Key changes:

- `IPredictionModel.encode()` — default identity; override for learned latent space
- `IEvaluationModel.compute_performance()` — extended with optional `feature_std` parameter, returns 3-tuple `(avg, per_row, std_list)`
- `PredictionSystem` — adds `_fit_kde()`, `encode()`, `uncertainty()`, `kernel_similarity()`, `predict_for_calibration()`; KDE is fitted on unique training configs after `train()`
- `CalibrationSystem` — replaced surrogate model with injected `perf_fn` / `uncertainty_fn` / `similarity_fn` callables; Level 2 trajectory diversity discounting implemented
- `PfabAgent.initialize_systems()` — creates the `perf_fn` closure wiring `PredictionSystem.predict_for_calibration()` and `EvaluationSystem._evaluate_feature_dict()`

**Degenerate-dimension guard:** If all training configs share the same value on a latent dimension (e.g. fixed `dim_1`, `dim_2`), those dimensions are dropped before KDE fitting to prevent a singular covariance matrix. The active-dimension mask (`_kde_active_mask`) is stored and applied consistently in `uncertainty()` and `kernel_similarity()`.

**Mini-ensemble reference baseline** was not implemented — NatPN-light was validated directly against the existing test suite.

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
  2. For each unique effective parameter configuration c seen in training:
     z_c = PM.encode(x_params_c)
       non-trajectory experiments → one config per experiment (initial params)
       trajectory experiments     → K configs per experiment (one per trigger step)
     w_c = sqrt(n_rows_c) / Σ_c' sqrt(n_rows_c')   (segment-size weight; sqrt for diminishing returns)
  3. Fit weighted KDE on all { z_c } with weights { w_c }
     q_max    = max_c q_KDE(z_c)      (normalisation constant, computed once)
     N_exp    = total number of experiments (n_prior scale)

Inference (exploration):
  For query x_new:
  1. z_new = PM.encode(x_new)
  2. evidence  n^post = N_exp · q_KDE(z_new) / q_max        (n_prior = N_exp)
  3. uncertainty = 1 / (1 + n^post)    (→ 0 near training configs; → 1 for OOD)
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

## NatPN vs. NatPN-light: What Changes in Practice

NatPN-light preserves the full theoretical skeleton of NatPN but replaces one component — the normalizing flow — with a lightweight alternative. Understanding exactly what changes (and what doesn't) is important for assessing where the approximation is tight and where it weakens.

### What full NatPN uses: the normalizing flow

In the original NatPN, the latent-space density estimator is a **normalizing flow** (e.g., RealNVP or MAF). A normalizing flow is a learnable bijection `z = f(u)` that transforms a simple base distribution (e.g., standard Gaussian) into an arbitrarily complex target distribution via a chain of invertible, differentiable transformations. The density at any latent point `z` is computed exactly via the change-of-variables formula:

```
log q_flow(z) = log p_base(f⁻¹(z)) - log |det J_f⁻¹(z)|
```

This gives a **globally calibrated** density estimate over the entire latent space. The flow's key properties:

- **Expressiveness:** can model any continuous density, including multimodal, asymmetric, or manifold-structured latent distributions.
- **Formal OOD guarantee (Theorem 1):** for inputs whose latent representations fall outside the support of the training distribution, `q_flow(z) → 0` exactly — the flow produces zero density OOD by construction.
- **Training cost:** the flow has its own parameter set (typically comparable in size to a small encoder) and requires gradient-based training — either jointly with the encoder or post-hoc on frozen latents.

When trained jointly, the encoder and flow interact: the encoder may organise the latent space in ways that make flow training easier, which can subtly shift the learned representations away from what pure prediction training would produce.

### What NatPN-light substitutes: KDE

NatPN-light replaces the flow with **kernel density estimation (KDE)** on the experiment-level latent points `{z_exp_1, ..., z_exp_N}`:

```
q_KDE(z) = (1/N) Σ_i K_h(z - z_exp_i)
```

With a Gaussian kernel `K_h`, this is a sum of N Gaussians centred on the training latent representations. Key properties:

- **No training:** KDE is fit by storing the N latent points and choosing a bandwidth `h` (e.g., Silverman's rule). Zero gradient steps, zero additional parameters.
- **Purely post-hoc:** the PM is trained on its prediction objective alone; KDE is applied to frozen latents afterward. No coupling between density estimation and encoder training — the latent geometry is determined entirely by the prediction task.
- **Practical OOD detection:** for inputs far from all training points, `q_KDE(z) → 0` due to exponential Gaussian kernel decay. No formal theorem equivalent to Theorem 1, but equivalent practical behaviour for small-to-moderate N.
- **Expressiveness:** KDE is a Gaussian mixture with one component per training point. It cannot model densities with complex topology (holes, sharp boundaries, elongated manifolds) as accurately as a flow.

### What is preserved exactly

The following components of NatPN carry over unchanged in NatPN-light:

| Component | Full NatPN | NatPN-light |
|-----------|:---:|:---:|
| NIG conjugate prior | ✓ | ✓ |
| Evidence posterior update `n^post = n_prior · density(z)` | ✓ | ✓ (KDE instead of flow) |
| Uncertainty-to-evidence mapping `u = 1 / (1 + n^post)` | ✓ | ✓ |
| Aleatoric / epistemic decomposition via NIG parameters | ✓ | ✓ |
| Uncertainty propagation chain `σ_y → σ_perf → S(σ)` | ✓ | ✓ |
| PM as the exploitation term in UCB | ✓ | ✓ |

The NIG framework — the principled Bayesian decomposition of aleatoric uncertainty (captured in the NIG `α, β` parameters) and epistemic uncertainty (captured in the evidence `n^post`) — is unaffected by the choice of density estimator.

### What changes

| Dimension | Full NatPN | NatPN-light |
|-----------|-----------|-------------|
| **Density estimator** | Normalizing flow (parametric, trained) | KDE (non-parametric, zero training) |
| **OOD guarantee** | Formal — Theorem 1, exact zero density OOD | Practical — exponential kernel decay, no theorem |
| **Density expressiveness** | Arbitrary continuous densities | Gaussian mixtures (one component per explored parameter configuration — K per trajectory experiment) |
| **Training coupling** | Flow can be jointly trained with encoder | Purely post-hoc — PM training unaffected |
| **Additional parameters** | Flow weights (~order of a small encoder) | None — N stored latent vectors + scalar bandwidth |
| **Inference cost per query** | O(d · L_flow) | O(N · d_latent) |
| **Implementation complexity** | Flow architecture + training loop required | KDE fit in a few lines; scipy / sklearn |

### Where the approximation is tight and where it weakens

**Tight (NatPN-light ≈ full NatPN):**

- **Small N (3–30 experiments).** Both methods produce near-identical uncertainty landscapes. At N=3, only three latent clusters exist; everything outside them is low-density regardless of estimator. The flow cannot learn a meaningfully better density than KDE because there is insufficient data to train its parameters.
- **Well-separated experiments in latent space.** When explored experiments are well-distributed and non-degenerate, KDE's Gaussian mixture approximation is adequate.
- **Low-dimensional, physics-bounded parameter space.** For PFAB's operating regime, exponential kernel decay is a sufficient practical proxy for the formal OOD guarantee.

**Weakens (full NatPN becomes preferable):**

- **Large N (300+ experiments) with complex latent structure.** A flow can learn fine-grained density variations — a cluster with a gap in the middle, an elongated manifold, multimodal distributions — that KDE smooths over with its fixed Gaussian kernel shape.
- **High-dimensional latent spaces.** KDE suffers from the curse of dimensionality: bandwidth selection becomes unreliable and density estimates noisier as `d_latent` grows. The flow does not have this structural limitation.
- **Formally auditable systems.** Where a provable OOD guarantee (Theorem 1) is a hard requirement, the normalizing flow is necessary.

### Summary

NatPN-light makes one targeted approximation: the normalizing flow — which provides formal OOD guarantees and expressiveness — is replaced by KDE, which provides zero additional training cost and purely post-hoc applicability. For PFAB's expected dataset sizes (3–~100 experiments) and low-dimensional parameter space, the two methods are expected to converge in practical behaviour. The approximation becomes an increasingly poor substitute as N grows and latent geometry becomes complex. At that scale, migrating to full NatPN or falling back to a mini-ensemble would be warranted.

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

# New acquisition function (PM-based), single parameter vector:
def _acquisition_func(self, X, w_explore):
    y_pred = self.prediction_system.predict(X)          # PM forward pass
    sigma_y = self.prediction_system.uncertainty(X)    # NatPN-light
    perf_mean = self.evaluation_system.compute_performance(y_pred)
    perf_std = self.evaluation_system.compute_performance(y_pred, sigma_y)
    score = (1 - w_explore) * S(perf_mean) + w_explore * S(perf_std)
    return -score
```

### Trajectory scoring (Level 2 — non-redundant uncertainty)

For trajectory proposals `π = [(x_static, x_runtime_k) for k in 1..K]`, simple averaging of per-step UCB scores does not penalise within-trajectory redundancy: if two trigger steps propose nearly the same runtime config, they each contribute full uncertainty weight even though observing them together yields less than twice the information.

Level 2 scoring corrects for this using the same Gaussian kernel already in use for the KDE, applied between pairs of trajectory steps. Steps that are close in latent space share credit rather than each receiving full weight:

```
For trajectory π with K trigger steps:

  1. Compute per-step acquisition scores: a_k = α(x_static, x_runtime_k)   for k = 1..K

  2. Compute pairwise kernel similarities between step latent representations:
     sim(j, k) = exp( -||z_step_j - z_step_k||² / h² )
     where h is the same KDE bandwidth

  3. Non-redundant score:
     a_1_effective = a_1
     a_k_effective = a_k · (1 - max_{j<k} sim(z_step_j, z_step_k))   for k > 1

  4. Trajectory score: α(π) = (1/K) Σ_k a_k_effective
```

Step k's effective contribution is discounted by its maximum similarity to any earlier step: if it is identical to a previous step in latent space (`sim = 1`), it contributes zero. If it is fully orthogonal (`sim ≈ 0`), it contributes its full acquisition score. The kernel reuse is intentional — the density estimator and the diversity term operate on the same notion of distance in latent space, ensuring consistent geometry across the acquisition function.

**Caveat: trajectory experiments confound parameter effects with natural dimensional progression.**

When runtime parameters change across segments, the observed change in features between segments cannot be cleanly attributed to either the parameter change or the natural progression of the process (thermal buildup, accumulated material effects, layer-to-layer dynamics). A single-config experiment with constant parameters does not have this ambiguity.

This is not a flaw of trajectory exploration specifically — it is a fundamental property of any multi-config experiment — but it should be understood when designing datasets:

- **For causal attribution** (isolating the effect of a parameter value): use single-config experiments.
- **For parameter-space coverage** (mapping which regions are feasible and broadly characterising behaviour): trajectory experiments are appropriate and more efficient.
- **Recommended practice**: mix both types. Single-config experiments establish the clean baseline; trajectory experiments extend coverage efficiently.

**Resolution at the PM design level.** The confounding is resolved if the PM takes prior feature measurements as input alongside current parameters — e.g., `(x_params_t, pos_t, y_{t-1})`. In this formulation, the process state at step `t` is fully summarised by `y_{t-1}` (what the previous layer actually measured), making the model Markovian in feature space. Natural progression is absorbed into `y_{t-1}`; the parameter effect at step `t` is isolated in `x_params_t`. This is achievable with the current interface: `IPredictionModel.input_features` allows a PM to declare which prior feature values it requires as inputs. The specific architecture (last layer only, rolling window, aggregated statistics) is a PM implementation concern.

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

## Design Decisions

The following questions were resolved during design review:

1. **`encode()` is optional with an identity default.** `IPredictionModel` defines `encode(X)` with a default `return X` (input-space KDE fallback). No existing PM is required to change. PMs with learned encoders override it to expose their internal representation (e.g., penultimate-layer activations). `PredictionSystem` calls `encode()` post-training to build the KDE and at query time — individual PMs have no awareness of KDE.

2. **One latent point per unique effective parameter configuration (Option B), weighted by √(segment size).** Non-trajectory experiments contribute one latent point (initial params). Trajectory experiments contribute K latent points — one per trigger step, using effective runtime params at that segment. Each point's KDE weight is `w_c = sqrt(n_rows_c) / Σ sqrt(n_rows)`. The √n weighting reflects diminishing returns to parameter-space coverage: the first observations at a config establish whether it is feasible and what its gross behaviour is; further observations at the same config refine the estimate with shrinking marginal gain. This holds regardless of PM architecture — it is a property of exploration value, not of row-level information content.

3. **Silverman's rule for bandwidth selection.** Closed-form, O(N) computation, stable at N=3. The relative ordering of KDE densities — not their absolute calibration — drives the acquisition function, making the Gaussian assumption an acceptable approximation for PFAB's physics-bounded, low-dimensional parameter space.

4. **`n_prior = N_exp` (dynamic, = number of experiments).** Normalised evidence: `n^post = N_exp · q_KDE(z) / q_max`. Using experiment count (not config count) keeps the evidence scale anchored to real fabrication runs regardless of how many trajectory trigger steps they contained. Minimum achievable uncertainty is `1 / (1 + N_exp)` at the most familiar configuration — naturally shrinking as experiments accumulate (25% at N=3, 9% at N=10, 3% at N=30). Absolute values are high at small N, but the UCB acquisition function uses relative ordering between candidates, so this is correct behaviour.

5. **Level 2 non-redundant trajectory uncertainty.** Per-step UCB scores are discounted for within-trajectory similarity using the same Gaussian kernel as the KDE. Steps close in latent space share credit rather than each receiving full weight. Kernel reuse between the density estimator and the diversity term is intentional: both mechanisms operate on the same notion of distance in latent space. See the Trajectory Scoring section under Required Interface Changes.
