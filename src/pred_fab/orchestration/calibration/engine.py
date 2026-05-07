from dataclasses import dataclass, field
from typing import Any, Callable
import warnings

import numpy as np
import torch

from ...core import DataModule
from ...utils import PfabLogger, ProgressBar, profiler


@dataclass
class _OptResult:
    """Raw output from an optimizer backend."""
    best_x: np.ndarray | None
    nfev: int
    n_starts: int
    score: float  # negated objective (higher = better)
    convergence_history: list[float] = field(default_factory=list)  # best energy per iteration


warnings.filterwarnings("ignore", category=UserWarning)


# ======================================================================
# OptimizationEngine — pure optimization, no schema/calibration knowledge
# ======================================================================

class OptimizationEngine:
    """Numerical optimization backend: DE (integer phase) and GRADIENT (continuous)."""

    def __init__(self, logger: PfabLogger, random_seed: int | None = None):
        self.logger = logger
        self._random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        # DE optimizer parameters (population-based, used for integer phase).
        # Convergence is governed by "no improvement in K generations" via the
        # callback heuristic — same as the prior scipy wrapper.
        self.de_maxiter: int = 1000
        self.de_popsize: int = 64
        self.de_no_improve_window: int = 50  # generations without improvement → halt
        self.de_improvement_eps: float = 1e-10  # min Δbest to count as an improvement

        # GRADIENT optimizer parameters (autograd multi-start with sigmoid bound reparam).
        # LBFGS is the default — quasi-Newton with line search; converges in
        # ~5-30 iterations on the smooth, deterministic acquisition surface
        # we have. Adam is available for noisier/flatter surfaces.
        self.gradient_n_starts: int = 4
        self.gradient_n_iters: int = 100
        self.gradient_lr: float = 0.05
        self.gradient_method: str = "lbfgs"  # "lbfgs" | "adam"

        # Smart-init parameters (BoTorch gen_batch_initial_conditions
        # pattern). raw_samples Sobol points are batch-evaluated, top-K selected via
        # Boltzmann sampling with temperature `eta` to balance quality vs. diversity.
        # 0 disables (uniform random multi-start fallback).
        self.gradient_raw_samples: int = 256
        self.gradient_init_eta: float = 1.0

    def smart_maxiter(self, D: int) -> int:
        """Per-call DE maxiter scaled with decision-variable count, capped by
        the configured global ceiling and floored at 5 (scipy DE requirement).

        Same rule across baseline / exploration / inference / per-exp schedule
        — single source of truth so the X/cap ratio in DE progress bars is
        consistent across all phases. Override behaviour by setting
        ``self.de_maxiter`` (capping ceiling) at agent-config time.
        """
        return min(max(40, 15 * max(D, 1)), max(self.de_maxiter, 5))

    def run_acquisition_vectorized(
        self,
        objective_vectorized: Callable[[np.ndarray], np.ndarray],
        bounds: list[tuple[float, float]],
        *,
        x0: np.ndarray | None = None,
        n_restarts: int = 0,
        label: str = "Optimizing",
        show_progress: bool = False,
    ) -> _OptResult:
        """Vectorised DE for single-point acquisition (N=1, L=1).

        ``objective_vectorized`` takes ``(D, S)`` and returns ``(S,)`` — one negated
        score per candidate. scipy DE then evaluates the entire population in a
        single call, amortising forward_pass overhead across ``S = popsize × D``.

        L-BFGS-B path fallback for the optimizer enum is unsupported here — the
        vectorisation gain is DE-specific. Callers wanting L-BFGS-B should use
        the scalar ``run`` method.
        """
        return self._run_de(
            objective_vectorized,
            bounds,
            label=label,
            show_progress=show_progress,
            maxiter=self.smart_maxiter(len(bounds)),
            vectorized=True,
        )

    def run_acquisition_gradient(
        self,
        objective_tensor: Callable[[torch.Tensor], torch.Tensor],
        bounds: list[tuple[float, float]],
        *,
        n_starts: int | None = None,
        n_iters: int | None = None,
        lr: float | None = None,
        method: str | None = None,
        x0: np.ndarray | None = None,
        compile_objective: bool = False,
        label: str = "Optimizing",
        show_progress: bool = False,
    ) -> _OptResult:
        """Multi-start gradient optimisation with sigmoid bound reparameterisation.

        ``objective_tensor`` takes ``(S, D)`` and returns ``(S,)`` — one scalar
        per starting point. The optimiser stacks ``n_starts`` initial points
        in z-space, runs ``n_iters`` Adam (or L-BFGS) steps with all starts in
        parallel (one autograd graph per call), then picks the best.

        Bounds are enforced via ``x = sigmoid(z) · (hi - lo) + lo``: smooth
        gradient + strict ``x ∈ [lo, hi]``. No clipping, no penalty tuning.

        Convention matches DE: ``objective_tensor`` returns the **negated**
        score (lower is better). Score in the result is ``−min(objective)``.

        ``compile_objective=True`` wraps the objective
        with ``torch.compile(dynamic=True)``; first call traces, subsequent
        calls reuse the JIT graph. Falls back to eager silently on compile
        failure (e.g. unsupported ops in the objective graph).
        """
        n_starts = n_starts if n_starts is not None else self.gradient_n_starts
        n_iters = n_iters if n_iters is not None else self.gradient_n_iters
        lr = lr if lr is not None else self.gradient_lr
        method = (method or self.gradient_method).lower()
        if method not in ("adam", "lbfgs"):
            raise ValueError(f"unknown gradient method: {method!r}")

        D = len(bounds)
        if D == 0:
            return _OptResult(best_x=None, nfev=0, n_starts=0, score=0.0)

        # Optional torch.compile wrap of the objective.
        if compile_objective:
            try:
                objective_tensor = torch.compile(objective_tensor, dynamic=True)  # type: ignore[assignment]
            except Exception as e:
                self.logger.warning(
                    f"torch.compile failed for objective_tensor; running eager: {e!r}"
                )

        bounds_arr = np.asarray(bounds, dtype=np.float64)
        lo_t = torch.tensor(bounds_arr[:, 0], dtype=torch.float64)
        hi_t = torch.tensor(bounds_arr[:, 1], dtype=torch.float64)
        span_t = hi_t - lo_t  # (D,)

        # Smart initial conditions (BoTorch
        # gen_batch_initial_conditions pattern):
        #   1. Draw `raw_samples` Sobol points
        #   2. Forward the objective on all of them in one batched no-grad call
        #   3. Boltzmann-select top n_starts with temperature `eta` for diversity
        # Falls back to uniform random when raw_samples ≤ n_starts (e.g. for tests
        # that explicitly want simple multi-start).
        x_inits: list[np.ndarray] = []
        if x0 is not None:
            x_inits.append(np.clip(x0, bounds_arr[:, 0], bounds_arr[:, 1]).astype(np.float64))

        raw_samples = max(int(self.gradient_raw_samples), 0)
        eta = float(self.gradient_init_eta)
        n_more = max(n_starts - len(x_inits), 0)

        if raw_samples > n_more and n_more > 0:
            # Sobol cube samples in (raw_samples, D); evaluate without autograd.
            sobol_seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if self._random_seed is None \
                else int(self.rng.randint(0, 2**31 - 1))
            sobol = torch.quasirandom.SobolEngine(dimension=D, scramble=True, seed=sobol_seed)
            cand = sobol.draw(raw_samples).double() * span_t + lo_t  # (raw_samples, D)
            with torch.no_grad():
                vals = objective_tensor(cand)  # (raw_samples,)
            # Boltzmann selection: weight ∝ exp(−eta · z(value)) where z normalises to [0, 1].
            v = vals.detach().cpu().double()
            v_min, v_max = float(v.min().item()), float(v.max().item())
            if v_max - v_min > 1e-12:
                v_norm = (v - v_min) / (v_max - v_min)
            else:
                v_norm = torch.zeros_like(v)
            # Lower obj = better (negated convention). Boltzmann favours low values.
            logits = -eta * v_norm
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            probs = probs / probs.sum()
            chosen_idx = self.rng.choice(raw_samples, size=n_more, replace=False, p=probs)
            for idx in chosen_idx:
                x_inits.append(cand[int(idx)].cpu().numpy())
        else:
            # Uniform random fallback (used when raw_samples is disabled or tiny).
            for _ in range(n_more):
                x_inits.append(self.rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1]).astype(np.float64))

        x_inits_arr = np.stack(x_inits, axis=0)  # (n_starts, D)

        # Map x → z via inverse sigmoid: z = logit((x - lo) / span). Clamp the
        # interior so logit(0) / logit(1) don't blow up at the cube boundary.
        u = (x_inits_arr - bounds_arr[:, 0]) / np.where(
            bounds_arr[:, 1] - bounds_arr[:, 0] > 0,
            bounds_arr[:, 1] - bounds_arr[:, 0],
            1.0,
        )
        u = np.clip(u, 1e-4, 1.0 - 1e-4)
        z_init = np.log(u / (1.0 - u))  # (n_starts, D)
        z = torch.tensor(z_init, dtype=torch.float64, requires_grad=True)

        history: list[float] = []
        nfev = [0]
        bar = ProgressBar(label, max_iter=n_iters) if show_progress else None

        def _decode_x(z_tensor: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(z_tensor) * span_t + lo_t

        def _eval_obj(z_tensor: torch.Tensor) -> torch.Tensor:
            x = _decode_x(z_tensor)
            vals = objective_tensor(x)
            nfev[0] += int(z_tensor.shape[0])
            return vals

        with profiler.section("engine.run_acquisition_gradient"):
            if method == "adam":
                optimizer = torch.optim.Adam([z], lr=lr)
                for _it in range(n_iters):
                    optimizer.zero_grad()
                    vals = _eval_obj(z)
                    loss = vals.sum()
                    loss.backward()
                    optimizer.step()
                    best_now = float(vals.detach().min().item())
                    history.append(best_now)
                    if bar:
                        bar.step(obj=best_now)
            else:  # lbfgs
                optimizer = torch.optim.LBFGS(
                    [z], lr=lr, max_iter=n_iters, line_search_fn="strong_wolfe",
                )
                last_vals: list[torch.Tensor] = []
                _iter_count = [0]
                _prev_loss: list[float] = [float("inf")]

                def _closure() -> torch.Tensor:
                    optimizer.zero_grad()
                    vals = _eval_obj(z)
                    last_vals.clear()
                    last_vals.append(vals.detach())
                    loss = vals.sum()
                    loss.backward()
                    cur = float(vals.detach().min().item())
                    # Only count as a new iteration when loss improves or first call
                    # (line search re-evaluations don't advance the iteration counter).
                    if cur < _prev_loss[0] - 1e-12 or _iter_count[0] == 0:
                        _iter_count[0] += 1
                        _prev_loss[0] = cur
                        history.append(cur)
                        if bar:
                            bar.step(i=_iter_count[0], obj=cur)
                    return loss

                optimizer.step(_closure)
                if not history and last_vals:
                    history.append(float(last_vals[0].min().item()))

        with torch.no_grad():
            x_final = _decode_x(z).cpu().numpy()  # (n_starts, D)
            vals_final = objective_tensor(_decode_x(z)).cpu().numpy()
            nfev[0] += int(z.shape[0])

        best_idx = int(np.argmin(vals_final))
        best_val = float(vals_final[best_idx])
        best_x = x_final[best_idx]

        if bar:
            bar.finish(suffix=f"{len(history)}/{n_iters}  obj={best_val:.3f}")

        return _OptResult(
            best_x=best_x,
            nfev=nfev[0],
            n_starts=n_starts,
            score=float(-best_val),
            convergence_history=history,
        )

    def _run_de(
        self,
        objective: Callable,
        bounds: list[tuple[float, float]],
        *,
        init_pop: np.ndarray | None = None,
        integrality: list[bool] | None = None,
        label: str = "Optimizing",
        show_progress: bool = False,
        maxiter: int | None = None,
        popsize: int | None = None,
        vectorized: bool = False,
        no_improve_window: int | None = None,
        improvement_eps: float | None = None,
    ) -> _OptResult:
        """Torch-native differential evolution.

        Replaces ``scipy.optimize.differential_evolution`` with a pure-torch
        implementation. Same `_run_de` signature; population update is one
        vectorised tensor step per generation. ``vectorized=True`` means the
        ``objective`` callable accepts ``(D, S)`` numpy arrays and returns
        ``(S,)`` — the standard scipy convention. Scalar objective (one
        ``(D,)`` input → float) is also supported and evaluated row-by-row.

        Mutation: rand/1/bin (DE/rand/1) with F~U[0.5, 1.0] and CR=0.7.
        Integer params (``integrality[d]=True``) are rounded after each
        crossover. No polish step (gradient path is the polish).

        Convergence: same callback heuristic as the prior scipy wrapper —
        halt if ``best_so_far`` hasn't moved by ``improvement_eps`` for
        ``no_improve_window`` generations.
        """
        if maxiter is None:
            maxiter = self.de_maxiter
        if popsize is None:
            popsize = self.de_popsize

        D = len(bounds)
        if D == 0:
            return _OptResult(best_x=None, nfev=0, n_starts=0, score=0.0)

        bounds_arr = np.asarray(bounds, dtype=np.float64)
        lo = torch.tensor(bounds_arr[:, 0], dtype=torch.float64)
        hi = torch.tensor(bounds_arr[:, 1], dtype=torch.float64)
        span = hi - lo

        # Population size matches scipy convention: popsize × D total individuals.
        N_pop = max(int(popsize) * D, 5)
        # Torch RNG seeded off the same engine seed for determinism.
        gen = torch.Generator()
        if self._random_seed is not None:
            gen.manual_seed(int(self.rng.randint(0, 2**31 - 1)))
        else:
            gen.manual_seed(int(np.random.randint(0, 2**31 - 1)))

        # Initial population: provided init_pop, else Latin-hypercube via Sobol.
        if init_pop is not None:
            X = torch.from_numpy(np.asarray(init_pop, dtype=np.float64))
            if X.shape[0] != N_pop:
                # Pad / truncate to N_pop.
                if X.shape[0] < N_pop:
                    pad_count = N_pop - X.shape[0]
                    sobol = torch.quasirandom.SobolEngine(
                        dimension=D, scramble=True,
                        seed=int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item()),
                    )
                    pad = sobol.draw(pad_count).double() * span + lo
                    X = torch.cat([X, pad], dim=0)
                else:
                    X = X[:N_pop]
        else:
            sobol = torch.quasirandom.SobolEngine(
                dimension=D, scramble=True,
                seed=int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item()),
            )
            X = sobol.draw(N_pop).double() * span + lo

        # Round integer dims after init.
        if integrality is not None and any(integrality):
            int_mask = torch.tensor(integrality, dtype=torch.bool)
            X[:, int_mask] = X[:, int_mask].round()
        X = torch.minimum(torch.maximum(X, lo), hi)

        def _eval_pop(P: torch.Tensor) -> torch.Tensor:
            """Evaluate objective at each row of P (N_pop, D). Returns (N_pop,)."""
            P_np = P.cpu().numpy()
            if vectorized:
                vals = np.asarray(objective(P_np.T), dtype=np.float64)  # (S,)
            else:
                vals = np.array([float(objective(P_np[i])) for i in range(P_np.shape[0])], dtype=np.float64)
            return torch.from_numpy(vals)

        f = _eval_pop(X)
        nfev = int(N_pop)

        self.logger.info(
            f"DE init pop: best={float(f.min()):.6f}  worst={float(f.max()):.6f}  "
            f"std={float(f.std()):.6f}  N_pop={N_pop}"
        )

        bar = ProgressBar(label, max_iter=maxiter) if show_progress else None
        history: list[float] = []
        best_so_far = float(f.min().item())
        best_at_last_check = best_so_far
        gens_no_improve = 0
        _eps = improvement_eps if improvement_eps is not None else self.de_improvement_eps
        _window = no_improve_window if no_improve_window is not None else self.de_no_improve_window
        recombination = 0.7
        iter_count = 0

        with profiler.section("engine._run_de [torch DE]"):
            for gen_idx in range(maxiter):
                iter_count += 1
                # Mutation: pick 3 distinct other-than-self indices per individual.
                # Sample with rejection: indices are uniform [0, N_pop), then we
                # bump indices >= self by 1 to skip self for the first column,
                # then re-sample distinct second / third in same way.
                N = N_pop
                rand_idx = torch.randint(0, N - 1, (N, 3), generator=gen)
                self_idx = torch.arange(N).unsqueeze(-1)
                rand_idx = rand_idx + (rand_idx >= self_idx).long()
                # Ensure cols 1, 2 are distinct from col 0 and each other (rough — re-sample collisions).
                for k in range(1, 3):
                    collisions = (rand_idx[:, k:k + 1] == rand_idx[:, :k]).any(dim=-1)
                    while bool(collisions.any().item()):
                        new_vals = torch.randint(0, N - 1, (int(collisions.sum().item()),), generator=gen)
                        new_vals = new_vals + (new_vals >= self_idx[collisions].squeeze(-1)).long()
                        rand_idx[collisions, k] = new_vals
                        collisions = (rand_idx[:, k:k + 1] == rand_idx[:, :k]).any(dim=-1)

                F_val = 0.5 + 0.5 * float(torch.rand(1, generator=gen).item())  # F ∈ [0.5, 1.0]
                a = X[rand_idx[:, 0]]
                b = X[rand_idx[:, 1]]
                c = X[rand_idx[:, 2]]
                mutant = a + F_val * (b - c)
                # Reflect/clamp to bounds.
                mutant = torch.minimum(torch.maximum(mutant, lo), hi)

                # Binomial crossover with X.
                cr_mask = torch.rand(N, D, generator=gen) < recombination
                # Force at least one dim from mutant per individual.
                forced = torch.randint(0, D, (N,), generator=gen)
                cr_mask[torch.arange(N), forced] = True
                trial = torch.where(cr_mask, mutant, X)

                # Round integer dims.
                if integrality is not None and any(integrality):
                    trial[:, int_mask] = trial[:, int_mask].round()  # type: ignore[has-type]

                # Evaluate trial population.
                f_trial = _eval_pop(trial)
                nfev += int(N)

                # Selection: keep trial if better.
                better = f_trial < f
                X = torch.where(better.unsqueeze(-1), trial, X)
                f = torch.where(better, f_trial, f)

                cur_best = float(f.min().item())
                history.append(cur_best)
                if cur_best < best_so_far:
                    best_so_far = cur_best
                if bar:
                    bar.step(obj=best_so_far)

                if best_so_far < best_at_last_check - _eps:
                    best_at_last_check = best_so_far
                    gens_no_improve = 0
                else:
                    gens_no_improve += 1
                if gens_no_improve >= _window:
                    break

        best_idx = int(torch.argmin(f).item())
        best_x = X[best_idx].cpu().numpy()
        best_val = float(f[best_idx].item())

        if bar:
            bar.finish(suffix=f"{iter_count}/{maxiter} iter  obj={best_val:.3f}")

        return _OptResult(
            best_x=best_x,
            nfev=nfev,
            n_starts=1,
            score=float(-best_val),
            convergence_history=history,
        )

    def _wrap_mpc_objective(
        self,
        base_objective: Callable,
        datamodule: DataModule,
        bounds_fn: Callable[[DataModule, dict[str, Any]], np.ndarray],
        depth: int,
        discount: float,
    ) -> Callable:
        """Wrap base_objective with MPC lookahead for online adaptation.

        At each candidate X, simulates ``depth`` future steps via short
        torch-native DE inner solves (5 generations × popsize 4 = ~20 evals
        budget per step) within trust-region bounds. Returns discounted sum
        of scores.

        replaced ``scipy.optimize.minimize`` with
        the torch-native DE inside this engine. Same per-step eval budget.
        """
        if depth <= 0:
            return base_objective

        engine = self  # capture for closure

        class _MpcObjective:
            """Callable MPC wrapper that tracks total inner evaluations."""

            def __init__(self, weight_sum: float):
                self._eval_counter = [0]
                self._weight_sum = weight_sum

            def __call__(self, X: np.ndarray) -> float:
                base_score = base_objective(X)
                self._eval_counter[0] += 1
                total = base_score
                X_cur = X.copy()
                for j in range(depth):
                    try:
                        params_cur = datamodule.array_to_params(X_cur)
                        bounds_ahead_arr = bounds_fn(datamodule, params_cur)
                        bounds_ahead = [(float(lo), float(hi)) for lo, hi in bounds_ahead_arr]
                        # Torch-native short-DE step: 5 generations × popsize 4.
                        # Same ~20-eval budget as the prior scipy L-BFGS-B path,
                        # but no scipy dependency.
                        res = engine._run_de(
                            base_objective, bounds_ahead,
                            maxiter=5, popsize=4,
                            init_pop=X_cur.reshape(1, -1),  # seed near current X
                            label="mpc",
                        )
                        self._eval_counter[0] += res.nfev
                        if res.best_x is not None:
                            X_cur = res.best_x
                        step_score = base_objective(X_cur)
                        self._eval_counter[0] += 1
                        total += discount ** (j + 1) * step_score
                    except Exception:
                        break
                return total / self._weight_sum

        weight_sum = 1.0 + sum(discount ** (j + 1) for j in range(depth))
        return _MpcObjective(weight_sum)
