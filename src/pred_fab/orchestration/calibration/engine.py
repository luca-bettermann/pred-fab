from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import warnings

import numpy as np
import torch
# Strategy D commit 9: scipy.optimize fully replaced — torch DE for global,
# torch.optim.LBFGS for local. Module is scipy-free.

from ...core import DataModule
from ...utils import PfabLogger, ProgressBar, profiler


class Optimizer(Enum):
    """Optimization backend for the calibration acquisition function."""
    LBFGSB   = "lbfgsb"    # gradient-based multi-start, scipy numpy (fast, local)
    DE       = "de"         # differential evolution, scipy numpy (global, slower)
    GRADIENT = "gradient"   # autograd multi-start, torch.optim (Strategy D commit 5)


@dataclass
class _OptResult:
    """Raw output from an optimizer backend."""
    best_x: np.ndarray | None
    nfev: int
    n_starts: int
    score: float  # negated objective (higher = better)
    convergence_history: list[float] = field(default_factory=list)  # best energy per iteration


# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ======================================================================
# OptimizationEngine — pure optimization, no schema/calibration knowledge
# ======================================================================

class OptimizationEngine:
    """Numerical optimization backend: DE and L-BFGS-B with joint schedule support."""

    def __init__(self, logger: PfabLogger, random_seed: int | None = None):
        self.logger = logger
        self._random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        # DE optimizer parameters (global, population-based + L-BFGS-B polish)
        # Convergence is governed by "no improvement in K generations" via the
        # callback rather than scipy's std-vs-mean criterion, which misfires
        # on low-D / integer landscapes by declaring victory after one gen
        # with a clustered population. Setting scipy's tol to 0 disables that
        # fragile check; the callback is the authoritative exit signal.
        self.de_maxiter: int = 1000
        # scipy treats popsize as a *multiplier* of D — total population is
        # popsize × D individuals per generation. Default 8 (was scipy's 15)
        # gives sufficient diversity for our smooth-ish acquisition landscapes
        # while halving evaluations per generation. Tunable via configure_optimizer.
        self.de_popsize: int = 8
        self.de_tol: float = 0.0  # passed to scipy; 0 effectively disables std/mean exit
        self.de_no_improve_window: int = 10  # generations without improvement → halt
        self.de_improvement_eps: float = 1e-6  # min Δbest to count as an improvement

        # L-BFGS-B optimizer parameters (gradient-based, multi-start)
        self.lbfgsb_maxfun: int | None = None
        self.lbfgsb_eps: float = 1e-3

        # GRADIENT optimizer parameters (autograd multi-start with sigmoid bound reparam)
        self.gradient_n_starts: int = 4
        self.gradient_n_iters: int = 60
        self.gradient_lr: float = 0.05
        self.gradient_method: str = "adam"  # "adam" | "lbfgs"

        # Smart-init parameters (Strategy D commit 9b — BoTorch gen_batch_initial_conditions
        # pattern). raw_samples Sobol points are batch-evaluated, top-K selected via
        # Boltzmann sampling with temperature `eta` to balance quality vs. diversity.
        # 0 disables (uniform random multi-start fallback).
        self.gradient_raw_samples: int = 256
        self.gradient_init_eta: float = 1.0

        # Schedule smoothing: penalizes speed changes between adjacent layers
        self.schedule_smoothing: float = 0.05

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

    def run(
        self,
        per_step_fn: Callable[[np.ndarray], float],
        N: int,
        D_static: int,
        D_sched: int,
        L: int,
        static_bounds: list[tuple[float, float]],
        sched_bounds: list[tuple[float, float]],
        sched_deltas: np.ndarray,
        *,
        optimizer: Optimizer | None = None,
        default_optimizer: Optimizer = Optimizer.DE,
        init_pop: np.ndarray | None = None,
        integrality_static: list[bool] | None = None,
        x0: np.ndarray | None = None,
        n_restarts: int = 0,
        label: str = "Optimizing",
        show_progress: bool = False,
    ) -> tuple[_OptResult, np.ndarray, np.ndarray]:
        """Single optimization engine for all calibration use cases.

        Vector layout per unit: [static, sched_step0, offset_1, ..., offset_{L-1}]
        Total vars: N x (D_static + D_sched + (L-1) x D_sched)

        Returns (opt_result, static_out[N, D_static], sched_out[N, L, D_sched]).
        """
        active_optimizer = optimizer or default_optimizer
        D_unit = D_static + D_sched + max(L - 1, 0) * D_sched
        n_vars = N * D_unit

        # --- 1. Build bounds ---
        all_bounds: list[tuple[float, float]] = []
        integrality: list[bool] | None = None
        if integrality_static is not None and any(integrality_static):
            integrality = []

        for _u in range(N):
            for d in range(D_static):
                all_bounds.append(static_bounds[d])
                if integrality is not None:
                    integrality.append(integrality_static[d])  # type: ignore[index]
            for d in range(D_sched):
                all_bounds.append(sched_bounds[d])
                if integrality is not None:
                    integrality.append(False)
            for _k in range(1, L):
                for d in range(D_sched):
                    dn = float(sched_deltas[d])
                    all_bounds.append((-dn, dn))
                    if integrality is not None:
                        integrality.append(False)

        # --- 2. Build objective wrapper ---
        def _objective(x_flat: np.ndarray) -> float:
            units = x_flat.reshape(N, D_unit)
            step_sum = 0.0

            for k in range(L):
                pts = np.zeros((N, D_static + D_sched))
                for u in range(N):
                    pts[u, :D_static] = units[u, :D_static]
                    if D_sched > 0:
                        step0 = units[u, D_static:D_static + D_sched]
                        abs_val = step0.copy()
                        for kk in range(1, k + 1):
                            off_start = D_static + D_sched + (kk - 1) * D_sched
                            abs_val = abs_val + units[u, off_start:off_start + D_sched]
                        pts[u, D_static:] = abs_val

                step_sum += per_step_fn(pts)

            return step_sum / L

        # --- 3. Run optimizer ---
        if active_optimizer == Optimizer.DE:
            opt = self._run_de(
                _objective,
                all_bounds,
                init_pop=init_pop,
                integrality=integrality,
                label=label,
                show_progress=show_progress,
                maxiter=self.smart_maxiter(n_vars),
            )
        else:
            x0_list: list[np.ndarray] = []
            if x0 is not None:
                if x0.size == D_static:
                    x0_list.append(x0)
                else:
                    x0_list.append(x0[:n_vars] if x0.size >= n_vars else x0)
            bounds_arr = np.array(all_bounds)
            for _ in range(n_restarts):
                x0_list.append(self.rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1]))
            if not x0_list:
                x0_list.append(self.rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1]))

            opt = self._run_lbfgsb(
                _objective, bounds_arr.tolist(), x0_list=x0_list,
                label=label, show_progress=show_progress,
            )

        # --- 4. Decode result ---
        if opt.best_x is not None:
            units = opt.best_x.reshape(N, D_unit)
        else:
            units = np.full((N, D_unit), 0.5)

        static_out = units[:, :D_static]
        sched_out = np.zeros((N, L, D_sched))
        if D_sched > 0:
            for u in range(N):
                step0 = units[u, D_static:D_static + D_sched]
                sched_out[u, 0] = step0
                for k in range(1, L):
                    off_start = D_static + D_sched + (k - 1) * D_sched
                    sched_out[u, k] = sched_out[u, k - 1] + units[u, off_start:off_start + D_sched]

        return opt, static_out, sched_out

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

        bounds_arr = np.asarray(bounds, dtype=np.float64)
        lo_t = torch.tensor(bounds_arr[:, 0], dtype=torch.float64)
        hi_t = torch.tensor(bounds_arr[:, 1], dtype=torch.float64)
        span_t = hi_t - lo_t  # (D,)

        # Smart initial conditions (Strategy D commit 9b — BoTorch
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

                def _closure() -> torch.Tensor:
                    optimizer.zero_grad()
                    vals = _eval_obj(z)
                    last_vals.clear()
                    last_vals.append(vals.detach())
                    loss = vals.sum()
                    loss.backward()
                    history.append(float(vals.detach().min().item()))
                    if bar:
                        bar.step(obj=history[-1])
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
            bar.finish(suffix=f"obj={best_val:.3f}")

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
    ) -> _OptResult:
        """Torch-native differential evolution (Strategy D commit 9).

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

        bar = ProgressBar(label, max_iter=maxiter) if show_progress else None
        history: list[float] = []
        best_so_far = float(f.min().item())
        best_at_last_check = best_so_far
        gens_no_improve = 0
        improvement_eps = self.de_improvement_eps
        no_improve_window = self.de_no_improve_window
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

                if best_so_far < best_at_last_check - improvement_eps:
                    best_at_last_check = best_so_far
                    gens_no_improve = 0
                else:
                    gens_no_improve += 1
                if gens_no_improve >= no_improve_window:
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

    def _run_lbfgsb(
        self,
        objective: Callable,
        bounds: list[tuple[float, float]],
        *,
        x0_list: list[np.ndarray] | None = None,
        label: str = "Optimizing",
        show_progress: bool = False,
    ) -> _OptResult:
        """Multi-start L-BFGS optimisation via torch.optim (Strategy D commit 9 step B).

        Replaces ``scipy.optimize.minimize(method='L-BFGS-B')`` with
        ``torch.optim.LBFGS`` driven by finite-difference gradients on the
        numpy scalar objective. Bounds enforced via sigmoid reparameterisation
        (same trick as ``run_acquisition_gradient``); each start runs a
        single LBFGS solve with strong-Wolfe line search.

        This path will be deleted entirely in Phase 5 commit 18 — the
        ``Optimizer`` enum collapses to a single ``run_acquisition`` method
        — but it's functional and scipy-free in the interim.
        """
        bounds_arr = np.array(bounds, dtype=np.float64)
        if x0_list is None or not x0_list:
            x0_list = [self.rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1])]

        n_dims = len(bounds)
        if n_dims == 0:
            return _OptResult(best_x=None, nfev=0, n_starts=0, score=0.0)

        max_fun = self.lbfgsb_maxfun if self.lbfgsb_maxfun is not None else max(100, 10 * (n_dims + 1))
        eps = self.lbfgsb_eps
        total_starts = len(x0_list)

        lo_t = torch.tensor(bounds_arr[:, 0], dtype=torch.float64)
        hi_t = torch.tensor(bounds_arr[:, 1], dtype=torch.float64)
        span_t = hi_t - lo_t

        def _decode_x(z_tensor: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(z_tensor) * span_t + lo_t

        def _eval_fd(x_np: np.ndarray) -> tuple[float, np.ndarray]:
            """Numpy scalar objective + central finite-difference gradient."""
            f0 = float(objective(x_np))
            grad = np.zeros(n_dims, dtype=np.float64)
            for d in range(n_dims):
                x_plus = x_np.copy()
                x_plus[d] += eps
                x_plus[d] = min(x_plus[d], bounds_arr[d, 1])
                x_minus = x_np.copy()
                x_minus[d] -= eps
                x_minus[d] = max(x_minus[d], bounds_arr[d, 0])
                f_plus = float(objective(x_plus))
                f_minus = float(objective(x_minus))
                grad[d] = (f_plus - f_minus) / (2.0 * eps)
            return f0, grad

        best_x, best_val = None, np.inf
        total_nfev = 0
        bar = ProgressBar(label, max_iter=total_starts) if show_progress else None
        for i, x0_i in enumerate(x0_list):
            if bar:
                bar.step()
            try:
                # Map x0 → z via inverse sigmoid (clamped interior to avoid logit blowup).
                u0 = (np.clip(x0_i, bounds_arr[:, 0], bounds_arr[:, 1]) - bounds_arr[:, 0]) / np.where(
                    bounds_arr[:, 1] - bounds_arr[:, 0] > 0, bounds_arr[:, 1] - bounds_arr[:, 0], 1.0,
                )
                u0 = np.clip(u0, 1e-4, 1.0 - 1e-4)
                z0 = np.log(u0 / (1.0 - u0))
                z = torch.tensor(z0, dtype=torch.float64, requires_grad=True)
                opt = torch.optim.LBFGS(
                    [z], max_iter=max(int(max_fun // (n_dims * 2 + 1)), 1),
                    line_search_fn="strong_wolfe",
                )
                eval_count = [0]

                def _closure(_z: torch.Tensor = z) -> torch.Tensor:
                    opt.zero_grad()
                    x_np = _decode_x(_z).detach().cpu().numpy()
                    f_val, grad_np = _eval_fd(x_np)
                    eval_count[0] += 2 * n_dims + 1
                    # Chain rule through sigmoid: dx/dz = sigmoid * (1 - sigmoid) * span.
                    sig = torch.sigmoid(_z)
                    dx_dz = sig * (1.0 - sig) * span_t
                    grad_t = torch.tensor(grad_np, dtype=torch.float64) * dx_dz
                    _z.grad = grad_t
                    return torch.tensor(f_val, dtype=torch.float64)

                opt.step(_closure)
                with torch.no_grad():
                    x_final = _decode_x(z).cpu().numpy()
                    f_final = float(objective(x_final))
                eval_count[0] += 1
                total_nfev += eval_count[0]
                if f_final < best_val:
                    best_val = f_final
                    best_x = x_final
                self.logger.debug(
                    f"  start {i + 1}/{total_starts}: val={f_final:.6f}, nfev={eval_count[0]}"
                )
            except Exception as e:
                self.logger.warning(f"L-BFGS round {i + 1} failed: {e}")

        if bar:
            obj_str = f"obj={best_val:.3f}" if best_val < np.inf else "no solution"
            bar.finish(suffix=obj_str)

        return _OptResult(
            best_x=best_x,
            nfev=total_nfev,
            n_starts=total_starts,
            score=float(-best_val) if best_val != np.inf else 0.0,
        )

    @staticmethod
    def _schedule_smoothing_factor(
        scheduled_values: np.ndarray,
        deltas: np.ndarray,
        lam: float,
    ) -> float:
        """Multiplicative penalty in (0, 1] for schedule jumps."""
        if lam <= 0 or scheduled_values.shape[0] <= 1:
            return 1.0
        factor = 1.0
        for k in range(1, scheduled_values.shape[0]):
            for d in range(scheduled_values.shape[1]):
                if deltas[d] <= 0:
                    continue
                change = abs(scheduled_values[k, d] - scheduled_values[k - 1, d])
                frac = min(change / deltas[d], 1.0)
                factor *= (1.0 - lam * frac)
        return factor

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

        Strategy D commit 9 step B: replaced ``scipy.optimize.minimize`` with
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
