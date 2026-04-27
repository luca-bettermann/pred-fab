from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import warnings

import numpy as np
from scipy.optimize import minimize, differential_evolution

from ...core import DataModule
from ...utils import PfabLogger, ProgressBar


class Optimizer(Enum):
    """Optimization backend for the calibration acquisition function."""
    LBFGSB = "lbfgsb"  # gradient-based multi-start (fast, local)
    DE     = "de"       # differential evolution (global, slower)


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
        self.de_popsize: int = 15
        self.de_tol: float = 0.0  # passed to scipy; 0 effectively disables std/mean exit
        self.de_no_improve_window: int = 10  # generations without improvement → halt
        self.de_improvement_eps: float = 1e-6  # min Δbest to count as an improvement

        # L-BFGS-B optimizer parameters (gradient-based, multi-start)
        self.lbfgsb_maxfun: int | None = None
        self.lbfgsb_eps: float = 1e-3

        # Schedule smoothing: penalizes speed changes between adjacent layers
        self.schedule_smoothing: float = 0.05

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
    ) -> _OptResult:
        """Unified differential evolution wrapper."""
        if maxiter is None:
            maxiter = self.de_maxiter
        if popsize is None:
            popsize = self.de_popsize
        has_int = integrality is not None and any(integrality)
        bar = ProgressBar(label, max_iter=maxiter) if show_progress else None
        iter_count = [0]
        best_so_far = [np.inf]
        best_at_last_check = [np.inf]
        gens_no_improve = [0]
        history: list[float] = []
        improvement_eps = self.de_improvement_eps
        no_improve_window = self.de_no_improve_window

        def _tracked_objective(x: np.ndarray) -> float:
            val = objective(x)
            if val < best_so_far[0]:
                best_so_far[0] = val
            return val

        def _progress(xk: Any, convergence: Any) -> bool:
            iter_count[0] += 1
            history.append(best_so_far[0])
            if bar:
                bar.step(obj=best_so_far[0])
            # Improvement-window check: halt if best_so_far hasn't moved by at
            # least improvement_eps in no_improve_window consecutive generations.
            if best_so_far[0] < best_at_last_check[0] - improvement_eps:
                best_at_last_check[0] = best_so_far[0]
                gens_no_improve[0] = 0
            else:
                gens_no_improve[0] += 1
            return gens_no_improve[0] >= no_improve_window

        de_kwargs: dict[str, Any] = dict(
            func=_tracked_objective,
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            mutation=(0.5, 1.0),
            recombination=0.7,
            tol=self.de_tol,
            # scipy declares convergence when std(energies) <= atol + tol*|mean|.
            # On low-D / integer landscapes the population can collapse to
            # identical energies after gen 1 (std=0) — atol=0 then trips even
            # if the global optimum hasn't been found. Negative atol disables
            # that exit; the callback's no-improvement-window is authoritative.
            atol=-1.0,
            polish=not has_int,
            callback=_progress,
        )
        if init_pop is not None:
            de_kwargs["init"] = init_pop
        else:
            de_kwargs["init"] = "latinhypercube"
        if integrality is not None:
            de_kwargs["integrality"] = integrality
        if self._random_seed is not None:
            de_kwargs["seed"] = int(self.rng.randint(0, 2**31 - 1))

        result = differential_evolution(**de_kwargs)  # type: ignore[call-overload]

        if bar:
            bar.finish(suffix=f"{iter_count[0]}/{maxiter} iter  obj={result.fun:.3f}")

        return _OptResult(
            best_x=result.x,
            nfev=result.nfev,
            n_starts=1,
            score=float(-result.fun),
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
        """Multi-start L-BFGS-B optimization."""
        bounds_arr = np.array(bounds)
        if x0_list is None or not x0_list:
            x0_list = [self.rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1])]

        n_dims = len(bounds)
        max_fun = self.lbfgsb_maxfun if self.lbfgsb_maxfun is not None else max(100, 10 * (n_dims + 1))
        eps = self.lbfgsb_eps
        total_starts = len(x0_list)

        best_x, best_val = None, np.inf
        total_nfev = 0
        bar = ProgressBar(label, max_iter=total_starts) if show_progress else None
        for i, x0_i in enumerate(x0_list):
            if bar:
                bar.step()
            try:
                res = minimize(
                    fun=objective, x0=x0_i, bounds=bounds_arr, method='L-BFGS-B',
                    options={'eps': eps, 'maxfun': max_fun},
                )
                total_nfev += res.nfev
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
                self.logger.debug(
                    f"  start {i + 1}/{total_starts}: val={res.fun:.6f}, nfev={res.nfev}, converged={res.success}"
                )
            except Exception as e:
                self.logger.warning(f"L-BFGS-B round {i + 1} failed: {e}")

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

        At each candidate X, simulates `depth` future steps via inner L-BFGS-B
        solves within trust-region bounds. Returns discounted sum of scores.
        """
        if depth <= 0:
            return base_objective

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
                        bounds_ahead = bounds_fn(datamodule, params_cur)
                        res = minimize(
                            fun=base_objective,
                            x0=X_cur,
                            bounds=bounds_ahead,
                            method='L-BFGS-B',
                            options={'maxfun': 20},
                        )
                        self._eval_counter[0] += res.nfev
                        X_cur = res.x
                        step_score = base_objective(X_cur)
                        self._eval_counter[0] += 1
                        total += discount ** (j + 1) * step_score
                    except Exception:
                        break
                return total / self._weight_sum

        weight_sum = 1.0 + sum(discount ** (j + 1) for j in range(depth))
        return _MpcObjective(weight_sum)
