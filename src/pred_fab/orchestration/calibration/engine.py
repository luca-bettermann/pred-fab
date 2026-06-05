from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from ...utils import PfabLogger, ProgressBar, profiler


@dataclass
class _OptResult:
    """Raw output from an optimizer backend."""
    best_x: np.ndarray | None
    nfev: int
    n_starts: int
    score: float  # negated objective (higher = better)
    convergence_history: list[list[float]] = field(default_factory=list)


class OptimizationEngine:
    """Sobol → top-N → independent LBFGS → pick best.

    Two modes:
      - ``raw_u=False`` (default): sigmoid bound reparameterisation inside the
        engine. Sobol draws in [lo, hi], LBFGS operates in logit u-space,
        objective receives bounded x.
      - ``raw_u=True``: engine operates directly in u-space. Sobol draws
        in the provided bounds (u-bounds), LBFGS operates on raw u,
        objective receives u and handles its own decode.

    (``u`` is the unconstrained optimiser variable; the code historically
    called this frame ``z``.)
    """

    def __init__(self, logger: PfabLogger, random_seed: int | None = None):
        self.logger = logger
        self._random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        self.n_starts: int = 16
        self.n_sobol: int = 512
        self.lr: float = 0.05
        self.sobol_batch_size: int = 64

    def optimize(
        self,
        objective: Callable[[torch.Tensor], torch.Tensor],
        bounds: list[tuple[float, float]],
        *,
        raw_u: bool = False,
        d_param: int | None = None,
        n_starts: int | None = None,
        n_sobol: int | None = None,
        lr: float | None = None,
        compile_objective: bool = False,
        label: str = "Optimizing",
        show_progress: bool = False,
    ) -> _OptResult:
        """Sobol → top-N → independent LBFGS → pick best.

        ``objective`` takes ``(S, D)`` and returns ``(S,)`` — one scalar per
        candidate (lower is better).

        When ``raw_u=True``, bounds are u-space ranges for Sobol sampling,
        and the objective receives raw u-values (no sigmoid in the engine).
        """
        n_starts = n_starts if n_starts is not None else self.n_starts
        n_sobol_base = n_sobol if n_sobol is not None else self.n_sobol
        lr = lr if lr is not None else self.lr

        D = len(bounds)
        D_display = d_param if d_param is not None else D
        if D == 0:
            return _OptResult(best_x=None, nfev=0, n_starts=0, score=0.0)

        if compile_objective:
            try:
                objective = torch.compile(objective, dynamic=True)  # type: ignore[assignment]
            except Exception as e:
                self.logger.warning(f"torch.compile failed; running eager: {e!r}")

        bounds_arr = np.asarray(bounds, dtype=np.float64)
        lo_t = torch.tensor(bounds_arr[:, 0], dtype=torch.float64)
        hi_t = torch.tensor(bounds_arr[:, 1], dtype=torch.float64)
        span_t = hi_t - lo_t

        nfev = [0]

        if raw_u:
            def _eval(u: torch.Tensor) -> torch.Tensor:
                vals = objective(u)
                nfev[0] += int(u.shape[0])
                return vals
        else:
            def _decode(u: torch.Tensor) -> torch.Tensor:
                return torch.sigmoid(u) * span_t + lo_t

            def _encode(x: torch.Tensor) -> torch.Tensor:
                t = (x - lo_t) / span_t
                t = t.clamp(1e-4, 1.0 - 1e-4)
                return torch.log(t / (1.0 - t))

            def _eval(u: torch.Tensor) -> torch.Tensor:
                x = _decode(u)
                vals = objective(x)
                nfev[0] += int(u.shape[0])
                return vals

        # --- Phase 1: Sobol global ---
        n_sobol_scaled = max(n_sobol_base, 32 * D_display)
        sobol_x, sobol_vals = self._sobol_phase(
            objective, n_sobol_scaled, D, lo_t, hi_t, span_t, nfev,
            D_display=D_display,
            show_progress=show_progress,
        )

        # --- Phase 2: Top-N selection ---
        top_idx = torch.argsort(sobol_vals)[:n_starts]
        x_starts = sobol_x[top_idx]
        if raw_u:
            u_starts = x_starts
        else:
            u_starts = _encode(x_starts)  # type: ignore[possibly-undefined]

        # --- Phase 3: Independent LBFGS per start ---
        per_start_history: list[list[float]] = []
        info: dict[str, Any] = {"D": D_display}
        if d_param is not None:
            info["V"] = D
        bar = ProgressBar(label, info=info) if show_progress else None

        best_u: torch.Tensor | None = None
        best_val = float("inf")

        with profiler.section("engine.lbfgs"):
            for s in range(n_starts):
                start_history: list[float] = []
                per_start_history.append(start_history)
                u_s = u_starts[s].clone().detach().unsqueeze(0).requires_grad_(True)
                start_best = [float("inf")]

                def _closure(u_ref: torch.Tensor = u_s, _sh: list[float] = start_history, _s: int = s) -> torch.Tensor:
                    optimizer.zero_grad()  # type: ignore[has-type]
                    vals = _eval(u_ref)
                    loss = vals.sum()
                    if loss.requires_grad:
                        loss.backward()
                    cur = float(vals.detach().item())
                    if cur < start_best[0] - 1e-15:
                        start_best[0] = cur
                    _sh.append(cur)
                    if bar:
                        bar.step(
                            fill=(_s + 1) / n_starts,
                            obj=min(best_val, start_best[0]),
                            starts=_s + 1,
                            iters=len(_sh),
                        )
                    return loss

                optimizer = torch.optim.LBFGS(
                    [u_s], lr=lr, max_iter=100,
                    line_search_fn="strong_wolfe",
                )
                optimizer.step(_closure)

                with torch.no_grad():
                    final_val = float(_eval(u_s).item())
                    nfev[0] -= 1  # _eval already counted; undo double-count

                if final_val < best_val:
                    best_val = final_val
                    best_u = u_s.detach().clone()

        if bar:
            bar.finish()

        # --- Decode best ---
        if best_u is not None:
            if raw_u:
                best_x = best_u.squeeze(0).cpu().numpy()
            else:
                with torch.no_grad():
                    best_x = _decode(best_u).squeeze(0).cpu().numpy()  # type: ignore[possibly-undefined]
        else:
            best_x = None

        return _OptResult(
            best_x=best_x,
            nfev=nfev[0],
            n_starts=n_starts,
            score=float(-best_val),
            convergence_history=per_start_history,
        )

    def _sobol_phase(
        self,
        objective: Callable[[torch.Tensor], torch.Tensor],
        n_sobol: int,
        D: int,
        lo_t: torch.Tensor,
        hi_t: torch.Tensor,
        span_t: torch.Tensor,
        nfev: list[int],
        *,
        D_display: int | None = None,
        show_progress: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw Sobol candidates and evaluate in batches."""
        sobol_seed = (
            int(torch.randint(0, 2**31 - 1, (1,)).item())
            if self._random_seed is None
            else int(self.rng.randint(0, 2**31 - 1))
        )
        sobol = torch.quasirandom.SobolEngine(dimension=D, scramble=True, seed=sobol_seed)
        cand = sobol.draw(n_sobol).double() * span_t + lo_t

        batch_size = self.sobol_batch_size
        n_batches = (n_sobol + batch_size - 1) // batch_size
        sobol_info: dict[str, Any] = {"D": D_display or D}
        if D_display is not None:
            sobol_info["V"] = D
        bar = ProgressBar("Global sampling", info=sobol_info) if show_progress else None

        with torch.no_grad():
            val_chunks: list[torch.Tensor] = []
            batch_num = 0
            for b_start in range(0, n_sobol, batch_size):
                chunk = cand[b_start : b_start + batch_size]
                chunk_vals = objective(chunk)
                val_chunks.append(chunk_vals)
                nfev[0] += int(chunk.shape[0])
                batch_num += 1
                if bar:
                    best_so_far = float(torch.cat(val_chunks, dim=0).min().item())
                    bar.step(fill=batch_num / n_batches, obj=best_so_far, batches=batch_num)
            vals = torch.cat(val_chunks, dim=0)

        if bar:
            bar.finish()

        return cand, vals

    def _wrap_mpc_objective(
        self,
        base_objective: Callable,
        datamodule: Any,
        bounds_fn: Callable,
        depth: int,
        discount: float,
    ) -> Callable:
        """Wrap base_objective with MPC lookahead via short LBFGS inner solves."""
        if depth <= 0:
            return base_objective

        engine = self

        class _MpcObjective:
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

                        def _obj_tensor(x_S: torch.Tensor) -> torch.Tensor:
                            with torch.no_grad():
                                return torch.tensor(
                                    [base_objective(x_S[i].cpu().numpy()) for i in range(x_S.shape[0])],
                                    dtype=x_S.dtype,
                                )

                        res = engine.optimize(
                            _obj_tensor, bounds_ahead,
                            n_starts=1,
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
