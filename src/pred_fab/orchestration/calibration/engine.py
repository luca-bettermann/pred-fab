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
    convergence_history: list[float] = field(default_factory=list)


warnings.filterwarnings("ignore", category=UserWarning)


class OptimizationEngine:
    """Numerical optimization backend — LBFGS with sigmoid bound reparameterisation."""

    def __init__(self, logger: PfabLogger, random_seed: int | None = None):
        self.logger = logger
        self._random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        self.gradient_n_starts: int = 4
        self.gradient_n_iters: int = 100
        self.gradient_lr: float = 0.05
        self.gradient_method: str = "lbfgs"
        self.gradient_raw_samples: int = 256
        self.gradient_init_eta: float = 1.0

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
        raw_samples: int | None = None,
        compile_objective: bool = False,
        label: str = "Optimizing",
        show_progress: bool = False,
    ) -> _OptResult:
        """Multi-start gradient optimisation with sigmoid bound reparameterisation.

        ``objective_tensor`` takes ``(S, D)`` and returns ``(S,)`` — one scalar
        per starting point. Bounds enforced via ``x = sigmoid(z) · (hi - lo) + lo``.
        """
        n_starts = n_starts if n_starts is not None else self.gradient_n_starts
        n_iters = n_iters if n_iters is not None else self.gradient_n_iters
        lr = lr if lr is not None else self.gradient_lr
        method = (method or self.gradient_method).lower()
        if method not in ("adam", "lbfgs", "sgd"):
            raise ValueError(f"unknown gradient method: {method!r}")

        D = len(bounds)
        if D == 0:
            return _OptResult(best_x=None, nfev=0, n_starts=0, score=0.0)

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
        span_t = hi_t - lo_t

        x_inits: list[np.ndarray] = []
        if x0 is not None:
            x_inits.append(np.clip(x0, bounds_arr[:, 0], bounds_arr[:, 1]).astype(np.float64))

        _raw_base = raw_samples if raw_samples is not None else self.gradient_raw_samples
        raw_samples = max(int(_raw_base * (D ** 0.5)), 0)
        eta = float(self.gradient_init_eta)
        n_more = max(n_starts - len(x_inits), 0)

        if raw_samples > n_more and n_more > 0:
            sobol_seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if self._random_seed is None \
                else int(self.rng.randint(0, 2**31 - 1))
            sobol = torch.quasirandom.SobolEngine(dimension=D, scramble=True, seed=sobol_seed)
            cand = sobol.draw(raw_samples).double() * span_t + lo_t
            with torch.no_grad():
                vals = objective_tensor(cand)
            v = vals.detach().cpu().double()
            v_min, v_max = float(v.min().item()), float(v.max().item())
            if v_max - v_min > 1e-12:
                v_norm = (v - v_min) / (v_max - v_min)
            else:
                v_norm = torch.zeros_like(v)
            logits = -eta * v_norm
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            probs = probs / probs.sum()
            chosen_idx = self.rng.choice(raw_samples, size=n_more, replace=False, p=probs)
            for idx in chosen_idx:
                x_inits.append(cand[int(idx)].cpu().numpy())
        else:
            for _ in range(n_more):
                x_inits.append(self.rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1]).astype(np.float64))

        x_inits_arr = np.stack(x_inits, axis=0)

        u = (x_inits_arr - bounds_arr[:, 0]) / np.where(
            bounds_arr[:, 1] - bounds_arr[:, 0] > 0,
            bounds_arr[:, 1] - bounds_arr[:, 0],
            1.0,
        )
        u = np.clip(u, 1e-4, 1.0 - 1e-4)
        z_init = np.log(u / (1.0 - u))
        z = torch.tensor(z_init, dtype=torch.float64, requires_grad=True)

        history: list[float] = []
        nfev = [0]
        bar = ProgressBar(label) if show_progress else None

        def _decode_x(z_tensor: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(z_tensor) * span_t + lo_t

        # Scale objective to keep gradients above LBFGS line-search threshold.
        # Evidence values shrink with dimensionality; scaling preserves the
        # optimum while making the gradient landscape navigable.
        obj_scale = float(max(D, 1)) * 100.0

        def _eval_obj(z_tensor: torch.Tensor) -> torch.Tensor:
            x = _decode_x(z_tensor)
            vals = objective_tensor(x)
            nfev[0] += int(z_tensor.shape[0])
            return vals * obj_scale

        with profiler.section("engine.run_acquisition_gradient"):
            if method == "sgd":
                optimizer = torch.optim.SGD([z], lr=lr)
                for _it in range(n_iters):
                    optimizer.zero_grad()
                    vals = _eval_obj(z)
                    loss = vals.sum()
                    if loss.requires_grad:
                        loss.backward()
                    optimizer.step()
                    best_now = float(vals.detach().min().item()) / obj_scale
                    history.append(best_now)
                    if bar:
                        bar.step(obj=best_now)
            elif method == "adam":
                optimizer = torch.optim.Adam([z], lr=lr)
                for _it in range(n_iters):
                    optimizer.zero_grad()
                    vals = _eval_obj(z)
                    loss = vals.sum()
                    if loss.requires_grad:
                        loss.backward()
                    optimizer.step()
                    best_now = float(vals.detach().min().item()) / obj_scale
                    history.append(best_now)
                    if bar:
                        bar.step(obj=best_now)
            else:
                last_vals: list[torch.Tensor] = []
                _iter_count = [0]
                _best_obj: list[float] = [float("inf")]

                def _closure() -> torch.Tensor:
                    optimizer.zero_grad()  # type: ignore[has-type]
                    vals = _eval_obj(z)
                    last_vals.clear()
                    last_vals.append(vals.detach())
                    loss = vals.sum()
                    if loss.requires_grad:
                        loss.backward()
                    cur = float(vals.detach().min().item()) / obj_scale
                    if cur < _best_obj[0] - 1e-15 / obj_scale or _iter_count[0] == 0:
                        _iter_count[0] += 1
                        _best_obj[0] = cur
                        history.append(cur)
                        if bar:
                            bar.step(i=_iter_count[0], obj=cur)
                    return loss

                optimizer = torch.optim.LBFGS(
                    [z], lr=lr, max_iter=n_iters,
                    line_search_fn="strong_wolfe",
                    tolerance_grad=0, tolerance_change=0,
                )
                optimizer.step(_closure)
                if not history and last_vals:
                    history.append(float(last_vals[0].min().item()))

        with torch.no_grad():
            x_final = _decode_x(z).cpu().numpy()
            vals_final = objective_tensor(_decode_x(z)).cpu().numpy()
            nfev[0] += int(z.shape[0])

        best_idx = int(np.argmin(vals_final))
        best_val = float(vals_final[best_idx])
        best_x = x_final[best_idx]

        if bar:
            bar.finish()

        return _OptResult(
            best_x=best_x,
            nfev=nfev[0],
            n_starts=n_starts,
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

                        res = engine.run_acquisition_gradient(
                            _obj_tensor, bounds_ahead,
                            x0=X_cur, n_starts=1, n_iters=5,
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
