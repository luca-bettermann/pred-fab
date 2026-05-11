"""Trajectory optimizer — coordinate descent over per-experiment trajectories.

Each experiment's trajectory is optimized one at a time. Non-active experiments
are placed into the KDE as "old" kernels (via push_virtual), and the active
experiment's layers are passed as "new" candidates through the standard
acquisition function. The objective evaluates total evidence across ALL
experiments, with gradient flowing only through the active one.

Global solution maps to the trajectory midpoint, allowing the trajectory
to extend in both directions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from ...core import DataModule, ExperimentSpec, ParameterProposal, ParameterTrajectory
from ...utils.enum import SourceStep
from ...utils import ProgressBar
from .engine import OptimizationEngine, _OptResult


@dataclass
class TrajectoryState:
    """Mutable state for the coordinate-descent trajectory loop."""
    n: int
    flat_specs: list[ExperimentSpec]
    sched_params: list[tuple[str, float, float]]
    static_params: list[tuple[str, float, float]]
    per_exp_L: list[int]
    primary_dim_code: str
    sched_delta_norms: list[float]
    static_delta_norms: list[float]

    static_norms: np.ndarray
    schedule_norms: list[np.ndarray]

    n_dm_cols: int
    exp_base_rows: list[np.ndarray]
    sched_col_map: list[tuple[int, int]]
    static_col_map: list[tuple[int, int]]

    @property
    def D_sched(self) -> int:
        return len(self.sched_params)

    @property
    def D_static(self) -> int:
        return len(self.static_params)


class TrajectoryOptimizer:
    """Coordinate-descent trajectory optimizer with global objective.

    Non-active experiments go into the KDE as "old" kernels (via push/pop
    virtual). The active experiment's layers are "new" candidates. The
    acquisition function evaluates total evidence (old + new), with
    gradient only through the active experiment.
    """

    def __init__(
        self,
        engine: OptimizationEngine,
        acquisition_fn: Callable[[torch.Tensor, float, Any], torch.Tensor],
        *,
        push_virtual_fn: Callable[[list[dict[str, Any]], list[float] | None, DataModule | None], None] | None = None,
        pop_virtual_fn: Callable[[], None] | None = None,
        sanitize_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        smoothness_weight: float = 0.01,
        max_rounds: int = 5,
        step_callback: Callable[[int, int, TrajectoryState], None] | None = None,
    ):
        self._engine = engine
        self._acquisition_fn = acquisition_fn
        self._push_virtual_fn = push_virtual_fn
        self._pop_virtual_fn = pop_virtual_fn
        self._sanitize = sanitize_fn or (lambda d: d)
        self.smoothness_weight = smoothness_weight
        self.max_rounds = max_rounds
        self.step_callback = step_callback

        self.last_trajectory_points: np.ndarray | None = None
        self.last_trajectory_exp_ids: list[int] | None = None
        self.last_trajectory_schedule_norms: list[np.ndarray] | None = None
        self.last_trajectory_sched_params: list[tuple[str, float, float]] | None = None
        self.last_trajectory_per_exp_L: list[int] | None = None
        self.total_iters: int = 0
        self.convergence_history: list[float] = []

    def optimize(
        self,
        state: TrajectoryState,
        datamodule: DataModule | None,
        console: bool = False,
    ) -> list[ExperimentSpec]:
        if state.D_sched == 0 or max(state.per_exp_L) <= 1:
            return list(state.flat_specs)

        self._run_coordinate_descent(state, datamodule, console)
        return self._decode_specs(state)

    # ------------------------------------------------------------------
    # Virtual point helpers
    # ------------------------------------------------------------------

    def _build_virtual_params(
        self, state: TrajectoryState, exclude_idx: int = -1,
    ) -> tuple[list[dict[str, Any]], list[float]]:
        """Build param dicts + 1/L weights for non-active experiments."""
        params_list: list[dict[str, Any]] = []
        weights_list: list[float] = []
        for j in range(state.n):
            if j == exclude_idx and exclude_idx >= 0:
                continue
            L_j = state.per_exp_L[j]
            w = 1.0 / L_j
            for k in range(L_j):
                p: dict[str, Any] = {}
                for si, (code, lo, hi) in enumerate(state.static_params):
                    p[code] = float(state.static_norms[j, si] * (hi - lo) + lo)
                for si, (code, lo, hi) in enumerate(state.sched_params):
                    p[code] = float(state.schedule_norms[j][k, si] * (hi - lo) + lo)
                base_dict = state.flat_specs[j].initial_params.to_dict()
                for code, val in base_dict.items():
                    if code not in p:
                        p[code] = val
                params_list.append(p)
                weights_list.append(w)
        return params_list, weights_list

    def _push_others(self, state: TrajectoryState, exp_idx: int, datamodule: DataModule | None) -> None:
        """Push non-active experiments into KDE as old kernels."""
        if self._push_virtual_fn is None:
            return
        others, weights = self._build_virtual_params(state, exclude_idx=exp_idx)
        if others:
            self._push_virtual_fn(others, weights, datamodule)

    def _pop(self) -> None:
        if self._pop_virtual_fn is not None:
            self._pop_virtual_fn()

    # ------------------------------------------------------------------
    # Trajectory decode (decision vector → absolute layer positions)
    # ------------------------------------------------------------------

    def _decode_trajectory_from_midpoint(
        self, x_S: torch.Tensor, D_static: int, D_sched: int, L_i: int,
        sched_delta_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode decision vector with Global anchor at midpoint.

        Returns (stat, traj) where traj is (S, L_i, D_sched).
        """
        S = int(x_S.shape[0])
        stat = x_S[:, :D_static]
        mid_idx = L_i // 2
        mid_val = x_S[:, D_static:D_static + D_sched]

        traj = torch.zeros(S, L_i, D_sched, dtype=x_S.dtype)
        traj[:, mid_idx, :] = mid_val

        if L_i > 1:
            d_vars = x_S[:, D_static + D_sched:].reshape(S, L_i - 1, D_sched)
            # Forward from midpoint: layers mid_idx+1 ... L_i-1
            prev = mid_val
            for k in range(mid_idx, L_i - 1):
                lo_k = (prev - sched_delta_t).clamp(min=0.0)
                hi_k = (prev + sched_delta_t).clamp(max=1.0)
                traj[:, k + 1, :] = lo_k + d_vars[:, k, :] * (hi_k - lo_k)
                prev = traj[:, k + 1, :]
            # Backward from midpoint: layers mid_idx-1 ... 0
            prev = mid_val
            for k in range(mid_idx - 1, -1, -1):
                lo_k = (prev - sched_delta_t).clamp(min=0.0)
                hi_k = (prev + sched_delta_t).clamp(max=1.0)
                traj[:, k, :] = lo_k + d_vars[:, k, :] * (hi_k - lo_k)
                prev = traj[:, k, :]

        return stat, traj

    def _decode_trajectory_from_midpoint_np(
        self, best_x: np.ndarray, D_static: int, D_sched: int, L_i: int,
        sched_deltas_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Numpy version of midpoint decode for storing results."""
        new_static = best_x[:D_static]
        mid_idx = L_i // 2
        mid_val = best_x[D_static:D_static + D_sched]

        new_sched = np.zeros((L_i, D_sched))
        new_sched[mid_idx] = mid_val

        if L_i > 1:
            d_vars = best_x[D_static + D_sched:].reshape(L_i - 1, D_sched)
            prev = mid_val.copy()
            for k in range(mid_idx, L_i - 1):
                lo_k = np.maximum(0.0, prev - sched_deltas_np)
                hi_k = np.minimum(1.0, prev + sched_deltas_np)
                new_sched[k + 1] = lo_k + d_vars[k] * (hi_k - lo_k)
                prev = new_sched[k + 1].copy()
            prev = mid_val.copy()
            for k in range(mid_idx - 1, -1, -1):
                lo_k = np.maximum(0.0, prev - sched_deltas_np)
                hi_k = np.minimum(1.0, prev + sched_deltas_np)
                new_sched[k] = lo_k + d_vars[k] * (hi_k - lo_k)
                prev = new_sched[k].copy()

        return new_static, new_sched

    # ------------------------------------------------------------------
    # Coordinate descent
    # ------------------------------------------------------------------

    def _run_coordinate_descent(
        self,
        state: TrajectoryState,
        datamodule: DataModule | None,
        console: bool,
    ) -> None:
        n = state.n
        D_static = state.D_static
        D_sched = state.D_sched
        per_exp_L = list(state.per_exp_L)

        base_rows_t = torch.from_numpy(np.stack(state.exp_base_rows)).to(dtype=torch.float64)
        n_dm_cols = base_rows_t.shape[1]
        static_col_idxs = torch.tensor(
            [col_idx for _, col_idx in state.static_col_map], dtype=torch.long,
        )
        sched_col_idxs = torch.tensor(
            [col_idx for _, col_idx in state.sched_col_map], dtype=torch.long,
        )
        static_delta_norms_list = state.static_delta_norms
        sched_delta_norms_list = state.sched_delta_norms
        sched_delta_t = torch.tensor(sched_delta_norms_list, dtype=torch.float64) if sched_delta_norms_list else torch.zeros(D_sched, dtype=torch.float64)
        sched_deltas_np = np.array(sched_delta_norms_list)
        kappa = 1.0

        def _make_per_exp_objective(exp_idx: int):
            L_i = per_exp_L[exp_idx]
            D_exp = D_static + L_i * D_sched

            def _objective(x_S: torch.Tensor) -> torch.Tensor:
                S = int(x_S.shape[0])
                stat, traj = self._decode_trajectory_from_midpoint(
                    x_S, D_static, D_sched, L_i, sched_delta_t,
                )

                cand_i = base_rows_t[exp_idx].unsqueeze(0).unsqueeze(0).expand(S, L_i, n_dm_cols).clone().to(dtype=x_S.dtype)
                if D_static > 0:
                    cand_i = cand_i.scatter(-1, static_col_idxs.unsqueeze(0).unsqueeze(0).expand(S, L_i, -1),
                                            stat.unsqueeze(1).expand(S, L_i, D_static))
                if D_sched > 0:
                    cand_i = cand_i.scatter(-1, sched_col_idxs.unsqueeze(0).unsqueeze(0).expand(S, L_i, -1), traj)

                # Active experiment's layers as candidates; non-active are in KDE via virtual points
                full_S_NL = cand_i.unsqueeze(1)  # (S, 1, L_i, n_dm_cols)
                scores_neg = self._acquisition_fn(full_S_NL, kappa, None)

                # Smoothness penalties on the active trajectory
                if D_sched > 0 and L_i > 2:
                    obj_scale = scores_neg.detach().abs()
                    diffs = traj[:, 1:, :] - traj[:, :-1, :]

                    products = diffs[:, 1:, :] * diffs[:, :-1, :]
                    reversal_penalty = (-products).clamp(min=0.0).sum(dim=(1, 2))
                    scores_neg = scores_neg + obj_scale * self.smoothness_weight * reversal_penalty

                    t = torch.arange(L_i, dtype=x_S.dtype)
                    t_mean = t.mean()
                    t_var = ((t - t_mean) ** 2).sum()
                    y_mean = traj.mean(dim=1, keepdim=True)
                    ss_tot = ((traj - y_mean) ** 2).sum(dim=1)
                    has_variation = ss_tot > 1e-6
                    if bool(has_variation.any().item()):
                        cov = ((t[None, :, None] - t_mean) * (traj - y_mean)).sum(dim=1)
                        slope = cov / t_var.clamp(min=1e-12)
                        y_hat = y_mean + slope[:, None, :] * (t[None, :, None] - t_mean)
                        ss_res = ((traj - y_hat) ** 2).sum(dim=1)
                        r2 = torch.where(has_variation, 1.0 - ss_res / ss_tot.clamp(min=1e-6), torch.ones_like(ss_tot))
                        r2_penalty = (1.0 - r2).clamp(min=0.0).sum(dim=-1)
                        scores_neg = scores_neg + obj_scale * self.smoothness_weight * r2_penalty

                return scores_neg

            return _objective, D_exp

        max_rounds = min(self.max_rounds, max(1, self._engine.gradient_n_iters // max(n, 1)))
        total_iters = 0

        for round_idx in range(max_rounds):
            if console:
                import sys
                sys.stdout.write(f"  \033[2mTrajectory round {round_idx+1}\033[0m\n")
                sys.stdout.flush()
            improved_this_round = False
            for exp_idx in range(n):
                L_i = per_exp_L[exp_idx]
                mid_idx = L_i // 2

                # Non-active experiments → KDE "old" kernels
                self._pop()
                self._push_others(state, exp_idx, datamodule)

                objective, D_exp = _make_per_exp_objective(exp_idx)

                # Bounds: [static drift | midpoint drift | L-1 deltas ∈ [0,1]]
                bounds_i: list[tuple[float, float]] = []
                for si in range(D_static):
                    c = float(state.static_norms[exp_idx, si])
                    d = static_delta_norms_list[si]
                    bounds_i.append((max(0.0, c - d), min(1.0, c + d)))
                for si in range(D_sched):
                    c = float(state.schedule_norms[exp_idx][mid_idx, si])
                    d = sched_delta_norms_list[si]
                    bounds_i.append((max(0.0, c - d), min(1.0, c + d)))
                for _k in range(L_i - 1):
                    for _si in range(D_sched):
                        bounds_i.append((0.0, 1.0))

                # Multi-start: warm start from midpoint + random noise on deltas
                rng = self._engine.rng
                n_traj_starts = 3
                best_opt = None
                bar = ProgressBar(
                    f"Exp {exp_idx+1}/{n}", D=D_exp, V=D_exp,
                    max_starts=n_traj_starts,
                ) if console else None
                for _si in range(n_traj_starts):
                    x0_i = np.zeros(D_exp)
                    x0_i[:D_static] = state.static_norms[exp_idx]
                    x0_i[D_static:D_static + D_sched] = state.schedule_norms[exp_idx][mid_idx]
                    if L_i > 1 and D_sched > 0:
                        delta_start = D_static + D_sched
                        for di in range((L_i - 1) * D_sched):
                            x0_i[delta_start + di] = 0.5 + rng.uniform(-0.15, 0.15)
                    trial = self._engine.run_acquisition_gradient(
                        objective, bounds_i, x0=x0_i,
                        label=f"Exp {exp_idx+1}/{n}",
                        show_progress=False,
                        n_starts=1, raw_samples=0,
                    )
                    if best_opt is None or trial.score > best_opt.score:
                        best_opt = trial
                    if bar:
                        bar.step(obj=-best_opt.score)
                    if _si < n_traj_starts - 1 and bar:
                        bar.new_start()
                opt = best_opt  # type: ignore[assignment]
                if bar:
                    bar.finish()
                total_iters += len(opt.convergence_history)

                if opt.best_x is not None:
                    new_static, new_sched = self._decode_trajectory_from_midpoint_np(
                        opt.best_x, D_static, D_sched, L_i, sched_deltas_np,
                    )
                    if not np.allclose(new_static, state.static_norms[exp_idx], atol=1e-6) or \
                       not np.allclose(new_sched, state.schedule_norms[exp_idx], atol=1e-6):
                        improved_this_round = True
                    state.static_norms[exp_idx] = new_static.copy()
                    state.schedule_norms[exp_idx] = new_sched.copy()

                if self.step_callback is not None:
                    self.step_callback(round_idx, exp_idx, state)

            if not improved_this_round:
                break

        # Clean up virtual points
        self._pop()

        self.total_iters = total_iters
        self.convergence_history = []

    # ------------------------------------------------------------------
    # Decode to ExperimentSpec
    # ------------------------------------------------------------------

    def _decode_specs(self, state: TrajectoryState) -> list[ExperimentSpec]:
        self.last_trajectory_points = np.concatenate(state.schedule_norms, axis=0)
        exp_ids: list[int] = []
        for i in range(state.n):
            exp_ids.extend([i] * state.per_exp_L[i])
        self.last_trajectory_exp_ids = exp_ids

        specs_out: list[ExperimentSpec] = []
        for i in range(state.n):
            L_i = state.per_exp_L[i]
            base_params = dict(state.flat_specs[i].initial_params.to_dict())
            traj = state.schedule_norms[i]

            for si, (code, lo, hi) in enumerate(state.static_params):
                base_params[code] = float(state.static_norms[i, si] * (hi - lo) + lo)
            for si, (code, lo, hi) in enumerate(state.sched_params):
                base_params[code] = float(traj[0, si] * (hi - lo) + lo)

            base_params = self._sanitize(base_params)
            initial = ParameterProposal.from_dict(base_params, source_step=SourceStep.BASELINE)

            entries: list[tuple[int, ParameterProposal]] = []
            for k in range(1, L_i):
                sp: dict[str, Any] = {}
                for si, (code, lo, hi) in enumerate(state.sched_params):
                    sp[code] = float(traj[k, si] * (hi - lo) + lo)
                sp = self._sanitize(sp)
                entries.append((k, ParameterProposal.from_dict(sp, source_step=SourceStep.BASELINE)))

            trajectories: dict[str, ParameterTrajectory] = {}
            if entries:
                trajectories[state.primary_dim_code] = ParameterTrajectory(
                    dimension=state.primary_dim_code, entries=entries
                )

            specs_out.append(ExperimentSpec(initial_params=initial, trajectories=trajectories))

        self.last_trajectory_schedule_norms = [s.copy() for s in state.schedule_norms]
        self.last_trajectory_sched_params = list(state.sched_params)
        self.last_trajectory_per_exp_L = list(state.per_exp_L)

        return specs_out


# ------------------------------------------------------------------
# State initialization
# ------------------------------------------------------------------

def init_trajectory_state(
    flat_specs: list[ExperimentSpec],
    sched_params: list[tuple[str, float, float]],
    static_params: list[tuple[str, float, float]],
    per_exp_L: list[int],
    primary_dim_code: str,
    trust_regions: dict[str, float],
    datamodule: DataModule,
    bounds_fn: Callable[[str], tuple[float, float]],
) -> TrajectoryState | None:
    n = len(flat_specs)
    if datamodule is None:
        return None

    sched_delta_norms = [
        (trust_regions.get(code, (hi - lo) / 10.0) / (hi - lo) if hi - lo > 0 else 0.0)
        for code, lo, hi in sched_params
    ]
    static_delta_norms = [
        (trust_regions.get(code, (hi - lo) / 10.0) / (hi - lo) if hi - lo > 0 else 0.0)
        for code, lo, hi in static_params
    ]

    # Global solution → midpoint of each trajectory
    midpoint_warmstart = _warmstart_from_specs(flat_specs, sched_params, n)
    static_warmstart = _warmstart_from_specs(flat_specs, static_params, n)

    static_norms = static_warmstart.copy()
    schedule_norms = [
        np.tile(midpoint_warmstart[i], (per_exp_L[i], 1)) for i in range(n)
    ]

    n_dm_cols = len(datamodule.input_columns)
    sched_col_map = [
        (si, datamodule.input_columns.index(code))
        for si, (code, _, _) in enumerate(sched_params)
        if code in datamodule.input_columns
    ]
    static_col_map = [
        (si, datamodule.input_columns.index(code))
        for si, (code, _, _) in enumerate(static_params)
        if code in datamodule.input_columns
    ]

    sched_code_set = {code for code, _, _ in sched_params}
    static_code_set = {code for code, _, _ in static_params}
    exp_base_rows: list[np.ndarray] = []
    for i_exp in range(n):
        row = np.full(n_dm_cols, 0.5)
        static_dict = flat_specs[i_exp].initial_params.to_dict()
        for c_idx, col in enumerate(datamodule.input_columns):
            if col in static_dict and col not in sched_code_set and col not in static_code_set:
                val = static_dict[col]
                try:
                    lo_s, hi_s = bounds_fn(col)
                    span_s = hi_s - lo_s
                    row[c_idx] = (float(val) - lo_s) / span_s if span_s > 0 else 0.5
                except (ValueError, KeyError):
                    row[c_idx] = 0.5
        exp_base_rows.append(row)

    return TrajectoryState(
        n=n, flat_specs=flat_specs,
        sched_params=sched_params, static_params=static_params,
        per_exp_L=per_exp_L, primary_dim_code=primary_dim_code,
        sched_delta_norms=sched_delta_norms, static_delta_norms=static_delta_norms,
        static_norms=static_norms, schedule_norms=schedule_norms,
        n_dm_cols=n_dm_cols, exp_base_rows=exp_base_rows,
        sched_col_map=sched_col_map, static_col_map=static_col_map,
    )


def _warmstart_from_specs(
    flat_specs: list[ExperimentSpec],
    params: list[tuple[str, float, float]],
    n: int,
) -> np.ndarray:
    out = np.zeros((n, len(params)))
    for i, spec in enumerate(flat_specs):
        p_dict = spec.initial_params.to_dict()
        for si, (code, lo, hi) in enumerate(params):
            raw_val = float(p_dict.get(code, (lo + hi) / 2.0))
            span = hi - lo
            out[i, si] = (raw_val - lo) / span if span > 0 else 0.5
    return out
