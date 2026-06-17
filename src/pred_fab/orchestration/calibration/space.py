"""Decision-vector layout, variable types, and decode for N-experiment optimisation.

All bound enforcement uses ``sigmoid(K · u)`` with a single sharpness
constant K.  Steeper sigmoid → the full bounded range is reachable with
modest u-values, and the optimizer has strong gradients throughout.

``u`` is the *unconstrained* optimiser variable (logit space) — not the [0,1]
decision vector (``sigmoid(K·u)``) nor the z-score model input. Full frame chain
(u → [0,1] → z-score / physical): see ``ORCHESTRATION_CONTEXT.md``.
(The code historically called this unconstrained frame ``z``; the ``z_bounds``
property and ``Z_RANGE`` constant keep that name.)

Variable types own their decode:
  - StaticVariable: sigmoid(Ku) → normalised value
  - TrajectoryVariable: sigmoid(K · (u_mid + offset · slope)) → per-layer values
    where slope = sigmoid(Ku) · 2·sm - sm  (bounded by its own sigmoid)

SolutionSpace maps Variables to a flat u-space decision vector, provides
z-bounds for Sobol sampling, and builds ExperimentSpecs from optimised u-vectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from ...core import DataModule, ExperimentSpec, ParameterProposal, ParameterTrajectory
from ...core.frames import param_value_to_fill
from ...core.data_objects import DataObject
from .bounds import BoundsManager


SIGMOID_K = 3.0
Z_RANGE = 4.6 / SIGMOID_K


def _sigmoid_k(u: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(SIGMOID_K * u)

# ---------------------------------------------------------------------------
# Variable types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Variable:
    """A parameter to be optimized. Wraps the schema DataObject."""
    data_object: DataObject
    lo: float = 0.0
    hi: float = 1.0

    @property
    def code(self) -> str:
        return self.data_object.code

    @property
    def span(self) -> float:
        return self.hi - self.lo

    @property
    def n_dims(self) -> int:
        return 1

    @property
    def z_bounds(self) -> list[tuple[float, float]]:
        return [(-Z_RANGE, Z_RANGE)]

    def decode(self, u: torch.Tensor) -> torch.Tensor:
        return _sigmoid_k(u)

    def to_real(self, norm: float) -> float:
        return float(norm * self.span + self.lo)


@dataclass(frozen=True)
class StaticVariable(Variable):
    """Single value per experiment (continuous or integer)."""
    is_integer: bool = False

    @property
    def int_range(self) -> int | None:
        return int(self.hi - self.lo) if self.is_integer else None

    @property
    def z_bounds(self) -> list[tuple[float, float]]:
        if self.is_integer:
            return [(0.0, float(self.hi - self.lo))]
        return [(-Z_RANGE, Z_RANGE)]

    def decode(self, u: torch.Tensor) -> torch.Tensor:
        if self.is_integer:
            # STE round in the [0,range] optimiser frame, then ÷range so the
            # norm frame is uniformly [0,1] (matches the continuous sigmoid).
            # Clamp to [0, range]: LBFGS is unconstrained (engine.py uses
            # torch.optim.LBFGS with no bounds), and unlike the sigmoid path
            # the integer linear decode is not self-bounding, so without this
            # the objective is evaluated at integers outside [lo, hi].
            rounded = u + (u.round() - u).detach()
            rounded = rounded.clamp(0.0, self.span)
            return rounded / self.span if self.span > 0 else torch.zeros_like(u)
        return _sigmoid_k(u)

    def to_real(self, norm: float) -> float | int:
        if self.is_integer:
            return int(np.clip(np.round(norm * self.span) + self.lo, self.lo, self.hi))
        return float(norm * self.span + self.lo)


@dataclass(frozen=True)
class TrajectoryVariable(Variable):
    """Midpoint + slope per experiment, decoded to L layers via sigmoid.

    The slope decision variable is unbounded; sigmoid maps it to
    [-slope_max, slope_max] differentiably (no clamping needed).
    """
    dimension_code: str = ""
    slope_max: float = 0.8

    @property
    def n_dims(self) -> int:
        return 2

    @property
    def z_bounds(self) -> list[tuple[float, float]]:
        return [(-Z_RANGE, Z_RANGE), (-Z_RANGE, Z_RANGE)]

    def decode_trajectory(
        self,
        u_mid: torch.Tensor,
        u_slope_raw: torch.Tensor,
        L: int,
    ) -> torch.Tensor:
        """Decode midpoint + slope into per-layer normalised values.

        slope = sigmoid(K · u_slope_raw) · 2·slope_max - slope_max
        value(k) = sigmoid(K · (u_mid + offset · slope))
        """
        slope = _sigmoid_k(u_slope_raw) * 2.0 * self.slope_max - self.slope_max
        mid_idx = L // 2
        offsets = torch.arange(L, dtype=u_mid.dtype, device=u_mid.device) - mid_idx
        u_all = u_mid.unsqueeze(-2) + offsets.reshape(
            *([1] * (u_mid.ndim - 1)), L, 1,
        ) * slope.unsqueeze(-2)
        return _sigmoid_k(u_all)

    def to_real(self, norm: float) -> float:
        return float(norm * self.span + self.lo)


# ---------------------------------------------------------------------------
# SolutionSpace
# ---------------------------------------------------------------------------

class SolutionSpace:
    """Maps Variable objects to a flat u-space decision vector for the optimizer.

    The engine operates in u-space (no sigmoid). SolutionSpace provides z-bounds
    for Sobol sampling and applies per-variable decode (sigmoid) in its decode()
    method — called by the objective function during optimisation.
    """

    def __init__(
        self,
        variables: list[Variable],
        n_experiments: int,
        bounds_manager: BoundsManager,
        datamodule: DataModule,
        fixed_params: dict[str, Any],
        cat_codes: list[str],
        cat_assignments: list[tuple[Any, ...]],
        schema_sanitize: Callable[[dict[str, Any]], dict[str, Any]],
        dimension_derivations: dict[str, Callable[[dict[str, Any]], int]] | None = None,
        source_step: str | None = None,
    ):
        self._n_experiments = n_experiments
        self._bounds_manager = bounds_manager
        self._datamodule = datamodule
        self._fixed_params = fixed_params
        self._cat_codes = cat_codes
        self._cat_assignments = cat_assignments
        self._schema_sanitize = schema_sanitize
        self._dimension_derivations = dimension_derivations or {}
        self._source_step = source_step

        # Separate variables by type
        self._statics: list[StaticVariable] = []
        self._trajectories: list[TrajectoryVariable] = []
        for v in variables:
            if isinstance(v, TrajectoryVariable):
                self._trajectories.append(v)
            elif isinstance(v, StaticVariable):
                self._statics.append(v)

        self._D_static = len(self._statics)
        self._D_traj = len(self._trajectories)

        # Per-experiment layout: [static_0..D_s-1 | mid_0..D_t-1 | slope_0..D_t-1]
        self._D_per_exp = self._D_static + 2 * self._D_traj
        self._D_param = self._D_static + self._D_traj

        # z-bounds for the optimizer (Sobol sampling range)
        self._bounds_list: list[tuple[float, float]] = []
        for _ in range(n_experiments):
            for sv in self._statics:
                self._bounds_list.extend(sv.z_bounds)
            for tv in self._trajectories:
                self._bounds_list.extend(tv.z_bounds)

        # Column mapping: variable code -> datamodule column index
        dm_cols = datamodule.input_columns
        self._static_dm_idx: list[int | None] = [
            dm_cols.index(sv.code) if sv.code in dm_cols else None
            for sv in self._statics
        ]
        self._traj_dm_idx: list[int | None] = [
            dm_cols.index(tv.code) if tv.code in dm_cols else None
            for tv in self._trajectories
        ]
        self._n_dm_cols = len(dm_cols)

        # Prior fill: frozen columns template (N, D_dm)
        self._prior_fill = self._build_prior_fill()

        # Primary dimension code for trajectory ExperimentSpec construction
        dim_codes = sorted({tv.dimension_code for tv in self._trajectories})
        self._primary_dim_code = dim_codes[0] if dim_codes else ""

    def _build_prior_fill(self) -> torch.Tensor:
        """Build template tensor for frozen columns (fixed params, categoricals)."""
        dm_cols = self._datamodule.input_columns
        # Default 0.5 in [0,1] bounds space = midpoint of parameter range.
        # _decode_frames converts to raw / z-score after decode().
        fill = torch.full(
            (self._n_experiments, self._n_dm_cols), 0.5, dtype=torch.float64,
        )
        opt_codes = {sv.code for sv in self._statics} | {tv.code for tv in self._trajectories}

        for i in range(self._n_experiments):
            merged: dict[str, Any] = dict(self._fixed_params)
            for d_cat, code in enumerate(self._cat_codes):
                merged[code] = self._cat_assignments[i][d_cat]

            for c_idx, col in enumerate(dm_cols):
                if col not in merged or col in opt_codes:
                    continue
                # Frozen column → prior-fill frame. Categoricals carry no normaliser
                # stats (decode passes them through), so they hold the raw cat-index;
                # numeric columns are [0,1] by bounds. param_value_to_fill owns that
                # rule (inverse of _raw_row_to_params). Unknown label / missing bounds
                # → leave the 0.5 midpoint default.
                cats = self._datamodule.categorical_mappings.get(col)
                try:
                    bounds = None if cats is not None else self._bounds_manager.get_hierarchical_bounds_for_code(col)
                    fill[i, c_idx] = param_value_to_fill(merged[col], categories=cats, bounds=bounds)
                except (ValueError, KeyError):
                    pass
        return fill

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return self._bounds_list

    @property
    def total_vars(self) -> int:
        return self._n_experiments * self._D_per_exp

    def _derive_L_per_exp(self, static_u: torch.Tensor) -> list[int]:
        """Layers per experiment, from the primary trajectory dimension's derivation."""
        deriv = self._dimension_derivations.get(self._primary_dim_code)
        if self._D_traj == 0 or deriv is None:
            return [1] * self._n_experiments
        Ls = []
        for i in range(self._n_experiments):
            p: dict[str, Any] = {}
            for si, sv in enumerate(self._statics):
                u_val = static_u[i, si].detach()
                norm = float(sv.decode(u_val).item())
                p[sv.code] = sv.to_real(norm)
            Ls.append(max(1, int(deriv(p))))
        return Ls

    def decode(self, u_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """u-space decision vector -> (raw_points, zscore_points, weights).

        Per-variable sigmoid/STE decode lands in [0,1] norm, then _decode_frames
        produces the physical-frame (raw, for perf) and z-score (for evidence)
        tensors. Single definition — called by the acquisition objective.
        """
        S = int(u_flat.shape[0])
        u = u_flat.reshape(S, self._n_experiments, self._D_per_exp)

        static_u = u[:, :, : self._D_static]
        mid_u = u[:, :, self._D_static : self._D_static + self._D_traj]
        slope_u = u[:, :, self._D_static + self._D_traj :]

        # Decode static variables via their own decode method
        static_norm = torch.empty_like(static_u)
        for si, sv in enumerate(self._statics):
            static_norm[:, :, si] = sv.decode(static_u[:, :, si])

        # Derive L per experiment (uses first candidate in batch for consistency)
        L_per_exp = self._derive_L_per_exp(static_u[0])

        # Build base rows from prior fill
        base = self._prior_fill.unsqueeze(0).expand(S, -1, -1).clone().to(dtype=u_flat.dtype)

        # Scatter static values (already decoded to normalised)
        for si, dm_idx in enumerate(self._static_dm_idx):
            if dm_idx is None:
                continue
            base[:, :, dm_idx] = static_norm[:, :, si]

        # Expand to per-layer rows with trajectory decode
        layer_rows: list[torch.Tensor] = []
        weights: list[float] = []

        for i in range(self._n_experiments):
            L_i = L_per_exp[i]
            expanded = base[:, i, :].unsqueeze(1).expand(S, L_i, self._n_dm_cols).clone()

            if self._D_traj > 0:
                traj_layers = torch.empty(S, L_i, self._D_traj, dtype=u_flat.dtype)
                for ti, tv in enumerate(self._trajectories):
                    traj_layers[:, :, ti] = tv.decode_trajectory(
                        mid_u[:, i, ti : ti + 1], slope_u[:, i, ti : ti + 1], L_i,
                    ).squeeze(-1)
                for ti, dm_idx in enumerate(self._traj_dm_idx):
                    if dm_idx is not None:
                        expanded[:, :, dm_idx] = traj_layers[:, :, ti]

            layer_rows.append(expanded)
            weights.extend([1.0 / L_i] * L_i)

        norm_points = torch.cat(layer_rows, dim=1)  # (S, total_points, D_dm) in [0,1]
        raw, zscore = self._decode_frames(norm_points)
        w = torch.tensor(weights, dtype=u_flat.dtype)
        w = w.unsqueeze(0).expand(S, -1)  # (S, total_points)
        return raw, zscore, w

    def _decode_frames(self, norm_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map [0,1] norm columns to (physical-frame, z-score) tensors.

        Bounded columns: [0,1] → raw via bounds → z-score via normaliser.
        Unbounded / degenerate columns (context, iterators): training mean →
        z-score 0.0. No-stats columns pass through unchanged in both frames.
        Gradient flows through the normaliser's forward().
        """
        dm = self._datamodule
        raw = norm_points.clone()
        zscore = norm_points.clone()
        for c_idx, col in enumerate(dm.input_columns):
            stats = dm._parameter_stats.get(col)
            if stats is None:
                continue
            try:
                lo, hi = self._bounds_manager.get_hierarchical_bounds_for_code(col)
                span = hi - lo
                if span <= 0:
                    raise ValueError("degenerate bounds")
                raw_col = norm_points[..., c_idx] * span + lo
                raw[..., c_idx] = raw_col
                zscore[..., c_idx] = stats.forward(raw_col)
            except (ValueError, KeyError):
                # No/degenerate bounds (context, iterators) → training mean (z-score 0)
                zscore[..., c_idx] = 0.0
                raw[..., c_idx] = stats.reverse(torch.zeros_like(norm_points[..., c_idx]))
        return raw, zscore

    def decode_to_specs(self, best_u: np.ndarray) -> list[ExperimentSpec]:
        """Convert optimised u-vector to ExperimentSpec instances."""
        u_t = torch.tensor(best_u, dtype=torch.float64)
        u_per_exp = u_t.reshape(self._n_experiments, self._D_per_exp)

        # Decode static u-values for L derivation
        static_u = u_per_exp[:, : self._D_static]
        L_per_exp = self._derive_L_per_exp(static_u)

        specs: list[ExperimentSpec] = []
        for i in range(self._n_experiments):
            off = i * self._D_per_exp
            L_i = L_per_exp[i]

            bp: dict[str, Any] = dict(self._fixed_params)
            for d_cat, code in enumerate(self._cat_codes):
                bp[code] = self._cat_assignments[i][d_cat]

            # Static params: decode u → normalised → real
            for si, sv in enumerate(self._statics):
                u_val = best_u[off + si]
                norm = float(sv.decode(torch.tensor(u_val, dtype=torch.float64)).item())
                bp[sv.code] = sv.to_real(norm)

            # Trajectory params: decode via sigmoid(u_mid + offset * u_slope)
            traj_vals: np.ndarray | None = None
            if self._D_traj > 0:
                mid_t = torch.tensor(
                    best_u[off + self._D_static : off + self._D_static + self._D_traj],
                    dtype=torch.float64,
                ).unsqueeze(0)
                slp_t = torch.tensor(
                    best_u[off + self._D_static + self._D_traj : off + self._D_per_exp],
                    dtype=torch.float64,
                ).unsqueeze(0)
                traj_norm = torch.empty(1, L_i, self._D_traj, dtype=torch.float64)
                for ti, tv in enumerate(self._trajectories):
                    traj_norm[0, :, ti] = tv.decode_trajectory(
                        mid_t[:, ti : ti + 1], slp_t[:, ti : ti + 1], L_i,
                    ).squeeze(-1)
                traj_vals = traj_norm[0].cpu().numpy()
                for ti, tv in enumerate(self._trajectories):
                    bp[tv.code] = tv.to_real(float(traj_vals[0, ti]))

            bp = self._schema_sanitize(bp)
            initial = ParameterProposal.from_dict(bp, source_step=self._source_step)

            # Trajectory entries (layers 1..L-1)
            entries: list[tuple[int, ParameterProposal]] = []
            if traj_vals is not None:
                for k in range(1, L_i):
                    sp: dict[str, Any] = {}
                    for ti, tv in enumerate(self._trajectories):
                        sp[tv.code] = tv.to_real(float(traj_vals[k, ti]))
                    sp = self._schema_sanitize(sp)
                    entries.append((k, ParameterProposal.from_dict(sp, source_step=self._source_step)))

            trajectories: dict[str, ParameterTrajectory] = {}
            if entries and self._primary_dim_code:
                trajectories[self._primary_dim_code] = ParameterTrajectory(
                    dimension=self._primary_dim_code, entries=entries,
                )

            specs.append(ExperimentSpec(initial_params=initial, trajectories=trajectories))

        return specs

    def get_trajectory_plot_data(
        self, best_u: np.ndarray,
    ) -> tuple[list[np.ndarray], list[tuple[str, float, float]], list[int]]:
        """Extract trajectory norms, param info, and L per experiment for plotting."""
        u_per_exp = torch.tensor(best_u, dtype=torch.float64).reshape(
            self._n_experiments, self._D_per_exp,
        )
        L_per_exp = self._derive_L_per_exp(u_per_exp[:, : self._D_static])

        traj_norms: list[np.ndarray] = []
        for i in range(self._n_experiments):
            off = i * self._D_per_exp
            L_i = L_per_exp[i]
            mid_t = torch.tensor(
                best_u[off + self._D_static : off + self._D_static + self._D_traj],
                dtype=torch.float64,
            ).unsqueeze(0)
            slp_t = torch.tensor(
                best_u[off + self._D_static + self._D_traj : off + self._D_per_exp],
                dtype=torch.float64,
            ).unsqueeze(0)
            exp_norms = torch.empty(L_i, self._D_traj, dtype=torch.float64)
            for ti, tv in enumerate(self._trajectories):
                exp_norms[:, ti] = tv.decode_trajectory(
                    mid_t[:, ti : ti + 1], slp_t[:, ti : ti + 1], L_i,
                ).squeeze(0).squeeze(-1)
            traj_norms.append(exp_norms.cpu().numpy())

        traj_params = [(tv.code, tv.lo, tv.hi) for tv in self._trajectories]
        return traj_norms, traj_params, L_per_exp
