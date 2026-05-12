"""Decision-vector layout, variable types, and decode for N-experiment optimisation.

Variable types define the contract between step methods and the optimizer.
SolutionSpace maps Variables to a flat decision vector, owns the single decode
pipeline (Sobol eval, LBFGS eval, spec construction), and builds ExperimentSpecs.

Tanh-slope trajectory decode:
    z(k) = z_mid + (k - L//2) * z_slope
    value(k) = 0.5 + 0.5 * tanh(z(k))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from ...core import DataModule, ExperimentSpec, ParameterProposal, ParameterTrajectory
from ...core.data_objects import DataObject
from .bounds import BoundsManager


# ---------------------------------------------------------------------------
# Variable types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Variable:
    """A parameter to be optimized. Wraps the schema DataObject — no info duplicated."""
    data_object: DataObject

    @property
    def code(self) -> str:
        return self.data_object.code


@dataclass(frozen=True)
class StaticVariable(Variable):
    """Single value per experiment (continuous or integer)."""
    is_integer: bool = False


@dataclass(frozen=True)
class TrajectoryVariable(Variable):
    """Midpoint + slope per experiment, decoded to L layers via tanh."""
    dimension_code: str


# ---------------------------------------------------------------------------
# Tanh-slope trajectory decode
# ---------------------------------------------------------------------------

def _decode_slope_trajectory(
    midpoint_norm: torch.Tensor,
    z_slope: torch.Tensor,
    L: int,
) -> torch.Tensor:
    """Decode midpoint + slope into per-layer normalised values in (0, 1).

    tanh'(0) = 1, so z_slope ~ real-space normalised step near center.
    """
    mid_idx = L // 2
    x_centered = (2.0 * midpoint_norm - 1.0).clamp(-1 + 1e-4, 1 - 1e-4)
    z_mid = torch.atanh(x_centered)
    offsets = torch.arange(L, dtype=z_mid.dtype, device=z_mid.device) - mid_idx
    z_all = z_mid.unsqueeze(-2) + offsets.reshape(*([1] * (z_mid.ndim - 1)), L, 1) * z_slope.unsqueeze(-2)
    return 0.5 + 0.5 * torch.tanh(z_all)


# ---------------------------------------------------------------------------
# SolutionSpace
# ---------------------------------------------------------------------------

class SolutionSpace:
    """Maps Variable objects to a flat decision vector for the optimizer.

    Owns the complete decode pipeline: flat vector -> evaluable tensor + weights -> ExperimentSpec.
    The decode method is defined once and used by Sobol evaluation, LBFGS optimization, and
    final spec construction.
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
        derive_L_fn: Callable[[dict[str, Any]], int] | None = None,
        source_step: str | None = None,
    ):
        self._n_experiments = n_experiments
        self._bounds_manager = bounds_manager
        self._datamodule = datamodule
        self._fixed_params = fixed_params
        self._cat_codes = cat_codes
        self._cat_assignments = cat_assignments
        self._schema_sanitize = schema_sanitize
        self._derive_L_fn = derive_L_fn
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

        # Real-space bounds from BoundsManager (for decode)
        self._static_bounds: list[tuple[float, float]] = []
        self._static_int_ranges: list[int | None] = []
        for sv in self._statics:
            lo, hi = bounds_manager._get_hierarchical_bounds_for_code(sv.code)
            self._static_bounds.append((lo, hi))
            self._static_int_ranges.append(int(hi - lo) if sv.is_integer else None)

        self._traj_bounds: list[tuple[float, float]] = []
        self._slope_maxes: list[float] = []
        for tv in self._trajectories:
            lo, hi = bounds_manager._get_hierarchical_bounds_for_code(tv.code)
            self._traj_bounds.append((lo, hi))
            span = hi - lo
            trust = bounds_manager.trust_regions.get(tv.code, span / 10.0)
            self._slope_maxes.append(trust / span if span > 0 else 0.1)

        # Optimizer bounds (normalised)
        self._bounds_list: list[tuple[float, float]] = []
        for _ in range(n_experiments):
            for i, sv in enumerate(self._statics):
                r = self._static_int_ranges[i]
                if r is not None:
                    self._bounds_list.append((0.0, float(r)))
                else:
                    self._bounds_list.append((0.0, 1.0))
            for _ in self._trajectories:
                self._bounds_list.append((0.0, 1.0))
            for ti in range(self._D_traj):
                sm = self._slope_maxes[ti]
                self._bounds_list.append((-sm, sm))

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
        fill = torch.full(
            (self._n_experiments, self._n_dm_cols), 0.5, dtype=torch.float64,
        )
        opt_codes = {sv.code for sv in self._statics} | {tv.code for tv in self._trajectories}

        for i in range(self._n_experiments):
            merged: dict[str, Any] = dict(self._fixed_params)
            for d_cat, code in enumerate(self._cat_codes):
                merged[code] = self._cat_assignments[i][d_cat]

            for c_idx, col in enumerate(dm_cols):
                if col in merged and col not in opt_codes:
                    try:
                        lo, hi = self._bounds_manager._get_hierarchical_bounds_for_code(col)
                        span = hi - lo
                        fill[i, c_idx] = (float(merged[col]) - lo) / span if span > 0 else 0.5
                    except (ValueError, KeyError):
                        pass
        return fill

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return self._bounds_list

    @property
    def total_vars(self) -> int:
        return self._n_experiments * self._D_per_exp

    def _derive_L_per_exp(self, static_vals: torch.Tensor) -> list[int]:
        """Compute layers per experiment from current static variable values."""
        if self._derive_L_fn is None or self._D_traj == 0:
            return [1] * self._n_experiments
        Ls = []
        for i in range(self._n_experiments):
            p: dict[str, Any] = {}
            for si, sv in enumerate(self._statics):
                lo, hi = self._static_bounds[si]
                r = self._static_int_ranges[si]
                if r is not None:
                    p[sv.code] = int(static_vals[i, si].detach().round().item()) + int(lo)
                else:
                    p[sv.code] = float(static_vals[i, si].detach().item()) * (hi - lo) + lo
            Ls.append(max(1, self._derive_L_fn(p)))
        return Ls

    def decode(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decision vector -> (S, total_points, D_dm) tensor + (S, total_points) weights.

        Single definition — called by Sobol eval, LBFGS eval, and spec decode.
        """
        S = int(x_flat.shape[0])
        x = x_flat.reshape(S, self._n_experiments, self._D_per_exp)

        static_vals = x[:, :, : self._D_static]
        midpoints = x[:, :, self._D_static : self._D_static + self._D_traj]
        slopes = x[:, :, self._D_static + self._D_traj :]

        # STE rounding for integers
        for si, r in enumerate(self._static_int_ranges):
            if r is not None:
                raw = static_vals[:, :, si]
                static_vals = static_vals.clone()
                static_vals[:, :, si] = raw + (raw.round() - raw).detach()

        # Derive L per experiment (uses first candidate in batch for consistency)
        L_per_exp = self._derive_L_per_exp(static_vals[0])

        # Build base rows from prior fill
        base = self._prior_fill.unsqueeze(0).expand(S, -1, -1).clone().to(dtype=x_flat.dtype)

        # Scatter static values
        for si, dm_idx in enumerate(self._static_dm_idx):
            if dm_idx is None:
                continue
            r = self._static_int_ranges[si]
            if r is not None:
                lo = self._static_bounds[si][0]
                hi = self._static_bounds[si][1]
                span = hi - lo
                base[:, :, dm_idx] = (static_vals[:, :, si] + lo - lo) / span if span > 0 else 0.5
            else:
                base[:, :, dm_idx] = static_vals[:, :, si]

        # Expand to per-layer rows with trajectory decode
        # Unified path: _decode_slope_trajectory works for L=1 too (returns single layer).
        # When D_traj=0, midpoints/slopes are empty and the scatter loop is a no-op.
        layer_rows: list[torch.Tensor] = []
        weights: list[float] = []

        for i in range(self._n_experiments):
            L_i = L_per_exp[i]
            expanded = base[:, i, :].unsqueeze(1).expand(S, L_i, self._n_dm_cols).clone()

            if self._D_traj > 0:
                traj_layers = _decode_slope_trajectory(
                    midpoints[:, i, :], slopes[:, i, :], L_i,
                )  # (S, L_i, D_traj)
                for ti, dm_idx in enumerate(self._traj_dm_idx):
                    if dm_idx is not None:
                        expanded[:, :, dm_idx] = traj_layers[:, :, ti]

            layer_rows.append(expanded)
            weights.extend([1.0 / L_i] * L_i)

        points = torch.cat(layer_rows, dim=1)  # (S, total_points, D_dm)
        w = torch.tensor(weights, dtype=x_flat.dtype)
        w = w.unsqueeze(0).expand(S, -1)  # (S, total_points)
        return points, w

    def decode_to_specs(self, best_x: np.ndarray) -> list[ExperimentSpec]:
        """Convert optimized vector to ExperimentSpec instances."""
        x_t = torch.tensor(best_x, dtype=torch.float64)
        x_per_exp = x_t.reshape(self._n_experiments, self._D_per_exp)
        L_per_exp = self._derive_L_per_exp(x_per_exp[:, : self._D_static])

        specs: list[ExperimentSpec] = []
        for i in range(self._n_experiments):
            off = i * self._D_per_exp
            L_i = L_per_exp[i]

            bp: dict[str, Any] = dict(self._fixed_params)
            for d_cat, code in enumerate(self._cat_codes):
                bp[code] = self._cat_assignments[i][d_cat]

            # Static params
            for si, sv in enumerate(self._statics):
                lo, hi = self._static_bounds[si]
                raw = best_x[off + si]
                if sv.is_integer:
                    bp[sv.code] = int(np.clip(np.round(raw) + lo, lo, hi))
                else:
                    bp[sv.code] = float(raw * (hi - lo) + lo)

            # Trajectory params — always decode via tanh slope (works for L=1 too)
            traj_vals: np.ndarray | None = None
            if self._D_traj > 0:
                mid_t = torch.tensor(
                    best_x[off + self._D_static : off + self._D_static + self._D_traj],
                    dtype=torch.float64,
                ).unsqueeze(0)
                slp_t = torch.tensor(
                    best_x[off + self._D_static + self._D_traj : off + self._D_per_exp],
                    dtype=torch.float64,
                ).unsqueeze(0)
                traj_vals = _decode_slope_trajectory(mid_t, slp_t, L_i)[0].cpu().numpy()
                for ti, tv in enumerate(self._trajectories):
                    lo, hi = self._traj_bounds[ti]
                    bp[tv.code] = float(traj_vals[0, ti] * (hi - lo) + lo)

            bp = self._schema_sanitize(bp)
            initial = ParameterProposal.from_dict(bp, source_step=self._source_step)

            # Trajectory entries (layers 1..L-1 — empty when L=1)
            entries: list[tuple[int, ParameterProposal]] = []
            if traj_vals is not None:
                for k in range(1, L_i):
                    sp: dict[str, Any] = {}
                    for ti, tv in enumerate(self._trajectories):
                        lo, hi = self._traj_bounds[ti]
                        sp[tv.code] = float(traj_vals[k, ti] * (hi - lo) + lo)
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
        self, best_x: np.ndarray,
    ) -> tuple[list[np.ndarray], list[tuple[str, float, float]], list[int]]:
        """Extract trajectory norms, param info, and L per experiment for plotting."""
        x_per_exp = torch.tensor(best_x, dtype=torch.float64).reshape(
            self._n_experiments, self._D_per_exp,
        )
        L_per_exp = self._derive_L_per_exp(x_per_exp[:, : self._D_static])

        traj_norms: list[np.ndarray] = []
        for i in range(self._n_experiments):
            off = i * self._D_per_exp
            L_i = L_per_exp[i]
            mid_t = torch.tensor(
                best_x[off + self._D_static : off + self._D_static + self._D_traj],
                dtype=torch.float64,
            ).unsqueeze(0)
            slp_t = torch.tensor(
                best_x[off + self._D_static + self._D_traj : off + self._D_per_exp],
                dtype=torch.float64,
            ).unsqueeze(0)
            traj_norms.append(
                _decode_slope_trajectory(mid_t, slp_t, L_i)[0].cpu().numpy()
            )

        traj_params = [(tv.code, *self._traj_bounds[ti]) for ti, tv in enumerate(self._trajectories)]
        return traj_norms, traj_params, L_per_exp
