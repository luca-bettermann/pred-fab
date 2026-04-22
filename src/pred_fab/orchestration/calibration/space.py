from typing import Any, Callable

import numpy as np

from ...core import ExperimentSpec, ParameterProposal, ParameterSchedule
from ...utils import SourceStep
from .engine import OptimizationEngine


# ======================================================================
# SolutionSpace — decision vector layout, bounds, decode/encode
# ======================================================================

class SolutionSpace:
    """Decision vector layout, bounds, and decode/encode for optimization.

    Handles both regular (one point per experiment) and scheduled
    (multiple points per experiment with delta-constrained offsets) cases.
    """

    def __init__(
        self,
        n_experiments: int,
        static_params: list[tuple[str, float, float]],   # (code, lo, hi) — raw
        sched_params: list[tuple[str, float, float]],     # (code, lo, hi) — raw
        per_exp_L: list[int],                              # layers per experiment
        trust_regions: dict[str, float],                   # delta per sched param (raw)
        int_set: set[int],                                 # integer param indices
        int_ranges_map: dict[int, int],                    # integer ranges
        schedule_smoothing: float = 0.05,
        riesz_p: float = 2.0,
        static_de_bounds: list[tuple[float, float]] | None = None,
        sched_de_bounds: list[tuple[float, float]] | None = None,
        sched_delta_norms: list[float] | None = None,
        step0_values: np.ndarray | None = None,           # (n_exp, D_sched) fixed step0 values (normalized)
    ):
        self._n_experiments = n_experiments
        self._static_params = static_params
        self._sched_params = sched_params
        self._per_exp_L = per_exp_L
        self._trust_regions = trust_regions
        self._int_set = int_set
        self._int_ranges_map = int_ranges_map
        self._schedule_smoothing = schedule_smoothing
        self._riesz_p = riesz_p
        self._step0_values = step0_values

        self._D_static = len(static_params)
        self._D_sched = len(sched_params)
        self._L_max = max(per_exp_L) if per_exp_L else 1
        self._is_scheduled = self._L_max > 1 and self._D_sched > 0
        self._N_total = sum(per_exp_L)
        self._D_point = self._D_static + self._D_sched

        self._step0_fixed = step0_values is not None
        self._exp_offsets: list[int] = []
        total = 0
        for i in range(n_experiments):
            self._exp_offsets.append(total)
            step0_vars = 0 if self._step0_fixed else self._D_sched
            total += self._D_static + step0_vars + max(per_exp_L[i] - 1, 0) * self._D_sched
        self._total_vars = total

        self._sched_bounds_list = list(sched_de_bounds) if sched_de_bounds is not None else [(0.0, 1.0)] * self._D_sched
        if sched_delta_norms is not None:
            self._sched_deltas_norm = list(sched_delta_norms)
        else:
            self._sched_deltas_norm = [
                (trust_regions.get(code, 0.0) / (hi - lo) if hi - lo > 0 else 0.0)
                for code, lo, hi in sched_params
            ]
        self._sched_delta_arr = np.array(self._sched_deltas_norm) if self._sched_deltas_norm else np.array([])

        self._static_bounds_list: list[tuple[float, float]] = []
        self._integrality_mask: list[bool] = []
        for d, (code, lo, hi) in enumerate(static_params):
            if d in int_set:
                self._static_bounds_list.append((0.0, float(int_ranges_map[d])))
                self._integrality_mask.append(True)
            elif static_de_bounds is not None:
                self._static_bounds_list.append(static_de_bounds[d])
                self._integrality_mask.append(False)
            else:
                self._static_bounds_list.append((0.0, 1.0))
                self._integrality_mask.append(False)

        self._bounds_list: list[tuple[float, float]] = []
        self._integrality_list: list[bool] | None = [] if any(self._integrality_mask) else None
        for i in range(n_experiments):
            self._bounds_list.extend(self._static_bounds_list)
            if self._integrality_list is not None:
                self._integrality_list.extend(self._integrality_mask)
            if step0_values is None:
                # Step0 is optimizable
                self._bounds_list.extend(self._sched_bounds_list)
                if self._integrality_list is not None:
                    self._integrality_list.extend([False] * self._D_sched)
            # Offsets for steps 1..L_i-1
            for _k in range(1, per_exp_L[i]):
                for d_s in range(self._D_sched):
                    self._bounds_list.append((-self._sched_deltas_norm[d_s], self._sched_deltas_norm[d_s]))
                    if self._integrality_list is not None:
                        self._integrality_list.append(False)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return self._bounds_list

    @property
    def total_vars(self) -> int:
        return self._total_vars

    @property
    def n_total_points(self) -> int:
        return self._N_total

    @property
    def integrality(self) -> list[bool] | None:
        return self._integrality_list

    def decode(self, x_flat: np.ndarray) -> np.ndarray:
        """Decode flat decision vector into (N_total, D_point) array."""
        pts = np.zeros((self._N_total, self._D_point))
        pt_idx = 0
        for i in range(self._n_experiments):
            off = self._exp_offsets[i]
            static_vals = x_flat[off:off + self._D_static]
            static_norm = static_vals.copy()
            for si in range(self._D_static):
                if si in self._int_set:
                    r = self._int_ranges_map[si]
                    static_norm[si] = static_vals[si] / r if r > 0 else 0.5

            if self._D_sched > 0:
                if self._step0_fixed and self._step0_values is not None:
                    # Step0 is fixed from Process — not in decision vector
                    step0 = self._step0_values[i].copy()
                    offset_base = off + self._D_static
                else:
                    # Step0 is optimizable — in decision vector
                    step0 = x_flat[off + self._D_static:off + self._D_static + self._D_sched]
                    offset_base = off + self._D_static + self._D_sched
                abs_speed = step0.copy()
            else:
                abs_speed = np.array([])
                offset_base = off + self._D_static

            L_i = self._per_exp_L[i]
            for k in range(L_i):
                if k > 0 and self._D_sched > 0:
                    off_k = offset_base + (k - 1) * self._D_sched
                    abs_speed = abs_speed + x_flat[off_k:off_k + self._D_sched]
                pts[pt_idx, :self._D_static] = static_norm
                if self._D_sched > 0:
                    pts[pt_idx, self._D_static:self._D_static + self._D_sched] = abs_speed
                pt_idx += 1
        return pts

    def smoothing_penalty(self, x_flat: np.ndarray, base_energy: float = 0.0) -> float:
        """Compute absolute schedule smoothing penalty.

        penalty = lam · Σ min(|Δv| / delta, 1.0) / n_experiments.
        Decoupled from objective magnitude. base_energy is ignored (kept for API compat).
        """
        lam = self._schedule_smoothing
        if lam <= 0 or self._D_sched == 0:
            return 0.0
        total = 0.0
        for i in range(self._n_experiments):
            L_i = self._per_exp_L[i]
            if L_i <= 1:
                continue
            off = self._exp_offsets[i]
            sv = np.zeros((L_i, self._D_sched))
            if self._step0_fixed and self._step0_values is not None:
                sv[0] = self._step0_values[i]
                offset_base = off + self._D_static
            else:
                sv[0] = x_flat[off + self._D_static:off + self._D_static + self._D_sched]
                offset_base = off + self._D_static + self._D_sched
            for kk in range(1, L_i):
                off_k = offset_base + (kk - 1) * self._D_sched
                sv[kk] = sv[kk - 1] + x_flat[off_k:off_k + self._D_sched]
            # Absolute cost per adjacent pair per dimension
            for kk in range(1, L_i):
                for d in range(self._D_sched):
                    if self._sched_delta_arr[d] <= 0:
                        continue
                    change = abs(sv[kk, d] - sv[kk - 1, d])
                    frac = min(change / self._sched_delta_arr[d], 1.0)
                    total += lam * frac
        return total / self._n_experiments

    def build_init_population(self, rng: np.random.RandomState, init_norm: np.ndarray) -> np.ndarray:
        """Build warm-started initial DE population from init_norm (n_experiments, n_numeric)."""
        init_flat = np.zeros(self._total_vars)
        for i in range(self._n_experiments):
            off = self._exp_offsets[i]
            for si in range(self._D_static):
                if si < init_norm.shape[1]:
                    init_flat[off + si] = init_norm[i, si]
            for si in range(self._D_sched):
                src = self._D_static + si
                init_flat[off + self._D_static + si] = init_norm[i, src] if src < init_norm.shape[1] else 0.5
        pop_total = max(15, 2) * self._total_vars
        init_pop = np.empty((pop_total, self._total_vars))
        bds = self._bounds_list
        for ii in range(pop_total):
            for v in range(self._total_vars):
                lo_b, hi_b = bds[v]
                init_pop[ii, v] = lo_b if lo_b == hi_b else rng.uniform(lo_b + 0.001, hi_b - 0.001)
        init_pop[0] = init_flat
        for j in range(1, min(pop_total, 6)):
            cand = init_flat + rng.normal(0, 0.05, size=self._total_vars)
            for v in range(self._total_vars):
                lo_b, hi_b = bds[v]
                cand[v] = lo_b if lo_b == hi_b else np.clip(cand[v], lo_b + 0.001, hi_b - 0.001)
            init_pop[j] = cand
        return init_pop

    def decode_to_specs(
        self, x_flat: np.ndarray, fixed_params: dict[str, Any],
        cat_codes: list[str], cat_assignments: list[tuple[Any, ...]],
        structural_values: list[dict[str, int]] | None, primary_dim_code: str,
        schema_sanitize: Callable[[dict[str, Any]], dict[str, Any]],
        integer_params: list[tuple[str, int, int]], source_step: str = "baseline_step",
    ) -> list[ExperimentSpec]:
        """Convert optimized vector back to ExperimentSpecs."""
        specs: list[ExperimentSpec] = []
        for i in range(self._n_experiments):
            off = self._exp_offsets[i]
            bp: dict[str, Any] = dict(fixed_params)
            if structural_values is not None:
                for sv_code, sv_val in structural_values[i].items():
                    bp[sv_code] = sv_val
            for si, (code, lo, hi) in enumerate(self._static_params):
                val = x_flat[off + si]
                bp[code] = int(np.round(val) + lo) if si in self._int_set else float(val * (hi - lo) + lo)
            step0 = x_flat[off + self._D_static:off + self._D_static + self._D_sched] if self._D_sched > 0 else np.array([])
            for si, (code, lo, hi) in enumerate(self._sched_params):
                bp[code] = float(step0[si] * (hi - lo) + lo)
            for d_cat, code in enumerate(cat_codes):
                bp[code] = cat_assignments[i][d_cat]
            for _, (code_i, lo_i, hi_i) in enumerate(integer_params):
                if code_i in bp and isinstance(bp[code_i], (int, float)):
                    bp[code_i] = int(np.clip(np.round(bp[code_i]), lo_i, hi_i))
            bp = schema_sanitize(bp)
            initial = ParameterProposal.from_dict(bp, source_step=source_step)
            entries: list[tuple[int, ParameterProposal]] = []
            if self._D_sched > 0:
                abs_val = step0.copy()
                for k in range(1, self._per_exp_L[i]):
                    off_k = off + self._D_static + self._D_sched + (k - 1) * self._D_sched
                    abs_val = abs_val + x_flat[off_k:off_k + self._D_sched]
                    sp: dict[str, Any] = {}
                    for si, (code, lo, hi) in enumerate(self._sched_params):
                        sp[code] = float(abs_val[si] * (hi - lo) + lo)
                    entries.append((k, ParameterProposal.from_dict(schema_sanitize(sp), source_step=source_step)))
            schedules: dict[str, ParameterSchedule] = {}
            if entries:
                schedules[primary_dim_code] = ParameterSchedule(dimension=primary_dim_code, entries=entries)
            specs.append(ExperimentSpec(initial_params=initial, schedules=schedules))
        return specs

    def decode_optimized_positions(self, x_flat: np.ndarray) -> np.ndarray:
        """Return (n_experiments, D_static) normalized static positions."""
        out = np.zeros((self._n_experiments, self._D_static))
        for i in range(self._n_experiments):
            off = self._exp_offsets[i]
            for si in range(self._D_static):
                val = x_flat[off + si]
                if si in self._int_set:
                    r = self._int_ranges_map[si]
                    out[i, si] = val / r if r > 0 else 0.5
                else:
                    out[i, si] = val
        return out
