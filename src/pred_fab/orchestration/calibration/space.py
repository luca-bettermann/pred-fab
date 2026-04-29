from typing import Any, Callable

import numpy as np

from ...core import ExperimentSpec, ParameterProposal


# ======================================================================
# SolutionSpace — decision vector layout, bounds, decode/encode
# ======================================================================

class SolutionSpace:
    """Decision vector layout, bounds, and decode/encode for the static
    DE acquisition path (one point per experiment).

    Schedule trajectories are owned by the gradient backend
    (``_optimise_schedule_for_experiment`` with absolute-step + sigmoid
    reparam) — this class is DE static-only.
    """

    def __init__(
        self,
        n_experiments: int,
        static_params: list[tuple[str, float, float]],   # (code, lo, hi) — raw
        trust_regions: dict[str, float],                   # delta per param (raw); kept for API compat
        int_set: set[int],                                 # integer param indices
        int_ranges_map: dict[int, int],                    # integer ranges
        static_de_bounds: list[tuple[float, float]] | None = None,
    ):
        del trust_regions  # static-only space — schedule trust regions handled by gradient path
        self._n_experiments = n_experiments
        self._static_params = static_params
        self._int_set = int_set
        self._int_ranges_map = int_ranges_map

        self._D_static = len(static_params)
        self._N_total = n_experiments
        self._D_point = self._D_static
        self._total_vars = n_experiments * self._D_static

        self._static_bounds_list: list[tuple[float, float]] = []
        self._integrality_mask: list[bool] = []
        for d, (_code, _lo, _hi) in enumerate(static_params):
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
        for _i in range(n_experiments):
            self._bounds_list.extend(self._static_bounds_list)
            if self._integrality_list is not None:
                self._integrality_list.extend(self._integrality_mask)

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
        """Decode flat decision vector into ``(n_experiments, D_static)`` array.

        Integer-valued static params (indices in ``int_set``) are normalised
        back to ``[0, 1]`` via the per-index range; continuous params pass
        through unchanged.
        """
        units = x_flat.reshape(self._n_experiments, self._D_static)
        pts = units.copy()
        for si in range(self._D_static):
            if si in self._int_set:
                r = self._int_ranges_map[si]
                if r > 0:
                    pts[:, si] = units[:, si] / r
                else:
                    pts[:, si] = 0.5
        return pts

    def build_init_population(self, rng: np.random.RandomState, init_norm: np.ndarray) -> np.ndarray:
        """Build warm-started initial DE population from ``init_norm`` ``(n_experiments, D_static)``."""
        init_flat = np.zeros(self._total_vars)
        for i in range(self._n_experiments):
            off = i * self._D_static
            for si in range(self._D_static):
                if si < init_norm.shape[1]:
                    init_flat[off + si] = init_norm[i, si]
        pop_total = 15 * 2 * self._total_vars
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
        """Convert optimized vector back to ``ExperimentSpec`` instances (one per experiment)."""
        del primary_dim_code  # static-only; schedules added by the gradient phase
        specs: list[ExperimentSpec] = []
        for i in range(self._n_experiments):
            off = i * self._D_static
            bp: dict[str, Any] = dict(fixed_params)
            if structural_values is not None:
                for sv_code, sv_val in structural_values[i].items():
                    bp[sv_code] = sv_val
            for si, (code, lo, hi) in enumerate(self._static_params):
                val = x_flat[off + si]
                bp[code] = int(np.round(val) + lo) if si in self._int_set else float(val * (hi - lo) + lo)
            for d_cat, code in enumerate(cat_codes):
                bp[code] = cat_assignments[i][d_cat]
            for _, (code_i, lo_i, hi_i) in enumerate(integer_params):
                if code_i in bp and isinstance(bp[code_i], (int, float)):
                    bp[code_i] = int(np.clip(np.round(bp[code_i]), lo_i, hi_i))
            bp = schema_sanitize(bp)
            initial = ParameterProposal.from_dict(bp, source_step=source_step)
            specs.append(ExperimentSpec(initial_params=initial, schedules={}))
        return specs

    def decode_optimized_positions(self, x_flat: np.ndarray) -> np.ndarray:
        """Return ``(n_experiments, D_static)`` normalised static positions."""
        return self.decode(x_flat)
