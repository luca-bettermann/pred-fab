from typing import Any, Callable

import numpy as np

from ...core import ExperimentSpec, ParameterProposal


class SolutionSpace:
    """Decision vector layout, bounds, and decode/encode for N-experiment optimisation.

    All params live in [0, 1] normalised space. Integer params use continuous
    relaxation during optimisation; rounding happens in decode via STE in the
    objective or hard rounding in decode_to_specs.
    """

    def __init__(
        self,
        n_experiments: int,
        static_params: list[tuple[str, float, float]],
        int_set: set[int],
        int_ranges_map: dict[int, int],
        **_kwargs: Any,
    ):
        self._n_experiments = n_experiments
        self._static_params = static_params
        self._int_set = int_set
        self._int_ranges_map = int_ranges_map

        self._D_static = len(static_params)
        self._total_vars = n_experiments * self._D_static

        self._static_bounds_list: list[tuple[float, float]] = []
        for d, (_code, _lo, _hi) in enumerate(static_params):
            if d in int_set:
                self._static_bounds_list.append((0.0, float(int_ranges_map[d])))
            else:
                self._static_bounds_list.append((0.0, 1.0))

        self._bounds_list: list[tuple[float, float]] = []
        for _i in range(n_experiments):
            self._bounds_list.extend(self._static_bounds_list)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return self._bounds_list

    @property
    def total_vars(self) -> int:
        return self._total_vars

    @property
    def n_total_points(self) -> int:
        return self._n_experiments

    def decode(self, x_flat: np.ndarray) -> np.ndarray:
        """Decode flat decision vector into ``(n_experiments, D_static)`` normalised array."""
        units = x_flat.reshape(self._n_experiments, self._D_static)
        pts = units.copy()
        for si in range(self._D_static):
            if si in self._int_set:
                r = self._int_ranges_map[si]
                pts[:, si] = units[:, si] / r if r > 0 else 0.5
        return pts

    def decode_to_specs(
        self, x_flat: np.ndarray, fixed_params: dict[str, Any],
        cat_codes: list[str], cat_assignments: list[tuple[Any, ...]],
        structural_values: list[dict[str, int]] | None, primary_dim_code: str,
        schema_sanitize: Callable[[dict[str, Any]], dict[str, Any]],
        integer_params: list[tuple[str, int, int]], source_step: str = "baseline_step",
    ) -> list[ExperimentSpec]:
        """Convert optimized vector back to ``ExperimentSpec`` instances."""
        del primary_dim_code
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
            specs.append(ExperimentSpec(initial_params=initial, trajectories={}))
        return specs

    def decode_optimized_positions(self, x_flat: np.ndarray) -> np.ndarray:
        """Return ``(n_experiments, D_static)`` normalised positions."""
        return self.decode(x_flat)
