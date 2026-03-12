"""Baseline experiment generation via Latin Hypercube Sampling.

This module is private to the orchestration package.  All public surface goes
through ``CalibrationSystem.generate_baseline_experiments``.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy.stats import qmc

from ..core import DataInt, DataReal, DataBool, DataCategorical, DataDimension
from ..core import ParameterProposal, ParameterSchedule, ExperimentSpec
from ..utils import PfabLogger, SamplingStrategy, SourceStep


class BaselineSampler:
    """Generates baseline experiment designs using Latin Hypercube Sampling.

    Holds *references* to the shared config dicts owned by ``CalibrationSystem``
    (``data_objects``, ``schema_bounds``, ``fixed_params``, ``param_bounds``,
    ``trajectory_configs``).  Because Python dicts are mutable objects, any
    ``configure_*`` call on ``CalibrationSystem`` is automatically visible here
    without re-wiring.
    """

    def __init__(
        self,
        parameters: Any,
        data_objects: Dict[str, Any],
        schema_bounds: Dict[str, Tuple[float, float]],
        fixed_params: Dict[str, Any],
        param_bounds: Dict[str, Tuple[float, float]],
        trajectory_configs: Dict[str, str],
        rng: np.random.RandomState,
        logger: PfabLogger,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            parameters: ``DatasetSchema.parameters`` — used for ``sanitize_values``.
            data_objects: Shared ``{code: DataObject}`` registry.
            schema_bounds: Shared schema-level ``{code: (low, high)}`` bounds.
            fixed_params: Shared ``{code: value}`` fixed-param registry.
            param_bounds: Shared ``{code: (low, high)}`` explicit override bounds.
            trajectory_configs: Shared ``{param_code: dim_code}`` trajectory map.
            rng: Shared ``numpy.random.RandomState`` for reproducible sampling.
            logger: Logger instance.
            random_seed: Passed to ``scipy.stats.qmc.LatinHypercube`` for the LHS
                sampler (kept separate from *rng* for consistency with the original
                implementation).
        """
        self.parameters = parameters
        self.data_objects = data_objects
        self.schema_bounds = schema_bounds
        self.fixed_params = fixed_params
        self.param_bounds = param_bounds
        self.trajectory_configs = trajectory_configs
        self.rng = rng
        self.logger = logger
        self.random_seed = random_seed

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        n_samples: int,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        n_trajectory_segments: int = 3,
    ) -> List[ExperimentSpec]:
        """Generate an initial design using Latin Hypercube Sampling.

        Args:
            n_samples: Number of baseline experiments to generate.
            param_bounds: Optional per-parameter override bounds
                ``{code: (low, high)}``.  Falls back to configured bounds when
                absent.
            n_trajectory_segments: Number of evenly-spaced trajectory trigger
                points to generate for runtime-adjustable parameters that have
                a ``trajectory_configs`` entry.  Only used when
                ``trajectory_configs`` is non-empty and ``n_trajectory_segments
                > 1``.

        Returns:
            List of ``ExperimentSpec`` objects — one per LHS sample row.
        """
        self.logger.info(
            f"Generating {n_samples} baseline experiments using Latin Hypercube Sampling..."
        )

        # 1. Define Sampling Space (Physics)
        sampling_specs = self._get_sampling_specs(param_bounds)
        if not sampling_specs:
            self.logger.warning("No valid parameters for baseline generation.")
            return []

        param_names = sorted(sampling_specs.keys())
        d = len(param_names)

        # 2. Generate Stratified Samples (Geometry) — (n_samples, d) in [0, 1]
        lhs_samples = self._lhs_unit_samples(d, n_samples)

        # 3. Build ExperimentSpec per row
        experiments: List[ExperimentSpec] = []
        for row in lhs_samples:
            initial_dict = self._transform_lhs_sample(row, param_names, sampling_specs)
            initial_proposal = ParameterProposal.from_dict(
                initial_dict, source_step=SourceStep.BASELINE_SAMPLING
            )

            # 4. Generate trajectory schedules for configured runtime params
            schedules: Dict[str, ParameterSchedule] = {}
            if self.trajectory_configs and n_trajectory_segments > 1:
                schedules = self._generate_baseline_schedules(
                    initial_dict, n_trajectory_segments
                )

            spec = ExperimentSpec(initial_params=initial_proposal, schedules=schedules)
            experiments.append(spec)
            self.logger.debug(
                f"Generated baseline experiment: {initial_dict}, "
                f"schedules={list(schedules.keys())}"
            )

        return experiments

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _generate_baseline_schedules(
        self,
        initial_dict: Dict[str, Any],
        n_segments: int,
    ) -> Dict[str, ParameterSchedule]:
        """Sample trajectory schedules for trajectory-configured runtime parameters."""
        # Group trajectory params by their dimension code
        params_per_dim: Dict[str, List[str]] = defaultdict(list)
        for code, dim_code in self.trajectory_configs.items():
            params_per_dim[dim_code].append(code)

        schedules: Dict[str, ParameterSchedule] = {}
        for dim_code, param_codes in params_per_dim.items():
            dim_obj = self.data_objects.get(dim_code)
            if not isinstance(dim_obj, DataDimension):
                self.logger.warning(
                    f"Dimension '{dim_code}' not found or not a DataDimension for "
                    f"trajectory params {param_codes}. Skipping schedule generation."
                )
                continue

            dim_size = int(initial_dict.get(dim_code, dim_obj.constraints.get("min", 1)))
            n_triggers = n_segments - 1
            trigger_steps = sorted({
                max(1, int((k + 1) * dim_size / n_segments))
                for k in range(n_triggers)
            })

            entries: List[Tuple[int, ParameterProposal]] = []
            for step_idx in trigger_steps:
                step_values: Dict[str, Any] = {}
                for code in param_codes:
                    low, high = self.schema_bounds.get(code, (-np.inf, np.inf))
                    if low == -np.inf or high == np.inf:
                        try:
                            low, high = self._get_hierarchical_bounds_for_code(code)
                        except ValueError:
                            continue
                    step_values[code] = float(self.rng.uniform(low, high))
                if step_values:
                    entries.append((
                        step_idx,
                        ParameterProposal.from_dict(
                            step_values, source_step=SourceStep.BASELINE_TRAJECTORY
                        ),
                    ))

            if entries:
                schedules[dim_code] = ParameterSchedule(
                    dimension=dim_code, entries=entries
                )

        return schedules

    def _get_sampling_specs(
        self,
        param_bounds: Optional[Dict[str, Tuple[float, float]]],
    ) -> Dict[str, Dict[str, Any]]:
        """Build sampling specifications for each parameter."""
        sampling_specs = {}

        for code, data_obj in self.data_objects.items():

            # 1. Determine Effective Bounds (Continuous Only)
            if param_bounds and code in param_bounds:
                low, high = param_bounds[code]
            else:
                low, high = self._get_hierarchical_bounds_for_code(code)

            # 2. Check for Infinite Bounds (Safeguard)
            if isinstance(data_obj, (DataReal, DataInt)) and (low == -np.inf or high == np.inf):
                self.logger.warning(
                    f"Parameter '{code}' has infinite bounds; skipping in baseline generation."
                )
                continue

            # 3. Create Specification
            if isinstance(data_obj, DataCategorical):
                # Use retrieved bounds to detect fixed functionality
                sampling_specs[code] = {
                    'type': SamplingStrategy.CATEGORICAL,
                    'categories': data_obj.constraints['categories'] if low != high else [low]
                }
            elif isinstance(data_obj, DataBool):
                # Use retrieved bounds to detect fixed functionality
                if low == high:
                    sampling_specs[code] = {
                        'type': SamplingStrategy.CATEGORICAL,
                        'categories': [low]
                    }
                else:
                    sampling_specs[code] = {
                        'type': SamplingStrategy.BOOL
                    }
            else:
                # Continuous / Integer
                dtype = int if isinstance(data_obj, DataInt) else float
                sampling_specs[code] = {
                    'type': SamplingStrategy.NUMERICAL,
                    'low': low,
                    'high': high,
                    'dtype': dtype
                }

            self.logger.debug(f"Included '{code}' in baseline generation specs.")

        return sampling_specs

    def _transform_lhs_sample(
        self,
        row: np.ndarray,
        param_names: List[str],
        sampling_specs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Transform a single [0, 1] LHS vector into a valid parameter dictionary."""
        params = {}
        for val, name in zip(row, param_names):
            spec = sampling_specs[name]

            if spec['type'] == SamplingStrategy.NUMERICAL:
                # Scale: [0, 1] -> [low, high]
                scaled_val = spec['low'] + val * (spec['high'] - spec['low'])
                params[name] = spec['dtype'](scaled_val)

            elif spec['type'] == SamplingStrategy.BOOL:
                # Scale: [0, 1] -> {True, False}
                params[name] = bool(val > 0.5)

            elif spec['type'] == SamplingStrategy.CATEGORICAL:
                # Scale: [0, 1] -> Category Index
                cats = spec['categories']
                idx = int(val * len(cats))
                idx = min(idx, len(cats) - 1)  # clip to be safe
                params[name] = cats[idx]

        # Reuse canonical parameter coercion/rounding rules.
        return self.parameters.sanitize_values(params, ignore_unknown=True)

    def _lhs_unit_samples(self, d: int, n: int) -> np.ndarray:
        """Return an ``(n, d)`` array of LHS samples in ``[0, 1]^d``."""
        return qmc.LatinHypercube(d=d, seed=self.random_seed).random(n=n)

    def _get_hierarchical_bounds_for_code(self, code: str) -> Tuple[float, float]:
        """Resolve effective bounds for *code* with priority: fixed > param_bounds > schema."""
        if code in self.fixed_params:
            val = self.fixed_params[code]
            return val, val
        elif code in self.param_bounds:
            return self.param_bounds[code]
        elif code in self.schema_bounds:
            return self.schema_bounds[code]
        else:
            raise ValueError(
                f"No bounds found for '{code}'. Cannot determine optimization bounds."
            )
