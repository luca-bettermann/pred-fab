from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from ...core import DataModule, Dataset, DatasetSchema
from ...core import DataInt, DataObject, DataBool, DataCategorical, DataDomainAxis
from ...core import ParameterProposal, ExperimentSpec
from ...utils import PfabLogger, NormMethod, SourceStep, SplitType, combined_score, profiler
from ..base_system import BaseOrchestrationSystem
from .engine import OptimizationEngine
from .bounds import BoundsManager
from .space import SolutionSpace, StaticVariable, TrajectoryVariable, Variable

@dataclass
class EvidenceBackend:
    """Δ∫E callbacks the acquisition objective dispatches to.

    ``batched_tensor`` covers per-candidate single-point Δ∫E; ``joint_batched_tensor``
    covers per-candidate joint multi-point (trajectory / N-batch) Δ∫E. Both
    are gradient-traversable. Either may be ``None`` when unavailable —
    the κ-blend skips that arm.
    """
    batched_tensor: Callable[..., torch.Tensor] | None = None
    joint_batched_tensor: Callable[..., torch.Tensor] | None = None


# ======================================================================
# CalibrationSystem — orchestrator, composes the other two
# ======================================================================

class CalibrationSystem(BaseOrchestrationSystem):
    """Active-learning calibration engine: acquisition-driven exploration, inference, batch discovery, and joint schedule optimization."""

    def __init__(
        self,
        schema: DatasetSchema,
        logger: PfabLogger,
        *,
        evidence: EvidenceBackend | None = None,
        n_exp_fn: Callable[[], int] | None = None,
        fit_empty_kde_fn: Callable[[DataModule, int], None] | None = None,
        push_virtual_fn: Callable[[list[dict[str, Any]], list[float] | None, DataModule | None], None] | None = None,
        pop_virtual_fn: Callable[[], None] | None = None,
        perf_fn_tensor: Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]] | None = None,
        random_seed: int | None = None,
    ):
        super().__init__(logger, random_seed=random_seed)
        self.perf_fn_tensor = perf_fn_tensor
        self.evidence = evidence if evidence is not None else EvidenceBackend()
        self._n_exp_fn = n_exp_fn
        self._fit_empty_kde_fn = fit_empty_kde_fn
        self._push_virtual_fn = push_virtual_fn
        self._pop_virtual_fn = pop_virtual_fn

        # Composed subsystems
        self.engine = OptimizationEngine(logger, random_seed=random_seed)
        self.bounds = BoundsManager(schema, logger)

        # Active datamodule — set before each optimization run so that
        # _acquisition_objective can call array_to_params.
        self._active_datamodule: DataModule | None = None

        # Set after each optimization call for external inspection.
        self.last_opt_nfev: int = 0
        self.last_opt_n_starts: int = 0
        self.last_opt_score: float = 0.0
        self.last_opt_perf: float = 0.0
        self.last_opt_unc: float = 0.0
        self.last_trajectory: list[dict[str, Any]] | None = None
        self.convergence_history: dict[str, list[list[float]]] = {}  # label → per-start convergence
        # Phase data for validation plots
        self.last_domain_values: list[dict[str, int]] | None = None
        self.last_process_points: list[dict[str, Any]] | None = None
        self.last_trajectory_points: np.ndarray | None = None
        self.last_trajectory_exp_ids: list[int] | None = None
        self.acquisition_scale: float = 1000.0
        self.post_global_callback: Callable[[list[ExperimentSpec]], None] | None = None
        self.derive_L_fn: Callable[[dict[str, Any]], int] | None = None  # deprecated, use dimension_derivations
        self.dimension_derivations: dict[str, Callable[[dict[str, Any]], int]] = {}

        # Set ordered weights
        self.schema = schema
        self.perf_names_order = list(schema.performance_attrs.keys())
        self.performance_weights: dict[str, float] = {perf: 1.0 for perf in self.perf_names_order}
        self.parameters = schema.parameters


        # Persistent κ default for acquisition_step / exploration_step. Overridable
        # per call. inference_step ignores it (κ=0 is the inference semantic).
        self.kappa_default: float = 0.5

        # Running min/max of predicted system performance across training data.

        self._schedule_joint_var_limit: int = 200  # threshold for auto-selecting joint vs sequential
        self._suppress_opt_print: bool = False

    # ==================================================================
    # § Properties — proxy to BoundsManager
    # ==================================================================

    @property
    def data_objects(self) -> dict[str, DataObject]:
        return self.bounds.data_objects

    @property
    def schema_bounds(self) -> dict[str, tuple[float, float]]:
        return self.bounds.schema_bounds

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return self.bounds.param_bounds

    @property
    def fixed_params(self) -> dict[str, Any]:
        return self.bounds.fixed_params

    @property
    def trust_regions(self) -> dict[str, float]:
        return self.bounds.trust_regions

    @property
    def trajectory_configs(self) -> dict[str, str]:
        return self.bounds.trajectory_configs

    # ==================================================================
    # § Configuration
    # ==================================================================

    @property
    def random_seed(self) -> int | None:
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value: int | None) -> None:
        self._random_seed = value
        self.rng = np.random.RandomState(value)
        self.engine._random_seed = value
        self.engine.rng = np.random.RandomState(value)

    # ==================================================================
    # § Delegated config methods
    # ==================================================================

    def configure_param_bounds(self, bounds: dict[str, tuple[float, float]], force: bool = False) -> None:
        """Configure parameter ranges for offline calibration."""
        self.bounds.configure_param_bounds(bounds, force=force)

    def configure_fixed_params(self, fixed_params: dict[str, Any], force: bool = False) -> None:
        """Configure fixed parameter values."""
        self.bounds.configure_fixed_params(fixed_params, force=force)

    def configure_adaptation_delta(self, deltas: dict[str, float], force: bool = False) -> None:
        """Configure trust region deltas for online calibration."""
        self.bounds.configure_adaptation_delta(deltas, force=force)

    def configure_trajectory_parameter(self, code: str, dimension_code: str, force: bool = False) -> None:
        """Declare that a runtime-adjustable parameter should be re-optimised at each step of the given dimension."""
        self.bounds.configure_trajectory_parameter(code, dimension_code, force=force)

    def get_tunable_params(self, datamodule: DataModule) -> list[str]:
        """Return codes of parameters the optimizer can actually vary."""
        return self.bounds.get_tunable_params(datamodule)

    def _get_hierarchical_bounds_for_code(self, code: str) -> tuple[float, float]:
        return self.bounds._get_hierarchical_bounds_for_code(code)

    def _get_global_bounds(self, datamodule: DataModule) -> np.ndarray:
        return self.bounds._get_global_bounds(datamodule)

    def _get_trust_region_bounds(self, datamodule: DataModule, current_params: dict[str, Any]) -> np.ndarray:
        return self.bounds._get_trust_region_bounds(datamodule, current_params)

    def _get_n_exp(self) -> int:
        """Current experiment count from the prediction system."""
        if self._n_exp_fn is not None:
            return self._n_exp_fn()
        return 1

    def state_report(self) -> None:
        """Log the current calibration configuration state."""
        _B = "\033[1m"
        _D = "\033[2m"
        _R = "\033[0m"

        lines = [f"\n  {_B}Calibration{_R}"]

        pw_parts = [f"{k}={v:g}" for k, v in self.performance_weights.items()]
        lines.append(f"    {_D}Weights: {', '.join(pw_parts)}{_R}")


        lines.append(f"\n    {_D}{'Parameter':<20s} {'Bounds':<20s} {'Delta':<8s}{_R}")
        for code in self.data_objects.keys():
            low, high = self.bounds._get_hierarchical_bounds_for_code(code)
            bounds_str = f"[{low}, {high}]"
            delta = self.trust_regions.get(code, "\u2500")
            lines.append(f"    {code:<20s} {bounds_str:<20s} {delta:<8}")

        self.logger.console_summary("\n".join(lines))
        self.logger.console_new_line()

    def get_config_snapshot(self, kappa: float, sigma: float | None = None) -> dict[str, Any]:
        """Serializable snapshot of user-facing settings that affect the proposal."""
        param_bounds = {}
        for code in self.data_objects.keys():
            try:
                lo, hi = self.bounds._get_hierarchical_bounds_for_code(code)
                param_bounds[code] = [lo, hi]
            except (ValueError, KeyError):
                pass
        snapshot: dict[str, Any] = {
            "kappa": kappa,
            "performance_weights": dict(self.performance_weights),
            "param_bounds": param_bounds,
            "trust_regions": {k: v for k, v in self.trust_regions.items()},
            "fixed_params": {k: v for k, v in self.fixed_params.items()
                             if isinstance(v, (int, float, str, bool))},
            "trajectory_configs": dict(self.trajectory_configs),
            "n_experiments": self._get_n_exp(),
            "optimizer": {
                "n_starts": self.engine.n_starts,
                "n_sobol": self.engine.n_sobol,
                "lr": self.engine.lr,
            },
        }
        if sigma is not None:
            snapshot["sigma"] = sigma
        return snapshot

    def set_performance_weights(self, weights: dict[str, float]) -> None:
        """Set weights for system performance calculation. Default is 1.0 for all."""
        for name, value in weights.items():
            if name in self.performance_weights:
                self.performance_weights[name] = value
                self.logger.debug(f"Set performance weight: {name} -> {value}")
            else:
                self.logger.console_warning(f"Performance attribute '{name}' not in schema; ignoring weight.")

    # ==================================================================
    # § Public API — single-candidate evaluation from params dict
    # ==================================================================

    def predict_features(self, params: dict[str, Any]) -> dict[str, float | None]:
        """Predict per-feature values for a single candidate.

        Returns ``{feature_code: float}`` (NaN → None).
        """
        return self._compute_perf_dict_for_params(params)

    def system_performance(self, params: dict[str, Any]) -> float:
        """Weighted system performance P_sys for a single candidate (raw, unnormalized)."""
        return self._compute_normalised_perf_for_params(params)

    def _candidate_weight(self, params: dict[str, Any]) -> float:
        """1/L where L is derived from params. Matches SolutionSpace.decode()."""
        if not self.derive_L_fn:
            return 1.0
        return 1.0 / max(1, self.derive_L_fn(params))

    def evidence_gain(self, params: dict[str, Any]) -> float:
        """Evidence gain ΔE for a single candidate, weighted by 1/L."""
        return self._compute_evidence_gain_for_params(params, self._candidate_weight(params))

    def acquisition(self, params: dict[str, Any], kappa: float) -> float:
        """Acquisition score A = (1-κ)·P_sys + κ·ΔE for a single candidate."""
        p = self.system_performance(params) if kappa < 1.0 else 0.0
        e = self.evidence_gain(params) if kappa > 0.0 else 0.0
        return (1.0 - kappa) * p + kappa * e

    def compute_acquisition_grids(
        self,
        x_key: str,
        y_key: str,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        fixed_params: dict[str, Any],
        kappa: float,
        resolution: int = 60,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute 2D grids by slicing through the full-D acquisition at fixed_params.

        Sweeps x_key and y_key over their bounds while holding all other
        params fixed. Uses the same evidence_gain() and system_performance()
        the optimizer uses — no separate computation path.

        Returns (xs, ys, evidence_grid, perf_grid, acq_grid).
        """
        xs = np.linspace(*x_bounds, resolution)
        ys = np.linspace(*y_bounds, resolution)
        ev_grid = np.zeros((resolution, resolution))
        perf_grid = np.zeros((resolution, resolution))

        for i in range(resolution):
            for j in range(resolution):
                params = dict(fixed_params)
                params[x_key] = float(xs[i])
                params[y_key] = float(ys[j])
                for code, derive_fn in self.dimension_derivations.items():
                    if code not in params:
                        params[code] = derive_fn(params)
                if kappa > 0.0:
                    ev_grid[j, i] = self.evidence_gain(params)
                if kappa < 1.0:
                    perf_grid[j, i] = self.system_performance(params)

        acq_grid = (1.0 - kappa) * perf_grid + kappa * ev_grid
        return xs, ys, ev_grid, perf_grid, acq_grid

    # ==================================================================
    # § Acquisition objectives — κ-blended evidence + performance
    # ==================================================================
    #
    # Unified acquisition (higher is better):
    #
    #     score(batch) = (1 − κ) · mean_k combined_score(perf(z_k), w)
    #                   + κ · Δ∫_[0,1]^D E(z | data_old ∪ batch) − E(z | data_old) dz
    #
    # κ=1 discovery drops the perf term; κ=0 inference drops the Δ∫E term.
    # Single-candidate phases pass batch with shape (1, D).

    def _compute_perf_dict_for_params(self, params: dict[str, Any]) -> dict[str, float | None]:
        """Compute a single-candidate ``{perf_code: float}`` via the tensor perf closure."""
        if self.perf_fn_tensor is None:
            raise RuntimeError("_compute_perf_dict_for_params: perf_fn_tensor is None")
        with torch.no_grad():
            t_out = self.perf_fn_tensor([params])
        out: dict[str, float | None] = {}
        for code, t in t_out.items():
            v = float(t[0].item())
            out[code] = None if v != v else v  # NaN → None
        return out

    def _compute_normalised_perf_for_params(
        self,
        params: dict[str, Any],
        perf_range: tuple[float, float] | None = None,
    ) -> float:
        """Predict + evaluate single-candidate weighted system perf; normalise into ``[0, 1]`` if ``perf_range`` is given."""
        perf_dict = self._compute_perf_dict_for_params(params)
        return self._normalize_perf_dict(perf_dict, perf_range)

    def _compute_evidence_gain_for_params(
        self, params: dict[str, Any], candidate_weight: float = 1.0,
    ) -> float:
        """Single-candidate evidence gain ΔE for a params dict."""
        dm = self._active_datamodule
        if dm is None or self.evidence.batched_tensor is None:
            return 0.0
        x_norm = dm.params_to_array(params)
        X_SD = torch.from_numpy(np.atleast_2d(x_norm)).double()
        w = torch.tensor([candidate_weight], dtype=torch.float64)
        with torch.no_grad():
            ev = self.evidence.batched_tensor(X_SD, w)
        return float(ev[0].item())

    def _per_candidate_perf_batched(
        self,
        X_SD: np.ndarray,
        perf_range: tuple[float, float] | None,
    ) -> np.ndarray:
        """No-grad numpy shim around `_per_candidate_perf_tensor`."""
        S = X_SD.shape[0]
        if S == 0:
            return np.empty((0,), dtype=np.float64)
        with torch.no_grad():
            out_t = self._per_candidate_perf_tensor(
                torch.from_numpy(np.ascontiguousarray(X_SD)).double(), perf_range,
            )
        return out_t.detach().cpu().numpy().astype(np.float64)

    def _normalize_perf_dict(
        self,
        perf_dict: dict[str, float | None],
        perf_range: tuple[float, float] | None,
    ) -> float:
        """Combine per-feature perf into a system score, then normalise into [0, 1]."""
        perf_values = [
            float(perf_dict[name]) if perf_dict.get(name) is not None else 0.0  # type: ignore[arg-type]
            for name in self.perf_names_order
            if name in perf_dict
        ]
        sys_perf = self._compute_system_performance(perf_values) if perf_values else 0.0
        if perf_range is not None:
            pmin, pmax = perf_range
            span = pmax - pmin
            sys_perf = (sys_perf - pmin) / span if span > 1e-10 else 0.5
        return float(sys_perf)

    def _acquisition_objective(
        self,
        x_flat: np.ndarray,
        kappa: float,
        perf_range: tuple[float, float] | None = None,
    ) -> float:
        """DE-compatible (negated) acquisition for single-candidate phases.

        Thin no-grad numpy shim around `_acquisition_objective_tensor` —
        wraps the 1-D x_flat as a (1, D) batch and pulls out the scalar.
        """
        with torch.no_grad():
            X_SD = torch.from_numpy(np.atleast_2d(x_flat)).double()
            out = self._acquisition_objective_tensor(X_SD, kappa, perf_range)
        return float(out[0].item())

    def _kappa_blend(self, scores, perfs, evidences, kappa: float):
        """A = (1-κ)·P + κ·ΔE, negated and scaled for minimisation."""
        if perfs is not None and kappa < 1.0:
            scores = scores + (1.0 - kappa) * perfs
        if evidences is not None and kappa > 0.0:
            scores = scores + kappa * evidences
        return -scores * self.acquisition_scale

    def _per_candidate_perf_tensor(
        self,
        X_SD: torch.Tensor,
        perf_range: tuple[float, float] | None,
    ) -> torch.Tensor:
        """Per-candidate weighted performance ``(S,)``, gradient-traversable.

        Routes through ``perf_fn_tensor`` for autograd. When the tensor closure
        is unavailable (test fixtures) or raises, falls back to a scalar loop
        — gradient is lost on that path.
        """
        S = int(X_SD.shape[0])
        if S == 0:
            return torch.zeros(0, dtype=X_SD.dtype)
        dm = self._active_datamodule
        if dm is None:
            raise RuntimeError("_per_candidate_perf_tensor: _active_datamodule is None")
        if self.perf_fn_tensor is None:
            raise RuntimeError("_per_candidate_perf_tensor: perf_fn_tensor is None")
        if not getattr(dm, "_is_fitted", True):
            raise RuntimeError("_per_candidate_perf_tensor: DataModule not fitted")

        # Build per-candidate params dicts. Continuous values stay as 0-D
        # tensors for autograd; categorical / int / domain values resolve
        # to concrete Python types at decode time (no grad through them).
        params_list: list[dict[str, Any]] = []
        for s in range(S):
            try:
                p_np = dm.array_to_params(X_SD[s].detach().cpu().numpy().reshape(-1))
                p_with_grad = self._reattach_tensor_continuous(p_np, X_SD[s], dm)
                params_list.append(p_with_grad)
            except (ValueError, KeyError) as exc:
                self.logger.warning(f"_per_candidate_perf_tensor: candidate {s} decode failed: {exc}")
                params_list.append(None)  # type: ignore[arg-type]

        valid_idx = [i for i, p in enumerate(params_list) if p is not None]
        if not valid_idx:
            raise RuntimeError(f"_per_candidate_perf_tensor: all {S} candidates failed to decode")

        perf_dict_S = self.perf_fn_tensor(
            [params_list[i] for i in valid_idx]  # type: ignore[index]
        )

        n_valid = len(valid_idx)
        out_valid = torch.zeros(n_valid, dtype=X_SD.dtype)
        for k in range(n_valid):
            perf_k = {
                name: perf_dict_S[name][k].to(dtype=X_SD.dtype)
                for name in self.perf_names_order
                if name in perf_dict_S and not torch.isnan(perf_dict_S[name][k])
            }
            out_valid[k] = combined_score(perf_k, self.performance_weights)
        if perf_range is not None:
            pmin, pmax = perf_range
            span = pmax - pmin
            if span > 1e-10:
                out_valid = (out_valid - float(pmin)) / float(span)
            else:
                out_valid = torch.full_like(out_valid, 0.5)

        out_full = torch.zeros(S, dtype=X_SD.dtype)
        for k, i in enumerate(valid_idx):
            out_full[i] = out_valid[k]
        return out_full

    def _reattach_tensor_continuous(
        self,
        params_np: dict[str, Any],
        x_norm: torch.Tensor,
        dm: DataModule,
    ) -> dict[str, Any]:
        """Replace continuous numeric values in ``params_np`` with their
        gradient-bearing counterparts decoded from ``x_norm``.

        ``array_to_params`` returns Python floats / ints — the gradient on
        ``x_norm`` is lost during that decode. For autograd, we preserve the
        tensor entries for continuous params by applying the normaliser's
        differentiable ``reverse()`` to the z-score tensor values.
        Categorical / integer / domain params stay as Python.
        """
        out = dict(params_np)
        reattached = 0
        for j, col_name in enumerate(dm.input_columns):
            if col_name not in out:
                continue
            stats = dm._parameter_stats.get(col_name)
            if stats is None:
                continue
            if col_name in dm.categorical_mappings:
                continue
            val = stats.reverse(x_norm[j])
            if isinstance(val, torch.Tensor) and x_norm.requires_grad and not val.requires_grad:
                self.logger.console_warning(
                    f"Gradient lost in _reattach_tensor_continuous for '{col_name}'"
                )
            out[col_name] = val
            reattached += 1
        if reattached == 0 and x_norm.requires_grad:
            self.logger.console_warning(
                "_reattach_tensor_continuous: no columns reattached — performance gradient is zero"
            )
        return out

    def _acquisition_objective_tensor(
        self,
        X_SD: torch.Tensor,
        kappa: float,
        perf_range: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Vectorised (negated) acquisition for the gradient optimiser.

        Takes ``(S, D)`` torch tensor — one row per candidate — and returns
        ``(S,)`` of negated scores (lower = better). Mirrors
        ``_acquisition_objective_vectorized`` but routes through the tensor
        APIs added in so gradient flows from each
        candidate's score back through the per-candidate parameters.
        """
        with profiler.section("acq._acquisition_objective_tensor"):
            S = int(X_SD.shape[0])

            perfs: torch.Tensor | None = None
            if kappa < 1.0:
                perfs = self._per_candidate_perf_tensor(X_SD, perf_range)

            evidences: torch.Tensor | None = None
            if kappa > 0.0:
                if self.evidence.batched_tensor is None:
                    raise RuntimeError(
                        f"kappa={kappa} requires evidence but evidence.batched_tensor is None"
                    )
                evidences = self.evidence.batched_tensor(X_SD).to(dtype=X_SD.dtype)

            return self._kappa_blend(torch.zeros(S, dtype=X_SD.dtype), perfs, evidences, kappa)

    def _compute_system_performance(self, performance: list[float]) -> float:
        """Compute weighted system performance from an ordered list of scores."""
        if not performance:
            return 0.0
        perf_dict = {
            name: performance[i]
            for i, name in enumerate(self.perf_names_order)
            if i < len(performance)
        }
        return combined_score(perf_dict, self.performance_weights)

    # ==================================================================
    # § Helpers — spec construction, datamodule, perf range
    # ==================================================================

    def set_active_datamodule(self, datamodule: DataModule) -> None:
        """Set the active DataModule for acquisition evaluation."""
        self._active_datamodule = datamodule

    def _build_schema_datamodule(self) -> DataModule:
        """Build a schema-only DataModule (no training data) for discovery generation.

        Uses NormMethod.NONE; excludes params with infinite bounds (logged as warnings).
        """
        active_codes: list[str] = []
        for code, data_obj in self.data_objects.items():
            if code in self.fixed_params:
                continue
            if isinstance(data_obj, (DataCategorical, DataBool)):
                active_codes.append(code)
                continue
            try:
                lo, hi = self.bounds._get_hierarchical_bounds_for_code(code)
            except ValueError:
                continue
            if lo == -np.inf or hi == np.inf:
                self.logger.warning(
                    f"Parameter '{code}' has infinite bounds; "
                    "skipping in discovery generation."
                )
                continue
            active_codes.append(code)

        dataset = Dataset(schema=self.schema, debug_flag=True)
        datamodule = DataModule(dataset, normalize=NormMethod.NONE)
        datamodule.initialize(
            input_parameters=active_codes,
            input_features=[],
            output_columns=list(self.schema.performance_attrs.keys()),
        )
        datamodule.fit_without_data()
        return datamodule

    # ==================================================================
    # § Variable composition — shared by discovery + acquisition
    # ==================================================================

    def _compose_variables(
        self,
        n: int,
    ) -> tuple[list[Variable], list[str], list[tuple[Any, ...]]]:
        """Classify schema parameters into Variables and stratify categoricals.

        Returns ``(variables, cat_codes, cat_assignments)`` ready for ``_optimize``.
        """
        categorical_params: list[tuple[str, list[Any]]] = []
        optimizable: list[tuple[str, DataObject]] = []
        traj_set = set(self.trajectory_configs.keys())

        for code, data_obj in self.data_objects.items():
            if code in self.fixed_params:
                continue
            if isinstance(data_obj, DataCategorical):
                categorical_params.append((code, list(data_obj.constraints["categories"])))
                continue
            if isinstance(data_obj, DataBool):
                categorical_params.append((code, [False, True]))
                continue
            try:
                lo, hi = self.bounds._get_hierarchical_bounds_for_code(code)
            except ValueError:
                continue
            if lo == -np.inf or hi == np.inf:
                self.logger.warning(f"Parameter '{code}' has infinite bounds; skipping.")
                continue
            if lo == hi:
                fixed_val = int(lo) if isinstance(data_obj, (DataInt, DataDomainAxis)) else float(lo)
                self.bounds.fixed_params[code] = fixed_val
                self.logger.debug(f"Auto-fixed '{code}' = {fixed_val} (degenerate bounds).")
                continue
            optimizable.append((code, data_obj))

        variables: list[Variable] = []
        for code, obj in optimizable:
            lo, hi = self.bounds._get_hierarchical_bounds_for_code(code)
            if code in traj_set:
                span = hi - lo
                trust = self.bounds.trust_regions.get(code, span / 10.0)
                from .space import SIGMOID_K
                slope_max = trust / (SIGMOID_K * 0.25 * span) if span > 0 else 0.4
                variables.append(TrajectoryVariable(
                    data_object=obj, lo=lo, hi=hi,
                    dimension_code=self.trajectory_configs[code],
                    slope_max=slope_max,
                ))
            else:
                variables.append(StaticVariable(
                    data_object=obj, lo=lo, hi=hi,
                    is_integer=isinstance(obj, (DataInt, DataDomainAxis)),
                ))

        # Stratify categoricals
        if categorical_params:
            import itertools as _it
            cat_combos = list(_it.product(*(cats for _, cats in categorical_params)))
            cat_codes = [code for code, _ in categorical_params]
        else:
            cat_combos = [()]
            cat_codes = []
        n_strata = len(cat_combos)

        n_per_stratum = [n // n_strata] * n_strata
        for i in range(n % n_strata):
            n_per_stratum[i] += 1

        cat_assignments: list[tuple[Any, ...]] = []
        for combo, n_i in zip(cat_combos, n_per_stratum):
            cat_assignments.extend([combo] * n_i)

        return variables, cat_codes, cat_assignments

    # ==================================================================
    # § Discovery — batch proposal generation
    # ==================================================================

    def run_discovery(self, n: int) -> list["ExperimentSpec"]:
        """Generate n discovery proposals via Sobol -> LBFGS acquisition (kappa=1)."""
        if n == 0:
            return []

        variables, cat_codes, cat_assignments = self._compose_variables(n)
        if not variables and not cat_codes:
            self.logger.warning("No valid parameters for discovery generation.")
            return []

        dm = self._build_schema_datamodule()
        if self._fit_empty_kde_fn is not None:
            self._fit_empty_kde_fn(dm, n)

        specs = self._optimize(
            n, variables, cat_codes, cat_assignments,
            datamodule=dm,
            kappa=1.0, source_step=SourceStep.DISCOVERY,
            label="Local optimizer",
        )

        n_static = sum(1 for v in variables if isinstance(v, StaticVariable))
        n_traj = sum(1 for v in variables if isinstance(v, TrajectoryVariable))
        self.logger.info(
            f"Discovery: {n} experiments "
            f"({n_static} static, {n_traj} trajectory, "
            f"{len(cat_codes)} categorical)."
        )
        return specs

    # ==================================================================
    # § Acquisition — single-experiment exploration / inference
    # ==================================================================

    def run_acquisition(
        self,
        datamodule: DataModule,
        kappa: float,
        source_step: str,
    ) -> ExperimentSpec:
        """Single-experiment acquisition via Sobol -> LBFGS.

        Same pipeline as discovery but with n=1 and a trained datamodule.
        kappa>0 blends evidence + performance (exploration);
        kappa=0 is pure performance (inference).
        """
        variables, cat_codes, cat_assignments = self._compose_variables(1)

        specs = self._optimize(
            1, variables, cat_codes, cat_assignments,
            datamodule=datamodule,
            kappa=kappa, source_step=source_step,
            label="Local optimizer",
        )
        return specs[0]

    # ==================================================================
    # § _optimize — single unified optimization path
    # ==================================================================

    def _optimize(
        self,
        n: int,
        variables: list[Variable],
        cat_codes: list[str],
        cat_assignments: list[tuple[Any, ...]],
        datamodule: DataModule,
        kappa: float,
        source_step: str | None = None,
        label: str = "Optimizing",
    ) -> list[ExperimentSpec]:
        """Single optimization path: SolutionSpace + Engine.

        Constructs the decision vector layout from Variable objects, defines
        the objective closure, runs Sobol -> LBFGS, and decodes the result.
        The caller provides the datamodule (schema-only for discovery,
        trained for exploration/inference).
        """
        self._active_datamodule = datamodule

        space = SolutionSpace(
            variables=variables,
            n_experiments=n,
            bounds_manager=self.bounds,
            datamodule=datamodule,
            fixed_params=dict(self.fixed_params),
            cat_codes=cat_codes,
            cat_assignments=cat_assignments,
            schema_sanitize=lambda d: self.schema.parameters.sanitize_values(d, ignore_unknown=True),
            derive_L_fn=self.derive_L_fn,
            source_step=source_step,
        )

        if space.total_vars == 0:
            return space.decode_to_specs(np.zeros(0))

        console = self.logger._console_output_enabled
        perf_range = None

        def objective(x_flat: torch.Tensor) -> torch.Tensor:
            points, weights = space.decode(x_flat)
            full_S_NL = points.unsqueeze(1)  # (S, 1, total_points, D)
            return self._acquisition_joint_batched_tensor(full_S_NL, kappa, perf_range, weights)

        self.logger.info(
            f"{label}: N={n}, D={space._D_per_exp}, V={space.total_vars}"
        )

        opt = self.engine.optimize(
            objective, space.bounds,
            raw_z=True,
            d_param=space._D_param,
            label=label,
            show_progress=console,
        )
        self.convergence_history[label] = opt.convergence_history
        self.last_discovery_nfev = getattr(self, 'last_discovery_nfev', 0) + opt.nfev

        best_x = opt.best_x if opt.best_x is not None else np.zeros(space.total_vars)
        specs = space.decode_to_specs(best_x)

        config = self.get_config_snapshot(kappa)
        for spec in specs:
            spec.config_snapshot = config


        # Store trajectory plot data
        if space._D_traj > 0:
            traj_norms, traj_params, L_per_exp = space.get_trajectory_plot_data(best_x)
            self.last_traj_norms = traj_norms
            self.last_traj_params = traj_params
            self.last_trajectory_per_exp_L = L_per_exp
            if any(L > 1 for L in L_per_exp):
                self.last_trajectory_points = np.concatenate(traj_norms, axis=0)
                exp_ids: list[int] = []
                for i, L_i in enumerate(L_per_exp):
                    exp_ids.extend([i] * L_i)
                self.last_trajectory_exp_ids = exp_ids

        self.last_global_specs = list(specs)
        if self.post_global_callback is not None:
            self.post_global_callback(self.last_global_specs)

        if specs and kappa < 1.0 and console:
            self._print_proposal_metrics(specs[0], kappa)

        return specs

    def _print_proposal_metrics(self, spec: Any, kappa: float) -> None:
        """Print ΔE, P_sys, κ, A at the proposed point."""
        params = dict(spec.initial_params.to_dict())
        for code, obj in self.schema.parameters.items():
            if isinstance(obj, DataDomainAxis) and code not in params:
                if code in self.dimension_derivations:
                    params[code] = self.dimension_derivations[code](params)
        perf_dict = self._compute_perf_dict_for_params(params)
        p_sys = combined_score(perf_dict, self.performance_weights)
        de = self.evidence_gain(params)
        a = (1.0 - kappa) * p_sys + kappa * de
        self.logger.console_info(
            f"\n    {'ΔE':>8s}  {'P_sys':>8s}  {'κ':>6s}  {'A':>8s}\n"
            f"  {'─' * 38}\n"
            f"    {de:8.4f}  {p_sys:8.4f}  {kappa:6.2f}  {a:8.4f}\n"
        )


    def _acquisition_joint_batched_tensor(
        self,
        full_S_NL: torch.Tensor,
        kappa: float,
        perf_range: tuple[float, float] | None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Negated κ-weighted joint acquisition ``(S,)``, gradient-traversable.

        ``weights`` shape ``(S, NL)`` — per-point evidence weight. None = 1.0.
        """
        with profiler.section("acq._acquisition_joint_batched_tensor"):
            S, N, L, D = full_S_NL.shape
            NL = N * L
            dtype = full_S_NL.dtype

            perfs_S: torch.Tensor | None = None
            if kappa < 1.0:
                flat_rows = full_S_NL.reshape(S * NL, D)
                perfs_flat = self._per_candidate_perf_tensor(flat_rows, perf_range)
                perfs_S = perfs_flat.reshape(S, NL).mean(dim=-1).to(dtype=dtype)

            evidence_S: torch.Tensor | None = None
            if kappa > 0.0 and self.evidence.joint_batched_tensor is not None:
                flat_per_candidate = full_S_NL.reshape(S, NL, D)
                w = weights.reshape(S, NL) if weights is not None else None
                evidence_S = self.evidence.joint_batched_tensor(flat_per_candidate, w).to(dtype=dtype)

            return self._kappa_blend(torch.zeros(S, dtype=dtype), perfs_S, evidence_S, kappa)

    # ==================================================================
    # § Model wrappers
    # ==================================================================

    def get_models(self) -> list[Any]:
        """Return empty list (no internal ML models owned by CalibrationSystem)."""
        return []

    def get_model_specs(self) -> dict[str, list[str]]:
        return {}

