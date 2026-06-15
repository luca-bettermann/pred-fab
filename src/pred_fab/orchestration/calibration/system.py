from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from ...core import DataModule, Dataset, DatasetSchema
from ...core import DataInt, DataObject, DataBool, DataCategorical, DataDomainAxis
from ...core import ParameterProposal, ExperimentSpec
from ...core.designs import SobolDesign
from ...core.frames import raw_scalar_to_param
from ...utils import PfabLogger, NormMethod, SourceStep, SplitType, combined_score, profiler
from ...utils.console import _B, _D, _R
from ..base_system import BaseOrchestrationSystem
from ..evidence import evidence_from_density
from .engine import OptimizationEngine
from .bounds import BoundsManager
from .space import SolutionSpace, StaticVariable, TrajectoryVariable, Variable


def _design_from_source_step(source_step: Any | None) -> str | None:
    """Map a ``SourceStep`` (or its string value) to its design label.

    The design is the queryable provenance axis — ``discovery`` / ``exploration`` /
    ``inference`` / ``adaptation`` / ``sobol`` — derived from the ``SourceStep`` that
    generated the proposal (``'discovery_step'`` → ``'discovery'``). See the KB note
    *First-class dataset concept in pred-fab*.
    """
    if source_step is None:
        return None
    value = getattr(source_step, "value", source_step)
    return str(value).removesuffix("_step")


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
    """Active-learning calibration engine: acquisition-driven exploration, inference, batch discovery, and joint trajectory optimization."""

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
        self._evidence_backend = evidence if evidence is not None else EvidenceBackend()
        self._density_fn: Callable[[np.ndarray], float] | None = None
        self._n_exp_fn = n_exp_fn
        self._fit_empty_kde_fn = fit_empty_kde_fn
        self._push_virtual_fn = push_virtual_fn
        self._pop_virtual_fn = pop_virtual_fn

        # Composed subsystems
        self.engine = OptimizationEngine(logger, random_seed=random_seed)
        self.bounds = BoundsManager(schema, logger)

        # Active datamodule — set before each optimization run so the
        # objective can decode candidates to parameter dicts.
        self._active_datamodule: DataModule | None = None

        # Set after each optimization call for external inspection.
        self.last_opt_nfev: int = 0
        self.last_opt_n_starts: int = 0
        self.last_opt_score: float = 0.0
        self.last_trajectory: list[dict[str, Any]] | None = None
        self.convergence_history: dict[str, list[list[float]]] = {}  # label → per-start convergence
        # Phase data for validation plots
        self.last_domain_values: list[dict[str, int]] | None = None
        self.last_process_points: list[dict[str, Any]] | None = None
        self.last_trajectory_points: np.ndarray | None = None
        self.last_trajectory_exp_ids: list[int] | None = None
        self.post_global_callback: Callable[[list[ExperimentSpec]], None] | None = None
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

        self._trajectory_joint_var_limit: int = 200  # threshold for auto-selecting joint vs sequential
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
        return self.bounds.get_hierarchical_bounds_for_code(code)

    def _get_global_bounds(self, datamodule: DataModule) -> np.ndarray:
        return self.bounds.get_global_bounds(datamodule)

    def _get_trust_region_bounds(self, datamodule: DataModule, current_params: dict[str, Any]) -> np.ndarray:
        return self.bounds.get_trust_region_bounds(datamodule, current_params)

    def _get_n_exp(self) -> int:
        """Current experiment count from the prediction system."""
        if self._n_exp_fn is not None:
            return self._n_exp_fn()
        return 1

    def state_report(self) -> None:
        """Log the current calibration configuration state."""
        lines = [f"\n  {_B}Calibration{_R}"]

        pw_parts = [f"{k}={v:g}" for k, v in self.performance_weights.items()]
        lines.append(f"    {_D}Weights: {', '.join(pw_parts)}{_R}")


        lines.append(f"\n    {_D}{'Parameter':<20s} {'Bounds':<20s} {'Delta':<8s}{_R}")
        for code in self.data_objects.keys():
            low, high = self.bounds.get_hierarchical_bounds_for_code(code)
            bounds_str = f"[{low}, {high}]"
            delta = self.trust_regions.get(code, "\u2500")
            lines.append(f"    {code:<20s} {bounds_str:<20s} {delta:<8}")

        self.logger.console_summary("\n".join(lines))
        self.logger.console_new_line()

    def get_config_snapshot(
        self,
        kappa: float | None,
        sigma: float | None = None,
        source_step: Any | None = None,
    ) -> dict[str, Any]:
        """Serializable snapshot of the settings that generated a proposal — its provenance.

        Records the *design* (the queryable axis, derived from ``source_step``) and the
        generative settings — ``kappa`` (``None`` for data-independent designs like Sobol),
        ``seed``, bounds, trust regions, fixed params, trajectory configs, optimizer knobs —
        so an experiment's origin is reproducible given the same known data. See the KB note
        *First-class dataset concept in pred-fab*.
        """
        param_bounds = {}
        for code in self.data_objects.keys():
            try:
                lo, hi = self.bounds.get_hierarchical_bounds_for_code(code)
                param_bounds[code] = [lo, hi]
            except (ValueError, KeyError):
                pass
        snapshot: dict[str, Any] = {
            "design": _design_from_source_step(source_step),
            "kappa": kappa,
            "seed": self._random_seed,
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

    def density(self, params: dict[str, Any]) -> float:
        """Raw kernel density D(z) = Σ w_j exp(-||z-c_j||²/2σ²). Unbounded.

        Distances are normalized by domain_bounds to [0,1] before applying σ,
        so σ=0.05 means "5% of the parameter range." This differs from the
        ANOVA integration path (evidence_gain) which operates in raw z-score
        space. The normalization makes density meaningful for 2D slice
        visualization — without it, σ=0.05 is negligible in z-score space
        and D≈0 everywhere except at kernel centers.

        See backlog: "Fix ANOVA marginal evidence saturation" for the plan
        to unify both paths under consistent σ semantics.
        """
        dm = self._active_datamodule
        if dm is None or self._density_fn is None:
            return 0.0
        return self._density_fn(dm.params_to_array(params))

    def evidence(self, params: dict[str, Any]) -> float:
        """Pointwise evidence E(z) = D/(1+D) ∈ [0, 1). High near data.

        Built from density() — same [0,1]-normalized distances and same
        σ semantics. Bounded by construction: D=0 → E=0, D→∞ → E→1.
        """
        return evidence_from_density(self.density(params))

    def _candidate_weight(self, params: dict[str, Any]) -> float:
        """1/L where L is derived from params. Matches SolutionSpace.decode()."""
        dims = sorted(set(self.trajectory_configs.values()))
        deriv = self.dimension_derivations.get(dims[0]) if dims else None
        if deriv is None:
            return 1.0
        return 1.0 / max(1, int(deriv(params)))

    def evidence_gain(self, params: dict[str, Any]) -> float:
        """Evidence gain ΔE for a single candidate, weighted by 1/L."""
        return self._compute_evidence_gain_for_params(params, self._candidate_weight(params))

    def acquisition(self, params: dict[str, Any], kappa: float) -> float:
        """Acquisition score A = (1-κ)·P_sys + κ·ΔE for a single candidate."""
        p = self.system_performance(params) if kappa < 1.0 else 0.0
        e = self.evidence_gain(params) if kappa > 0.0 else 0.0
        return (1.0 - kappa) * p + kappa * e

    def _sweep_2d(
        self,
        x_key: str,
        y_key: str,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        fixed_params: dict[str, Any],
        cell_fn: Callable[[int, int, dict[str, Any]], None],
        resolution: int = 60,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sweep (x_key, y_key) over their bounds, calling ``cell_fn`` per cell.

        Shared 2D-slice skeleton for the grid builders: holds all other params
        fixed, fills any ``dimension_derivations`` for derived codes, and invokes
        ``cell_fn(i, j, params)`` where ``i`` indexes ``xs`` and ``j`` indexes
        ``ys`` (grids are filled ``[j, i]`` by the callable). Returns (xs, ys).
        """
        xs = np.linspace(*x_bounds, resolution)
        ys = np.linspace(*y_bounds, resolution)

        for i in range(resolution):
            for j in range(resolution):
                params = dict(fixed_params)
                params[x_key] = float(xs[i])
                params[y_key] = float(ys[j])
                for code, derive_fn in self.dimension_derivations.items():
                    if code not in params:
                        params[code] = derive_fn(params)
                cell_fn(i, j, params)

        return xs, ys

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
        ev_grid = np.zeros((resolution, resolution))
        perf_grid = np.zeros((resolution, resolution))

        def cell(i: int, j: int, params: dict[str, Any]) -> None:
            if kappa > 0.0:
                ev_grid[j, i] = self.evidence_gain(params)
            if kappa < 1.0:
                perf_grid[j, i] = self.system_performance(params)

        xs, ys = self._sweep_2d(
            x_key, y_key, x_bounds, y_bounds, fixed_params, cell, resolution,
        )

        acq_grid = (1.0 - kappa) * perf_grid + kappa * ev_grid
        return xs, ys, ev_grid, perf_grid, acq_grid

    def compute_evidence_grids(
        self,
        x_key: str,
        y_key: str,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        fixed_params: dict[str, Any],
        resolution: int = 60,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute 2D density and evidence grids for visualization.

        Same sweep as compute_acquisition_grids but calls density() and
        evidence() — the pointwise [0,1]-normalized path. For evidence
        topology plots where per-point coverage matters more than
        integrated gain.

        Returns (xs, ys, density_grid, evidence_grid).
        """
        d_grid = np.zeros((resolution, resolution))
        e_grid = np.zeros((resolution, resolution))

        def cell(i: int, j: int, params: dict[str, Any]) -> None:
            d_grid[j, i] = self.density(params)
            e_grid[j, i] = evidence_from_density(d_grid[j, i])

        xs, ys = self._sweep_2d(
            x_key, y_key, x_bounds, y_bounds, fixed_params, cell, resolution,
        )

        return xs, ys, d_grid, e_grid

    def compute_evidence_marginal_grids(
        self,
        x_key: str,
        y_key: str,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        experiment_params: list[dict[str, Any]],
        sigma: float,
        weights: list[float] | None = None,
        resolution: int = 60,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Marginal density + evidence over (x_key, y_key).

        Param-space KDE: projects training points onto the two visible
        axes in [0,1]-normalized space and sums isotropic Gaussian
        kernels. Marginalizing over hidden dims is analytic — the 2D
        marginal of an isotropic Gaussian is itself a Gaussian with the
        same σ and weights, evaluated on the projected centers.

        Returns (xs, ys, density_grid, evidence_grid).
        """
        xs = np.linspace(*x_bounds, resolution)
        ys = np.linspace(*y_bounds, resolution)

        x_span = x_bounds[1] - x_bounds[0]
        y_span = y_bounds[1] - y_bounds[0]
        if x_span < 1e-10 or y_span < 1e-10:
            return xs, ys, np.zeros((resolution, resolution)), np.zeros((resolution, resolution))

        n = len(experiment_params)
        if n == 0:
            return xs, ys, np.zeros((resolution, resolution)), np.zeros((resolution, resolution))

        centers = np.empty((n, 2))
        for k, p in enumerate(experiment_params):
            centers[k, 0] = (float(p[x_key]) - x_bounds[0]) / x_span
            centers[k, 1] = (float(p[y_key]) - y_bounds[0]) / y_span

        w = np.array(weights) if weights is not None else np.ones(n)
        inv_2s2 = 1.0 / (2.0 * sigma ** 2)

        xs_norm = (xs - x_bounds[0]) / x_span
        ys_norm = (ys - y_bounds[0]) / y_span

        d_grid = np.zeros((resolution, resolution))
        for i in range(resolution):
            dx = xs_norm[i] - centers[:, 0]
            for j in range(resolution):
                dy = ys_norm[j] - centers[:, 1]
                d_grid[j, i] = float(np.sum(w * np.exp(-(dx ** 2 + dy ** 2) * inv_2s2)))

        e_grid = d_grid / (1.0 + d_grid)
        return xs, ys, d_grid, e_grid

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
        if dm is None or self._evidence_backend.batched_tensor is None:
            return 0.0
        x_norm = dm.params_to_array(params)
        X_SD = torch.from_numpy(np.atleast_2d(x_norm)).double()
        w = torch.tensor([candidate_weight], dtype=torch.float64)
        with torch.no_grad():
            ev = self._evidence_backend.batched_tensor(X_SD, w)
        return float(ev[0].item())

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

    def _kappa_blend(self, scores, perfs, evidences, kappa: float):
        """A = (1-κ)·P + κ·ΔE, negated for minimisation."""
        if perfs is not None and kappa < 1.0:
            scores = scores + (1.0 - kappa) * perfs
        if evidences is not None and kappa > 0.0:
            scores = scores + kappa * evidences
        return -scores

    def _point_perf(
        self,
        raw_SD: torch.Tensor,
        perf_range: tuple[float, float] | None,
    ) -> torch.Tensor:
        """Per-point weighted performance ``(S,)``, gradient-traversable.

        ``raw_SD`` is physical-frame (one row per point). Builds a param dict
        per row directly from raw — continuous values stay grad-bearing tensors,
        categorical / integer / domain values resolve to Python — and scores via
        ``perf_fn_tensor``. No normalization round-trip (the model re-normalises).
        """
        S = int(raw_SD.shape[0])
        if S == 0:
            return torch.zeros(0, dtype=raw_SD.dtype)
        dm = self._active_datamodule
        if dm is None:
            raise RuntimeError("_point_perf: _active_datamodule is None")
        if self.perf_fn_tensor is None:
            raise RuntimeError("_point_perf: perf_fn_tensor is None")
        if not getattr(dm, "_is_fitted", True):
            raise RuntimeError("_point_perf: DataModule not fitted")

        params_list = [self._raw_row_to_params(raw_SD[s], dm) for s in range(S)]
        perf_dict_S = self.perf_fn_tensor(params_list)

        out = torch.zeros(S, dtype=raw_SD.dtype)
        for k in range(S):
            perf_k = {
                name: perf_dict_S[name][k].to(dtype=raw_SD.dtype)
                for name in self.perf_names_order
                if name in perf_dict_S and not torch.isnan(perf_dict_S[name][k])
            }
            out[k] = combined_score(perf_k, self.performance_weights)
        if perf_range is not None:
            pmin, pmax = perf_range
            span = pmax - pmin
            out = (out - float(pmin)) / float(span) if span > 1e-10 else torch.full_like(out, 0.5)
        return out

    def _raw_row_to_params(self, raw_row: torch.Tensor, dm: DataModule) -> dict[str, Any]:
        """Physical-frame ``(D,)`` row → param dict (perf-path decode).

        Mirrors ``DataModule.array_to_params`` minus the reverse-normalisation
        (input is already physical). Continuous values stay grad-bearing tensors;
        categorical → label, integer / domain → Python int. Context columns are
        injected by the perf closure, so they are skipped here.
        """
        params: dict[str, Any] = {}
        ctx = set(dm._context_feature_codes)
        for j, col in enumerate(dm.input_columns):
            if col in ctx:
                continue
            params[col] = raw_scalar_to_param(
                raw_row[j],
                categories=dm.categorical_mappings.get(col),
                is_integer=isinstance(self.data_objects.get(col), DataInt),
            )
        return params

    def _candidate_perf(
        self,
        raw_pts: torch.Tensor,
        perf_range: tuple[float, float] | None,
    ) -> torch.Tensor:
        """Per-candidate performance ``(S,)`` — mean over the candidate's points."""
        S, NL, D = raw_pts.shape
        perf_flat = self._point_perf(raw_pts.reshape(S * NL, D), perf_range)
        return perf_flat.reshape(S, NL).mean(dim=-1)

    def _candidate_evidence(
        self,
        zscore_pts: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-candidate Δ∫E ``(S,)`` from z-score points via the evidence backend."""
        if self._evidence_backend.joint_batched_tensor is None:
            raise RuntimeError("evidence requested but evidence.joint_batched_tensor is None")
        return self._evidence_backend.joint_batched_tensor(zscore_pts, weights).to(dtype=zscore_pts.dtype)

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

    def _classify_param(
        self, code: str, data_obj: DataObject,
    ) -> tuple[str, tuple[float, float] | None]:
        """Classify a schema parameter for optimisation/discovery composition.

        Single source of truth for the cascade shared by
        ``_build_schema_datamodule`` and ``_compose_variables``. Returns a
        ``(kind, bounds)`` verdict; callers map each kind to their own outcome
        (the categorical choices / degenerate auto-fix / warning text differ):

        - ``"skip_fixed"``     — in ``fixed_params`` (silent).
        - ``"categorical"``    — ``DataCategorical`` / ``DataBool``.
        - ``"skip_no_bounds"`` — no hierarchical bounds (silent).
        - ``"skip_infinite"``  — infinite bounds (caller logs); ``bounds`` set.
        - ``"degenerate"``     — ``lo == hi``; ``bounds`` set.
        - ``"optimizable"``    — finite, non-degenerate; ``bounds`` set.
        """
        if code in self.fixed_params:
            return "skip_fixed", None
        if isinstance(data_obj, (DataCategorical, DataBool)):
            return "categorical", None
        try:
            lo, hi = self.bounds.get_hierarchical_bounds_for_code(code)
        except ValueError:
            return "skip_no_bounds", None
        if lo == -np.inf or hi == np.inf:
            return "skip_infinite", (lo, hi)
        if lo == hi:
            return "degenerate", (lo, hi)
        return "optimizable", (lo, hi)

    def _build_schema_datamodule(self) -> DataModule:
        """Build a schema-only DataModule (no training data) for discovery generation.

        Uses NormMethod.NONE; excludes params with infinite bounds (logged as warnings).
        """
        active_codes: list[str] = []
        for code, data_obj in self.data_objects.items():
            kind, _ = self._classify_param(code, data_obj)
            if kind == "skip_fixed" or kind == "skip_no_bounds":
                continue
            if kind == "skip_infinite":
                self.logger.warning(
                    f"Parameter '{code}' has infinite bounds; "
                    "skipping in discovery generation."
                )
                continue
            # categorical / degenerate / optimizable are all active here:
            # the schema datamodule keeps degenerate + categorical columns.
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
            kind, bnds = self._classify_param(code, data_obj)
            if kind == "skip_fixed" or kind == "skip_no_bounds":
                continue
            if kind == "categorical":
                if isinstance(data_obj, DataCategorical):
                    categorical_params.append((code, list(data_obj.constraints["categories"])))
                else:  # DataBool
                    categorical_params.append((code, [False, True]))
                continue
            if kind == "skip_infinite":
                self.logger.warning(f"Parameter '{code}' has infinite bounds; skipping.")
                continue
            if kind == "degenerate":
                lo, _ = bnds  # type: ignore[misc]
                fixed_val = int(lo) if isinstance(data_obj, (DataInt, DataDomainAxis)) else float(lo)
                self.bounds.fixed_params[code] = fixed_val
                self.logger.debug(f"Auto-fixed '{code}' = {fixed_val} (degenerate bounds).")
                continue
            optimizable.append((code, data_obj))

        variables: list[Variable] = []
        for code, obj in optimizable:
            lo, hi = self.bounds.get_hierarchical_bounds_for_code(code)
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
    # § Sobol — data-independent space-filling test/validation design
    # ==================================================================

    def run_sobol(self, n: int) -> list["ExperimentSpec"]:
        """Generate ``n`` space-filling (Sobol) proposals — a data-independent test design.

        Unlike :meth:`run_discovery` (evidence-optimised, κ=1), this draws low-discrepancy
        points directly in normalised ``[0,1]`` space and maps them to real parameters via
        each ``Variable.to_real`` — uniform over the parameter bounds, independent of model
        and collected data, so it serves as a fair generalisation yardstick (the held-out
        Sobol probe; see the KB note *First-class dataset concept in pred-fab*). Sampling
        in u-space would inherit the optimiser's sigmoid, ~25× denser at the centre than the
        bounds — unusable for space-filling — hence the direct ``to_real`` path.

        Categoricals are stratified (reusing ``_compose_variables``); integers are int-typed;
        deterministic given ``random_seed``. Trajectory parameters need their own Sobol design
        (space-filling over layer trajectories) and are not yet covered here — raises if any
        are configured rather than silently emitting incomplete test experiments.
        """
        if n <= 0:
            return []

        variables, cat_codes, cat_assignments = self._compose_variables(n)
        traj_vars = [v for v in variables if isinstance(v, TrajectoryVariable)]
        if traj_vars:
            raise NotImplementedError(
                "run_sobol does not yet cover trajectory parameters "
                f"({[v.code for v in traj_vars]}); a Sobol test design over layer "
                "trajectories is a follow-up — see 'First-class dataset concept in pred-fab'."
            )
        statics = [v for v in variables if isinstance(v, StaticVariable)]
        if not statics and not cat_codes:
            self.logger.warning("No valid parameters for Sobol test design.")
            return []

        pts = SobolDesign().unit_points(n, len(statics), seed=self._random_seed)
        specs: list[ExperimentSpec] = []
        for i in range(n):
            bp: dict[str, Any] = dict(self.fixed_params)
            for d_cat, code in enumerate(cat_codes):
                bp[code] = cat_assignments[i][d_cat]
            for di, sv in enumerate(statics):
                bp[sv.code] = sv.to_real(float(pts[i, di]))
            bp = self.schema.parameters.sanitize_values(bp, ignore_unknown=True)
            proposal = ParameterProposal.from_dict(bp, source_step=SourceStep.SOBOL)
            specs.append(ExperimentSpec(initial_params=proposal, trajectories={}))

        # Stamp generative provenance (design='sobol', seed, bounds, ...); κ is N/A.
        snapshot = self.get_config_snapshot(None, source_step=SourceStep.SOBOL)
        for spec in specs:
            spec.config_snapshot = snapshot

        self.logger.info(
            f"Sobol test design: {n} experiments "
            f"({len(statics)} static, {len(cat_codes)} categorical)."
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
            dimension_derivations=self.dimension_derivations,
            source_step=source_step,
        )

        if space.total_vars == 0:
            return space.decode_to_specs(np.zeros(0))

        console = self.logger._console_output_enabled
        perf_range = None

        def objective(x_flat: torch.Tensor) -> torch.Tensor:
            raw_pts, zscore_pts, weights = space.decode(x_flat)
            return self._blend_objective(raw_pts, zscore_pts, weights, kappa, perf_range)

        self.logger.info(
            f"{label}: N={n}, D={space._D_per_exp}, V={space.total_vars}"
        )

        opt = self.engine.optimize(
            objective, space.bounds,
            raw_u=True,
            d_param=space._D_param,
            label=label,
            show_progress=console,
        )
        self.convergence_history[label] = opt.convergence_history
        self.last_opt_nfev = opt.nfev
        self.last_opt_n_starts = opt.n_starts
        self.last_opt_score = opt.score
        self.last_discovery_nfev = getattr(self, 'last_discovery_nfev', 0) + opt.nfev

        best_x = opt.best_x if opt.best_x is not None else np.zeros(space.total_vars)
        specs = space.decode_to_specs(best_x)

        config = self.get_config_snapshot(kappa, source_step=source_step)
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


    def _blend_objective(
        self,
        raw_pts: torch.Tensor,
        zscore_pts: torch.Tensor,
        weights: torch.Tensor | None,
        kappa: float,
        perf_range: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Negated κ-blend acquisition ``(S,)``, gradient-traversable — THE objective.

        Single path for production (``_optimize`` closure) and tests. Perf reads
        the physical-frame ``raw_pts``; evidence reads ``zscore_pts``; both
        ``(S, NL, D)`` with per-point ``weights`` ``(S, NL)`` (None = 1.0).
        """
        with profiler.section("acq._blend_objective"):
            S = int(raw_pts.shape[0])
            dtype = raw_pts.dtype
            perfs = self._candidate_perf(raw_pts, perf_range) if kappa < 1.0 else None
            if perfs is not None:
                perfs = perfs.to(dtype=dtype)
            evidence = (
                self._candidate_evidence(zscore_pts, weights)
                if kappa > 0.0 and self._evidence_backend.joint_batched_tensor is not None
                else None
            )
            return self._kappa_blend(torch.zeros(S, dtype=dtype), perfs, evidence, kappa)

    # ==================================================================
    # § Model wrappers
    # ==================================================================

    def get_models(self) -> list[Any]:
        """Return empty list (no internal ML models owned by CalibrationSystem)."""
        return []

    def get_model_specs(self) -> dict[str, list[str]]:
        return {}

