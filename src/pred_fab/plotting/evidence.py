"""Evidence density panels — reusable grid + scatter + trajectory plotting.

Three panel types built on a single grid computation core:
  - ``plot_density_panel``       — raw kernel density D(z)
  - ``plot_evidence_panel``      — evidence integrand 1/(1+D)
  - ``plot_evidence_gain_panel`` — ΔE between two point sets

All accept ``ExperimentSpec`` lists directly and handle trajectory
expansion (1/L weighting) automatically.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt

from ._style import (
    AxisSpec,
    apply_style,
    clean_spines,
    subplot_label,
    subplot_topology,
    save_fig,
    STEEL_500,
    ZINC_400,
)


# ======================================================================
# Experiment expansion — specs → flat points with trajectory weights
# ======================================================================

def expand_experiments(
    experiments: list[Any],
    param_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[int], list[float]]:
    """Expand experiments into flat point lists with per-layer 1/L weights.

    Accepts:
      - ``ExperimentSpec`` — expands via ``.trajectories``
      - ``ExperimentData`` — expands via ``.parameter_updates``
      - ``dict`` — single point, no expansion
    """
    transform = param_transform or (lambda d: d)
    all_pts: list[dict[str, Any]] = []
    exp_ids: list[int] = []
    weights: list[float] = []

    for i, exp in enumerate(experiments):
        if hasattr(exp, "initial_params"):
            base = dict(exp.initial_params.to_dict())
            layers: list[dict[str, Any]] = [transform(base)]
            for _dim, traj in exp.trajectories.items():
                for _step, proposal in traj.entries:
                    layer_p = dict(base)
                    layer_p.update(transform(proposal.to_dict()))
                    layers.append(layer_p)
        elif hasattr(exp, "parameter_updates") and hasattr(exp, "parameters"):
            base_params = exp.parameters.get_values_dict()
            if exp.parameter_updates:
                dim_names = exp.parameters.get_dim_names()
                n_steps = int(exp.parameters.get_value(dim_names[0])) if dim_names else 1
                layers = []
                for step in range(n_steps):
                    ctx = {exp.parameters.get_dim_iterator_codes(dim_names[:1])[0]: step} if dim_names else {}
                    layers.append(transform(exp.get_effective_parameters_for_context(ctx)))
            else:
                layers = [transform(base_params)]
        else:
            layers = [transform(dict(exp))]

        w = 1.0 / len(layers)
        for lp in layers:
            all_pts.append(lp)
            exp_ids.append(i)
            weights.append(w)

    return all_pts, exp_ids, weights


# ======================================================================
# Grid computation — single core with pluggable cell function
# ======================================================================

def _compute_grid(
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    points: list[dict[str, Any]],
    weights: list[float],
    sigma: float,
    cell_fn: Callable[[float], float],
    resolution: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 2D grid by evaluating ``cell_fn(density)`` at each cell.

    ``cell_fn`` maps density array ``D`` (one value per point set) to the
    displayed quantity. For raw density: ``lambda D: D``. For evidence:
    ``lambda D: 1/(1+D)``.
    """
    x_lo, x_hi = x_axis.bounds  # type: ignore[misc]
    y_lo, y_hi = y_axis.bounds  # type: ignore[misc]
    xs = np.linspace(x_lo, x_hi, resolution)
    ys = np.linspace(y_lo, y_hi, resolution)
    x_range = x_hi - x_lo
    y_range = y_hi - y_lo

    px = np.array([p.get(x_axis.key, 0) for p in points])
    py = np.array([p.get(y_axis.key, 0) for p in points])
    w = np.array(weights)
    inv_2s2 = 1.0 / (2.0 * sigma ** 2)

    grid = np.zeros((resolution, resolution))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            d2 = ((px - x) / x_range) ** 2 + ((py - y) / y_range) ** 2
            density = np.sum(w * np.exp(-d2 * inv_2s2))
            grid[j, i] = cell_fn(density)

    return xs, ys, grid


def _density_fn(D: float) -> float:
    return D

def evidence_cell_fn(D: float) -> float:
    """Evidence saturation D/(1+D) — high where points are, bounded [0, 1]."""
    return D / (1.0 + D)


def evidence_gain_cell_fn(D: float) -> float:
    """Evidence gain 1/(1+D) — high where points are missing, bounded [0, 1]."""
    return 1.0 / (1.0 + D)


def make_marginalized_evidence_fn(
    evidence_fn: Callable[[dict[str, Any]], float],
    all_axes: list[AxisSpec],
    x_key: str,
    y_key: str,
    n_quadrature: int = 5,
) -> Callable[[dict[str, Any]], float]:
    """Build a ``params → float`` wrapper that marginalizes over hidden dims.

    Uses Gauss-Legendre quadrature (same integration method as the
    KernelField shell probes) over hidden dimensions. Evaluates the full
    evidence pipeline at each quadrature point — no simplified proxy.

    For ``H`` hidden dims, each call evaluates ``evidence_fn`` at
    ``n_quadrature ** H`` points.
    """
    hidden = [a for a in all_axes if a.key not in (x_key, y_key)]
    if not hidden:
        return evidence_fn

    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(n_quadrature)
    nodes_01 = (nodes + 1.0) / 2.0
    weights_01 = weights / 2.0

    hidden_keys = [a.key for a in hidden]
    hidden_lo = np.array([a.bounds[0] for a in hidden])  # type: ignore[index]
    hidden_hi = np.array([a.bounds[1] for a in hidden])  # type: ignore[index]
    hidden_range = hidden_hi - hidden_lo

    grids = np.meshgrid(*[nodes_01] * len(hidden), indexing="ij")
    quad_points = np.stack([g.ravel() for g in grids], axis=-1)
    w_grids = np.meshgrid(*[weights_01] * len(hidden), indexing="ij")
    quad_weights = np.ones(len(quad_points))
    for wg in w_grids:
        quad_weights *= wg.ravel()

    quad_params = hidden_lo + quad_points * hidden_range

    def _fn(params: dict[str, Any]) -> float:
        total = 0.0
        for row, w in zip(quad_params, quad_weights):
            p = dict(params)
            for k, v in zip(hidden_keys, row):
                p[k] = float(v)
            total += w * evidence_fn(p)
        return total

    return _fn


def _normal_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _marginal_factors(
    points: list[dict[str, Any]],
    hidden_axes: list[AxisSpec],
    sigma: float,
) -> np.ndarray:
    """Per-point marginal weight: how much of each kernel falls inside bounds
    along the hidden dimensions.

    Analytical Gaussian integral over [lo, hi] per hidden axis, multiplied
    across all hidden axes. Returns shape ``(N_points,)``.
    """
    n = len(points)
    factors = np.ones(n)
    for ax in hidden_axes:
        lo, hi = ax.bounds  # type: ignore[misc]
        rng = hi - lo
        for j in range(n):
            u = (float(points[j].get(ax.key, (lo + hi) / 2)) - lo) / rng
            f = _normal_cdf((1.0 - u) / sigma) - _normal_cdf(-u / sigma)
            factors[j] *= f
    return factors


def compute_grid_marginalized(
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    all_axes: list[AxisSpec],
    points: list[dict[str, Any]],
    weights: list[float],
    sigma: float,
    cell_fn: Callable[[float], float],
    resolution: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D grid with analytical marginalization over hidden dimensions.

    Each point's kernel contribution is scaled by the fraction of its
    Gaussian that falls inside the bounds of the non-plotted dimensions.
    Boundary points automatically show lower density/evidence.
    """
    hidden = [a for a in all_axes if a.key not in (x_axis.key, y_axis.key)]
    m_factors = _marginal_factors(points, hidden, sigma)

    x_lo, x_hi = x_axis.bounds  # type: ignore[misc]
    y_lo, y_hi = y_axis.bounds  # type: ignore[misc]
    xs = np.linspace(x_lo, x_hi, resolution)
    ys = np.linspace(y_lo, y_hi, resolution)
    x_range = x_hi - x_lo
    y_range = y_hi - y_lo

    px = np.array([p.get(x_axis.key, 0) for p in points])
    py = np.array([p.get(y_axis.key, 0) for p in points])
    w = np.array(weights) * m_factors
    inv_2s2 = 1.0 / (2.0 * sigma ** 2)

    grid = np.zeros((resolution, resolution))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            d2 = ((px - x) / x_range) ** 2 + ((py - y) / y_range) ** 2
            density = np.sum(w * np.exp(-d2 * inv_2s2))
            grid[j, i] = cell_fn(density)

    return xs, ys, grid


# ======================================================================
# Scatter + trajectory overlay
# ======================================================================

def _overlay_points(
    ax: plt.Axes,  # type: ignore[name-defined]
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    points: list[dict[str, Any]],
    exp_ids: list[int],
) -> None:
    """Draw scatter points with trajectory lines, colored by layer index."""
    from matplotlib.colors import Normalize as MplNorm
    px = [float(p.get(x_axis.key, 0)) for p in points]
    py = [float(p.get(y_axis.key, 0)) for p in points]

    # Compute per-point layer fraction: 0 = first layer, 1 = last
    layer_frac = np.zeros(len(points))
    n_exp = max(exp_ids) + 1 if exp_ids else 0
    for eid in range(n_exp):
        mask = [j for j, e in enumerate(exp_ids) if e == eid]
        if len(mask) > 1:
            for k, j in enumerate(mask):
                layer_frac[j] = k / (len(mask) - 1)
            ex = [px[j] for j in mask]
            ey = [py[j] for j in mask]
            ax.plot(ex, ey, color="#E8913A", linewidth=0.6, alpha=0.4, zorder=1)
        if mask:
            ax.annotate(f"{eid+1}", (px[mask[0]], py[mask[0]]), fontsize=7,
                       ha="center", va="bottom", xytext=(0, 5),
                       textcoords="offset points", color="#B06A1E")

    cm = plt.get_cmap("Oranges")
    colors = [cm(0.3 + 0.55 * layer_frac[j]) for j in range(len(points))]
    ax.scatter(px, py, s=60, c=colors, edgecolors="white",
               linewidth=0.8, zorder=5)


# ======================================================================
# Panel functions — one axis pair, one subplot
# ======================================================================

def plot_density_panel(
    ax: plt.Axes,  # type: ignore[name-defined]
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    experiments: list[Any],
    sigma: float,
    *,
    label: str | None = None,
    param_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    resolution: int = 40,
) -> None:
    """Kernel density D(z) panel."""
    pts, exp_ids, weights = expand_experiments(experiments, param_transform)
    xs, ys, grid = _compute_grid(x_axis, y_axis, pts, weights, sigma, _density_fn, resolution)
    subplot_topology(ax, x_axis, y_axis, xs, ys, grid,
                     cmap_name="density", contour_overlay=False, show_colorbar=True)
    if label:
        subplot_label(ax, label)
    _overlay_points(ax, x_axis, y_axis, pts, exp_ids)


def plot_evidence_panel(
    ax: plt.Axes,  # type: ignore[name-defined]
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    experiments: list[Any],
    sigma: float,
    *,
    all_axes: list[AxisSpec] | None = None,
    label: str | None = None,
    param_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    resolution: int = 40,
) -> None:
    """Evidence integrand D/(1+D) panel with faithful marginalization.

    When ``all_axes`` is provided, hidden dimensions are marginalized
    analytically — boundary points show reduced evidence because their
    kernels extend outside the parameter space.
    """
    pts, exp_ids, weights = expand_experiments(experiments, param_transform)
    if all_axes is not None:
        xs, ys, grid = compute_grid_marginalized(
            x_axis, y_axis, all_axes, pts, weights, sigma, evidence_cell_fn, resolution,
        )
    else:
        xs, ys, grid = _compute_grid(x_axis, y_axis, pts, weights, sigma, evidence_cell_fn, resolution)
    grid_max = float(grid.max()) if grid.size > 0 else 1.0
    subplot_topology(ax, x_axis, y_axis, xs, ys, grid,
                     cmap_name="evidence", contour_overlay=False, show_colorbar=True,
                     vmin=0.0, vmax=1.0, cbar_lim=max(grid_max, 1e-6))
    if label:
        subplot_label(ax, label)
    _overlay_points(ax, x_axis, y_axis, pts, exp_ids)


def plot_evidence_gain_panel(
    ax: plt.Axes,  # type: ignore[name-defined]
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    experiments_before: list[Any],
    experiments_after: list[Any],
    sigma: float,
    *,
    label: str | None = None,
    param_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    resolution: int = 40,
) -> None:
    """Evidence gain ΔE panel — difference between after and before."""
    pts_b, _, w_b = expand_experiments(experiments_before, param_transform)
    pts_a, exp_ids_a, w_a = expand_experiments(experiments_after, param_transform)

    _, _, grid_before = _compute_grid(x_axis, y_axis, pts_b, w_b, sigma, evidence_cell_fn, resolution)
    _, _, grid_after = _compute_grid(x_axis, y_axis, pts_a, w_a, sigma, evidence_cell_fn, resolution)
    xs = np.linspace(x_axis.bounds[0], x_axis.bounds[1], resolution)  # type: ignore[index]
    ys = np.linspace(y_axis.bounds[0], y_axis.bounds[1], resolution)  # type: ignore[index]
    grid = grid_after - grid_before

    subplot_topology(ax, x_axis, y_axis, xs, ys, grid,
                     cmap_name="evidence_gain", contour_overlay=False, show_colorbar=True)
    if label:
        subplot_label(ax, label)
    _overlay_points(ax, x_axis, y_axis, pts_a, exp_ids_a)


# ======================================================================
# Multi-angle composition
# ======================================================================

def plot_multi_angle(
    focus_axis: AxisSpec,
    other_axes: list[AxisSpec],
    experiments: list[Any],
    sigma: float,
    path: str,
    *,
    panel_fn: Callable | None = None,
    param_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    resolution: int = 40,
    title: str | None = None,
) -> None:
    """Plot ``focus_axis`` against each axis in ``other_axes`` in a single row.

    When using the default ``plot_evidence_panel``, hidden dimensions are
    marginalized analytically so the 2D projection faithfully represents
    the full-D evidence landscape.
    """
    fn = panel_fn or plot_evidence_panel
    all_axes = [focus_axis] + list(other_axes)
    n = len(other_axes)
    if n == 0:
        return

    apply_style()
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for i, other in enumerate(other_axes):
        label = f"{focus_axis.label} × {other.label}"
        kwargs: dict[str, Any] = dict(
            label=label, param_transform=param_transform, resolution=resolution,
        )
        if fn is plot_evidence_panel:
            kwargs["all_axes"] = all_axes
        fn(axes[i], focus_axis, other, experiments, sigma, **kwargs)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
