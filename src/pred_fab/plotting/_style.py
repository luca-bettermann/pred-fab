"""Shared style utilities, palette, and semantic colormaps for PFAB plots.

Single source of truth for visual identity across pred-fab and pred-fab-mock.
Five semantic colormaps:
    density        -> Greys     raw kernel sum (unbounded)
    evidence       -> Blues     saturated D/(1+D) ∈ [0, 1)
    evidence_gain  -> Purples   ΔI acquisition signal
    performance    -> RdYlGn    perf score, bad → good
    mixed          -> cividis   (1−κ)·perf + κ·ΔI, distinct from components
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap


# Visual Identity — core palette
STEEL_100 = "#D6E4F0"
STEEL_300 = "#8BB0CC"
STEEL_500 = "#4A7FA5"
STEEL_700 = "#2D5F85"
STEEL_900 = "#1A3A5C"
EMERALD_100 = "#D1FAE5"
EMERALD_300 = "#6EE7B7"
EMERALD_500 = "#10B981"
EMERALD_700 = "#047857"
EMERALD_900 = "#064E3B"
ZINC_50 = "#FAFAFA"
ZINC_100 = "#F4F4F5"
ZINC_200 = "#E4E4E7"
ZINC_300 = "#D4D4D8"
ZINC_400 = "#A1A1AA"
ZINC_500 = "#71717A"
ZINC_600 = "#52525B"
ZINC_700 = "#3F3F46"
ZINC_800 = "#27272A"
ZINC_900 = "#18181B"
ACCENT_YELLOW = "#EAB308"
ACCENT_RED = "#DC2626"
RED = ACCENT_RED
YELLOW = ACCENT_YELLOW


# ---------------------------------------------------------------------------
# Colormap registry
# ---------------------------------------------------------------------------

_CMAP_REGISTRY: dict[str, str] = {
    "density": "Greys",
    "evidence": "Blues",
    "evidence_gain": "Purples",
    "performance": "RdYlGn",
    "mixed": "cividis",
}


def cmap(name: str) -> Colormap:
    """Return the canonical matplotlib colormap for a semantic surface.

    Valid names: density, evidence, evidence_gain, performance, mixed.
    """
    if name not in _CMAP_REGISTRY:
        valid = ", ".join(sorted(_CMAP_REGISTRY))
        raise ValueError(f"unknown cmap {name!r}; expected one of: {valid}")
    return plt.get_cmap(_CMAP_REGISTRY[name])


# ---------------------------------------------------------------------------
# rcParams / style application
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Set matplotlib rcParams to the PFAB visual-identity defaults."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titlecolor": ZINC_700,
        "axes.labelsize": 9,
        "axes.labelcolor": ZINC_600,
        "axes.edgecolor": ZINC_300,
        "axes.linewidth": 0.8,
        "xtick.color": ZINC_500,
        "ytick.color": ZINC_500,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# Axis / spine helpers
# ---------------------------------------------------------------------------

def clean_spines(ax) -> None:
    """Remove top/right spines; tone down left/bottom."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(ZINC_300)
    ax.spines["bottom"].set_color(ZINC_300)


def clean_3d_panes(ax) -> None:
    """Transparent panes, light edges, muted ticks — 3blue1brown-style 3-D."""
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.0)
        pane.set_edgecolor(ZINC_300)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_tick_params(colors=ZINC_500, labelsize=7, pad=1)
        axis.label.set_color(ZINC_600)
        axis.line.set_color(ZINC_300)
    ax.grid(False)


def subplot_label(
    ax,
    text: str,
    *,
    color: str = ZINC_700,
    fontsize: float = 9.5,
    pad: float = 8.0,
    loc: str = "left",
) -> None:
    """Subplot title placed *above* the axes (PFAB convention).

    Renders via ``ax.set_title(loc=loc, pad=pad)`` so it sits outside the plot
    area and never overlaps plot contents. Pair with ``figure_subtitle`` for
    figure-wide context — the figure subtitle sits above all subplot titles.
    """
    ax.set_title(text, loc=loc, fontsize=fontsize, color=color, pad=pad)


def figure_subtitle(
    fig,
    text: str,
    *,
    y: float = 0.995,
    fontsize: float = 8.0,
    color: str = ZINC_400,
    style: str = "italic",
) -> None:
    """Figure-level subtitle (info common to all subplots), top-center.

    Sits *above* subplot titles. Marks the figure so :func:`save_fig`
    reserves top room — preventing subplot titles from colliding with this
    band.
    """
    fig.text(0.5, y, text, ha="center", va="top",
             fontsize=fontsize, color=color, style=style)
    setattr(fig, "_pfab_has_subtitle", True)


# ---------------------------------------------------------------------------
# Geometric overlays for evidence figures
# ---------------------------------------------------------------------------

def add_kernel_radii_2d(
    ax,
    center: np.ndarray,
    sigma: float,
    multipliers: Iterable[float],
    *,
    color_scale: bool = True,
    base_color: str = STEEL_500,
    alpha_max: float = 0.7,
    lw: float = 0.7,
    n_points: int = 240,
) -> None:
    """Concentric circles at σ·multipliers around `center` (atom-style 2-D).

    If `color_scale` is True, each ring is shaded by `exp(-r²/2σ²)` (peak-density
    fraction at that radius) using the evidence cmap.
    """
    cm = cmap("evidence") if color_scale else None
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy = float(center[0]), float(center[1])
    for m in multipliers:
        r = float(m) * sigma
        density_frac = float(np.exp(-(r ** 2) / (2.0 * sigma ** 2)))
        if cm is not None:
            color = cm(0.25 + 0.55 * density_frac)
        else:
            color = base_color
        ax.plot(cx + r * cos_t, cy + r * sin_t,
                color=color, lw=lw, alpha=alpha_max * (0.4 + 0.6 * density_frac),
                zorder=2)


def add_kernel_radii_3d(
    ax,
    center: np.ndarray,
    sigma: float,
    multipliers: Iterable[float],
    *,
    color_scale: bool = True,
    base_color: str = STEEL_500,
    alpha_max: float = 0.5,
    lw: float = 0.7,
    n_points: int = 200,
    orbitals_per_shell: int = 3,
) -> None:
    """Great-circle orbitals at σ·multipliers around `center` (atom-style 3-D).

    `orbitals_per_shell` controls how many great circles are drawn per shell:
    1 → minimal (one tilted ring per radius), 3 → xy, xz, yz planes (default,
    reads as a sphere wireframe-lite), 2 → two orthogonal rings.
    """
    cm = cmap("evidence") if color_scale else None
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    zero = np.zeros_like(theta)
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    multipliers = list(multipliers)

    # Per-orbital base curves (unit circle in three principal planes).
    # Index 0: xy-plane;  1: xz-plane;  2: yz-plane.
    plane_curves = [
        (cos_t, sin_t, zero),
        (cos_t, zero, sin_t),
        (zero, cos_t, sin_t),
    ]

    n_orbitals = max(1, min(int(orbitals_per_shell), 3))

    for k, m in enumerate(multipliers):
        r = float(m) * sigma
        density_frac = float(np.exp(-(r ** 2) / (2.0 * sigma ** 2)))
        color = cm(0.25 + 0.55 * density_frac) if cm is not None else base_color
        alpha = alpha_max * (0.4 + 0.6 * density_frac)

        if n_orbitals == 1:
            # One tilted ring per shell — old "atomic orbital" look.
            tilt_y = (k * 35.0) * np.pi / 180.0
            tilt_z = (k * 50.0) * np.pi / 180.0
            x, y, z = r * cos_t, r * sin_t, zero
            cy_, sy_ = np.cos(tilt_y), np.sin(tilt_y)
            x2, z2 = x * cy_ - z * sy_, x * sy_ + z * cy_
            cz_, sz_ = np.cos(tilt_z), np.sin(tilt_z)
            x3, y3 = x2 * cz_ - y * sz_, x2 * sz_ + y * cz_
            ax.plot(cx + x3, cy + y3, cz + z2,
                    color=color, lw=lw, alpha=alpha, zorder=2)
            continue

        for plane in plane_curves[:n_orbitals]:
            xs, ys, zs = plane
            ax.plot(cx + r * xs, cy + r * ys, cz + r * zs,
                    color=color, lw=lw, alpha=alpha, zorder=2)


def style_colorbar(cbar) -> None:
    """Apply PFAB tick / outline styling to a matplotlib Colorbar."""
    cbar.ax.tick_params(colors=ZINC_500, labelsize=8)
    if cbar.outline is not None:
        cbar.outline.set_edgecolor(ZINC_300)  # type: ignore[attr-defined]
        cbar.outline.set_linewidth(0.6)  # type: ignore[attr-defined]


def _resolve_cmap(name: str):
    """Resolve a colormap by semantic registry name first, then fall back to mpl."""
    if name in _CMAP_REGISTRY:
        return cmap(name)
    return plt.get_cmap(name)


def subplot_topology(
    ax,
    x_axis: "AxisSpec",
    y_axis: "AxisSpec",
    x_values: np.ndarray,
    y_values: np.ndarray,
    grid: np.ndarray,
    *,
    cmap_name: str = "performance",
    label: str | None = None,
    levels: int = 20,
    contour_overlay: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    points: list[dict[str, Any]] | None = None,
    schedules: dict[str, list[dict[str, Any]]] | None = None,
    codes: list[str] | None = None,
    point_color: str = "white",
    point_edge: str = ZINC_700,
    point_size: float = 18,
    point_alpha: float = 1.0,
    show_colorbar: bool = True,
    cbar_label: str | None = None,
):
    """Render a topology panel: contourf + optional white contours + scatter + colorbar.

    Returns the QuadContourSet so callers can attach further markers or replace
    the colorbar. ``cmap_name`` accepts a semantic key from the registry
    (density, evidence, evidence_gain, performance, mixed) or any raw
    matplotlib colormap name.
    """
    cm = _resolve_cmap(cmap_name)
    im = ax.contourf(x_values, y_values, grid, levels=levels, cmap=cm,
                     vmin=vmin, vmax=vmax)
    if contour_overlay:
        ax.contour(x_values, y_values, grid, levels=max(levels // 2, 4),
                   colors="white", linewidths=0.3, alpha=0.5)

    if points:
        _plot_schedule_ranges(ax, points, x_axis, y_axis, schedules, codes,
                              color=point_color, alpha=0.6 * point_alpha)
        px, py = _extract_xy(points, x_axis, y_axis)
        ax.scatter(px, py, s=point_size, c=point_color, edgecolors=point_edge,
                   linewidth=0.5, zorder=5, alpha=point_alpha)

    _apply_axes(ax, x_axis, y_axis)
    clean_spines(ax)
    if label is not None:
        subplot_label(ax, label)

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, label=cbar_label or "")
        style_colorbar(cbar)

    return im


def cube_wireframe(
    ax,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    color: str = STEEL_300,
    lw: float = 0.7,
    alpha: float = 0.55,
    view: tuple[float, float] | None = None,
    front_zorder: float = 10.0,
    back_zorder: float = -10.0,
) -> None:
    """Draw the 12 edges of an axis-aligned cube on a 3-D axis.

    When ``view=(elev_deg, azim_deg)`` is given, edges are split into a back
    layer (drawn behind contents) and a front layer (drawn in front), so
    points/glyphs inside the cube read as being inside it. Without ``view``,
    all edges share one low zorder (legacy behaviour).
    """
    corners = np.array([
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ], dtype=float)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    if view is None:
        for i, j in edges:
            p1, p2 = corners[i], corners[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=color, lw=lw, alpha=alpha, zorder=-1)
        return

    # Project edge midpoints onto the camera view direction; near half = front.
    elev, azim = np.deg2rad(view[0]), np.deg2rad(view[1])
    view_vec = np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        np.sin(elev),
    ])
    center = 0.5 * (lo + hi)
    spans = np.maximum(hi - lo, 1e-12)
    proj = []
    for i, j in edges:
        mid = 0.5 * (corners[i] + corners[j])
        # Normalize mid by span so scale doesn't bias the split.
        proj.append(np.dot((mid - center) / spans, view_vec))
    median = float(np.median(proj))
    for (i, j), p in zip(edges, proj):
        zo = front_zorder if p >= median else back_zorder
        p1, p2 = corners[i], corners[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=color, lw=lw, alpha=alpha, zorder=zo)


def square_wireframe(
    ax,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    color: str = STEEL_300,
    lw: float = 0.8,
    alpha: float = 0.7,
) -> None:
    """Draw the 4 edges of an axis-aligned square on a 2-D axis."""
    xs = [lo[0], hi[0], hi[0], lo[0], lo[0]]
    ys = [lo[1], lo[1], hi[1], hi[1], lo[1]]
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=-1)


# ---------------------------------------------------------------------------
# AxisSpec + existing utilities (kept for plotting subpackage callers)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisSpec:
    """Definition of a plot axis tied to a schema parameter."""

    key: str
    label: str
    unit: str = ""
    bounds: tuple[float, float] | None = None
    integer: bool = False

    @property
    def display_label(self) -> str:
        return f"{self.label} [{self.unit}]" if self.unit else self.label


def save_fig(path: str, dpi: int = 150) -> None:
    """Save current figure and close.

    If the active figure carries a PFAB subtitle (set by
    :func:`figure_subtitle`), reserves the top band so subplot titles do not
    collide with it.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig = plt.gcf()
    if getattr(fig, "_pfab_has_subtitle", False):
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    else:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _extract_xy(
    points: list[dict[str, Any]],
    x_axis: AxisSpec,
    y_axis: AxisSpec,
) -> tuple[list[float], list[float]]:
    return (
        [float(p[x_axis.key]) for p in points],
        [float(p[y_axis.key]) for p in points],
    )


def _add_fixed_subtitle(
    fig: plt.Figure,  # type: ignore[name-defined]
    fixed_params: dict[str, Any] | None,
) -> None:
    if not fixed_params:
        return
    parts = [f"{k} = {v}" for k, v in fixed_params.items()]
    figure_subtitle(fig, "fixed: " + ", ".join(parts), fontsize=7)


def _plot_schedule_ranges(
    ax: "plt.Axes",  # type: ignore[name-defined]
    points: list[dict[str, Any]],
    x_axis: "AxisSpec",
    y_axis: "AxisSpec",
    schedules: dict[str, list[dict[str, Any]]] | None,
    codes: list[str] | None,
    *,
    color: str = ZINC_400,
    alpha: float = 0.5,
    linewidth: float = 1.2,
    cap_size: float = 3.0,
    step_dot_cmap: str = "Blues",
    step_dot_size: float = 14,
) -> None:
    """Draw T-ended range lines + per-step dots for scheduled parameters."""
    if not schedules or not codes:
        return

    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(step_dot_cmap)

    for i, code in enumerate(codes):
        if code not in schedules or len(schedules[code]) <= 1:
            continue

        steps = schedules[code]
        x_vals = [float(s.get(x_axis.key, points[i].get(x_axis.key, 0))) for s in steps]
        y_vals = [float(s.get(y_axis.key, points[i].get(y_axis.key, 0))) for s in steps]

        x_varies = max(x_vals) - min(x_vals) > 1e-8
        y_varies = max(y_vals) - min(y_vals) > 1e-8

        if y_varies:
            x_center = float(points[i].get(x_axis.key, 0))
            ax.plot([x_center, x_center], [min(y_vals), max(y_vals)],
                    color=color, alpha=alpha, linewidth=linewidth, zorder=1)
            x_span = (x_axis.bounds[1] - x_axis.bounds[0]) if x_axis.bounds else 1.0
            cap = x_span * 0.008 * cap_size
            for y_end in [min(y_vals), max(y_vals)]:
                ax.plot([x_center - cap, x_center + cap], [y_end, y_end],
                        color=color, alpha=alpha, linewidth=linewidth, zorder=1)

        if x_varies:
            y_center = float(points[i].get(y_axis.key, 0))
            ax.plot([min(x_vals), max(x_vals)], [y_center, y_center],
                    color=color, alpha=alpha, linewidth=linewidth, zorder=1)
            y_span = (y_axis.bounds[1] - y_axis.bounds[0]) if y_axis.bounds else 1.0
            cap = y_span * 0.008 * cap_size
            for x_end in [min(x_vals), max(x_vals)]:
                ax.plot([x_end, x_end], [y_center - cap, y_center + cap],
                        color=color, alpha=alpha, linewidth=linewidth, zorder=1)

        # Per-step dots, light → dark across the step index so first/last
        # layer ordering is visible at a glance.
        n = len(steps)
        for k, (xv, yv) in enumerate(zip(x_vals, y_vals)):
            t = k / max(n - 1, 1)
            # Stay above the band where Blues is too pale to be visible
            # against light backgrounds; map step index into [0.35, 0.95].
            ax.scatter([xv], [yv], s=step_dot_size,
                       color=cmap(0.35 + 0.6 * t),
                       edgecolors="white", linewidths=0.4,
                       zorder=3)


def _apply_axes(
    ax: plt.Axes,  # type: ignore[name-defined]
    x_axis: AxisSpec,
    y_axis: AxisSpec,
) -> None:
    from matplotlib.ticker import MaxNLocator
    ax.set_xlabel(x_axis.display_label)
    ax.set_ylabel(y_axis.display_label)
    if x_axis.bounds:
        ax.set_xlim(*x_axis.bounds)
    if y_axis.bounds:
        ax.set_ylim(*y_axis.bounds)
    if x_axis.integer:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if y_axis.integer:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
