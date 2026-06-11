"""Figure-level smoke + behaviour tests for the topology plotting path."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from pred_fab.plotting import AxisSpec
from pred_fab.plotting.exploration import plot_acquisition
from pred_fab.plotting.performance import radar_chart


@pytest.fixture
def axes_2d():
    x = AxisSpec("water_ratio", "Water Ratio", bounds=(0.3, 0.5))
    y = AxisSpec("print_speed", "Print Speed", unit="mm/s", bounds=(20, 60))
    return x, y


@pytest.fixture
def grids():
    xs = np.linspace(0.3, 0.5, 12)
    ys = np.linspace(20, 60, 12)
    gx, gy = np.meshgrid(xs, ys)
    perf = (gx - 0.3) / 0.2
    gain = 0.05 * (1 - perf)
    combined = 0.5 * perf + 0.5 * gain
    return xs, ys, perf, gain, combined


def test_plot_acquisition_renders(tmp_path, axes_2d, grids):
    """The combined panel uses the registered 'acquisition' surface (was 'mixed')."""
    x, y = axes_2d
    xs, ys, perf, gain, combined = grids
    path = str(tmp_path / "acq.png")
    plot_acquisition(path, x, y, xs, ys, perf, gain, combined,
                     proposed={"water_ratio": 0.42, "print_speed": 45.0})
    assert (tmp_path / "acq.png").exists()


def test_radar_legend_uses_passed_labels():
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    radar_chart(
        ax, ["a", "b", "c"],
        values=[0.5, 0.6, 0.7], stds=[0.1, 0.1, 0.1],
        ref_values=[0.4, 0.5, 0.6],
        label="EXP-01", ref_label="dataset avg",
    )
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "EXP-01" in texts
    assert "dataset avg" in texts
    plt.close(fig)


def test_bounded_surface_defaults_to_unit_scale(axes_2d, grids):
    """Bounded semantics render on [0,1] regardless of the data range."""
    from pred_fab.plotting._style import subplot_topology
    x, y = axes_2d
    xs, ys, perf, _, _ = grids
    fig, ax = plt.subplots()
    im = subplot_topology(ax, x, y, xs, ys, perf * 0.5 + 0.2,
                          cmap_name="performance", show_colorbar=False)
    assert im.norm.vmin == 0.0
    assert im.norm.vmax == 1.0
    plt.close(fig)


def test_fit_to_data_overrides_bounded_default(axes_2d, grids):
    from pred_fab.plotting._style import subplot_topology
    x, y = axes_2d
    xs, ys, perf, _, _ = grids
    data = perf * 0.5 + 0.2  # range [0.2, 0.7]
    fig, ax = plt.subplots()
    im = subplot_topology(ax, x, y, xs, ys, data,
                          cmap_name="performance", fit_to_data=True,
                          show_colorbar=False)
    assert im.norm.vmax < 1.0
    plt.close(fig)


def test_explicit_bounds_always_win(axes_2d, grids):
    from pred_fab.plotting._style import subplot_topology
    x, y = axes_2d
    xs, ys, perf, _, _ = grids
    fig, ax = plt.subplots()
    im = subplot_topology(ax, x, y, xs, ys, perf,
                          cmap_name="performance", vmin=0.25, vmax=0.75,
                          show_colorbar=False)
    assert im.norm.vmin == 0.25
    assert im.norm.vmax == 0.75
    plt.close(fig)


def test_comparison_figures_render_shared_scale(tmp_path, axes_2d, grids):
    from pred_fab.plotting.discovery import plot_parameter_space
    from pred_fab.plotting.prediction import plot_topology_comparison
    x, y = axes_2d
    xs, ys, perf, _, _ = grids
    pts = [{"water_ratio": 0.35, "print_speed": 30.0}]
    plot_parameter_space(str(tmp_path / "ps.png"), x, y, xs, ys, pts,
                         perf, perf * 0.8)
    plot_parameter_space(str(tmp_path / "ps_fit.png"), x, y, xs, ys, pts,
                         perf, perf * 0.8, fit_to_data=True)
    plot_topology_comparison(str(tmp_path / "tc.png"), x, y, xs, ys,
                             {"truth": perf, "model": perf * 0.8})
    assert (tmp_path / "ps.png").exists()
    assert (tmp_path / "ps_fit.png").exists()
    assert (tmp_path / "tc.png").exists()


def test_radar_legend_defaults_without_labels():
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    radar_chart(ax, ["a", "b", "c"], values=[0.5, 0.6, 0.7])
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert texts[0] == "mean"
    plt.close(fig)


def test_inference_with_marginals(tmp_path, axes_2d, grids):
    from pred_fab.plotting.inference import plot_inference_result
    x, y = axes_2d
    xs, ys, perf, _, _ = grids
    path = tmp_path / "inf_marg.png"
    plot_inference_result(str(path), x, y, xs, ys, perf,
                          {"water_ratio": 0.42, "print_speed": 40.0}, 0.81,
                          optimum={"water_ratio": 0.44, "print_speed": 38.0},
                          optimum_score=0.85, marginals=True)
    assert path.exists()


def test_marginal_slices_match_grid_rows():
    from pred_fab.plotting._style import draw_marginal_slices, marginal_layout
    xs = np.linspace(0, 1, 10)
    ys = np.linspace(0, 1, 10)
    grid = np.outer(np.linspace(0, 1, 10), np.linspace(0.5, 1, 10))
    fig, ax, ax_top, ax_right = marginal_layout((7, 6.6))
    draw_marginal_slices(ax, ax_top, ax_right, xs, ys, grid, 0.5, 0.5)
    iy = int(np.abs(ys - 0.5).argmin())
    np.testing.assert_allclose(ax_top.lines[0].get_ydata(), grid[iy, :])
    plt.close(fig)
