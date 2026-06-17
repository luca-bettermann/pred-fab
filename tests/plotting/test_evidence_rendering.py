"""Evidence-aware rendering: fade overlay + trust boundary."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from pred_fab.plotting import AxisSpec
from pred_fab.plotting._style import subplot_topology


@pytest.fixture
def setup():
    x = AxisSpec("a", "A", bounds=(0, 1))
    y = AxisSpec("b", "B", bounds=(0, 1))
    xs = np.linspace(0, 1, 15)
    ys = np.linspace(0, 1, 15)
    gx, _ = np.meshgrid(xs, ys)
    perf = gx
    return x, y, xs, ys, perf


def test_evidence_grid_adds_fade_overlay(setup):
    x, y, xs, ys, perf = setup
    evidence = np.tile(np.linspace(0, 1, 15), (15, 1))
    fig, ax = plt.subplots()
    subplot_topology(ax, x, y, xs, ys, perf, evidence_grid=evidence,
                     show_colorbar=False)
    assert len(ax.images) == 1
    alpha = ax.images[0].get_array()[..., 3]
    assert alpha.max() > 0.5  # low-evidence side washed out
    assert alpha.min() < 0.01  # saturated side untouched
    plt.close(fig)


def test_no_fade_without_evidence_grid(setup):
    x, y, xs, ys, perf = setup
    fig, ax = plt.subplots()
    subplot_topology(ax, x, y, xs, ys, perf, show_colorbar=False)
    assert len(ax.images) == 0
    plt.close(fig)


def test_trust_contour_only_when_threshold_crossed(setup):
    x, y, xs, ys, perf = setup
    fig, (ax1, ax2) = plt.subplots(1, 2)
    crossing = np.tile(np.linspace(0, 1, 15), (15, 1))
    subplot_topology(ax1, x, y, xs, ys, perf, evidence_grid=crossing,
                     show_colorbar=False)
    saturated = np.full((15, 15), 0.9)
    subplot_topology(ax2, x, y, xs, ys, perf, evidence_grid=saturated,
                     show_colorbar=False)
    # The crossing panel carries the inline "E = 0.5" label; the saturated one doesn't.
    labels1 = [t.get_text() for t in ax1.texts]
    labels2 = [t.get_text() for t in ax2.texts]
    assert any("E = 0.5" in t for t in labels1)
    assert not any("E = 0.5" in t for t in labels2)
    plt.close(fig)


def test_figure_functions_accept_evidence(tmp_path, setup):
    from pred_fab.plotting.discovery import plot_parameter_space
    from pred_fab.plotting.exploration import plot_acquisition
    from pred_fab.plotting.inference import plot_inference_result
    from pred_fab.plotting.prediction import plot_topology_comparison

    x, y, xs, ys, perf = setup
    ev = np.tile(np.linspace(0, 1, 15), (15, 1))
    plot_parameter_space(str(tmp_path / "ps.png"), x, y, xs, ys, [],
                         perf, perf, evidence_grid=ev)
    plot_acquisition(str(tmp_path / "acq.png"), x, y, xs, ys,
                     perf, perf * 0.05, perf, evidence_grid=ev)
    plot_topology_comparison(str(tmp_path / "tc.png"), x, y, xs, ys,
                             {"truth": perf, "model": perf},
                             evidence_grids={"model": ev})
    plot_inference_result(str(tmp_path / "inf.png"), x, y, xs, ys, perf,
                          {"a": 0.5, "b": 0.5}, 0.8, evidence_grid=ev)
    for name in ("ps.png", "acq.png", "tc.png", "inf.png"):
        assert (tmp_path / name).exists()
