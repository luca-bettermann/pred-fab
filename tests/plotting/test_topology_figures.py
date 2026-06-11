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


def test_radar_legend_defaults_without_labels():
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    radar_chart(ax, ["a", "b", "c"], values=[0.5, 0.6, 0.7])
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert texts[0] == "mean"
    plt.close(fig)
