"""Progression strips: shared scale, single colorbar, per-round overlays."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import pytest

from pred_fab.plotting import AxisSpec, plot_topology_evolution


@pytest.fixture
def setup():
    x = AxisSpec("a", "A", bounds=(0, 1))
    y = AxisSpec("b", "B", bounds=(0, 1))
    xs = np.linspace(0, 1, 12)
    ys = np.linspace(0, 1, 12)
    gx, _ = np.meshgrid(xs, ys)
    rounds = [gx * s for s in (0.4, 0.7, 1.0)]
    return x, y, xs, ys, rounds


def test_evolution_strip_renders(tmp_path, setup):
    x, y, xs, ys, rounds = setup
    path = tmp_path / "evo.png"
    plot_topology_evolution(str(path), x, y, xs, ys, rounds,
                            round_labels=["R1", "R2", "R3"])
    assert path.exists()


def test_evolution_with_truth_evidence_and_points(tmp_path, setup):
    x, y, xs, ys, rounds = setup
    ev = [np.full((12, 12), v) for v in (0.2, 0.5, 0.9)]
    pts = [[{"a": 0.2, "b": 0.3}],
           [{"a": 0.2, "b": 0.3}, {"a": 0.6, "b": 0.5}],
           [{"a": 0.2, "b": 0.3}, {"a": 0.6, "b": 0.5}, {"a": 0.8, "b": 0.7}]]
    path = tmp_path / "evo_full.png"
    plot_topology_evolution(str(path), x, y, xs, ys, rounds,
                            truth_grid=rounds[-1], evidence_grids=ev,
                            points_per_round=pts, cmap_name="performance",
                            cbar_label="Combined Score")
    assert path.exists()


def test_evolution_empty_grids_is_noop(tmp_path, setup):
    x, y, xs, ys, _ = setup
    path = tmp_path / "none.png"
    plot_topology_evolution(str(path), x, y, xs, ys, [])
    assert not path.exists()
