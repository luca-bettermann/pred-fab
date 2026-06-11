"""Publication mode: dual output, metadata embedding, font/size scaling."""

import json
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from PIL import Image

from pred_fab.plotting import AxisSpec
from pred_fab.plotting._style import (
    FONT, PAGE_WIDTH_IN, fig_size, is_publication_mode,
    save_fig, set_publication_mode,
)


@pytest.fixture(autouse=True)
def reset_mode():
    yield
    set_publication_mode(False)


def test_mode_swaps_font_table_in_place():
    ref = FONT  # consumers hold this reference
    dev_title = FONT["title"]
    set_publication_mode(True)
    assert is_publication_mode()
    assert ref["title"] < dev_title
    set_publication_mode(False)
    assert ref["title"] == dev_title


def test_publication_save_writes_png_and_pdf(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = tmp_path / "fig.png"
    save_fig(str(path), publication=True)
    assert path.exists()
    assert (tmp_path / "fig.pdf").exists()


def test_dev_save_writes_png_only(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = tmp_path / "fig.png"
    save_fig(str(path))
    assert path.exists()
    assert not (tmp_path / "fig.pdf").exists()


def test_metadata_embedded_in_png(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = tmp_path / "fig.png"
    save_fig(str(path), metadata={"round": 3, "kappa": 0.5, "seed": 42})
    info = Image.open(path).text
    assert info["round"] == "3"
    assert info["kappa"] == "0.5"


def test_fig_size_scales_to_page_width_in_publication():
    dev_w, dev_h = fig_size(3, panel_w=5.0, panel_h=5.0)
    set_publication_mode(True)
    pub_w, pub_h = fig_size(3, panel_w=5.0, panel_h=5.0)
    assert pub_w == PAGE_WIDTH_IN
    assert pub_h / pub_w == pytest.approx(dev_h / dev_w)


def test_overlay_diagnosed_points():
    from pred_fab.orchestration.cross_validation import (
        MODEL_PROBLEM, TRUSTWORTHY)
    from pred_fab.plotting.prediction import overlay_diagnosed_points
    x = AxisSpec("a", "A", bounds=(0, 1))
    y = AxisSpec("b", "B", bounds=(0, 1))
    diag = SimpleNamespace(points=[
        SimpleNamespace(params={"a": 0.2, "b": 0.3}, label=MODEL_PROBLEM),
        SimpleNamespace(params={"a": 0.7, "b": 0.6}, label=TRUSTWORTHY),
    ])
    fig, ax = plt.subplots()
    n_before = len(ax.collections)
    overlay_diagnosed_points(ax, diag, x, y)
    assert len(ax.collections) == n_before + 2
    plt.close(fig)
