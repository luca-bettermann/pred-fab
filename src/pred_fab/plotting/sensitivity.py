"""Sensitivity heatmap — Sobol total-order indices as a matrix plot."""
from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._style import apply_style, save_fig, FONT, ZINC_400, ZINC_500, ZINC_600, ZINC_700


def plot_sensitivity_matrix(
    path: str,
    S_T: np.ndarray,
    inputs: list[str],
    outputs: list[str],
    S_T_conf: np.ndarray | None = None,
    title: str = "Sobol Total-Order Sensitivity",
    cmap: str = "YlOrRd",
    dpi: int = 200,
) -> None:
    """Render Sobol S_T matrix as an annotated heatmap.

    Rows = outputs (features/performance), columns = inputs (parameters).
    Cell text shows S_T value; if S_T_conf is provided, cells with
    confidence interval > S_T are marked with a dot.
    """
    apply_style()
    n_out, n_in = S_T.shape
    fig_w = max(4.0, 0.9 * n_in + 1.5)
    fig_h = max(3.0, 0.7 * n_out + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(S_T, cmap=cmap, aspect="auto", vmin=0, vmax=max(0.5, float(S_T.max())))

    ax.set_xticks(range(n_in))
    ax.set_xticklabels(inputs, fontsize=FONT["tick"], color=ZINC_600, rotation=45, ha="right")
    ax.set_yticks(range(n_out))
    ax.set_yticklabels(outputs, fontsize=FONT["tick"], color=ZINC_600)

    for i in range(n_out):
        for j in range(n_in):
            val = S_T[i, j]
            text_color = "white" if val > 0.3 else ZINC_700
            label = f"{val:.2f}"
            if S_T_conf is not None and S_T_conf[i, j] > val:
                # CI wider than index — unreliable
                label += "\n·"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=FONT["annotation"], color=text_color)

    ax.set_title(title, fontsize=FONT["title"], color=ZINC_700, pad=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.06)
    cbar.ax.tick_params(labelsize=FONT["tick"], colors=ZINC_500)

    fig.tight_layout()
    save_fig(path, dpi=dpi)
