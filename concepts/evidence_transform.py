"""Concept: the saturating evidence transform  E(D) = D / (1 + D).

`E` is bounded in `[0, 1)`, hits 0.5 at `D = 1`, and saturates as `D → ∞`.
`u = 1 − E = 1/(1 + D)` is the dual "uncertainty" quantity. The 2-D field
visualisation of D, E, and ΔE under the same kernel layout lives in
`evidence_to_objective.py`; this file just shows the scalar curves.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from _style import (
    apply_style, clean_spines,
    ZINC_300, ZINC_500,
    STEEL_500, EMERALD_500,
)


PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def figure_scalar_curve() -> Path:
    apply_style()
    D_vals = np.linspace(0.0, 10.0, 1000)
    E_vals = D_vals / (1.0 + D_vals)
    u_vals = 1.0 / (1.0 + D_vals)

    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)

    ax.plot(D_vals, E_vals, color=STEEL_500, lw=2.2, label="E = D / (1 + D)")
    ax.plot(D_vals, u_vals, color=EMERALD_500, lw=2.2, label="u = 1 − E")

    for D_pt in (1, 3, 9):
        ax.axvline(D_pt, color=ZINC_300, lw=0.6, ls="--", alpha=0.6, zorder=-1)
        ax.scatter([D_pt], [D_pt / (1 + D_pt)], c=STEEL_500, s=18, zorder=5, edgecolors="none")
        ax.annotate(
            f"D = {D_pt}\nE = {D_pt / (1 + D_pt):.2f}",
            xy=(D_pt, D_pt / (1 + D_pt)),
            xytext=(6, -22), textcoords="offset points",
            fontsize=7.5, color=ZINC_500,
        )

    ax.axhline(1.0, color=ZINC_300, lw=0.6, ls="--", alpha=0.6)

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("raw density  D")
    ax.set_ylabel("transformed value")
    ax.legend(loc="center right", fontsize=9, frameon=False)
    ax.grid(True, which="major", color="#E4E4E7", alpha=0.4, lw=0.5)
    clean_spines(ax)

    path = PLOTS_DIR / "evidence_scalar_curve.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("scalar E(D) curve ...")
    p = figure_scalar_curve()
    print(f"      saved: {p}")
