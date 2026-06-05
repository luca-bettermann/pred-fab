"""Trajectory phase plot: fixed-params vs trajectory-params comparison."""

import matplotlib.pyplot as plt

from ._style import (
    AxisSpec, FONT, save_fig, apply_style, clean_spines,
    STEEL_500, ZINC_300,
)


def plot_trajectory_comparison(
    save_path: str,
    fixed_scores: list[float],
    traj_scores: list[float],
    trajectories: list[dict[str, list[float]]],
    y_axis: AxisSpec,
    *,
    n_steps: int | None = None,
) -> None:
    """1x2: fixed vs trajectory scores + per-step parameter trajectories."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    n_fixed = len(fixed_scores)
    n_traj = len(traj_scores)
    ax1.bar(range(1, n_fixed + 1), fixed_scores, color="#DD8452",
            label="Fixed params", alpha=0.8)
    ax1.bar(range(n_fixed + 1, n_fixed + n_traj + 1), traj_scores,
            color=STEEL_500, label="Trajectory", alpha=0.8)
    ax1.set_xlabel("Exploration Round")
    ax1.set_ylabel("Combined Score")
    ax1.set_title("Performance per Round")
    ax1.legend(fontsize=FONT["legend"])
    ax1.grid(True, alpha=0.2, axis="y", color=ZINC_300)
    clean_spines(ax1)

    for i, traj in enumerate(trajectories):
        if y_axis.key in traj:
            vals = traj[y_axis.key]
            steps = n_steps or len(vals)
            ax2.plot(range(steps), vals[:steps], "o-",
                     label=f"traj_{i+1:02d}", lw=1.5, ms=5)
    ax2.set_xlabel("Step Index")
    ax2.set_ylabel(y_axis.display_label)
    ax2.set_title(f"Per-Step {y_axis.display_label}")
    if any(y_axis.key in s for s in trajectories):
        ax2.legend(fontsize=FONT["legend"])
    ax2.grid(True, alpha=0.2, color=ZINC_300)
    clean_spines(ax2)

    save_fig(save_path)
