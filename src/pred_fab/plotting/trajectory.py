"""Trajectory and adaptation phase plots."""

import numpy as np
import matplotlib.pyplot as plt

from ._style import save_fig, STEEL_500


def plot_trajectory_comparison(
    save_path: str,
    fixed_scores: list[float],
    traj_scores: list[float],
    traj_schedules: list[dict[str, list[float]]],
    schedule_key: str,
    schedule_label: str = "",
    *,
    n_steps: int | None = None,
    title: str = "Fixed vs Trajectory Exploration",
) -> None:
    """1x2: fixed vs trajectory scores + per-step parameter schedules."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    n_fixed = len(fixed_scores)
    n_traj = len(traj_scores)
    ax1.bar(range(1, n_fixed + 1), fixed_scores, color="#DD8452",
            label="Fixed params", alpha=0.8)
    ax1.bar(range(n_fixed + 1, n_fixed + n_traj + 1), traj_scores,
            color=STEEL_500, label="Trajectory", alpha=0.8)
    ax1.set_xlabel("Exploration Round")
    ax1.set_ylabel("Combined Score")
    ax1.set_title("Performance per Round")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2, axis="y")

    y_label = schedule_label or schedule_key
    for i, sched in enumerate(traj_schedules):
        if schedule_key in sched:
            vals = sched[schedule_key]
            steps = n_steps or len(vals)
            ax2.plot(range(steps), vals[:steps], "o-",
                     label=f"traj_{i+1:02d}", lw=1.5, ms=5)
    ax2.set_xlabel("Step Index")
    ax2.set_ylabel(y_label)
    ax2.set_title(f"Per-Step {y_label}")
    if any(schedule_key in s for s in traj_schedules):
        ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.2)

    save_fig(save_path)


def plot_adaptation(
    save_path: str,
    step_values: list[float],
    deviations: list[float],
    step_label: str = "Adapted Parameter",
    deviation_label: str = "Avg Deviation",
    *,
    counterfactual: list[float] | None = None,
    title: str = "Online Adaptation \u2014 Step-by-Step",
) -> None:
    """2x1: adapted parameter + deviation over steps, optionally with counterfactual."""
    n = len(step_values)
    steps = [f"L{i}" for i in range(n)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax1.plot(steps, step_values, "o-", color=STEEL_500, lw=2, ms=6)
    ax1.set_ylabel(step_label)
    ax1.set_title(step_label)
    ax1.grid(True, alpha=0.2)

    ax2.plot(steps, deviations, "o-", color="#D65F5F", lw=2, ms=6, label="Adapted")
    if counterfactual:
        ax2.plot(steps, counterfactual, "o--", color="#D65F5F", lw=1.5, ms=5,
                 alpha=0.5, label="No adaptation")
        ax2.fill_between(range(n), deviations, counterfactual,
                         alpha=0.15, color="#D65F5F", label="Deviation saved")
    ax2.set_xlabel("Step")
    ax2.set_ylabel(deviation_label)
    ax2.set_title(deviation_label)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    save_fig(save_path)
