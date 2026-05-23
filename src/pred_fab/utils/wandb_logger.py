"""Optional Weights & Biases logger for prediction model training.

Logs per-epoch loss, post-training validation metrics, and model
hyperparameters. Enabled via ``WandbLogger(project=..., config=...)``;
disabled by passing ``None`` where a logger is expected.

wandb is imported lazily — no import error if not installed, only
if ``WandbLogger`` is actually instantiated.
"""
from __future__ import annotations

from typing import Any


class WandbLogger:
    """Thin wrapper around wandb.init / .log / .finish."""

    def __init__(
        self,
        project: str = "pred-fab",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        **wandb_kwargs: Any,
    ):
        import wandb
        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            **wandb_kwargs,
        )

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        self._run.log({"epoch": epoch, **metrics}, step=epoch, commit=True)

    def log_validation(self, results: dict[str, dict[str, float]]) -> None:
        flat: dict[str, float] = {}
        keep = {"r2", "r2_inf", "mae"}
        for feat, metrics in results.items():
            for k, v in metrics.items():
                if k in keep:
                    flat[f"val/{feat}/{k}"] = v
        self._run.log(flat, commit=True)

    def log_summary(self, key: str, value: Any) -> None:
        self._run.summary[key] = value  # type: ignore[union-attr]

    def finish(self) -> None:
        self._wandb.finish()
