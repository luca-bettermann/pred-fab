"""Console output helpers for PFAB."""

import math
from collections.abc import Mapping
from typing import Any

from .logger import PfabLogger
from .metrics import combined_score as _combined_score

_R  = "\033[0m"   # reset
_B  = "\033[1m"   # bold
_D  = "\033[2m"   # dim
_G  = "\033[32m"  # green
_Y  = "\033[33m"  # yellow
_RD = "\033[31m"  # red
_C  = "\033[36m"  # cyan
_W  = 58          # banner width


def _score_color(v: float) -> str:
    if v >= 0.8:
        return _G
    if v >= 0.5:
        return _Y
    return _RD


class ProgressBar:
    """Inline ANSI progress bar with named metrics that flash green/red.

    Callers pass ``**counters`` (epoch=300) and ``**metrics`` via
    ``step()``. Each metric independently tracks its best value —
    green when improving, dim when stagnant.
    """

    LABEL_WIDTH = 22

    def __init__(self, label: str, *, info: dict[str, Any] | None = None,
                 bar_len: int = 12):
        self._label = label.ljust(self.LABEL_WIDTH)
        self._len = bar_len
        self._fill = 0.0
        self._counters: dict[str, int] = {}
        self._best: dict[str, float] = {}
        self._current: dict[str, float] = {}

        if info:
            parts = ", ".join(f"{k}={v}" for k, v in info.items())
            self._info_str = f" ({parts})"
        else:
            self._info_str = ""

    def step(self, *, fill: float | None = None,
             metrics: dict[str, float] | None = None,
             # legacy single-metric support
             obj: float | None = None,
             **counters: int) -> None:
        """Update bar and redraw."""
        import sys
        if fill is not None:
            self._fill = fill
        self._counters.update(counters)

        if obj is not None:
            metrics = {**(metrics or {}), "obj": obj}

        if metrics:
            for name, val in metrics.items():
                self._current[name] = val
                if name not in self._best or val < self._best[name] - 1e-15:
                    self._best[name] = val

        metric_parts: list[str] = []
        for name, val in self._current.items():
            improved = abs(val - self._best.get(name, val)) < 1e-15
            color = _G if improved else _RD
            metric_parts.append(f"{color}{name}={val:.4f}{_R}")
        metric_str = "  ".join(metric_parts)
        if metric_str:
            metric_str = "  " + metric_str

        filled = int(self._len * min(self._fill, 1.0))
        bar = "█" * filled + "░" * (self._len - filled)
        ctr = "  ".join(f"{k}={v}" for k, v in self._counters.items())
        sys.stdout.write(
            f"\r  {self._label}{self._info_str} [{bar}] {_D}{ctr}{_R}{metric_str}            "
        )
        sys.stdout.flush()

    def finish(self) -> None:
        """Print final summary line."""
        import sys
        bar = "█" * self._len
        ctr = "  ".join(f"{k}={v}" for k, v in self._counters.items())
        best_parts = [f"{k}={v:.4f}" for k, v in self._best.items()]
        if best_parts:
            ctr += "  " + "  ".join(best_parts)
        sys.stdout.write(
            f"\r{_G}✓{_R} {self._label}{self._info_str} [{bar}] {_D}{ctr}{_R}            \n"
        )
        sys.stdout.flush()


class ConsoleReporter:
    """Schema-aware formatted console output for agent steps.

    Reads schema metadata (parameter types, performance attributes) to produce
    formatted output that works with any domain — no hard-coded field names.
    """

    def __init__(
        self,
        logger: PfabLogger,
        param_codes: list[str],
        perf_codes: list[str],
        param_categories: dict[str, list[str]],
        perf_weights: dict[str, float] | None = None,
    ):
        self._logger = logger
        self._param_codes = param_codes
        self._perf_codes = perf_codes
        self._param_categories = param_categories  # code → list of categories (empty = numeric)
        self._perf_weights = perf_weights or {}

    @property
    def enabled(self) -> bool:
        return self._logger._console_output_enabled

    def _print(self, msg: str) -> None:
        """Print via logger so it respects console_output_enabled."""
        self._logger.console_summary(msg)

    # ── Formatting helpers ──────────────────────────────────────────────

    def _format_params(self, params: dict[str, Any]) -> str:
        """Format parameters as a compact key=value string."""
        parts: list[str] = []
        for code in self._param_codes:
            val = params.get(code)
            if val is None:
                continue
            if code in self._param_categories:
                # Categorical: show first 3 chars
                parts.append(f"{code[:3]}={str(val)[:3]}")
            elif isinstance(val, float):
                # Auto-format: small values get .2f, large get .1f
                if abs(val) < 1.0:
                    parts.append(f"{code[0]}={val:.2f}")
                else:
                    parts.append(f"{code[:3]}={val:.1f}")
            elif isinstance(val, int):
                parts.append(f"{code[:3]}={val}")
            else:
                parts.append(f"{code[:3]}={val}")
        return "  ".join(parts)

    def _format_perf(self, perf: Mapping[str, float | None]) -> str:
        """Format performance metrics with ANSI coloring."""
        parts: list[str] = []
        for code in self._perf_codes:
            val = perf.get(code)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            v = float(val)
            short = code[:3]
            parts.append(f"{short}={_score_color(v)}{v:.3f}{_R}")
        return "  ".join(parts)

    # ── Phase headers ───────────────────────────────────────────────────

    def print_phase_header(self, num: int, title: str, subtitle: str = "") -> None:
        """Print a phase banner."""
        if not self.enabled:
            return
        bar = "━" * _W
        lines = [f"\n{_B}{_C}{bar}{_R}"]
        lines.append(f"{_B}{_C}  PHASE {num}{_R}{_B} ▸ {title}{_R}")
        if subtitle:
            lines.append(f"  {_D}{subtitle}{_R}")
        lines.append(f"{_B}{_C}{bar}{_R}")
        self._print("\n".join(lines))

    def print_section(self, title: str) -> None:
        """Print a lightweight section label."""
        if not self.enabled:
            return
        self._print(f"\n  {_B}▸ {title}{_R}")

    def print_done(self, plots_dir: str = "./plots/") -> None:
        if not self.enabled:
            return
        bar = "━" * _W
        self._print(f"\n{_B}{_C}{bar}{_R}\n{_B}{_C}  Done.  Plots saved to {plots_dir}{_R}\n{_B}{_C}{bar}{_R}\n")

    # ── Experiment rows ─────────────────────────────────────────────────

    def print_experiment_row(
        self,
        exp_code: str,
        params: dict[str, Any],
        perf: Mapping[str, float | None],
        suffix: str = "",
    ) -> None:
        """Print one experiment result row: params + performance scores."""
        if not self.enabled:
            return
        meta = self._format_params(params)
        perf_s = self._format_perf(perf)
        tail = f"  {_D}{suffix}{_R}" if suffix else ""
        self._print(f"  {_B}{exp_code:<14}{_R}{_D}{meta}{_R}  {perf_s}{tail}")

    def print_exploration_row(
        self,
        exp_code: str,
        params: dict[str, Any],
        perf: Mapping[str, float | None],
        uncertainty: float,
        score: float,
    ) -> None:
        """Exploration experiment result: params + perf + uncertainty + acquisition score."""
        if not self.enabled:
            return
        meta = self._format_params(params)
        perf_s = self._format_perf(perf)
        u_s = f"{_score_color(uncertainty)}{uncertainty:.3f}{_R}"
        obj_s = f"{_score_color(score)}{score:.3f}{_R}"
        self._print(
            f"  {_B}{exp_code:<14}{_R}{_D}{meta}{_R}  {perf_s}  "
            f"u={u_s}  obj={obj_s}"
        )

    def print_inference_row(
        self,
        exp_code: str,
        params: dict[str, Any],
        perf: Mapping[str, float | None],
        score: float,
    ) -> None:
        """Inference experiment result: params + perf + objective score."""
        if not self.enabled:
            return
        meta = self._format_params(params)
        perf_s = self._format_perf(perf)
        obj_s = f"{_score_color(score)}{score:.3f}{_R}"
        self._print(
            f"  {_B}{exp_code:<14}{_R}{_D}{meta}{_R}  {perf_s}  "
            f"obj={obj_s}"
        )

    def print_proposal_row(
        self,
        proposals: list[dict[str, Any]],
        perf: float,
        unc: float,
        obj: float,
    ) -> None:
        """Print exploration proposal: objective components."""
        if not self.enabled:
            return
        perf_s = f"perf={_score_color(perf)}{perf:.3f}{_R}"
        unc_s = f"unc={_score_color(unc)}{unc:.3f}{_R}"
        obj_s = f"obj={_score_color(obj)}{obj:.3f}{_R}"
        self._print(f"  {_C}>{_R} {perf_s}  {unc_s}  {obj_s}")

    def print_proposal_table(
        self,
        title: str,
        exp_codes: list[str],
        params_list: list[dict[str, Any]],
        *,
        exclude_codes: set[str] | None = None,
        short_names: dict[str, str] | None = None,
    ) -> None:
        """Print a parameter table with one row per experiment."""
        if not self.enabled or not exp_codes:
            return
        exclude = exclude_codes or set()
        shorts = short_names or {}
        columns = [c for c in self._param_codes if c not in exclude]
        if not columns:
            return
        headers = [shorts.get(c, c[:8]) for c in columns]

        self._print(f"\n  {_D}{title}{_R}")
        self._print(f"  {'code':<16s}" + "  ".join(f"{h:>8s}" for h in headers))
        self._print(f"  {'\u2500' * (16 + 10 * len(headers))}")

        for code, params in zip(exp_codes, params_list):
            vals: list[str] = []
            for col in columns:
                v = params.get(col, "\u2014")
                if isinstance(v, float):
                    vals.append(f"{v:8.4f}")
                elif isinstance(v, int):
                    vals.append(f"{v:>8d}")
                else:
                    vals.append(f"{str(v):>8s}")
            self._print(f"  {code:<16s}" + "  ".join(vals))

        self._logger.console_new_line()

    def print_schedule_table(
        self,
        param_code: str,
        exp_codes: list[str],
        per_exp_values: list[list[float]],
        *,
        short_name: str | None = None,
    ) -> None:
        """Print a per-step trajectory table for one schedule parameter."""
        if not self.enabled or not per_exp_values:
            return
        n_steps = max(len(v) for v in per_exp_values)
        step_headers = [str(i + 1) for i in range(n_steps)]
        label = short_name or param_code

        self._print(f"\n  {_D}{label}{_R}")
        self._print(f"  {'code':<16s}" + "  ".join(f"{h:>7s}" for h in step_headers))
        self._print(f"  {'\u2500' * (16 + 9 * n_steps)}")

        for code, vals in zip(exp_codes, per_exp_values):
            val_strs = [f"{v:7.3f}" for v in vals]
            self._print(f"  {code:<16s}" + "  ".join(val_strs))

        self._logger.console_new_line()

    def print_optimizer_stats(self, n_starts: int, n_evals: int) -> None:
        """Dim optimizer summary line."""
        if not self.enabled:
            return
        self._print(f"  {_D}    {n_starts} starts · {n_evals} evals{_R}")

    # ── Training ────────────────────────────────────────────────────────

    def print_training_summary(self, feature_metrics: dict[str, dict[str, float]]) -> None:
        """Print R², R²_adj, and MAE per feature in a table."""
        if not self.enabled:
            return
        has_adj = any('r2_adj' in m for m in feature_metrics.values())
        has_mae = any('mae' in m for m in feature_metrics.values())

        header = f"  {'Feature':<30s}  {'R²':>8s}"
        if has_adj:
            header += f"  {'R²_adj':>8s}"
        if has_mae:
            header += f"  {'MAE':>10s}"
        self._print(f"\n  {_B}Model quality{_R}")
        self._print(header)
        self._print(f"  {'─' * len(header)}")
        for name, metrics in feature_metrics.items():
            r2 = metrics.get('r2', 0.0)
            line = f"  {name:<30s}  {r2:8.4f}"
            if has_adj:
                r2_adj = metrics.get('r2_adj')
                if r2_adj is not None:
                    line += f"  {r2_adj:8.4f}"
                else:
                    line += f"  {'—':>8s}"
            if has_mae:
                mae = metrics.get('mae', 0.0)
                line += f"  {mae:10.3f}"
            self._print(line)

    # ── Adaptation ──────────────────────────────────────────────────────

    def print_adaptation_row(
        self,
        layer_idx: int,
        speed_before: float,
        deviation: float,
        speed_after: float | None = None,
        n_evals: int | None = None,
    ) -> None:
        """Print one layer's adaptation step result."""
        if not self.enabled:
            return
        dev_color = _score_color(max(0.0, 1.0 - deviation / 0.003))
        dev_str = f"{dev_color}{deviation:.5f}{_R}"
        evals_str = f"  {_D}({n_evals} evals){_R}" if n_evals is not None else ""

        if speed_after is not None:
            spd_str = f"{speed_before:.1f} → {_B}{speed_after:.1f}{_R} mm/s"
            self._print(f"  Layer {layer_idx}  │  speed={spd_str}  │  dev={dev_str}{evals_str}")
        else:
            spd_str = f"{speed_before:.1f} mm/s{' ' * 13}"
            self._print(f"  Layer {layer_idx}  │  speed={spd_str}  │  dev={dev_str}")

    # ── Phase summaries ─────────────────────────────────────────────────

    def print_phase_summary(
        self,
        experiments: list[tuple[str, dict[str, Any], dict[str, float]]],
    ) -> None:
        """Print a one-line best-result summary at the end of a phase."""
        if not self.enabled or not experiments:
            return
        weights = self._perf_weights
        best_code, best_params, best_perf = max(
            experiments, key=lambda x: _combined_score(x[2], weights),
        )
        perf_parts = []
        for code in self._perf_codes:
            val = best_perf.get(code, float("nan"))
            if not math.isnan(val):
                perf_parts.append(f"{code[:3]}={_G}{val:.3f}{_R}")
        perf_s = "  ".join(perf_parts)

        # Show first two categorical params as context
        context_parts = []
        for code in self._param_codes:
            if code in self._param_categories:
                context_parts.append(str(best_params.get(code, "?")))
        context = ", ".join(context_parts)

        self._print(
            f"\n  {_G}✓{_R} Best: {_B}{best_code}{_R} "
            f"— {perf_s}  "
            f"{_D}({context}){_R}"
        )

    def print_run_summary(
        self,
        perf_history: list[tuple[dict[str, Any], dict[str, float]]],
        phases: list[str],
        exp_codes: list[str],
    ) -> None:
        """Print a final table comparing best results across all phases."""
        if not self.enabled or not perf_history:
            return
        weights = self._perf_weights

        all_scored = [
            (code, params, perf, phase)
            for (params, perf), code, phase
            in zip(perf_history, exp_codes, phases)
        ]
        best = max(all_scored, key=lambda x: _combined_score(x[2], weights))
        infer_items = [x for x in all_scored if x[3] == "inference"]
        best_infer = max(infer_items, key=lambda x: _combined_score(x[2], weights)) if infer_items else None

        bar = "─" * _W
        lines = [f"\n  {_B}Run Summary{_R}", f"  {_D}{bar}{_R}"]

        # Build header from schema
        param_header = "  ".join(f"{c[:5]:>5}" for c in self._param_codes if c not in self._param_categories)
        cat_header = "  ".join(f"{c[:8]:<8}" for c in self._param_codes if c in self._param_categories)
        perf_header = "  ".join(f"{c[:6]:>6}" for c in self._perf_codes)
        lines.append(f"  {_D}  {'Experiment':<16} {cat_header}  {param_header}  {perf_header}  {'combined':>8}{_R}")
        lines.append(f"  {_D}{bar}{_R}")

        def _row(label: str, params: dict[str, Any], perf: dict[str, float]) -> str:
            comb = _combined_score(perf, weights)
            cat_vals = "  ".join(
                f"{str(params.get(c, '?')):<8}" for c in self._param_codes if c in self._param_categories
            )
            num_vals = "  ".join(
                f"{float(params.get(c, 0)):>5.2f}" if abs(float(params.get(c, 0))) < 1 else f"{float(params.get(c, 0)):>5.1f}"
                for c in self._param_codes if c not in self._param_categories
            )
            perf_vals = "  ".join(
                f"{_score_color(float(perf.get(c, 0)))}{float(perf.get(c, 0)):>6.3f}{_R}"
                for c in self._perf_codes
            )
            return (
                f"  {_B}{label:<16}{_R}"
                f"{cat_vals}  {num_vals}  "
                f"{perf_vals}  "
                f"{_score_color(comb)}{comb:>8.3f}{_R}"
            )

        lines.append(_row("Best overall", best[1], best[2]))
        if best_infer:
            lines.append(_row("Best inference", best_infer[1], best_infer[2]))
        lines.append(f"  {_D}{bar}{_R}")
        self._print("\n".join(lines))
