"""Schema-aware ANSI console reporter for agent step output."""

import math
from collections.abc import Mapping
from typing import Any

from .logger import PfabLogger

# ANSI codes
_R  = "\033[0m"   # reset
_B  = "\033[1m"   # bold
_D  = "\033[2m"   # dim
_G  = "\033[32m"  # green
_Y  = "\033[33m"  # yellow
_RD = "\033[31m"  # red
_C  = "\033[36m"  # cyan
_W  = 58          # banner width


def _score_color(v: float) -> str:
    if v >= 0.70:
        return _G
    if v >= 0.45:
        return _Y
    return _RD


from .metrics import combined_score as _combined_score


class ProgressBar:
    """Inline ANSI progress bar for optimizer iterations.

    Usage::

        bar = ProgressBar("Optimizing", max_iter=100)
        # as DE callback:
        def callback(xk, convergence):
            bar.step()
        # after optimizer completes:
        bar.finish(nfev=result.nfev)
    """

    def __init__(self, label: str = "Optimizing", max_iter: int = 100, bar_len: int = 12):
        self._label = label
        self._max = max_iter
        self._len = bar_len
        self._i = 0

    def step(self, i: int | None = None, obj: float | None = None) -> None:
        """Advance by one (or jump to ``i``) and redraw. Optional ``obj`` is
        the current best objective value, displayed live (3 decimals)."""
        self._i = (self._i + 1) if i is None else i
        filled = int(self._len * min(self._i / self._max, 1.0))
        bar = "\u2588" * filled + "\u2591" * (self._len - filled)
        obj_str = f"  obj={obj:.3f}" if obj is not None else ""
        print(f"  {self._label:<14s} [{bar}] {_D}{self._i}/{self._max}{obj_str}{_R}", end="\r", flush=True)

    def finish(self, nfev: int | None = None, suffix: str = "") -> None:
        """Fill the bar completely and print final info."""
        bar = "\u2588" * self._len
        info = suffix.strip() if suffix else ""
        if nfev is not None:
            info += f"  nfev={nfev}" if info else f"nfev={nfev}"
        if not info:
            info = f"{self._i}/{self._max}"
        # Pad to overwrite any leftover characters from the longer step() line
        # that may have included an obj=X.XXX suffix during optimization.
        print(f"{_G}\u2713{_R} {self._label:<14s} [{bar}] {_D}{info}{_R}            ")


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

    def print_params_line(self, params: dict[str, Any]) -> None:
        """Print proposed parameters in grey on own line, with trailing blank line."""
        if not self.enabled:
            return
        parts: list[str] = []
        for code, val in params.items():
            if isinstance(val, float):
                fmt = f"{val:.2f}" if abs(val) < 1 else f"{val:.1f}"
                parts.append(f"{code}={fmt}")
            elif isinstance(val, int):
                parts.append(f"{code}={val}")
            else:
                parts.append(f"{code}={val}")
        if parts:
            self._print(f"    {_D}\u2192 {', '.join(parts)}{_R}")
        if self._logger._console_output_enabled:
            print("", flush=True)

    def print_schedule_table(
        self,
        proposals: list[dict[str, Any]],
        tunable_codes: set[str],
        schedule_configs: dict[str, str],
    ) -> None:
        """Print per-step parameter schedule as a vertical table."""
        if not self.enabled or not proposals:
            return
        # Identify which params vary per layer vs fixed across layers
        sched_params = set(schedule_configs.keys())
        fixed_params: dict[str, str] = {}
        for code in tunable_codes:
            if code not in sched_params:
                val = proposals[0].get(code)
                if val is not None:
                    fmt = f"{val:.2f}" if isinstance(val, float) and abs(val) < 1 else f"{val:.1f}" if isinstance(val, float) else str(val)
                    fixed_params[code] = fmt

        # Fixed params line
        if fixed_params:
            fixed_s = ", ".join(f"{k}={v}" for k, v in fixed_params.items())
            self._print(f"    {_D}\u2192 {fixed_s}{_R}")

        # Per-step table for schedule params
        sched_codes = sorted(sched_params & tunable_codes)
        if sched_codes:
            header = f"    {_D}{'step':<7s}"
            for code in sched_codes:
                header += f"  {code:>8s}"
            header += _R
            self._print(header)
            for i, p in enumerate(proposals):
                row = f"    {_D}{i+1:<7d}"
                for code in sched_codes:
                    val = p.get(code, 0)
                    row += f"  {float(val):8.1f}" if isinstance(val, (int, float)) else f"  {str(val):>8s}"
                row += _R
                self._print(row)

        if self._logger._console_output_enabled:
            print("", flush=True)

    def print_optimizer_stats(self, n_starts: int, n_evals: int) -> None:
        """Dim optimizer summary line."""
        if not self.enabled:
            return
        self._print(f"  {_D}    {n_starts} starts · {n_evals} evals{_R}")

    # ── Training ────────────────────────────────────────────────────────

    def print_training_summary(self, feature_metrics: dict[str, dict[str, float]]) -> None:
        """Print R² and R²_adj scores for each prediction model output."""
        if not self.enabled:
            return
        parts: list[str] = []
        for name, metrics in feature_metrics.items():
            r2 = metrics.get('r2', 0.0)
            r2_adj = metrics.get('r2_adj')
            r2_str = f"R²={_score_color(max(0.0, r2))}{r2:.3f}{_R}"
            if r2_adj is not None:
                adj_str = f"R²_adj={_score_color(max(0.0, r2_adj))}{r2_adj:.3f}{_R}"
                parts.append(f"{name}: {r2_str}  {adj_str}")
            else:
                parts.append(f"{name}: {r2_str}")
        self._print(f"\n  {_B}Model quality{_R}  {'  '.join(parts)}")

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
