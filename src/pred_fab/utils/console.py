"""Console output helpers for PFAB."""

_G = "\033[32m"   # green
_D = "\033[2m"    # dim
_B = "\033[1m"    # bold
_C = "\033[36m"   # cyan
_R = "\033[0m"    # reset


class ProgressBar:
    """Inline ANSI progress bar for optimizer iterations.

    Shows iteration count and objective while running, then a clean
    summary line on finish. No fake max — the bar fills based on
    improvement plateaus, not a predetermined budget.

    Usage::

        bar = ProgressBar("Optimizing")
        for i in range(n):
            bar.step(obj=current_obj)
        bar.finish()
    """

    def __init__(self, label: str = "Optimizing", bar_len: int = 12, max_iter: int | None = None):
        self._label = label
        self._len = bar_len
        self._i = 0
        self._best_obj: float | None = None
        self._prev_obj: float | None = None
        self._no_improve = 0
        self._max_iter = max_iter

    def step(self, i: int | None = None, obj: float | None = None) -> None:
        """Advance by one (or jump to ``i``) and redraw."""
        import sys
        self._i = (self._i + 1) if i is None else i

        if obj is not None:
            improved = self._best_obj is not None and obj < self._best_obj - 1e-15
            if improved or self._best_obj is None:
                self._best_obj = obj
                self._no_improve = 0
            else:
                self._no_improve += 1
            color = _G if improved else _D
            obj_str = f"  {color}obj={obj:.3f}{_R}"
            self._prev_obj = obj
        else:
            obj_str = ""

        if self._max_iter is not None:
            filled = int(self._len * min(self._i / self._max_iter, 1.0))
        else:
            filled = min(self._i, self._len)
        bar = "█" * filled + "░" * (self._len - filled)
        sys.stdout.write(f"\r  {self._label:<14s} [{bar}] {_D}{self._i} iters{_R}{obj_str}            ")
        sys.stdout.flush()

    def finish(self, nfev: int | None = None, suffix: str = "") -> None:
        """Print final summary line."""
        bar = "█" * self._len
        info = suffix.strip() if suffix else ""
        if not info:
            parts = []
            parts.append(f"{self._i} iters")
            if self._best_obj is not None:
                parts.append(f"obj={self._best_obj:.3f}")
            info = "  ".join(parts)
        import sys
        sys.stdout.write(f"\r{_G}✓{_R} {self._label:<14s} [{bar}] {_D}{info}{_R}            \n")
        sys.stdout.flush()


class ConsoleReporter:
    """Structured console output for calibration phases and steps.

    Minimal ANSI formatting: bold+cyan headers, dimmed sub-steps.
    Called by agent or CalibrationSystem — no schema/domain awareness.
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def phase_header(self, num: int, title: str) -> None:
        if not self._enabled:
            return
        lines: list[str] = []
        lines.append("")
        lines.append(f"{_B}{_C}  PHASE {num}{_R}{_B} ▸ {title}{_R}")
        print("\n".join(lines))

    def step_line(self, text: str) -> None:
        if not self._enabled:
            return
        print(f"  {_D}{text}{_R}")

    def sub_header(self, title: str) -> None:
        if not self._enabled:
            return
        self._print(f"\n  {_B}▸ {title}{_R}")

    def info(self, text: str) -> None:
        if not self._enabled:
            return
        self._print(f"  {text}")

    def _print(self, text: str) -> None:
        print(text)
