"""Lightweight inline profiler for PFAB hot paths.

Singleton profiler with named sections that accumulate total time + call count.
No-op when disabled — zero cost in production builds. Enable via env var
``PFAB_PROFILE=1`` or programmatically via ``profiler.enable()``. Print a
breakdown after a run via ``profiler.report()``.

Usage::

    from pred_fab.utils import profiler
    profiler.enable()
    # ... run workload ...
    print(profiler.report())

Sections aggregate by name across all callers. The hot-path sections in
``CalibrationSystem`` / ``PredictionSystem`` / ``EvaluationSystem`` /
``OptimizationEngine`` are pre-instrumented; new code can wrap any block
with ``with profiler.section("my-section"): ...``.
"""

import os
import time
from contextlib import contextmanager
from typing import Iterator


class _Profiler:
    """Singleton time accumulator. Use the module-level ``profiler`` instance."""

    def __init__(self) -> None:
        self.enabled: bool = os.environ.get("PFAB_PROFILE", "").lower() in ("1", "true", "yes")
        # name → [total_time_seconds, call_count]
        self._sections: dict[str, list[float | int]] = {}

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def reset(self) -> None:
        """Clear all accumulated timings."""
        self._sections.clear()

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Time the wrapped block and accumulate into ``name``.

        No-op (zero overhead) when the profiler is disabled.
        """
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            entry = self._sections.setdefault(name, [0.0, 0])
            entry[0] = float(entry[0]) + dt  # type: ignore[assignment]
            entry[1] = int(entry[1]) + 1     # type: ignore[assignment]

    def report(self, sort_by: str = "total") -> str:
        """Return a sorted text table of all recorded sections.

        ``sort_by``: ``"total"`` (default — total wall time) or ``"avg"``
        (mean per call) or ``"count"`` (call count).
        """
        if not self._sections:
            return (
                "(profiler: no sections recorded — was profiler.enable() "
                "called and PFAB_PROFILE=1 set?)"
            )

        max_total = max(float(v[0]) for v in self._sections.values()) or 1.0
        rows: list[tuple[str, float, int, float, float]] = []
        for name, (total, count) in self._sections.items():
            count_i = int(count)
            total_f = float(total)
            avg = total_f / count_i if count_i > 0 else 0.0
            pct = 100.0 * total_f / max_total
            rows.append((name, total_f, count_i, avg, pct))

        sort_keys = {"total": 1, "avg": 3, "count": 2}
        key = sort_keys.get(sort_by, 1)
        rows.sort(key=lambda r: r[key], reverse=True)

        name_w = max(len(name) for name, *_ in rows)
        name_w = max(name_w, 30)
        lines = [
            f"{'Section'.ljust(name_w)}  {'Total':>10}  {'Calls':>8}  {'Avg/call':>12}  {'%':>6}",
            "─" * (name_w + 50),
        ]
        for name, total, count, avg, pct in rows:
            lines.append(
                f"{name.ljust(name_w)}  {self._fmt_time(total):>10}  "
                f"{count:>8d}  {self._fmt_time(avg):>12}  {pct:>5.1f}%"
            )
        return "\n".join(lines)

    @staticmethod
    def _fmt_time(t: float) -> str:
        if t >= 1.0:
            return f"{t:.2f}s"
        elif t >= 1e-3:
            return f"{t * 1e3:.2f}ms"
        else:
            return f"{t * 1e6:.1f}us"


# Module-level singleton — import as ``from pred_fab.utils import profiler``.
profiler = _Profiler()
