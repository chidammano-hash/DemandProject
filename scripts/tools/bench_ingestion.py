"""Ingestion baseline benchmark harness (US2).

Times each stage of the data-ingestion pipeline and emits structured timing
records, flagging stages slower than the ``function_slow_s`` threshold in
``config/platform/perf_config.yaml``. Use it to capture the before/after
numbers that the Phase-3 performance stories (US8-US11) are measured against.

Library use (unit-tested)::

    from scripts.tools.bench_ingestion import time_stage, flag_slow, to_markdown
    timings = [time_stage("sales", "load", load_sales)]
    print(to_markdown(flag_slow(timings)))

CLI use (operator, against a live DB)::

    python scripts/tools/bench_ingestion.py --mode full
    python scripts/tools/bench_ingestion.py --mode refresh
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.services.perf_profiler import _load_perf_config  # noqa: E402 — after sys.path bootstrap

logger = logging.getLogger(__name__)

# Default threshold (seconds) if perf_config.yaml is unavailable.
_DEFAULT_SLOW_S = 10


@dataclass
class StageTiming:
    """One timed pipeline stage."""

    domain: str
    stage: str
    seconds: float
    slow: bool = False


def time_stage(domain: str, stage: str, fn, *args, **kwargs) -> StageTiming:
    """Run ``fn`` and return a :class:`StageTiming` capturing its wall time."""
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return StageTiming(domain=domain, stage=stage, seconds=elapsed)


def time_command(domain: str, stage: str, cmd: list[str]) -> StageTiming:
    """Run a shell command (e.g. a make target) and time it."""
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    elapsed = time.perf_counter() - t0
    return StageTiming(domain=domain, stage=stage, seconds=elapsed)


def _slow_threshold_s() -> float:
    cfg = _load_perf_config()
    return cfg.get("thresholds", {}).get("function_slow_s", _DEFAULT_SLOW_S)


def flag_slow(timings: list[StageTiming], threshold_s: float | None = None) -> list[StageTiming]:
    """Mark each timing ``slow`` when it exceeds the threshold.

    When ``threshold_s`` is omitted it is read from perf_config.yaml's
    ``thresholds.function_slow_s`` (never hardcoded at the call site).
    """
    if threshold_s is None:
        threshold_s = _slow_threshold_s()
    for t in timings:
        t.slow = t.seconds > threshold_s
    return timings


def to_markdown(timings: list[StageTiming]) -> str:
    """Render timings as a Markdown table for docs/RUNBOOK.md."""
    lines = ["| Domain | Stage | Seconds | Slow |", "|---|---|---|---|"]
    for t in timings:
        marker = "⚠" if t.slow else ""
        lines.append(f"| {t.domain} | {t.stage} | {t.seconds:.2f} | {marker} |")
    return "\n".join(lines)


# Canonical stages timed end-to-end by the CLI (operator runs against a live DB).
_FULL_STAGES = [
    ("all", "normalize-all", ["make", "normalize-all"]),
    ("all", "load-all", ["make", "load-all"]),
    ("all", "refresh-mvs-tiered", ["make", "refresh-mvs-tiered"]),
]
_REFRESH_STAGES = [
    ("all", "pipeline-refresh", ["make", "pipeline-refresh"]),
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Benchmark ingestion pipeline stages")
    parser.add_argument("--mode", choices=["full", "refresh"], default="full")
    args = parser.parse_args()

    stages = _FULL_STAGES if args.mode == "full" else _REFRESH_STAGES
    timings: list[StageTiming] = []
    for domain, stage, cmd in stages:
        logger.info("Timing %s/%s ...", domain, stage)
        timings.append(time_command(domain, stage, cmd))

    flag_slow(timings)
    logger.info("\n%s", to_markdown(timings))


if __name__ == "__main__":
    main()
