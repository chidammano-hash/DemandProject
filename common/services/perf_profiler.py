"""Performance profiler with full isolation and rule-based suggestions.

Provides decorators, context managers, and a suggestion engine for profiling
scripts, API endpoints, and pipeline stages. All DB connections are wrapped
in read-only, rollback-only transactions for production safety.
"""
from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import json
import logging
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

from common.core.sql_helpers import _elapsed
from common.core.utils import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class QueryMetrics:
    """Captured metrics for a single DB query."""
    sql_preview: str
    duration_ms: float
    rows_affected: int | None = None
    is_executemany: bool = False
    timestamp: float = 0.0


@dataclasses.dataclass
class SectionMetrics:
    """Captured metrics for a profiled code section."""
    name: str
    wall_time_s: float = 0.0
    cpu_time_s: float = 0.0
    memory_peak_mb: float = 0.0
    memory_delta_mb: float = 0.0
    queries: list[QueryMetrics] = dataclasses.field(default_factory=list)
    children: list[SectionMetrics] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class QuerySummary:
    """Aggregate query statistics for a profiling run."""
    total_queries: int = 0
    total_query_time_ms: float = 0.0
    slowest_query_ms: float = 0.0
    slowest_query_sql: str = ""
    n_plus_1_detected: bool = False
    unbatched_inserts: int = 0


@dataclasses.dataclass
class Suggestion:
    """A performance improvement suggestion."""
    severity: str  # "critical" | "warning" | "info"
    category: str  # "query" | "memory" | "cpu" | "pattern"
    message: str
    section: str | None = None
    evidence: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PerfReport:
    """Complete performance profiling report."""
    script_name: str
    started_at: str = ""
    total_wall_time_s: float = 0.0
    total_cpu_time_s: float = 0.0
    peak_memory_mb: float = 0.0
    sections: list[SectionMetrics] = dataclasses.field(default_factory=list)
    query_summary: QuerySummary = dataclasses.field(default_factory=QuerySummary)
    suggestions: list[Suggestion] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict representation."""
        return dataclasses.asdict(self)

    def to_json(self, path: Path) -> None:
        """Write report as JSON to *path*."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("Report written to %s", path)

    def to_console(self) -> str:
        """Human-readable console summary."""
        lines: list[str] = []
        lines.append(f"=== Performance Report: {self.script_name} ===")
        lines.append(
            f"Total: {self.total_wall_time_s:.1f}s wall | "
            f"{self.total_cpu_time_s:.1f}s CPU | "
            f"{self.peak_memory_mb:.0f}MB peak memory"
        )
        qs = self.query_summary
        lines.append(
            f"Queries: {qs.total_queries} total | "
            f"{qs.total_query_time_ms / 1000:.1f}s total | "
            f"slowest: {qs.slowest_query_ms / 1000:.1f}s"
        )
        lines.append("")
        lines.append("Sections:")
        for sec in self.sections:
            q_info = f"{len(sec.queries)} query"
            if sec.queries:
                q_time = sum(q.duration_ms for q in sec.queries)
                q_info += f" ({q_time / 1000:.1f}s)"
            lines.append(
                f"  {sec.name:<24} {sec.wall_time_s:.1f}s wall | "
                f"{sec.cpu_time_s:.1f}s CPU | "
                f"+{sec.memory_delta_mb:.0f}MB | {q_info}"
            )
            for child in sec.children:
                lines.append(
                    f"    {child.name:<22} {child.wall_time_s:.1f}s wall | "
                    f"{child.cpu_time_s:.1f}s CPU | "
                    f"+{child.memory_delta_mb:.0f}MB"
                )
        if self.suggestions:
            lines.append("")
            lines.append(f"Suggestions ({len(self.suggestions)}):")
            for s in self.suggestions:
                lines.append(f"  [{s.severity.upper()}] {s.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Collector (thread/async-safe via ContextVar)
# ---------------------------------------------------------------------------

_active_collector: contextvars.ContextVar[PerfCollector | None] = (
    contextvars.ContextVar("_active_collector", default=None)
)


class PerfCollector:
    """Accumulates profiling metrics for a single script run."""

    def __init__(self, script_name: str) -> None:
        self.script_name = script_name
        self.root_sections: list[SectionMetrics] = []
        self._section_stack: list[SectionMetrics] = []
        self._all_queries: list[QueryMetrics] = []
        self._wall_start: float = 0.0
        self._cpu_start: float = 0.0
        self._started_at: str = ""

    def push_section(self, name: str) -> SectionMetrics:
        section = SectionMetrics(name=name)
        if self._section_stack:
            self._section_stack[-1].children.append(section)
        else:
            self.root_sections.append(section)
        self._section_stack.append(section)
        return section

    def pop_section(self) -> SectionMetrics | None:
        if self._section_stack:
            return self._section_stack.pop()
        return None

    def record_query(self, qm: QueryMetrics) -> None:
        self._all_queries.append(qm)
        if self._section_stack:
            self._section_stack[-1].queries.append(qm)

    @property
    def current_section(self) -> SectionMetrics | None:
        return self._section_stack[-1] if self._section_stack else None


# ---------------------------------------------------------------------------
# Context managers & decorators
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def profile_script(script_name: str) -> Iterator[PerfCollector]:
    """Top-level context manager for profiling a script run.

    Starts tracemalloc, creates a PerfCollector, and yields it.
    On exit, stops tracemalloc and records totals.
    """
    collector = PerfCollector(script_name)
    collector._started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    collector._wall_start = time.perf_counter()
    collector._cpu_start = time.process_time()

    tracemalloc.start()
    token = _active_collector.set(collector)
    try:
        yield collector
    finally:
        _active_collector.reset(token)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        collector._wall_end = time.perf_counter()
        collector._cpu_end = time.process_time()
        collector._peak_memory_bytes = peak


@contextlib.contextmanager
def profiled_section(name: str, metadata: dict[str, Any] | None = None) -> Iterator[SectionMetrics]:
    """Profile a named code block within a script."""
    collector = _active_collector.get()
    if collector is None:
        # No active profiler — yield a dummy section
        yield SectionMetrics(name=name)
        return

    section = collector.push_section(name)
    if metadata:
        section.metadata = metadata

    mem_before = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        yield section
    finally:
        section.wall_time_s = time.perf_counter() - wall_start
        section.cpu_time_s = time.process_time() - cpu_start
        if tracemalloc.is_tracing():
            mem_current, mem_peak = tracemalloc.get_traced_memory()
            section.memory_delta_mb = (mem_current - mem_before) / (1024 * 1024)
            section.memory_peak_mb = mem_peak / (1024 * 1024)
        collector.pop_section()


def profile_function(func: Callable | None = None, *, name: str | None = None) -> Callable:
    """Decorator that profiles wall time, CPU time, and memory delta.

    Usage::

        @profile_function
        def my_func(): ...

        @profile_function(name="custom_name")
        def my_func(): ...
    """
    def decorator(fn: Callable) -> Callable:
        section_name = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with profiled_section(section_name):
                return fn(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# DB connection wrapper (read-only, rollback-only isolation)
# ---------------------------------------------------------------------------

def wrap_connection(conn: Any, *, readonly: bool = True) -> Any:
    """Wrap a psycopg3 connection to auto-log query timings.

    Production safety:
    - Sets ``default_transaction_read_only = true`` (PostgreSQL rejects writes)
    - Always calls ``conn.rollback()`` — never commits

    Returns the same connection object (mutated in place).
    """
    if readonly:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = true")
        conn.autocommit = False

    original_execute = conn.cursor().__class__.execute
    original_executemany = conn.cursor().__class__.executemany

    class _ProfiledCursor:
        """Thin wrapper that logs query timings to the active collector."""

        def __init__(self, real_cursor: Any) -> None:
            self._cur = real_cursor

        def execute(self, query: Any, params: Any = None, **kwargs: Any) -> Any:
            return self._timed_exec(
                self._cur.execute, query, params, is_many=False, **kwargs
            )

        def executemany(self, query: Any, params_seq: Any, **kwargs: Any) -> Any:
            return self._timed_exec(
                self._cur.executemany, query, params_seq, is_many=True, **kwargs
            )

        def _timed_exec(
            self, method: Callable, query: Any, params: Any,
            *, is_many: bool, **kwargs: Any,
        ) -> Any:
            collector = _active_collector.get()
            t0 = time.perf_counter()
            try:
                result = method(query, params, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if collector is not None:
                    sql_text = str(query)[:200] if query else ""
                    qm = QueryMetrics(
                        sql_preview=sql_text,
                        duration_ms=elapsed_ms,
                        rows_affected=getattr(self._cur, "rowcount", None),
                        is_executemany=is_many,
                        timestamp=time.time(),
                    )
                    collector.record_query(qm)
            return result

        def fetchone(self) -> Any:
            return self._cur.fetchone()

        def fetchall(self) -> Any:
            return self._cur.fetchall()

        def fetchmany(self, size: int | None = None) -> Any:
            return self._cur.fetchmany(size)

        def close(self) -> None:
            self._cur.close()

        def copy(self, *args: Any, **kwargs: Any) -> Any:
            return self._cur.copy(*args, **kwargs)

        @property
        def description(self) -> Any:
            return self._cur.description

        @property
        def rowcount(self) -> int:
            return self._cur.rowcount

        @property
        def statusmessage(self) -> Any:
            return getattr(self._cur, "statusmessage", None)

        def __enter__(self) -> _ProfiledCursor:
            self._cur.__enter__()
            return self

        def __exit__(self, *exc: Any) -> None:
            self._cur.__exit__(*exc)

        def __iter__(self) -> Any:
            return iter(self._cur)

    # Store original cursor factory and replace with profiled version
    _orig_cursor = conn.cursor

    def _profiled_cursor(*args: Any, **kwargs: Any) -> _ProfiledCursor:
        return _ProfiledCursor(_orig_cursor(*args, **kwargs))

    conn.cursor = _profiled_cursor  # type: ignore[assignment]
    conn._perf_wrapped = True  # type: ignore[attr-defined]
    return conn


@contextlib.contextmanager
def auto_wrap_connections(*, readonly: bool = False) -> Iterator[None]:
    """Monkey-patch ``psycopg.connect`` so every new connection is auto-wrapped.

    All queries executed by the target script are captured by the active
    :class:`PerfCollector`.  Set *readonly=False* (default) to allow normal
    writes — the wrapper only adds timing instrumentation.  Set
    *readonly=True* to enforce read-only isolation (production safety).
    """
    try:
        import psycopg  # type: ignore[import-untyped]
    except ImportError:
        yield  # psycopg not available — nothing to patch
        return

    _original_connect = psycopg.connect

    def _patched_connect(*args: Any, **kwargs: Any) -> Any:
        conn = _original_connect(*args, **kwargs)
        if not getattr(conn, "_perf_wrapped", False):
            conn = wrap_connection(conn, readonly=readonly)
        return conn

    psycopg.connect = _patched_connect  # type: ignore[assignment]
    try:
        yield
    finally:
        psycopg.connect = _original_connect  # type: ignore[assignment]


def ensure_rollback(conn: Any) -> None:
    """Roll back the connection. Call at the end of a profiled run."""
    try:
        conn.rollback()
        logger.debug("Connection rolled back (perf profiler isolation)")
    except Exception:
        pass  # connection may already be closed


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    collector: PerfCollector,
    *,
    include_suggestions: bool = True,
    config: dict[str, Any] | None = None,
) -> PerfReport:
    """Build a PerfReport from the accumulated profiling data."""
    wall_total = getattr(collector, "_wall_end", time.perf_counter()) - collector._wall_start
    cpu_total = getattr(collector, "_cpu_end", time.process_time()) - collector._cpu_start
    peak_bytes = getattr(collector, "_peak_memory_bytes", 0)

    # Build query summary
    all_queries = collector._all_queries
    qs = QuerySummary(
        total_queries=len(all_queries),
        total_query_time_ms=sum(q.duration_ms for q in all_queries),
    )
    if all_queries:
        slowest = max(all_queries, key=lambda q: q.duration_ms)
        qs.slowest_query_ms = slowest.duration_ms
        qs.slowest_query_sql = slowest.sql_preview

    # Detect N+1 pattern
    sql_prefixes = [q.sql_preview[:80] for q in all_queries if not q.is_executemany]
    prefix_counts = Counter(sql_prefixes)
    cfg = config or _load_perf_config()
    thresholds = cfg.get("thresholds", {})
    n1_min = int(thresholds.get("n_plus_1_min_count", 10))
    qs.n_plus_1_detected = any(c >= n1_min for c in prefix_counts.values())

    # Count unbatched inserts
    insert_prefixes = [
        p for p in sql_prefixes
        if p.strip().upper().startswith("INSERT")
    ]
    insert_counts = Counter(insert_prefixes)
    ub_min = int(thresholds.get("unbatched_insert_min", 5))
    qs.unbatched_inserts = sum(1 for c in insert_counts.values() if c >= ub_min)

    report = PerfReport(
        script_name=collector.script_name,
        started_at=collector._started_at,
        total_wall_time_s=wall_total,
        total_cpu_time_s=cpu_total,
        peak_memory_mb=peak_bytes / (1024 * 1024),
        sections=collector.root_sections,
        query_summary=qs,
    )

    if include_suggestions:
        engine = SuggestionEngine(cfg)
        report.suggestions = engine.analyze(report)

    return report


def _load_perf_config() -> dict[str, Any]:
    """Load perf_config.yaml with fallback to empty dict."""
    try:
        return load_config("perf_config.yaml")
    except FileNotFoundError:
        return {}


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------

def persist_report(report: PerfReport, conn: Any) -> int:
    """Persist a PerfReport to the perf_run / perf_section / perf_query /
    perf_suggestion tables. Uses a SEPARATE writable connection (not the
    profiled read-only one). Returns the run_id."""
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO perf_run
                   (script_name, mode, started_at, total_wall_s, total_cpu_s,
                    peak_memory_mb, total_queries, total_query_ms,
                    suggestion_count, report_json)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               RETURNING run_id""",
            [
                report.script_name,
                "script",
                report.started_at or time.strftime("%Y-%m-%dT%H:%M:%S"),
                round(report.total_wall_time_s, 3),
                round(report.total_cpu_time_s, 3),
                round(report.peak_memory_mb, 1),
                report.query_summary.total_queries,
                round(report.query_summary.total_query_time_ms, 3),
                len(report.suggestions),
                json.dumps(report.to_dict(), default=str),
            ],
        )
        run_id = cur.fetchone()[0]

        # Persist sections (flat — root + children)
        section_id_map: dict[int, int] = {}
        for section in report.sections:
            _persist_section(cur, run_id, section, parent_db_id=None,
                             section_id_map=section_id_map)

        # Persist query metrics
        all_queries = []
        for sec in report.sections:
            _collect_queries(sec, all_queries)
        if all_queries:
            cur.executemany(
                """INSERT INTO perf_query
                       (run_id, sql_preview, duration_ms, rows_affected,
                        is_executemany)
                   VALUES (%s, %s, %s, %s, %s)""",
                [
                    (run_id, q.sql_preview[:500], round(q.duration_ms, 3),
                     q.rows_affected, q.is_executemany)
                    for q in all_queries
                ],
            )

        # Persist suggestions
        if report.suggestions:
            cur.executemany(
                """INSERT INTO perf_suggestion
                       (run_id, severity, category, message, section_name,
                        evidence)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                [
                    (run_id, s.severity, s.category, s.message,
                     s.section, json.dumps(s.evidence, default=str))
                    for s in report.suggestions
                ],
            )

    conn.commit()
    logger.info("Persisted perf run_id=%d (%s)", run_id, report.script_name)
    return run_id


def _persist_section(
    cur: Any, run_id: int, section: SectionMetrics,
    parent_db_id: int | None, section_id_map: dict[int, int],
) -> None:
    """Recursively persist a section and its children."""
    cur.execute(
        """INSERT INTO perf_section
               (run_id, parent_id, name, wall_time_s, cpu_time_s,
                memory_peak_mb, memory_delta_mb, query_count)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
           RETURNING section_id""",
        [
            run_id, parent_db_id, section.name,
            round(section.wall_time_s, 3),
            round(section.cpu_time_s, 3),
            round(section.memory_peak_mb, 1),
            round(section.memory_delta_mb, 1),
            len(section.queries),
        ],
    )
    db_id = cur.fetchone()[0]
    section_id_map[id(section)] = db_id
    for child in section.children:
        _persist_section(cur, run_id, child, parent_db_id=db_id,
                         section_id_map=section_id_map)


def _collect_queries(section: SectionMetrics, out: list[QueryMetrics]) -> None:
    """Recursively collect all queries from a section tree."""
    out.extend(section.queries)
    for child in section.children:
        _collect_queries(child, out)


# ---------------------------------------------------------------------------
# Suggestion engine
# ---------------------------------------------------------------------------

class SuggestionEngine:
    """Rule-based analyzer that produces improvement suggestions."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.thresholds = config.get("thresholds", {})

    def analyze(self, report: PerfReport) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        suggestions.extend(self._check_slow_queries(report))
        suggestions.extend(self._check_n_plus_1(report))
        suggestions.extend(self._check_unbatched_inserts(report))
        suggestions.extend(self._check_memory_spikes(report))
        suggestions.extend(self._check_slow_functions(report))
        suggestions.extend(self._check_sequential_processing(report))
        suggestions.extend(self._check_large_memory_delta(report))
        suggestions.extend(self._check_query_dominance(report))
        return suggestions

    def _check_slow_queries(self, report: PerfReport) -> list[Suggestion]:
        threshold_ms = float(self.thresholds.get("query_slow_ms", 5000))
        results: list[Suggestion] = []
        for section in self._all_sections(report):
            for q in section.queries:
                if q.duration_ms > threshold_ms:
                    results.append(Suggestion(
                        severity="warning",
                        category="query",
                        message=(
                            f"Query in '{section.name}' took {q.duration_ms:.0f}ms "
                            f"(>{threshold_ms:.0f}ms) — consider adding an index or "
                            f"optimizing: {q.sql_preview[:80]}..."
                        ),
                        section=section.name,
                        evidence={"duration_ms": q.duration_ms, "sql": q.sql_preview},
                    ))
        return results

    def _check_n_plus_1(self, report: PerfReport) -> list[Suggestion]:
        min_count = int(self.thresholds.get("n_plus_1_min_count", 10))
        results: list[Suggestion] = []
        for section in self._all_sections(report):
            prefixes = [q.sql_preview[:80] for q in section.queries if not q.is_executemany]
            counts = Counter(prefixes)
            for prefix, count in counts.items():
                if count >= min_count:
                    results.append(Suggestion(
                        severity="critical",
                        category="pattern",
                        message=(
                            f"N+1 query pattern in '{section.name}': {count} calls "
                            f"with same prefix — batch into a single query"
                        ),
                        section=section.name,
                        evidence={"count": count, "sql_prefix": prefix},
                    ))
        return results

    def _check_unbatched_inserts(self, report: PerfReport) -> list[Suggestion]:
        min_count = int(self.thresholds.get("unbatched_insert_min", 5))
        results: list[Suggestion] = []
        for section in self._all_sections(report):
            inserts = [
                q for q in section.queries
                if not q.is_executemany
                and q.sql_preview.strip().upper().startswith("INSERT")
            ]
            if len(inserts) >= min_count:
                results.append(Suggestion(
                    severity="warning",
                    category="pattern",
                    message=(
                        f"Unbatched inserts in '{section.name}': {len(inserts)} "
                        f"individual INSERTs — use executemany() or COPY"
                    ),
                    section=section.name,
                    evidence={"insert_count": len(inserts)},
                ))
        return results

    def _check_memory_spikes(self, report: PerfReport) -> list[Suggestion]:
        threshold_mb = float(self.thresholds.get("memory_spike_mb", 1024))
        results: list[Suggestion] = []
        if report.peak_memory_mb > threshold_mb:
            results.append(Suggestion(
                severity="warning",
                category="memory",
                message=(
                    f"Peak memory {report.peak_memory_mb:.0f}MB exceeds "
                    f"{threshold_mb:.0f}MB — consider chunking large datasets"
                ),
                evidence={"peak_mb": report.peak_memory_mb},
            ))
        return results

    def _check_slow_functions(self, report: PerfReport) -> list[Suggestion]:
        threshold_s = float(self.thresholds.get("function_slow_s", 10))
        results: list[Suggestion] = []
        for section in self._all_sections(report):
            if section.wall_time_s > threshold_s:
                results.append(Suggestion(
                    severity="warning",
                    category="cpu",
                    message=(
                        f"'{section.name}' took {section.wall_time_s:.1f}s "
                        f"(>{threshold_s:.0f}s) — consider chunking or vectorizing"
                    ),
                    section=section.name,
                    evidence={"wall_time_s": section.wall_time_s},
                ))
        return results

    def _check_sequential_processing(self, report: PerfReport) -> list[Suggestion]:
        threshold_s = float(self.thresholds.get("sequential_child_min_s", 2))
        results: list[Suggestion] = []
        for section in self._all_sections(report):
            slow_children = [c for c in section.children if c.wall_time_s > threshold_s]
            if len(slow_children) >= 3:
                results.append(Suggestion(
                    severity="info",
                    category="pattern",
                    message=(
                        f"'{section.name}' has {len(slow_children)} sequential "
                        f"sub-stages each >{threshold_s:.0f}s — consider parallel execution"
                    ),
                    section=section.name,
                    evidence={
                        "slow_children": [
                            {"name": c.name, "wall_time_s": c.wall_time_s}
                            for c in slow_children
                        ]
                    },
                ))
        return results

    def _check_large_memory_delta(self, report: PerfReport) -> list[Suggestion]:
        threshold_mb = float(self.thresholds.get("memory_delta_mb", 200))
        results: list[Suggestion] = []
        for section in self._all_sections(report):
            if section.memory_delta_mb > threshold_mb:
                results.append(Suggestion(
                    severity="info",
                    category="memory",
                    message=(
                        f"'{section.name}' allocated {section.memory_delta_mb:.0f}MB "
                        f"(>{threshold_mb:.0f}MB) — review for large DataFrame copies"
                    ),
                    section=section.name,
                    evidence={"memory_delta_mb": section.memory_delta_mb},
                ))
        return results

    def _check_query_dominance(self, report: PerfReport) -> list[Suggestion]:
        threshold_pct = float(self.thresholds.get("total_query_time_pct", 0.5))
        results: list[Suggestion] = []
        if report.total_wall_time_s > 0:
            query_time_s = report.query_summary.total_query_time_ms / 1000
            ratio = query_time_s / report.total_wall_time_s
            if ratio > threshold_pct:
                results.append(Suggestion(
                    severity="warning",
                    category="query",
                    message=(
                        f"DB queries consume {ratio:.0%} of total wall time "
                        f"(>{threshold_pct:.0%}) — optimize queries or add caching"
                    ),
                    evidence={"query_pct": ratio, "query_time_s": query_time_s},
                ))
        return results

    @staticmethod
    def _all_sections(report: PerfReport) -> list[SectionMetrics]:
        """Flatten all sections (root + children) for analysis."""
        result: list[SectionMetrics] = []
        stack = list(report.sections)
        while stack:
            sec = stack.pop()
            result.append(sec)
            stack.extend(sec.children)
        return result
