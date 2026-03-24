"""Tests for common/services/perf_profiler.py — performance profiler."""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from common.services.perf_profiler import (
    PerfCollector,
    PerfReport,
    QueryMetrics,
    QuerySummary,
    SectionMetrics,
    Suggestion,
    SuggestionEngine,
    ensure_rollback,
    generate_report,
    profile_function,
    profile_script,
    profiled_section,
    wrap_connection,
)


# ---------------------------------------------------------------------------
# Section / timing tests
# ---------------------------------------------------------------------------


def test_profiled_section_captures_wall_time():
    """profiled_section records wall_time_s >= sleep duration."""
    with profile_script("test_wall") as collector:
        with profiled_section("slow_block") as sec:
            time.sleep(0.05)

    assert sec.wall_time_s >= 0.04, f"Expected >= 0.04s, got {sec.wall_time_s}"
    assert sec.name == "slow_block"


def test_profiled_section_captures_memory():
    """profiled_section tracks positive memory_delta_mb for large allocations."""
    with profile_script("test_mem") as collector:
        with profiled_section("alloc_block") as sec:
            # Allocate ~8 MB (1M floats * 8 bytes)
            _big = [0.0] * 1_000_000  # noqa: F841

    # memory_delta_mb should be positive (exact value is platform-dependent)
    assert sec.memory_delta_mb > 0, f"Expected positive delta, got {sec.memory_delta_mb}"


def test_profile_function_decorator():
    """@profile_function adds a section with the function's qualname."""
    with profile_script("test_decorator") as collector:

        @profile_function
        def do_work():
            time.sleep(0.01)
            return 42

        result = do_work()

    assert result == 42
    assert len(collector.root_sections) == 1
    assert "do_work" in collector.root_sections[0].name


def test_profile_function_with_custom_name():
    """@profile_function(name='custom') uses the provided name."""
    with profile_script("test_custom") as collector:

        @profile_function(name="custom_section")
        def helper():
            return "ok"

        helper()

    assert collector.root_sections[0].name == "custom_section"


def test_nested_sections():
    """Child sections appear in the parent's children list."""
    with profile_script("test_nested") as collector:
        with profiled_section("parent") as parent:
            with profiled_section("child1") as child1:
                pass
            with profiled_section("child2") as child2:
                pass

    assert len(parent.children) == 2
    assert parent.children[0].name == "child1"
    assert parent.children[1].name == "child2"
    # Root should only have the parent
    assert len(collector.root_sections) == 1


def test_profiled_section_without_collector():
    """When no profile_script is active, profiled_section yields a dummy without error."""
    with profiled_section("orphan") as sec:
        time.sleep(0.01)

    assert isinstance(sec, SectionMetrics)
    assert sec.name == "orphan"
    # The dummy section has default wall_time_s=0 since it skips timing
    assert sec.wall_time_s == 0.0


# ---------------------------------------------------------------------------
# DB connection wrapper tests
# ---------------------------------------------------------------------------


def _make_mock_conn(*, with_cursor_class: bool = True):
    """Build a mock psycopg3 connection with enough structure for wrap_connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 5
    mock_cursor.execute = MagicMock(return_value=None)
    mock_cursor.executemany = MagicMock(return_value=None)
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor
    # wrap_connection accesses cursor().__class__.execute — set real attributes
    type(mock_cursor).execute = mock_cursor.execute
    type(mock_cursor).executemany = mock_cursor.executemany
    return mock_conn, mock_cursor


def test_wrap_connection_readonly():
    """wrap_connection(readonly=True) sets default_transaction_read_only."""
    mock_conn, mock_cursor = _make_mock_conn()

    wrap_connection(mock_conn, readonly=True)

    # Should have set autocommit then executed SET command
    assert mock_conn.autocommit is False  # restored after SET
    # The SET statement was executed via cursor context manager
    mock_cursor.execute.assert_called_with("SET default_transaction_read_only = true")


def test_wrap_connection_records_queries():
    """Wrapped connection records query metrics into active collector."""
    # wrap_connection accesses conn.cursor().__class__.execute to store originals.
    # We need a real class with execute/executemany attributes for this to work.
    class FakeCursor:
        rowcount = 3

        def execute(self, query, params=None, **kwargs):
            return None

        def executemany(self, query, params_seq, **kwargs):
            return None

        def fetchone(self):
            return (1,)

        def fetchall(self):
            return []

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = FakeCursor()

    with profile_script("test_wrap") as collector:
        wrapped = wrap_connection(mock_conn, readonly=False)
        cur = wrapped.cursor()
        cur.execute("SELECT 1")

    assert len(collector._all_queries) >= 1
    qm = collector._all_queries[-1]
    assert "SELECT 1" in qm.sql_preview
    assert qm.duration_ms >= 0


def test_ensure_rollback():
    """ensure_rollback calls conn.rollback()."""
    mock_conn = MagicMock()
    ensure_rollback(mock_conn)
    mock_conn.rollback.assert_called_once()


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------


def _build_collector(
    *,
    script_name: str = "test_script",
    queries: list[QueryMetrics] | None = None,
    sections: list[SectionMetrics] | None = None,
    peak_bytes: int = 0,
) -> PerfCollector:
    """Helper to build a PerfCollector with preset data."""
    c = PerfCollector(script_name)
    c._started_at = "2026-01-01T00:00:00"
    c._wall_start = 0.0
    c._wall_end = 1.0  # 1 second total
    c._cpu_start = 0.0
    c._cpu_end = 0.5
    c._peak_memory_bytes = peak_bytes
    if queries:
        c._all_queries = queries
    if sections:
        c.root_sections = sections
    return c


def test_generate_report_structure():
    """generate_report produces a PerfReport with all required fields."""
    collector = _build_collector()
    config = {"thresholds": {}}
    report = generate_report(collector, config=config)

    assert isinstance(report, PerfReport)
    assert report.script_name == "test_script"
    assert report.total_wall_time_s > 0
    assert isinstance(report.query_summary, QuerySummary)
    assert isinstance(report.sections, list)
    assert isinstance(report.suggestions, list)


def test_no_suggestions_when_clean():
    """Fast, low-memory report with no queries produces zero suggestions."""
    collector = _build_collector(peak_bytes=100)
    config = {"thresholds": {"query_slow_ms": 5000, "memory_spike_mb": 1024}}
    report = generate_report(collector, config=config)

    assert len(report.suggestions) == 0


# ---------------------------------------------------------------------------
# Suggestion engine tests
# ---------------------------------------------------------------------------


def _make_config(**overrides: float) -> dict:
    defaults = {
        "query_slow_ms": 5000,
        "n_plus_1_min_count": 10,
        "unbatched_insert_min": 5,
        "memory_spike_mb": 1024,
        "function_slow_s": 10,
        "sequential_child_min_s": 2,
        "memory_delta_mb": 200,
        "total_query_time_pct": 0.5,
    }
    defaults.update(overrides)
    return {"thresholds": defaults}


def test_suggestion_slow_query():
    """A query exceeding query_slow_ms threshold triggers a slow query suggestion."""
    slow_q = QueryMetrics(sql_preview="SELECT * FROM big_table", duration_ms=6000)
    section = SectionMetrics(name="data_load", queries=[slow_q])
    collector = _build_collector(queries=[slow_q], sections=[section])

    report = generate_report(collector, config=_make_config())

    slow_suggestions = [s for s in report.suggestions if "took 6000ms" in s.message]
    assert len(slow_suggestions) >= 1
    assert slow_suggestions[0].category == "query"


def test_suggestion_n_plus_1():
    """15 queries with the same prefix triggers N+1 suggestion."""
    queries = [
        QueryMetrics(sql_preview="SELECT * FROM orders WHERE id = %s", duration_ms=5)
        for _ in range(15)
    ]
    section = SectionMetrics(name="fetch_loop", queries=queries)
    collector = _build_collector(queries=queries, sections=[section])

    report = generate_report(collector, config=_make_config())

    n1 = [s for s in report.suggestions if "N+1" in s.message]
    assert len(n1) >= 1
    assert n1[0].severity == "critical"
    # Also verify query_summary detected it
    assert report.query_summary.n_plus_1_detected is True


def test_suggestion_unbatched_inserts():
    """10 individual INSERT queries triggers unbatched insert suggestion."""
    queries = [
        QueryMetrics(sql_preview="INSERT INTO items VALUES (%s)", duration_ms=2)
        for _ in range(10)
    ]
    section = SectionMetrics(name="insert_loop", queries=queries)
    collector = _build_collector(queries=queries, sections=[section])

    report = generate_report(collector, config=_make_config())

    ub = [s for s in report.suggestions if "Unbatched" in s.message]
    assert len(ub) >= 1
    assert ub[0].category == "pattern"


def test_suggestion_memory_spike():
    """peak_memory_mb > threshold triggers memory spike suggestion."""
    # 2 GB peak
    collector = _build_collector(peak_bytes=2 * 1024 * 1024 * 1024)

    report = generate_report(collector, config=_make_config())

    mem = [s for s in report.suggestions if "Peak memory" in s.message]
    assert len(mem) >= 1
    assert mem[0].category == "memory"


def test_suggestion_slow_function():
    """Section with wall_time_s > threshold triggers slow function suggestion."""
    section = SectionMetrics(name="heavy_compute", wall_time_s=15.0)
    collector = _build_collector(sections=[section])

    report = generate_report(collector, config=_make_config())

    slow = [s for s in report.suggestions if "heavy_compute" in s.message and "took" in s.message]
    assert len(slow) >= 1
    assert slow[0].category == "cpu"


def test_suggestion_sequential_processing():
    """3+ children each >2s triggers parallel processing suggestion."""
    children = [
        SectionMetrics(name=f"stage_{i}", wall_time_s=3.0)
        for i in range(4)
    ]
    parent = SectionMetrics(name="pipeline", children=children)
    collector = _build_collector(sections=[parent])

    report = generate_report(collector, config=_make_config())

    seq = [s for s in report.suggestions if "sequential" in s.message]
    assert len(seq) >= 1
    assert seq[0].category == "pattern"


def test_suggestion_query_dominance():
    """Queries consuming >50% wall time triggers query dominance suggestion."""
    # Total wall = 1.0s (from _build_collector), query time = 800ms = 80%
    queries = [QueryMetrics(sql_preview="SELECT 1", duration_ms=800)]
    section = SectionMetrics(name="db_heavy", queries=queries)
    collector = _build_collector(queries=queries, sections=[section])

    report = generate_report(collector, config=_make_config())

    dom = [s for s in report.suggestions if "DB queries consume" in s.message]
    assert len(dom) >= 1
    assert dom[0].category == "query"


# ---------------------------------------------------------------------------
# Report serialization tests
# ---------------------------------------------------------------------------


def test_report_to_dict_json_serializable():
    """PerfReport.to_dict() produces a JSON-serializable dict."""
    collector = _build_collector()
    report = generate_report(collector, config=_make_config())

    d = report.to_dict()
    # Should not raise
    serialized = json.dumps(d, default=str)
    assert isinstance(serialized, str)
    parsed = json.loads(serialized)
    assert parsed["script_name"] == "test_script"


def test_report_to_json_writes_file(tmp_path):
    """PerfReport.to_json() writes valid JSON to the specified path."""
    collector = _build_collector()
    report = generate_report(collector, config=_make_config())

    out_path = tmp_path / "reports" / "perf.json"
    report.to_json(out_path)

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["script_name"] == "test_script"
    assert "total_wall_time_s" in data


def test_report_to_console_format():
    """PerfReport.to_console() produces human-readable output with key markers."""
    section = SectionMetrics(name="data_load", wall_time_s=2.5, cpu_time_s=1.0)
    collector = _build_collector(sections=[section])
    report = generate_report(collector, config=_make_config())

    output = report.to_console()

    assert "=== Performance Report:" in output
    assert "test_script" in output
    assert "data_load" in output
    assert "Sections:" in output


# ---------------------------------------------------------------------------
# PerfCollector internals
# ---------------------------------------------------------------------------


def test_collector_push_pop_section():
    """PerfCollector push/pop correctly manages section stack."""
    c = PerfCollector("test")
    s1 = c.push_section("a")
    s2 = c.push_section("b")

    assert c.current_section is s2
    assert len(s1.children) == 1
    assert s1.children[0] is s2

    popped = c.pop_section()
    assert popped is s2
    assert c.current_section is s1

    c.pop_section()
    assert c.current_section is None
    assert len(c.root_sections) == 1


def test_collector_record_query():
    """record_query adds to both _all_queries and current section."""
    c = PerfCollector("test")
    s = c.push_section("s1")
    qm = QueryMetrics(sql_preview="SELECT 1", duration_ms=10)
    c.record_query(qm)

    assert len(c._all_queries) == 1
    assert len(s.queries) == 1
    assert s.queries[0] is qm
