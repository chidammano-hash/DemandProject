# Performance Profiling

> Instruments pipeline scripts and API endpoints with zero-friction decorators and context managers to collect timing, memory, and query data, then generates rule-based suggestions for fixing detected bottlenecks.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (backend only) |
| **Key Files** | `common/services/perf_profiler.py`, `scripts/ops/run_perf_analysis.py`, `config/platform/perf_config.yaml` |

---

## Problem

Ad-hoc timing code (`time.time()`, `_elapsed()` helpers) is scattered across 69+ files with no centralized visibility into performance characteristics. There is no automated regression detection, no improvement suggestions, and no production-safe profiling capability. When a pipeline slows down, engineers manually instrument code, collect timings, and reason about bottlenecks in isolation -- a slow, error-prone process that leaves no audit trail.

---

## Solution

A unified performance profiler that provides:

1. **Decorator and context manager** instrumentation (`@profile_function`, `profiled_section()`) for zero-friction adoption
2. **DB query auto-tracking** via `wrap_connection()` that intercepts all SQL calls with timing and row counts
3. **Memory profiling** via `tracemalloc` integration for peak and per-section memory deltas
4. **Rule-based suggestion engine** that detects 8 categories of performance anti-patterns and generates actionable recommendations
5. **Full production isolation** -- read-only transactions with rollback-only semantics, zero side effects on the target database

---

## How It Works

### Instrumentation Layer

The profiler collects timing, memory, and query data into a `ProfileCollector` object. Three instrumentation methods cover all use cases:

**Top-level context manager** wraps an entire script or API endpoint:

```python
with profile_script("compute_safety_stock") as collector:
    # ... entire script body ...
```

**Section context manager** tracks named subsections:

```python
with profiled_section("load_data"):
    df = pd.read_sql(query, conn)

with profiled_section("compute_targets"):
    results = calculate_ss(df)
```

**Function decorator** auto-instruments any callable:

```python
@profile_function
def expensive_computation(data):
    return data.groupby("item_id").apply(calc)
```

### DB Query Tracking

`wrap_connection(conn)` returns a proxy connection that intercepts `execute()` and `executemany()` calls. Each query is logged with:

- SQL text (first 200 chars)
- Execution time
- Row count (from `rowcount`)
- Whether it was a batch operation

This enables detection of N+1 queries, unbatched inserts, and query-dominated pipelines without modifying any SQL code.

### Memory Profiling

When enabled (default for full analysis), `tracemalloc` tracks:

- Peak memory usage across the entire profile session
- Per-section memory deltas (allocation growth within each `profiled_section`)
- Top allocation sites for memory spike investigation

### Suggestion Engine

The `SuggestionEngine` applies 8 detection rules against the collected profile data and generates prioritized recommendations:

| # | Rule | Trigger | Suggestion |
|---|------|---------|------------|
| 1 | Slow Queries | Any query > 5s | Add index, rewrite query, or use materialized view |
| 2 | N+1 Pattern | Same query template executed > 10x | Batch into single query with `IN` clause or JOIN |
| 3 | Unbatched Inserts | > 50 individual INSERT statements | Use `executemany()` or `COPY` bulk load |
| 4 | Memory Spikes | Peak memory > 2GB or delta > 500MB in one section | Stream data, reduce DataFrame copies, use chunked processing |
| 5 | Slow Functions | Any function > 30s | Profile internals, consider caching or parallelization |
| 6 | Sequential Processing | Multiple independent sections each > 10s | Use `concurrent.futures` or async processing |
| 7 | Large Memory Delta | Section allocates > 200MB net | Check for DataFrame copies, unnecessary `.values` calls |
| 8 | Query Dominance | DB time > 70% of total wall time | Optimize queries first; code changes will have minimal impact |

Thresholds are configurable via `config/platform/perf_config.yaml`.

---

## Architecture

```
common/services/perf_profiler.py    Core module: ProfileCollector, decorators, suggestion engine
scripts/ops/run_perf_analysis.py    CLI entry point: 4 modes (script, api, pipeline, report)
config/platform/perf_config.yaml             Threshold configuration
data/perf_reports/                  Output directory (gitignored)
```

### Component Diagram

```
┌─────────────────────────────────────────────┐
│  Target Code (script / API / pipeline)      │
│                                             │
│  @profile_function   profiled_section()     │
│  wrap_connection()   profile_script()       │
└──────────────────┬──────────────────────────┘
                   │ timing + memory + query data
                   ▼
┌─────────────────────────────────────────────┐
│  ProfileCollector                           │
│  - sections: List[SectionProfile]           │
│  - queries: List[QueryProfile]              │
│  - memory_snapshots: List[MemSnapshot]      │
│  - wall_time, peak_memory                   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  SuggestionEngine (8 rules)                 │
│  - analyze(collector) -> List[Suggestion]   │
│  - each Suggestion: category, severity,     │
│    message, context                         │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  ReportBuilder                              │
│  - generate_report(collector) -> dict       │
│  - Sections: summary, queries, memory,      │
│    suggestions, section breakdown           │
│  - Output: JSON + console table             │
└─────────────────────────────────────────────┘
```

---

## Key APIs

Five public entry points cover all profiling scenarios:

### 1. `profile_script(name)` — Top-level context manager

```python
from common.services.perf_profiler import profile_script

with profile_script("compute_safety_stock") as collector:
    main()
# collector now contains full profile data
report = generate_report(collector)
```

Starts wall-clock timer, enables tracemalloc, and yields a `ProfileCollector`. On exit, finalizes timing and memory snapshots.

### 2. `profiled_section(name)` — Section context manager

```python
from common.services.perf_profiler import profiled_section

with profiled_section("query_inventory"):
    rows = cursor.fetchall()
```

Records start/end time and memory delta for a named code section. Sections can be nested.

### 3. `@profile_function` — Decorator

```python
from common.services.perf_profiler import profile_function

@profile_function
def calculate_eoq(demand, cost, holding):
    ...
```

Wraps the function in a `profiled_section` using the function's qualified name. Works with sync functions.

### 4. `wrap_connection(conn)` — DB wrapper

```python
from common.services.perf_profiler import wrap_connection

conn = psycopg.connect(**db_params)
conn = wrap_connection(conn)
# All subsequent queries are auto-tracked
```

Returns a proxy that intercepts `cursor.execute()` and `cursor.executemany()`. The underlying connection behavior is unchanged.

### 5. `generate_report(collector)` — Report builder

```python
from common.services.perf_profiler import generate_report

report = generate_report(collector)
# report is a dict with: summary, sections, queries, memory, suggestions
```

Runs the suggestion engine, formats results, and optionally writes to `data/perf_reports/`.

---

## Production Safety

The profiler is designed for safe use against production databases:

| Safety Mechanism | Implementation |
|---|---|
| Read-only transactions | `SET default_transaction_read_only = true` on wrapped connections |
| Rollback-only | All transactions end with `ROLLBACK`, never `COMMIT` |
| No schema changes | Profiler creates no tables, views, or indexes |
| No data mutation | No INSERT, UPDATE, or DELETE operations |
| Isolated output | Reports written to `data/perf_reports/` (gitignored) |
| Configurable overhead | Memory profiling can be disabled for minimal overhead |

---

## CLI Usage

The CLI script `scripts/ops/run_perf_analysis.py` supports 4 modes:

```bash
# Profile a specific script
make perf-script SCRIPT=compute_safety_stock

# Profile API endpoint response times
make perf-api

# Profile the full pipeline (normalize -> load -> compute)
make perf-pipeline

# Generate a summary report from the last profile run
make perf-report
```

### Direct invocation:

```bash
# Script mode — profile a single pipeline script
python scripts/ops/run_perf_analysis.py --mode script --target compute_safety_stock

# API mode — profile API endpoint latencies
python scripts/ops/run_perf_analysis.py --mode api

# Pipeline mode — profile end-to-end pipeline
python scripts/ops/run_perf_analysis.py --mode pipeline

# Report mode — generate report from saved profile data
python scripts/ops/run_perf_analysis.py --mode report --input data/perf_reports/latest.json
```

---

## Configuration

File: `config/platform/perf_config.yaml`

```yaml
# Profiling behavior
profiling:
  enable_memory: true          # tracemalloc integration
  enable_query_tracking: true  # DB query interception
  tracemalloc_frames: 10       # Stack depth for memory traces
  output_dir: data/perf_reports

# Suggestion engine thresholds
thresholds:
  slow_query_seconds: 5.0       # Rule 1: flag queries slower than this
  n_plus_one_count: 10          # Rule 2: same query template repeated N times
  unbatched_insert_count: 50    # Rule 3: individual INSERTs before suggesting batch
  memory_spike_mb: 2048         # Rule 4: peak memory warning (MB)
  memory_section_delta_mb: 500  # Rule 4: per-section memory growth (MB)
  slow_function_seconds: 30.0   # Rule 5: function duration warning
  sequential_section_seconds: 10.0  # Rule 6: independent sections each above this
  large_delta_mb: 200           # Rule 7: per-section allocation threshold (MB)
  query_dominance_pct: 70       # Rule 8: DB time as % of total wall time

# Report formatting
report:
  top_queries: 20               # Number of slowest queries to include
  top_sections: 15              # Number of slowest sections to include
  top_memory_sites: 10          # Number of top memory allocators
  format: json                  # Output format: json or text
```

---

## Data Model

The profiler does not create database tables. All output is written to local files:

| Output | Location | Format |
|---|---|---|
| Full profile data | `data/perf_reports/<name>_<timestamp>.json` | JSON |
| Summary report | `data/perf_reports/<name>_summary.txt` | Plain text |
| Latest symlink | `data/perf_reports/latest.json` | Symlink |

---

## Dependencies

- **Upstream:** Any script, API endpoint, or pipeline to be profiled
- **Libraries:** `tracemalloc` (stdlib), `time` (stdlib), `psycopg` (for connection wrapping)
- **Configuration:** `config/platform/perf_config.yaml`
- **Output:** `data/perf_reports/` (gitignored)

---

## See Also

- [01-infrastructure](01-infrastructure.md) -- Platform overview and implemented features list
- [04-planning-date](04-planning-date.md) -- Configurable date used in profiled pipeline runs
- [../../PLATFORM_GUIDE.md](../../PLATFORM_GUIDE.md) -- Quick start and feature summary
