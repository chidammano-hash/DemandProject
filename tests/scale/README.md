# Scale Test Suite

Regression gate for **scale-related** behavior on hot endpoints.  These
tests synthesize a parameterized dataset, load it into a temporary
PostgreSQL schema, and measure latency percentiles (p50/p95/p99) against
the real FastAPI app via in-process ASGI transport (no network).

## Why

Our default test suite mocks the database, so an N+1 query, missing index,
or O(N^2) Python loop sails through CI and only surfaces in prod. The
scale suite catches these by **actually loading data** at a configurable
size and asserting a latency budget.

## Running

```bash
# Quick (default ~100K rows; runs in a few minutes):
make scale-test

# Nightly / full (10M rows; runs in tens of minutes):
SCALE=10000000 make scale-test

# Custom size:
SCALE=500000 make scale-test

# Override p95 budget for a slow CI machine:
~/.local/bin/uv run pytest tests/scale/ -m scale --override-ini="norecursedirs=" \
    --scale=100000 --scale-p95-budget-ms=8000
```

The `-m scale` flag and `--override-ini="norecursedirs="` are required to
unwind the two default-exclusion gates (see "Default exclusion" below).
`make scale-test` already wires both for you.

## Default exclusion

Scale tests are NOT collected by `pytest tests/` (the default run).
Two gates keep them out:

1. `addopts = "-m 'not scale'"` in `pyproject.toml` — every test is marked
   `@pytest.mark.scale`.
2. `norecursedirs = ["tests/scale"]` — pytest doesn't even descend into
   the directory unless given an explicit path.

The `make scale-test` target passes `tests/scale/` explicitly **plus**
`-m scale --override-ini="norecursedirs="` to unwind both gates.

## Requirements

- A reachable PostgreSQL instance (uses `common.core.db.get_db_params()`).
- Permission to `CREATE SCHEMA` and `DROP SCHEMA` (a session-scoped temp
  schema is created and dropped on teardown).
- If the DB is unreachable, every test cleanly **skips** rather than
  errors.

## Adding a new scale test

1. Create `tests/scale/test_<feature>_scale.py`.
2. Mark the test with `@pytest.mark.scale`.
3. Take the `scale_dataset` and `latency_helper` fixtures.
4. Use `httpx.AsyncClient(transport=ASGITransport(app))` (in-process, no
   network) — same pattern as `tests/api/`.
5. Run a small **warmup** loop (5–10 requests) to prime caches, then
   measure 100 sequential requests.
6. Assert `rep.p95_ms <= budget_ms`. Default budget is **5000 ms**;
   override per-test via the `--scale-p95-budget-ms` CLI flag.

## Interpreting p95 thresholds

- p50 — typical-case latency. Sudden movement = a default-path regression.
- **p95** — the budget. Failing this is a hard "no, this got slower" signal.
- p99 — long-tail. Watch for sudden growth (lock contention, GC pauses).

The budget is intentionally generous (5 s) at the default 100K-row size so
that real regressions stand out without flaking on shared CI runners. The
nightly 10M-row job tightens budgets per-endpoint as we collect baselines.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Marks the directory as a package + module docstring |
| `conftest.py` | `--scale` CLI flag, `scale_dataset` fixture, `latency_helper` |
| `test_customer_analytics_scale.py` | Example: `/customer-analytics/kpis` p95 |
| `test_inv_planning_scale.py` | Example: `/inv-planning/action-feed` p95 |

## Synthetic data shape

`scale_dataset` populates `<tmp_schema>.fact_customer_demand_monthly` with
N rows (default 100,000) following:

- `item_id`     — `ITEM_000000`..`ITEM_<n_items>` (capped at 5,000 unique)
- `customer_no` — `CUST_000000`..`CUST_<n_cust>` (capped at 1,000 unique)
- `site` / `location_id` — single-site for now (extend if needed)
- `startdate`   — first-of-month, walked back 0–24 months from 2024-01
- `demand_qty`  — log-normal (mean=3, sigma=1.2), clipped to `[0, 1e5]`
- `oos_qty`     — sparse (5% of rows non-zero)
- `sales_qty`   — `max(0, demand_qty - oos_qty)`

Two small dim tables are derived (`dim_customer`, `dim_item`).

This is **not** a substitute for prod-shaped data; it is a *deterministic*
stand-in for measuring relative latency changes between commits. If your
endpoint hits tables not in this fixture, extend `conftest.py` rather than
rolling your own setup.
