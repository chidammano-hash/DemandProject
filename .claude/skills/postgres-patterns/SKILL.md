---
name: postgres-patterns
description: PostgreSQL database patterns for query optimization, schema design, indexing, and security. Based on Supabase best practices.
origin: ECC
---

# PostgreSQL Patterns

Quick reference for PostgreSQL best practices. For detailed guidance, use the `database-reviewer` agent.

## When to Activate

- Writing SQL queries or migrations
- Designing database schemas
- Troubleshooting slow queries
- Implementing Row Level Security
- Setting up connection pooling

## Quick Reference

### Index Cheat Sheet

| Query Pattern | Index Type | Example |
|--------------|------------|---------|
| `WHERE col = value` | B-tree (default) | `CREATE INDEX idx ON t (col)` |
| `WHERE col > value` | B-tree | `CREATE INDEX idx ON t (col)` |
| `WHERE a = x AND b > y` | Composite | `CREATE INDEX idx ON t (a, b)` |
| `WHERE jsonb @> '{}'` | GIN | `CREATE INDEX idx ON t USING gin (col)` |
| `WHERE tsv @@ query` | GIN | `CREATE INDEX idx ON t USING gin (col)` |
| Time-series ranges | BRIN | `CREATE INDEX idx ON t USING brin (col)` |

### Data Type Quick Reference

| Use Case | Correct Type | Avoid |
|----------|-------------|-------|
| IDs | `bigint` | `int`, random UUID |
| Strings | `text` | `varchar(255)` |
| Timestamps | `timestamptz` | `timestamp` |
| Money | `numeric(10,2)` | `float` |
| Flags | `boolean` | `varchar`, `int` |

### Common Patterns

**Composite Index Order:**
```sql
-- Equality columns first, then range columns
CREATE INDEX idx ON orders (status, created_at);
-- Works for: WHERE status = 'pending' AND created_at > '2024-01-01'
```

**Covering Index:**
```sql
CREATE INDEX idx ON users (email) INCLUDE (name, created_at);
-- Avoids table lookup for SELECT email, name, created_at
```

**Partial Index:**
```sql
CREATE INDEX idx ON users (email) WHERE deleted_at IS NULL;
-- Smaller index, only includes active users
```

**RLS Policy (Optimized):**
```sql
CREATE POLICY policy ON orders
  USING ((SELECT auth.uid()) = user_id);  -- Wrap in SELECT!
```

**UPSERT:**
```sql
INSERT INTO settings (user_id, key, value)
VALUES (123, 'theme', 'dark')
ON CONFLICT (user_id, key)
DO UPDATE SET value = EXCLUDED.value;
```

**Cursor Pagination:**
```sql
SELECT * FROM products WHERE id > $last_id ORDER BY id LIMIT 20;
-- O(1) vs OFFSET which is O(n)
```

**Queue Processing:**
```sql
UPDATE jobs SET status = 'processing'
WHERE id = (
  SELECT id FROM jobs WHERE status = 'pending'
  ORDER BY created_at LIMIT 1
  FOR UPDATE SKIP LOCKED
) RETURNING *;
```

### Anti-Pattern Detection

```sql
-- Find unindexed foreign keys
SELECT conrelid::regclass, a.attname
FROM pg_constraint c
JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)
WHERE c.contype = 'f'
  AND NOT EXISTS (
    SELECT 1 FROM pg_index i
    WHERE i.indrelid = c.conrelid AND a.attnum = ANY(i.indkey)
  );

-- Find slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
WHERE mean_exec_time > 100
ORDER BY mean_exec_time DESC;

-- Check table bloat
SELECT relname, n_dead_tup, last_vacuum
FROM pg_stat_user_tables
WHERE n_dead_tup > 1000
ORDER BY n_dead_tup DESC;
```

### Configuration Template

```sql
-- Connection limits (adjust for RAM)
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET work_mem = '8MB';

-- Timeouts
ALTER SYSTEM SET idle_in_transaction_session_timeout = '30s';
ALTER SYSTEM SET statement_timeout = '30s';

-- Monitoring
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Security defaults
REVOKE ALL ON SCHEMA public FROM public;

SELECT pg_reload_conf();
```

## DemandProject data loading (ETL pitfalls)
- **COPY uses `copy.write_row((...))`, never hand-built buffers.** Text-format COPY treats
  tab as delimiter, newline as row terminator, backslash as escape — so any free-text
  dimension value (`item_id`/`customer_group`/`loc` from external CSVs) containing one
  silently shifts columns, routes the WRONG row, or drops rows. Always
  `with cur.copy("COPY t (cols) FROM STDIN") as copy: copy.write_row((...))` with an
  explicit column list. JSONB columns need a `psycopg.types.json.Jsonb` wrapper.
- **Null normalization**: `''`, `'null'`, `'none'`, `'NA'` → NULL during load.
- **Sales filtering**: only `TYPE=1` rows enter `fact_sales_monthly`.
- **Forecast `model_id`** default `'external'` for source feeds; `UNIQUE(forecast_ck, model_id)`
  prevents dupes.
- **Execution-lag loading**: dual-path insert, archive loaded BEFORE staging mutation
  (`docs/specs/02-forecasting/03-backtest-framework.md`).
- **Staged CSVs land in `data/staged/`** (`DomainSpec.clean_file` embeds the prefix); no new
  normalized output at `data/` root.
- **Never write synthetic/random data to a consumed fact table** — guard with
  `--dry-run`/`--allow-synthetic` and fail loud.

## MV & connection pools
- **ALL MV refreshes go through `common/core/mv_refresh.py`** — the single table→MV
  dependency map. After committing a fact/dim write call
  `refresh_for_tables([tables written])`; never hand-pick an inline MV list (gate rule 7
  blocks new `REFRESH MATERIALIZED VIEW` outside that module). New MV DDL must register in
  `MV_SOURCES` in the same change — `tests/unit/test_mv_refresh.py` diffs the map against
  sql/ DDL and fails on unregistered or retired MVs. Operator CLI:
  `scripts/db/refresh_mvs.py --all | --tables t1,t2 | --mvs m1` (backs `refresh-mvs-tiered`
  and `refresh-accuracy-mvs`). Safety net: the `refresh_all_mvs` job runs nightly
  (`config/platform/jobs_config.yaml`).
- **Stale MV after refresh** → dependent refreshed before source → the map's dict order is
  the refresh order (topological; test-enforced). Stub-table pattern: MV on a future table →
  `CREATE TABLE IF NOT EXISTS` + LEFT JOIN → NULL neutral scores until real data flows.
- **`REFRESH ... CONCURRENTLY` cannot run inside a transaction block** — the service opens
  its own autocommit connection; call it AFTER your commit, not on your write cursor.
- **`FATAL: too many connections`** → the API runs THREE pools per gunicorn worker
  (`api/pool.py`): sync `POOL_MAX_SIZE` (12), async `ASYNC_POOL_MAX_SIZE` (20),
  read-replica `READ_POOL_MAX_SIZE` (12, only when `READ_REPLICA_URL` set). Primary-ceiling
  invariant: `WORKERS × (POOL_MAX_SIZE + ASYNC_POOL_MAX_SIZE) + overhead ≤ max_connections`;
  `make deploy-check` enforces it (trips >85%). Don't tune one pool past the invariant.
- **Hot analytical GETs should not repeatedly scan facts/MVs on the primary.** Use
  `@cached_sync` / `@cached_async` for repeated read endpoints and route stale-tolerant reads
  through `get_read_only_conn()` / `get_async_read_only_conn()`. Dashboard and Accuracy
  routers are mechanically checked for this pattern.
- Profiling: `profiled_section()` from `common/services/perf_profiler.py` (not raw
  `time.time()`); thresholds in `config/platform/perf_config.yaml`.

## Related
- Agent: `database-reviewer` · Skill: `api-design`, `forecasting-patterns`

---

*Base query patterns adapted from Supabase Agent Skills (credit: Supabase team, MIT License)*
