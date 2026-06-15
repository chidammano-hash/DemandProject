"""Load normalized CSV directly into PostgreSQL tables.

Usage:
    python scripts/load_dataset_postgres.py --dataset <domain> [--replace] [--skip-archive]

Single-pass loader: CSV → temp staging → main table. No intermediate layers.
"""
import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import psycopg
import psycopg.errors
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params
from common.core.domain_specs import DOMAIN_SPECS, DomainSpec, get_spec
from common.core.etl_helpers import (
    DFU_MATCH_DOMAINS,
    FK_CHECKS,
    drop_indexes,
    drop_unique_constraints,
    ensure_monthly_partition,
    filter_fk_orphans,
    filter_unmatched_dfus,
    get_secondary_indexes,
    get_unique_constraints,
    is_pg_partitioned,
    recreate_indexes,
    recreate_unique_constraints,
    unmatched_warn_pct,
)
from common.core.planning_date import get_planning_date
from common.core.sql_helpers import (
    EXTERNAL_MODEL_ID,
    HASH_CHUNK_SIZE,
    MV_REFRESH_ARCHIVE,
    NULL_SQL,
    _elapsed,
    business_key_expr,
    qident,
    typed_expr_sets,
)
from common.engines.medallion import complete_batch, create_batch, fail_batch, file_hash
from common.services.perf_profiler import profiled_section

logger = logging.getLogger(__name__)

# PG session tuning
_PG_WORK_MEM = "512MB"
_PG_MAINTENANCE_WORK_MEM = "1GB"



# ---------------------------------------------------------------------------
# Index management lives in common/core/etl_helpers.py (get_secondary_indexes,
# get_unique_constraints, drop_indexes, recreate_indexes, and the unique-
# constraint equivalents) — imported above and used in load_domain() below.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Partition helpers — for partitioned tables (e.g., fact_inventory_snapshot)
# ---------------------------------------------------------------------------

# _is_partitioned delegates to the shared partitioned-check (US6 convergence).
_is_partitioned = is_pg_partitioned


def _ensure_partition_exists(cur, parent: str, start_date: str, end_date: str) -> str:
    """Create a monthly partition if it doesn't exist. Returns partition name.

    Thin wrapper over common/core/etl_helpers.ensure_monthly_partition: the
    inventory path supplies YYYY-MM-DD month bounds from the snapshot filename;
    we parse the month start and delegate (end recomputed as next-month start).
    """
    return ensure_monthly_partition(cur, parent, date.fromisoformat(start_date))


# ---------------------------------------------------------------------------
# DFU match + FK orphan filters live in common/core/etl_helpers.py
# (filter_unmatched_dfus, filter_fk_orphans, DFU_MATCH_DOMAINS, FK_CHECKS) —
# imported above and used in load_domain() below.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Forecast-specific helpers
# ---------------------------------------------------------------------------

def _resolve_forecast_execution_lag(cur, stg_table: str) -> int:
    """Set lag and execution_lag from dim_sku on staging table.

    Assumes unmatched rows have already been deleted by filter_unmatched_dfus().
    All external forecasts are assumed to be at execution lag — the source
    file's lag/execution_lag fields are ignored and overwritten from dim_sku.
    Returns number of matched (updated) rows.
    """
    cur.execute("""
        SELECT EXISTS(
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'dim_sku' AND table_schema = 'public'
        )
    """)
    if not cur.fetchone()[0]:
        logger.warning("dim_sku not found — cannot resolve execution lag")
        return 0

    cur.execute(f"""
        UPDATE {qident(stg_table)} s
        SET "execution_lag" = d.execution_lag::text,
            "lag" = d.execution_lag::text
        FROM dim_sku d
        WHERE d.sku_ck = trim(s."item_id") || '_' || trim(s."customer_group") || '_' || trim(s."loc")
    """)
    return cur.rowcount


# NOTE: external forecasts intentionally do NOT populate backtest_lag_archive
# (see load_domain — "External forecasts skip archive"). The archive is owned by
# load_backtest_forecasts.py / load_ext_ml_forecasts.py, which stream rows in
# BATCH_SIZE chunks and use the ON-CONFLICT fast-path. A prior dead
# _load_forecast_archive() here implied a dual-load that never ran — removed (US9).


# ---------------------------------------------------------------------------
# Post-load hooks (domain-specific)
# ---------------------------------------------------------------------------

def _post_load_purchase_order(cur) -> None:
    """Populate lead time actuals + sync open POs after loading purchase orders."""
    cur.execute("""
        INSERT INTO fact_lead_time_actuals
            (po_number, line_number, supplier_id, item_id, loc,
             promised_delivery_date, actual_receipt_date,
             lead_time_days_promised, lead_time_days_actual, source_file)
        SELECT po.po_number,
            ROW_NUMBER() OVER (PARTITION BY po.po_number ORDER BY po.item_id, po.loc)::integer,
            po.supplier_id, po.item_id, po.loc,
            po.original_delivery_date, po.delivery_date,
            (po.original_delivery_date - po.original_ship_date),
            (po.delivery_date - po.original_ship_date),
            'purchase_orders.csv'
        FROM fact_purchase_orders po
        WHERE po.closure_code = 'CLOSED'
          AND po.delivery_date IS NOT NULL
          AND po.original_ship_date IS NOT NULL
        ON CONFLICT (po_number, line_number) DO UPDATE SET
            supplier_id             = EXCLUDED.supplier_id,
            actual_receipt_date     = EXCLUDED.actual_receipt_date,
            lead_time_days_actual   = EXCLUDED.lead_time_days_actual,
            lead_time_days_promised = EXCLUDED.lead_time_days_promised
    """)
    logger.info("  Upserted %s lead time actuals from closed POs", f"{cur.rowcount:,}")

    cur.execute("SAVEPOINT sp_open_po")
    try:
        cur.execute("""
            INSERT INTO fact_open_purchase_orders
                (po_number, po_line_number, item_id, loc, supplier_id,
                 po_date, ordered_qty, received_qty, unit_cost,
                 promised_delivery_date, po_status, line_status, source_file)
            SELECT po.po_number,
                ROW_NUMBER() OVER (PARTITION BY po.po_number ORDER BY po.item_id, po.loc)::integer,
                po.item_id, po.loc, po.supplier_id,
                po.original_ship_date, po.ordered_qty,
                COALESCE(po.orig_po_qty - po.ordered_qty, 0),
                po.net_price, po.delivery_date, 'open', 'open', 'purchase_orders.csv'
            FROM fact_purchase_orders po
            WHERE (po.closure_code IS NULL OR po.closure_code = '')
              AND po.ordered_qty IS NOT NULL
              AND po.original_ship_date IS NOT NULL
            ON CONFLICT (po_number, po_line_number) DO UPDATE SET
                ordered_qty            = EXCLUDED.ordered_qty,
                received_qty           = EXCLUDED.received_qty,
                unit_cost              = EXCLUDED.unit_cost,
                promised_delivery_date = EXCLUDED.promised_delivery_date,
                modified_ts            = NOW()
        """)
        open_po_count = cur.rowcount
        cur.execute("RELEASE SAVEPOINT sp_open_po")
        logger.info("  Upserted %s open POs", f"{open_po_count:,}")
    except psycopg.errors.ForeignKeyViolation:
        cur.execute("ROLLBACK TO SAVEPOINT sp_open_po")
        logger.warning("  Skipped open PO sync — dim_supplier not yet populated")


def _post_load_sourcing(cur) -> None:
    """Sync sourcing data into dim_item_supplier."""
    cur.execute("""
        SELECT EXISTS(
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'dim_item_supplier' AND table_schema = 'public'
        )
    """)
    if not cur.fetchone()[0]:
        logger.info("  dim_item_supplier not found, skipping sourcing sync")
        return

    cur.execute("SAVEPOINT sp_item_supplier")
    try:
        cur.execute("""
            INSERT INTO dim_item_supplier (item_id, loc, supplier_id, is_preferred, lead_time_days)
            SELECT DISTINCT ON (s.item_id, s.loc, s.supplier_id)
                s.item_id, s.loc, s.supplier_id, FALSE, NULL
            FROM dim_sourcing s
            WHERE s.supplier_id IS NOT NULL AND s.supplier_id != ''
            ON CONFLICT (item_id, loc, supplier_id) DO NOTHING
        """)
        logger.info("  Synced %s sourcing rows into dim_item_supplier", f"{cur.rowcount:,}")

        cur.execute("""
            UPDATE dim_item_supplier dis SET is_preferred = TRUE
            WHERE dis.id IN (
                SELECT DISTINCT ON (item_id, loc) id
                FROM dim_item_supplier ORDER BY item_id, loc, id
            ) AND NOT dis.is_preferred
        """)
        logger.info("  Marked %s preferred suppliers", f"{cur.rowcount:,}")
        cur.execute("RELEASE SAVEPOINT sp_item_supplier")
    except psycopg.errors.ForeignKeyViolation:
        cur.execute("ROLLBACK TO SAVEPOINT sp_item_supplier")
        logger.warning("  Skipped item-supplier sync — dim_supplier not yet populated")


# ---------------------------------------------------------------------------
# Main load function
# ---------------------------------------------------------------------------

def load_domain(spec: DomainSpec, csv_path: Path,
                replace_mode: bool = False,
                skip_archive: bool = False,
                incremental_delete: str | None = None) -> dict:
    """Load CSV directly into main table. Single transaction, minimal overhead.

    Returns summary dict: {domain, rows_in, rows_loaded}.
    """
    t_total = time.time()
    csv_size_mb = csv_path.stat().st_size / (1024 ** 2) if csv_path.exists() else 0
    logger.info("=" * 60)
    logger.info("Loading %s -> %s (%.0f MB)", spec.name, spec.table, csv_size_mb)
    logger.info("=" * 60)

    if not csv_path.exists():
        logger.info("SKIPPED — %s not found. Run 'make normalize-all' first.", csv_path.name)
        return {"domain": spec.name, "skipped": True}

    db = get_db_params()
    src_hash = file_hash(csv_path)
    is_forecast = spec.name == "forecast"
    stg_table = f"_stg_{spec.name}"
    stg_alias = "d"
    src_alias = "s"

    # Build SQL for staging table and COPY
    create_stg_sql = (
        f"CREATE TEMP TABLE {qident(stg_table)} ("
        f"_load_seq bigserial, "
        + ", ".join(f"{qident(c)} text" for c in spec.columns)
        + ") ON COMMIT DROP"
    )
    copy_sql = (
        f"COPY {qident(stg_table)} ("
        + ", ".join(qident(c) for c in spec.columns)
        + ") FROM STDIN WITH (FORMAT CSV, HEADER TRUE)"
    )

    # Build INSERT SQL: type-cast + optional dedup
    key_col = "_ck"
    target_cols = [spec.ck_field, *spec.columns]
    select_exprs = [
        f"{src_alias}.{qident(key_col)} AS {qident(spec.ck_field)}",
        *[
            f"{typed_expr_sets(c, spec.int_fields, spec.float_fields, spec.date_fields, src_alias, spec.bool_fields)} AS {qident(c)}"
            for c in spec.columns
        ],
    ]

    # Domains with unique-by-design data (e.g. inventory: item_id+loc+date is unique
    # in source) skip DISTINCT ON + ORDER BY for ~10x faster INSERT on large datasets.
    _SKIP_DEDUP_DOMAINS = {"inventory"}

    if spec.name in _SKIP_DEDUP_DOMAINS:
        # Direct INSERT: no DISTINCT ON, no ORDER BY — much faster for large datasets
        insert_sql = (
            f"INSERT INTO {qident(spec.table)} ("
            + ", ".join(qident(c) for c in target_cols)
            + ") SELECT "
            + ", ".join(select_exprs)
            + " FROM (SELECT *, "
            + business_key_expr(spec, stg_alias)
            + " AS "
            + qident(key_col)
            + " FROM "
            + qident(stg_table)
            + " "
            + stg_alias
            + ") "
            + src_alias
        )
    else:
        # Dedup INSERT: DISTINCT ON business key, keep latest row by _load_seq
        insert_sql = (
            f"INSERT INTO {qident(spec.table)} ("
            + ", ".join(qident(c) for c in target_cols)
            + ") SELECT "
            + ", ".join(select_exprs)
            + " FROM (SELECT DISTINCT ON ("
            + qident(key_col)
            + ") * FROM (SELECT *, "
            + business_key_expr(spec, stg_alias)
            + " AS "
            + qident(key_col)
            + " FROM "
            + qident(stg_table)
            + " "
            + stg_alias
            + ") x ORDER BY "
            + qident(key_col)
            + ", _load_seq DESC) "
            + src_alias
        )

    # Forecast: filter to execution-lag rows only (added after lag resolution)
    forecast_filter = ""

    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        # Session tuning for bulk load
        cur.execute(f"SET work_mem = '{_PG_WORK_MEM}'")
        cur.execute(f"SET maintenance_work_mem = '{_PG_MAINTENANCE_WORK_MEM}'")
        cur.execute("SET synchronous_commit = 'off'")
        cur.execute("SET max_parallel_maintenance_workers = 4")
        cur.execute("SET effective_io_concurrency = 200")

        # Batch tracking (for change detection in incremental mode)
        batch_id = create_batch(cur, spec.name, csv_path.name, src_hash)

        try:
            # Phase 1: COPY CSV into staging
            with profiled_section("create_staging"):
                logger.info("[1/4] COPY %s -> staging ...", csv_path.name)
                t0 = time.time()
                cur.execute(create_stg_sql)

            with profiled_section("copy_csv"):
                with cur.copy(copy_sql) as copy:
                    with csv_path.open("r", encoding="utf-8", newline="") as f:
                        while chunk := f.read(HASH_CHUNK_SIZE):
                            copy.write(chunk)
                cur.execute(f"SELECT count(*) FROM {qident(stg_table)}")
                stg_rows = cur.fetchone()[0]
                logger.info("  %s rows staged (%s)", f"{stg_rows:,}", _elapsed(t0))

                # Check once: is the target table partitioned?
                is_partitioned = _is_partitioned(cur, spec.table)

                # For large datasets (>1M rows), create an index on the business key
                # to speed up the DISTINCT ON dedup in the INSERT.
                # Skip for partitioned targets (e.g. inventory) — these datasets
                # have unique rows by design so the dedup index adds no value.
                if stg_rows > 1_000_000 and not is_partitioned:
                    t_idx = time.time()
                    # Build key expression without table alias (CREATE INDEX has no FROM clause)
                    key_cols = [f"trim({qident(f)})" for f in spec.key_fields]
                    sep = (spec.business_key_separator or "-").replace("'", "''")
                    bk_idx_expr = (
                        key_cols[0] if len(key_cols) == 1
                        else f" || '{sep}' || ".join(key_cols)
                    )
                    cur.execute(
                        f"CREATE INDEX ON {qident(stg_table)} "
                        f"(({bk_idx_expr}), _load_seq DESC)"
                    )
                    logger.info("  Staging index created (%s)", _elapsed(t_idx))

            # Phase 1b: DFU match filter — only load rows with a matching dim_sku entry
            if spec.name in DFU_MATCH_DOMAINS:
                with profiled_section("filter_unmatched_dfus"):
                    t0 = time.time()
                    dfu_deleted = filter_unmatched_dfus(cur, stg_table, spec.name)
                    if dfu_deleted:
                        logger.info("  DFU filter: kept %s, removed %s (%s)",
                                    f"{stg_rows - dfu_deleted:,}",
                                    f"{dfu_deleted:,}", _elapsed(t0))
                        pct = (dfu_deleted / stg_rows * 100) if stg_rows else 0.0
                        if pct > unmatched_warn_pct():
                            logger.warning(
                                "  DFU filter removed %.1f%% of %s rows (> %.1f%% threshold)",
                                pct, spec.name, unmatched_warn_pct())

            # Phase 1b2: FK orphan filter — remove rows referencing missing dimension values
            if spec.name in FK_CHECKS:
                with profiled_section("filter_fk_orphans"):
                    t0 = time.time()
                    fk_deleted = filter_fk_orphans(cur, stg_table, spec.name)
                    if fk_deleted:
                        logger.info("  FK orphan filter: removed %s rows (%s)",
                                    f"{fk_deleted:,}", _elapsed(t0))

            # Phase 1c: Forecast-specific — 12-month filter + execution lag
            archive_count = 0
            if is_forecast:
                # Keep only the last 12 months of forecast data
                planning_dt = get_planning_date()
                cutoff = date(planning_dt.year - 1, planning_dt.month, 1)
                cur.execute(f"""
                    DELETE FROM {qident(stg_table)}
                    WHERE lower(trim("startdate")) IN ({NULL_SQL})
                       OR "startdate"::date < %s
                """, [cutoff])
                trimmed = cur.rowcount
                if trimmed:
                    logger.info("  Trimmed %s rows with startdate before %s",
                                f"{trimmed:,}", cutoff.isoformat())

                # External forecasts skip archive — archive is only for
                # backtest models loaded via load_backtest_forecasts.py.

                # Set lag and execution_lag from dim_sku.
                logger.info("  Resolving execution lag from dim_sku ...")
                t0 = time.time()
                matched = _resolve_forecast_execution_lag(cur, stg_table)
                logger.info("  Set execution lag for %s rows (%s)",
                            f"{matched:,}", _elapsed(t0))
                # No lag filter — all rows are at execution lag by assumption

            # Phase 2: Clear target (partition-aware or traditional)
            saved_indexes = []
            saved_constraints = []

            if is_partitioned:
                with profiled_section("prepare_partitions"):
                    logger.info("[2/4] Preparing %s (partitioned) ...", spec.table)
                    t0 = time.time()
                    if incremental_delete:
                        cur.execute(f"DELETE FROM {qident(spec.table)} WHERE {incremental_delete}")
                        logger.info("  Incremental delete (%s)", _elapsed(t0))
                    else:
                        # Full reload: drop ALL indexes/constraints from parent
                        # (propagates to partitions), then TRUNCATE for fast INSERT
                        saved_indexes = get_secondary_indexes(cur, spec.table)
                        saved_constraints = get_unique_constraints(cur, spec.table)
                        drop_unique_constraints(cur, spec.table, saved_constraints)
                        drop_indexes(cur, saved_indexes)
                        cur.execute(f"TRUNCATE TABLE {qident(spec.table)} CASCADE")
                        logger.info("  Truncated + dropped %d indexes (%s)",
                                    len(saved_indexes) + len(saved_constraints), _elapsed(t0))
            else:
                with profiled_section("drop_indexes"):
                    logger.info("[2/4] Preparing %s ...", spec.table)
                    t0 = time.time()
                    saved_indexes = get_secondary_indexes(cur, spec.table)
                    saved_constraints = get_unique_constraints(cur, spec.table)
                    # Drop UNIQUE constraints FIRST (they depend on backing indexes)
                    drop_unique_constraints(cur, spec.table, saved_constraints)
                    drop_indexes(cur, saved_indexes)

                if replace_mode and is_forecast:
                    cur.execute(
                        f"DELETE FROM {qident(spec.table)} WHERE model_id = %s",
                        [EXTERNAL_MODEL_ID],
                    )
                    logger.info("  Deleted external rows, dropped %d indexes (%s)",
                                len(saved_indexes) + len(saved_constraints), _elapsed(t0))
                elif incremental_delete:
                    cur.execute(f"DELETE FROM {qident(spec.table)} WHERE {incremental_delete}")
                    logger.info("  Incremental delete + dropped %d indexes (%s)",
                                len(saved_indexes) + len(saved_constraints), _elapsed(t0))
                else:
                    cur.execute(f"TRUNCATE TABLE {qident(spec.table)} CASCADE")
                    logger.info("  Truncated + dropped %d indexes (%s)",
                                len(saved_indexes) + len(saved_constraints), _elapsed(t0))

            # Phase 3: INSERT
            if is_partitioned and not incremental_delete:
                # Per-month parallel loading: promote staging to a real table visible
                # to parallel connections, create fresh partitions, INSERT each month
                # in a separate thread. No partition routing, no constraints during load.
                with profiled_section("insert_per_partition"):
                    logger.info("[3/4] INSERT per-month (parallel) -> %s ...", spec.table)
                    t0 = time.time()

                    # Promote temp staging to a real table so parallel connections can see it
                    real_stg = f"_stg_{spec.name}_shared"
                    cur.execute(f"DROP TABLE IF EXISTS {qident(real_stg)}")
                    cur.execute(
                        f"CREATE UNLOGGED TABLE {qident(real_stg)} AS "
                        f"SELECT * FROM {qident(stg_table)}"
                    )
                    conn.commit()
                    logger.info("  Promoted staging to shared table (%s)", _elapsed(t0))

                    # Find distinct months
                    date_col = spec.date_fields and next(iter(spec.date_fields)) or "snapshot_date"
                    cur.execute(
                        f"SELECT DISTINCT date_trunc('month', {stg_alias}.{qident(date_col)}::date)::date "
                        f"FROM {qident(real_stg)} {stg_alias} ORDER BY 1"
                    )
                    months = [r[0] for r in cur.fetchall()]

                    # Create all fresh partitions
                    for m in months:
                        m_str = m.strftime("%Y-%m-%d")
                        y, mo = m.year, m.month
                        end_str = f"{y + 1:04d}-01-01" if mo == 12 else f"{y:04d}-{mo + 1:02d}-01"
                        part_name = f"{spec.table}_{y:04d}_{mo:02d}"

                        cur.execute("""
                            SELECT 1 FROM pg_class
                            WHERE relname = %s AND relnamespace = 'public'::regnamespace
                        """, (part_name,))
                        if cur.fetchone():
                            cur.execute(
                                f"ALTER TABLE {qident(spec.table)} "
                                f"DETACH PARTITION {qident(part_name)}"
                            )
                            cur.execute(f"DROP TABLE {qident(part_name)}")

                        cur.execute(
                            f"CREATE TABLE {qident(part_name)} PARTITION OF {qident(spec.table)} "
                            f"FOR VALUES FROM ('{m_str}') TO ('{end_str}')"
                        )
                    conn.commit()
                    logger.info("  Created %d partitions (%s)", len(months), _elapsed(t0))

                    # Build INSERT SQL targeting the shared staging table
                    # Replace the temp staging table name with the shared one
                    parallel_insert_sql = insert_sql.replace(
                        qident(stg_table), qident(real_stg)
                    )

                    def _insert_month(month_date):
                        """Insert one month's data using a separate DB connection."""
                        m_str = month_date.strftime("%Y-%m-%d")
                        y, mo = month_date.year, month_date.month
                        end_str = f"{y + 1:04d}-01-01" if mo == 12 else f"{y:04d}-{mo + 1:02d}-01"
                        part_name = f"{spec.table}_{y:04d}_{mo:02d}"

                        month_filter = (
                            f" WHERE {src_alias}.{qident(date_col)}::date >= '{m_str}' "
                            f"AND {src_alias}.{qident(date_col)}::date < '{end_str}'"
                        )
                        t_m = time.time()
                        with psycopg.connect(**db) as m_conn, m_conn.cursor() as m_cur:
                            m_cur.execute(f"SET work_mem = '{_PG_WORK_MEM}'")
                            m_cur.execute("SET synchronous_commit = 'off'")
                            m_cur.execute(parallel_insert_sql + month_filter)
                            m_rows = m_cur.rowcount
                            m_conn.commit()
                        elapsed = _elapsed(t_m)
                        logger.info("  %s: %s rows (%s)", part_name, f"{m_rows:,}", elapsed)
                        return m_rows

                    # Parallel INSERT — one thread per month (I/O-bound, not CPU)
                    row_count = 0
                    max_workers = min(len(months), 6)  # cap at 6 parallel connections
                    with ThreadPoolExecutor(max_workers=max_workers) as pool:
                        futures = {pool.submit(_insert_month, m): m for m in months}
                        for fut in as_completed(futures):
                            row_count += fut.result()

                    # Clean up shared staging table
                    cur.execute(f"DROP TABLE IF EXISTS {qident(real_stg)}")
                    conn.commit()

                    logger.info("  Total: %s rows (%s, %s rows/s)",
                                f"{row_count:,}", _elapsed(t0),
                                f"{row_count / max(time.time() - t0, 0.001):,.0f}")
            else:
                with profiled_section("insert_from_staging"):
                    logger.info("[3/4] INSERT -> %s ...", spec.table)
                    t0 = time.time()
                    cur.execute(insert_sql + forecast_filter)
                    row_count = cur.rowcount
                    rate = row_count / max(time.time() - t0, 0.001)
                    logger.info("  %s rows inserted (%s, %s rows/s)",
                                f"{row_count:,}", _elapsed(t0), f"{rate:,.0f}")

            # Phase 3b: Post-load hooks
            with profiled_section("post_load_hooks"):
                if spec.name == "purchase_order":
                    _post_load_purchase_order(cur)
                elif spec.name == "sourcing":
                    _post_load_sourcing(cur)

            # Phase 4: Rebuild indexes
            with profiled_section("recreate_indexes"):
                if not saved_indexes and not saved_constraints:
                    logger.info("[4/4] No indexes to rebuild")
                else:
                    logger.info("[4/4] Rebuilding %d indexes ...",
                                len(saved_indexes) + len(saved_constraints))
                    t0 = time.time()
                    recreate_unique_constraints(cur, spec.table, saved_constraints)
                    recreate_indexes(cur, saved_indexes)
                    logger.info("  Indexes rebuilt (%s)", _elapsed(t0))

            # Forecast: refresh archive views
            with profiled_section("refresh_views"):
                if is_forecast and not skip_archive:
                    logger.info("  Refreshing archive views ...")
                    for mv in MV_REFRESH_ARCHIVE:
                        cur.execute(f"REFRESH MATERIALIZED VIEW {mv}")

            # Complete batch + commit
            complete_batch(cur, batch_id, stg_rows, row_count)
            conn.commit()

        except Exception as exc:
            conn.rollback()
            try:
                with psycopg.connect(**db) as err_conn, err_conn.cursor() as err_cur:
                    fail_batch(err_cur, batch_id, str(exc))
                    err_conn.commit()
            except psycopg.Error:
                pass
            raise

    total = _elapsed(t_total)
    logger.info("Done: %s rows -> %s (%s)", f"{row_count:,}", spec.table, total)
    if is_forecast and not skip_archive:
        logger.info("  Archive: %s rows", f"{archive_count:,}")
    logger.info("=" * 60)

    return {
        "domain": spec.name,
        "rows_in": stg_rows,
        "rows_loaded": row_count,
        "elapsed": total,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    allowed = ", ".join(sorted(DOMAIN_SPECS))
    parser = argparse.ArgumentParser(description="Load normalized CSV into Postgres")
    parser.add_argument("--dataset", required=True, help=allowed)
    parser.add_argument("--replace", action="store_true",
                        help="(forecast only) Replace only model_id='external' rows")
    parser.add_argument("--skip-archive", action="store_true",
                        help="(forecast only) Skip loading all lags into backtest_lag_archive")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    spec = get_spec(args.dataset)
    csv_path = ROOT / "data" / spec.clean_file

    load_domain(
        spec, csv_path,
        replace_mode=args.replace,
        skip_archive=args.skip_archive,
    )


if __name__ == "__main__":
    load_dotenv()
    main()
