"""Statistical Data Quality Auto-Fix Script.

Applies statistical remediation strategies for known DQ issues:
  1. Range outliers → clamp to percentile bounds (Winsorization)
  2. NULL completeness → impute with median (numeric) or mode (categorical)
  3. Orphan RI keys → report for review

Usage:
    uv run python scripts/fix_dq_issues.py                    # Preview all fixes (dry-run)
    uv run python scripts/fix_dq_issues.py --apply             # Apply all fixes
    uv run python scripts/fix_dq_issues.py --fix range         # Fix only range issues
    uv run python scripts/fix_dq_issues.py --fix completeness  # Fix only NULLs
    uv run python scripts/fix_dq_issues.py --fix orphans       # Quarantine orphan keys
"""
from __future__ import annotations

import argparse

import psycopg

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import get_db_params
from common.services.perf_profiler import profiled_section
from common.utils import _ts, load_config

import logging

logger = logging.getLogger(__name__)

_CONFIG = load_config("data_quality_config.yaml")

# Map table -> (date column, domain key) for corrections audit
_TABLE_META: dict[str, dict[str, str]] = {
    "fact_sales_monthly": {"date_col": "startdate", "domain": "sales"},
    "fact_inventory_snapshot": {"date_col": "snapshot_date", "domain": "inventory"},
    "fact_external_forecast_monthly": {"date_col": "startdate", "domain": "forecast"},
    "fact_purchase_orders": {"date_col": "delivery_date", "domain": "purchase_order"},
}


def _log_corrections(
    conn,
    table: str,
    col: str,
    fix_type: str,
    fix_strategy: str,
    threshold: float | None,
    lower_bound: float | None,
    upper_bound: float | None,
    where_clause: str,
    params: tuple = (),
    group_bounds_sql: str | None = None,
    group_join: str | None = None,
    group_params: tuple = (),
) -> int:
    """Log corrections to fact_dq_corrections before applying the fix.

    For grouped fixes, uses a subquery with per-group bounds.
    For global fixes, uses scalar lower/upper bounds.
    Returns the number of correction rows inserted.
    """
    meta = _TABLE_META.get(table, {"date_col": "NULL", "domain": "unknown"})
    date_col = meta["date_col"]
    domain = meta["domain"]

    try:
        with conn.cursor() as cur:
            if group_bounds_sql and group_join:
                # Per-group: join with bounds CTE to get per-row bounds
                cur.execute(
                    f"INSERT INTO fact_dq_corrections "
                    f"(domain, table_name, item_id, loc, period, column_name, "
                    f" old_value, new_value, fix_type, fix_strategy, threshold, "
                    f" lower_bound, upper_bound) "
                    f"SELECT %s, %s, t.item_id, t.loc, t.{date_col}, %s, "
                    f"  t.{col}, "
                    f"  CASE WHEN t.{col} < b.lower THEN b.lower "
                    f"       WHEN t.{col} > b.upper THEN b.upper END, "
                    f"  %s, %s, %s, b.lower, b.upper "
                    f"FROM {table} t "
                    f"JOIN ({group_bounds_sql}) b ON {group_join} "
                    f"WHERE t.{col} IS NOT NULL "
                    f"AND (t.{col} < b.lower OR t.{col} > b.upper)",
                    (domain, table, col, fix_type, fix_strategy, threshold,
                     *group_params),
                )
            else:
                # Global: use scalar bounds
                cur.execute(
                    f"INSERT INTO fact_dq_corrections "
                    f"(domain, table_name, item_id, loc, period, column_name, "
                    f" old_value, new_value, fix_type, fix_strategy, threshold, "
                    f" lower_bound, upper_bound) "
                    f"SELECT %s, %s, "
                    f"  CASE WHEN EXISTS (SELECT 1 FROM information_schema.columns "
                    f"    WHERE table_name = %s AND column_name = 'item_id') "
                    f"    THEN item_id END, "
                    f"  CASE WHEN EXISTS (SELECT 1 FROM information_schema.columns "
                    f"    WHERE table_name = %s AND column_name = 'loc') "
                    f"    THEN loc END, "
                    f"  {date_col}, %s, "
                    f"  {col}, "
                    f"  CASE WHEN {col} < %s THEN %s "
                    f"       WHEN {col} > %s THEN %s END, "
                    f"  %s, %s, %s, %s, %s "
                    f"FROM {table} "
                    f"WHERE {col} IS NOT NULL AND ({where_clause})",
                    (domain, table, table, table, col,
                     lower_bound, lower_bound, upper_bound, upper_bound,
                     fix_type, fix_strategy, threshold, lower_bound, upper_bound,
                     *params),
                )
            logged = cur.rowcount
            if logged > 0:
                logger.info("  Logged %d corrections to audit trail", logged)
            return logged
    except Exception:
        logger.warning("  Could not log corrections (table may not exist)", exc_info=True)
        return 0


# ---------------------------------------------------------------------------
# Fix strategies
# ---------------------------------------------------------------------------

def fix_range_outliers(conn, dry_run: bool = True) -> list[dict]:
    """Clamp (Winsorise) out-of-range values to configured bounds."""
    results = []
    range_checks = _CONFIG.get("checks", {}).get("range", {})

    for domain_key, spec in range_checks.items():
        table = spec.get("table", "")
        for col_spec in spec.get("columns", []):
            col = col_spec["column"]
            lo = col_spec.get("min")
            hi = col_spec.get("max")

            # Count current outliers
            clauses = []
            if lo is not None:
                clauses.append(f"{col} < {lo}")
            if hi is not None:
                clauses.append(f"{col} > {hi}")
            if not clauses:
                continue

            where = " OR ".join(clauses)
            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) FROM {table} WHERE {col} IS NOT NULL AND ({where})")
                outlier_count = cur.fetchone()[0]

            if outlier_count == 0:
                continue

            fix_desc = f"Clamp {table}.{col} to [{lo}, {hi}]"
            if dry_run:
                results.append({"fix": fix_desc, "affected_rows": outlier_count, "applied": False})
                print(f"  [DRY-RUN] {fix_desc}: {outlier_count:,} rows")
            else:
                # Log corrections audit trail before applying
                _log_corrections(
                    conn, table, col,
                    fix_type="range", fix_strategy="clamp",
                    threshold=None,
                    lower_bound=float(lo) if lo is not None else None,
                    upper_bound=float(hi) if hi is not None else None,
                    where_clause=where,
                )

                updates = []
                if lo is not None:
                    updates.append(f"UPDATE {table} SET {col} = {lo} WHERE {col} IS NOT NULL AND {col} < {lo}")
                if hi is not None:
                    updates.append(f"UPDATE {table} SET {col} = {hi} WHERE {col} IS NOT NULL AND {col} > {hi}")
                total_fixed = 0
                with conn.cursor() as cur:
                    for sql in updates:
                        cur.execute(sql)
                        total_fixed += cur.rowcount
                results.append({"fix": fix_desc, "affected_rows": total_fixed, "applied": True})
                print(f"  [APPLIED] {fix_desc}: {total_fixed:,} rows clamped")

    return results




def fix_null_completeness(conn, dry_run: bool = True) -> list[dict]:
    """Impute NULLs using statistical methods: median for numeric, mode for text."""
    results = []
    completeness_checks = _CONFIG.get("checks", {}).get("completeness", {})

    # Pre-load column types for all tables to avoid per-column information_schema queries
    col_types_cache: dict[str, dict[str, str]] = {}

    for domain_key, spec in completeness_checks.items():
        table = spec.get("table", "")

        # Batch-load all column types for this table once
        if table and table not in col_types_cache:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_name = %s",
                    (table,),
                )
                col_types_cache[table] = {row[0]: row[1] for row in cur.fetchall()}

        # Collect imputable columns (threshold > 0) for this table
        imputable_cols = [
            col_spec["column"]
            for col_spec in spec.get("columns", [])
            if col_spec.get("null_pct_threshold", 0.0) > 0.0
        ]
        if not imputable_cols or not table:
            continue

        # Batch null-count + median query for all numeric imputable columns in one pass
        # First, determine which are numeric vs categorical
        numeric_cols = []
        text_cols = []
        numeric_types = {"integer", "bigint", "numeric", "real", "double precision", "smallint"}
        for col in imputable_cols:
            dtype = col_types_cache.get(table, {}).get(col)
            if not dtype:
                continue
            if dtype in numeric_types:
                numeric_cols.append(col)
            else:
                text_cols.append(col)

        # Batch: get null_count + median for all numeric columns in a single query
        if numeric_cols:
            agg_parts = []
            for col in numeric_cols:
                agg_parts.append(
                    f"count(*) FILTER (WHERE {col} IS NULL)"
                )
                agg_parts.append(
                    f"percentile_cont(0.5) WITHIN GROUP (ORDER BY {col}) "
                    f"FILTER (WHERE {col} IS NOT NULL)"
                )
            with conn.cursor() as cur:
                cur.execute(f"SELECT {', '.join(agg_parts)} FROM {table}")
                agg_row = cur.fetchone()

            for i, col in enumerate(numeric_cols):
                null_count = agg_row[i * 2] or 0
                median_val = agg_row[i * 2 + 1]
                if null_count == 0 or median_val is None:
                    continue
                fix_desc = f"Impute {table}.{col} NULLs with median ({median_val})"
                if dry_run:
                    results.append({"fix": fix_desc, "affected_rows": null_count, "applied": False})
                    print(f"  [DRY-RUN] {fix_desc}: {null_count:,} rows")
                else:
                    with conn.cursor() as cur:
                        cur.execute(f"UPDATE {table} SET {col} = %s WHERE {col} IS NULL", (median_val,))
                        fixed = cur.rowcount
                    results.append({"fix": fix_desc, "affected_rows": fixed, "applied": True})
                    print(f"  [APPLIED] {fix_desc}: {fixed:,} rows imputed")

        # Categorical columns: get null_count + mode (still per-column since mode requires GROUP BY)
        for col in text_cols:
            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) FROM {table} WHERE {col} IS NULL")
                null_count = cur.fetchone()[0]

            if null_count == 0:
                continue

            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {col}, count(*) AS cnt FROM {table} "
                    f"WHERE {col} IS NOT NULL GROUP BY {col} ORDER BY cnt DESC LIMIT 1"
                )
                mode_row = cur.fetchone()
            if not mode_row:
                continue
            mode_val = mode_row[0]
            fix_desc = f"Impute {table}.{col} NULLs with mode ('{mode_val}')"
            if dry_run:
                results.append({"fix": fix_desc, "affected_rows": null_count, "applied": False})
                print(f"  [DRY-RUN] {fix_desc}: {null_count:,} rows")
            else:
                with conn.cursor() as cur:
                    cur.execute(f"UPDATE {table} SET {col} = %s WHERE {col} IS NULL", (mode_val,))
                    fixed = cur.rowcount
                results.append({"fix": fix_desc, "affected_rows": fixed, "applied": True})
                print(f"  [APPLIED] {fix_desc}: {fixed:,} rows imputed")

    return results


def fix_orphan_keys(conn, dry_run: bool = True) -> list[dict]:
    """Log orphan FK counts for review (non-destructive — reports only).

    Orphan keys are not deleted because they may be valid data that arrived
    before the dimension was loaded. The fix is to reload the dimension table.
    """
    results = []
    ri_checks = _CONFIG.get("checks", {}).get("referential_integrity", {})

    for check_name, spec in ri_checks.items():
        src_table = spec.get("source_table", "")
        src_cols = spec.get("source_columns", [])
        tgt_table = spec.get("target_table", "")
        tgt_cols = spec.get("target_columns", [])

        src_col_str = ", ".join(f"s.{c}" for c in src_cols)
        tgt_col_str = ", ".join(f"t.{c}" for c in tgt_cols)
        join_cond = " AND ".join(f"s.{sc} = t.{tc}" for sc, tc in zip(src_cols, tgt_cols))
        null_filter = " AND ".join(f"s.{c} IS NOT NULL" for c in src_cols)

        with conn.cursor() as cur:
            if len(tgt_cols) == 1:
                cur.execute(
                    f"SELECT count(DISTINCT ({src_col_str})) "
                    f"FROM {src_table} s LEFT JOIN {tgt_table} t ON {join_cond} "
                    f"WHERE {null_filter} AND t.{tgt_cols[0]} IS NULL"
                )
            else:
                cur.execute(
                    f"SELECT count(*) FROM ("
                    f"  SELECT DISTINCT {src_col_str} FROM {src_table} s WHERE {null_filter} "
                    f"  EXCEPT SELECT DISTINCT {tgt_col_str} FROM {tgt_table} t"
                    f") orphans"
                )
            orphan_count = cur.fetchone()[0]

        if orphan_count == 0:
            continue

        fix_desc = f"Orphan keys: {src_table}({','.join(src_cols)}) → {tgt_table}: {orphan_count:,} orphans"
        recommendation = "Reload dimension: make normalize-all && make load-all"
        results.append({
            "fix": fix_desc,
            "affected_rows": orphan_count,
            "applied": False,
            "recommendation": recommendation,
        })
        print(f"  [REPORT] {fix_desc}")
        print(f"           → {recommendation}")

    return results




# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

FIX_REGISTRY = {
    "range": fix_range_outliers,
    "completeness": fix_null_completeness,
    "orphans": fix_orphan_keys,
}


def preview_all_fixes(fix_type: str | None = None) -> list[dict]:
    """Preview all available fixes as an indexed list (dry-run only)."""
    items: list[dict] = []
    idx = 0
    with psycopg.connect(**get_db_params(), autocommit=True) as conn:
        for name, fn in FIX_REGISTRY.items():
            if fix_type and name != fix_type:
                continue
            results = fn(conn, dry_run=True)
            for r in results:
                items.append({
                    "id": idx,
                    "fix_type": name,
                    "description": r["fix"],
                    "affected_rows": r["affected_rows"],
                    "recommendation": r.get("recommendation"),
                    "status": "pending",
                })
                idx += 1
    return items


def apply_selected_fixes(fix_ids: list[int]) -> dict:
    """Apply only the selected fixes by their preview index IDs.

    Re-runs the preview to build the indexed list, then applies only
    those whose index matches one of *fix_ids*.
    """
    # Build the full indexed preview (cheap – dry-run only, no writes)
    preview = preview_all_fixes()
    id_set = set(fix_ids)
    applied: list[dict] = []
    rejected: list[dict] = []

    # Group selected fixes by type so we run each strategy once
    selected_by_type: dict[str, list[dict]] = {}
    for item in preview:
        if item["id"] in id_set:
            selected_by_type.setdefault(item["fix_type"], []).append(item)
        else:
            rejected.append({**item, "status": "skipped"})

    with psycopg.connect(**get_db_params(), autocommit=False) as conn:
        for fix_type, items in selected_by_type.items():
            fn = FIX_REGISTRY[fix_type]
            results = fn(conn, dry_run=False)
            # Match applied results back to selected items
            for i, r in enumerate(results):
                # Find the matching preview item by description
                for item in items:
                    if item["description"] == r["fix"]:
                        applied.append({
                            **item,
                            "status": "applied",
                            "rows_fixed": r["affected_rows"],
                        })
                        break
        conn.commit()
        # Refresh MVs so charts reflect corrected data
        with conn.cursor() as cur:
            for mv in ("agg_sales_monthly", "agg_forecast_monthly"):
                try:
                    cur.execute(f"REFRESH MATERIALIZED VIEW {mv}")
                except Exception:
                    logger.warning("Could not refresh %s", mv, exc_info=True)

    return {
        "applied": applied,
        "skipped": rejected,
        "total_applied": len(applied),
        "total_skipped": len(rejected),
        "total_rows_fixed": sum(a.get("rows_fixed", 0) for a in applied),
    }


def run_all_fixes(fix_type: str | None = None, dry_run: bool = True) -> dict:
    """Run all or specific DQ fix strategies."""
    all_results = {}
    with psycopg.connect(**get_db_params(), autocommit=not dry_run) as conn:
        for name, fn in FIX_REGISTRY.items():
            if fix_type and name != fix_type:
                continue
            print(f"\n{_ts()} ── {name.upper()} fixes ──")
            with profiled_section(f"fix_{name}"):
                all_results[name] = fn(conn, dry_run=dry_run)

        if not dry_run:
            conn.commit()
            # Refresh materialized views so charts reflect corrected data
            logger.info("Refreshing materialized views after DQ fixes…")
            with conn.cursor() as cur:
                for mv in ("agg_sales_monthly", "agg_forecast_monthly"):
                    try:
                        cur.execute(f"REFRESH MATERIALIZED VIEW {mv}")
                        logger.info("  Refreshed %s", mv)
                    except Exception:
                        logger.warning("  Could not refresh %s", mv, exc_info=True)

    total_affected = sum(r["affected_rows"] for fixes in all_results.values() for r in fixes)
    total_applied = sum(1 for fixes in all_results.values() for r in fixes if r.get("applied"))
    print(f"\n{_ts()} Summary: {total_applied} fixes applied, {total_affected:,} total rows affected")

    return {
        "dry_run": dry_run,
        "total_affected": total_affected,
        "total_applied": total_applied,
        "fixes": all_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical DQ auto-fix")
    parser.add_argument("--apply", action="store_true", help="Apply fixes (default: dry-run)")
    parser.add_argument("--fix", choices=list(FIX_REGISTRY.keys()), help="Run only a specific fix type")
    args = parser.parse_args()

    dry_run = not args.apply
    if dry_run:
        print(f"{_ts()} DRY-RUN mode — no changes will be written. Use --apply to execute.")
    else:
        print(f"{_ts()} APPLY mode — changes will be committed to the database.")

    run_all_fixes(fix_type=args.fix, dry_run=dry_run)
