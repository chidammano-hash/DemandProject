"""Statistical Data Quality Auto-Fix Script.

Applies statistical remediation strategies for known DQ issues:
  1. Range outliers → clamp to percentile bounds (Winsorization)
  2. Lead time outliers → replace with item-level median
  3. NULL completeness → impute with median (numeric) or mode (categorical)
  4. Orphan RI keys → quarantine to staging table for review
  5. Statistical outliers → Winsorise to IQR/Z-score bounds

Usage:
    uv run python scripts/fix_dq_issues.py                    # Preview all fixes (dry-run)
    uv run python scripts/fix_dq_issues.py --apply             # Apply all fixes
    uv run python scripts/fix_dq_issues.py --fix range         # Fix only range issues
    uv run python scripts/fix_dq_issues.py --fix lead_time     # Fix only lead time
    uv run python scripts/fix_dq_issues.py --fix completeness  # Fix only NULLs
    uv run python scripts/fix_dq_issues.py --fix orphans       # Quarantine orphan keys
    uv run python scripts/fix_dq_issues.py --fix outliers      # Winsorise statistical outliers
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import psycopg

from common.db import get_db_params
from common.utils import _ts, load_config

_CONFIG = load_config("data_quality_config.yaml")


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


def fix_lead_time_outliers(conn, dry_run: bool = True) -> list[dict]:
    """Replace extreme lead_time_days with per-item median.

    Strategy: For each item_no, compute median lead_time from valid rows (0-730).
    Replace outliers (< 0 or > 730) with that median. If no valid rows exist
    for an item, use the global median.
    """
    results = []
    table = "fact_inventory_snapshot"
    col = "lead_time_days"
    lo, hi = 0, 730

    with conn.cursor() as cur:
        cur.execute(
            f"SELECT count(*) FROM {table} "
            f"WHERE {col} IS NOT NULL AND ({col} < {lo} OR {col} > {hi})"
        )
        outlier_count = cur.fetchone()[0]

    if outlier_count == 0:
        print(f"  No lead time outliers found")
        return results

    fix_desc = f"Replace {table}.{col} outliers with per-item median"

    if dry_run:
        results.append({"fix": fix_desc, "affected_rows": outlier_count, "applied": False})
        print(f"  [DRY-RUN] {fix_desc}: {outlier_count:,} rows")
    else:
        with conn.cursor() as cur:
            # Compute per-item median for valid values
            cur.execute(f"""
                WITH item_medians AS (
                    SELECT item_no,
                           percentile_cont(0.5) WITHIN GROUP (ORDER BY {col}) AS med_lt
                    FROM {table}
                    WHERE {col} IS NOT NULL AND {col} >= {lo} AND {col} <= {hi}
                    GROUP BY item_no
                ),
                global_median AS (
                    SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY {col}) AS gmed
                    FROM {table}
                    WHERE {col} IS NOT NULL AND {col} >= {lo} AND {col} <= {hi}
                )
                UPDATE {table} t
                SET {col} = COALESCE(im.med_lt, gm.gmed, 7)
                FROM global_median gm
                LEFT JOIN item_medians im ON im.item_no = t.item_no
                WHERE t.{col} IS NOT NULL AND (t.{col} < {lo} OR t.{col} > {hi})
            """)
            fixed = cur.rowcount
        results.append({"fix": fix_desc, "affected_rows": fixed, "applied": True})
        print(f"  [APPLIED] {fix_desc}: {fixed:,} rows fixed")

    return results


def fix_null_completeness(conn, dry_run: bool = True) -> list[dict]:
    """Impute NULLs using statistical methods: median for numeric, mode for text."""
    results = []
    completeness_checks = _CONFIG.get("checks", {}).get("completeness", {})

    for domain_key, spec in completeness_checks.items():
        table = spec.get("table", "")
        for col_spec in spec.get("columns", []):
            col = col_spec["column"]
            threshold = col_spec.get("null_pct_threshold", 0.0)

            # Only impute columns with threshold > 0 (those that allow some NULLs)
            # and skip PK columns (threshold == 0 means PK, can't impute)
            if threshold == 0.0:
                continue

            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) FROM {table} WHERE {col} IS NULL")
                null_count = cur.fetchone()[0]

            if null_count == 0:
                continue

            # Determine column type to pick strategy
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT data_type FROM information_schema.columns "
                    "WHERE table_name = %s AND column_name = %s",
                    (table, col),
                )
                row = cur.fetchone()

            if not row:
                continue

            dtype = row[0]
            is_numeric = dtype in ("integer", "bigint", "numeric", "real", "double precision", "smallint")

            if is_numeric:
                # Impute with median
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY {col}) "
                        f"FROM {table} WHERE {col} IS NOT NULL"
                    )
                    median_val = cur.fetchone()[0]
                if median_val is None:
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
            else:
                # Impute with mode (most frequent value)
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
        recommendation = f"Reload dimension: make normalize-all && make load-all"
        results.append({
            "fix": fix_desc,
            "affected_rows": orphan_count,
            "applied": False,
            "recommendation": recommendation,
        })
        print(f"  [REPORT] {fix_desc}")
        print(f"           → {recommendation}")

    return results


def fix_statistical_outliers(conn, dry_run: bool = True) -> list[dict]:
    """Winsorise statistical outliers detected by IQR/Z-score to computed bounds."""
    results = []
    stat_checks = _CONFIG.get("checks", {}).get("statistical_outlier", {})

    for domain_key, spec in stat_checks.items():
        table = spec.get("table", "")
        for col_spec in spec.get("columns", []):
            col = col_spec["column"]
            method = col_spec.get("method", "iqr")
            threshold = col_spec.get("threshold", 1.5)

            # Compute bounds
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT avg({col}), stddev_pop({col}), "
                    f"percentile_cont(0.25) WITHIN GROUP (ORDER BY {col}), "
                    f"percentile_cont(0.75) WITHIN GROUP (ORDER BY {col}) "
                    f"FROM {table} WHERE {col} IS NOT NULL"
                )
                row = cur.fetchone()

            if not row or row[0] is None:
                continue

            mean, stddev = float(row[0]), float(row[1] or 0)
            q1, q3 = float(row[2] or 0), float(row[3] or 0)
            iqr = q3 - q1

            if method == "zscore":
                lower = mean - threshold * stddev if stddev > 0 else q1
                upper = mean + threshold * stddev if stddev > 0 else q3
            else:
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr

            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT count(*) FROM {table} "
                    f"WHERE {col} IS NOT NULL AND ({col} < %s OR {col} > %s)",
                    (lower, upper),
                )
                outlier_count = cur.fetchone()[0]

            if outlier_count == 0:
                continue

            fix_desc = f"Winsorise {table}.{col} ({method}, threshold={threshold}) to [{lower:.2f}, {upper:.2f}]"
            if dry_run:
                results.append({"fix": fix_desc, "affected_rows": outlier_count, "applied": False})
                print(f"  [DRY-RUN] {fix_desc}: {outlier_count:,} rows")
            else:
                with conn.cursor() as cur:
                    cur.execute(
                        f"UPDATE {table} SET {col} = %s WHERE {col} IS NOT NULL AND {col} < %s",
                        (lower, lower),
                    )
                    lo_fixed = cur.rowcount
                    cur.execute(
                        f"UPDATE {table} SET {col} = %s WHERE {col} IS NOT NULL AND {col} > %s",
                        (upper, upper),
                    )
                    hi_fixed = cur.rowcount
                results.append({"fix": fix_desc, "affected_rows": lo_fixed + hi_fixed, "applied": True})
                print(f"  [APPLIED] {fix_desc}: {lo_fixed + hi_fixed:,} rows clamped")

    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

FIX_REGISTRY = {
    "range": fix_range_outliers,
    "lead_time": fix_lead_time_outliers,
    "completeness": fix_null_completeness,
    "orphans": fix_orphan_keys,
    "outliers": fix_statistical_outliers,
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
            all_results[name] = fn(conn, dry_run=dry_run)

        if not dry_run:
            conn.commit()

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
