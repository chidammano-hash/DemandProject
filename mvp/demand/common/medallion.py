"""Medallion pipeline: Bronze → Silver → Gold with DQ gating and audit trail.

Provides the core functions that `load_dataset_postgres.py --medallion` calls.
Each function is independently testable and operates within a caller-managed
transaction.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import psycopg

from common.domain_specs import DomainSpec
from common.utils import _ts, load_config

# Reuse existing SQL helpers from the loader
NULL_SQL = "'', 'null', 'none', 'na', 'n/a'"

_CFG: dict | None = None


def _config() -> dict:
    global _CFG
    if _CFG is None:
        _CFG = load_config("medallion")
    return _CFG


def _elapsed(t0: float) -> str:
    dt = time.time() - t0
    if dt < 60:
        return f"{dt:.1f}s"
    m, s = divmod(dt, 60)
    return f"{int(m)}m {s:.0f}s"


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def typed_expr(field: str, spec: DomainSpec, src_alias: str) -> str:
    """Generate SQL cast expression for a column based on DomainSpec types."""
    col = f"{src_alias}.{qident(field)}"
    if field in spec.int_fields:
        return (
            f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL "
            f"ELSE {col}::integer END"
        )
    if field in spec.float_fields:
        return (
            f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL "
            f"ELSE {col}::numeric END"
        )
    if field in spec.date_fields:
        return (
            f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL "
            f"ELSE {col}::date END"
        )
    return col


def business_key_expr(spec: DomainSpec, src_alias: str) -> str:
    """Generate SQL expression for the composite business key."""
    cols = [f"trim({src_alias}.{qident(f)})" for f in spec.key_fields]
    if len(cols) == 1:
        return cols[0]
    sep = (spec.business_key_separator or "-").replace("'", "''")
    return f" || '{sep}' || ".join(cols)


# ---------------------------------------------------------------------------
# Batch lifecycle
# ---------------------------------------------------------------------------

def create_batch(cur, domain: str, source_file: str | None = None,
                 source_hash: str | None = None,
                 metadata: dict | None = None) -> int:
    """Create a new load batch record. Returns batch_id."""
    cur.execute(
        """INSERT INTO audit_load_batch
               (domain, layer, source_file, source_hash, status, metadata)
           VALUES (%s, 'bronze', %s, %s, 'running', %s)
           RETURNING batch_id""",
        [domain, source_file, source_hash,
         json.dumps(metadata) if metadata else None],
    )
    return cur.fetchone()[0]


def complete_batch(cur, batch_id: int, row_count_in: int,
                   row_count_out: int, quarantined: int = 0) -> None:
    """Mark a batch as completed."""
    cur.execute(
        """UPDATE audit_load_batch
           SET status = 'completed', row_count_in = %s, row_count_out = %s,
               row_count_quarantined = %s, completed_at = now()
           WHERE batch_id = %s""",
        [row_count_in, row_count_out, quarantined, batch_id],
    )


def fail_batch(cur, batch_id: int, error: str) -> None:
    """Mark a batch as failed."""
    cur.execute(
        """UPDATE audit_load_batch
           SET status = 'failed', error_message = %s, completed_at = now()
           WHERE batch_id = %s""",
        [error, batch_id],
    )


def file_hash(path: Path) -> str:
    """Compute SHA-256 of a file (first 10 MB for speed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for _ in range(10):
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Phase 1: Bronze Ingest
# ---------------------------------------------------------------------------

def ingest_bronze(cur, spec: DomainSpec, csv_path: Path,
                  batch_id: int) -> int:
    """COPY clean CSV into bronze_<domain> table (all TEXT). Returns row count."""
    bronze_table = f"bronze_{spec.name}"
    col_list = ", ".join(qident(c) for c in spec.columns)

    copy_sql = (
        f"COPY {qident(bronze_table)} ({col_list}, _load_batch_id, _source_row_num) "
        f"FROM STDIN WITH (FORMAT CSV, HEADER FALSE)"
    )

    # We need to inject batch_id and row number per row.
    # Strategy: use a temp staging table then INSERT into bronze.
    stg = f"_stg_bronze_{spec.name}"
    cur.execute(
        f"CREATE TEMP TABLE {qident(stg)} ("
        + f"_row_num bigserial, "
        + ", ".join(f"{qident(c)} text" for c in spec.columns)
        + ") ON COMMIT DROP"
    )

    copy_stg_sql = (
        f"COPY {qident(stg)} ({col_list}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)"
    )
    with cur.copy(copy_stg_sql) as copy:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            while chunk := f.read(1024 * 1024):
                copy.write(chunk)

    # Count staging rows
    cur.execute(f"SELECT count(*) FROM {qident(stg)}")
    stg_count = cur.fetchone()[0]

    # INSERT into persistent bronze table
    cur.execute(
        f"INSERT INTO {qident(bronze_table)} ({col_list}, _load_batch_id, _source_row_num) "
        f"SELECT {col_list}, %s, _row_num FROM {qident(stg)}",
        [batch_id],
    )

    return stg_count


# ---------------------------------------------------------------------------
# Phase 2: Silver Promotion (type casting + dedup)
# ---------------------------------------------------------------------------

def promote_to_silver(cur, spec: DomainSpec, batch_id: int) -> tuple[int, int]:
    """Type-cast and dedup from bronze → silver. Returns (inserted, quarantined)."""
    bronze_table = f"bronze_{spec.name}"
    silver_table = f"silver_{spec.name}"
    src = "b"

    # Build typed SELECT expressions
    ck_expr = business_key_expr(spec, src)
    typed_cols = [
        f"{ck_expr} AS {qident(spec.ck_field)}",
        *[f"{typed_expr(c, spec, src)} AS {qident(c)}" for c in spec.columns],
    ]

    # First: try to cast all rows, quarantine failures
    # Use a CTE that attempts casting; rows that fail go to quarantine
    # For simplicity, we do a two-pass approach:
    # Pass 1: INSERT valid rows into silver
    # Pass 2: Identify and quarantine bad rows

    # Pass 1: Insert with error trapping via a function wrapper
    # Since Postgres doesn't have TRY_CAST, we use a safe approach:
    # attempt the full INSERT and catch rows that fail via savepoint

    # Simpler approach: INSERT all, let type errors fail the whole batch
    # then quarantine known bad patterns before retrying.
    # For MVP: insert all, quarantine is handled by DQ gate checks (Phase 3).

    select_sql = ", ".join(typed_cols)
    dedup_inner = (
        f"SELECT DISTINCT ON ({qident('_ck')}) *, "
        f"{ck_expr} AS _ck, {src}._bronze_id "
        f"FROM {qident(bronze_table)} {src} "
        f"WHERE {src}._load_batch_id = %s "
        f"ORDER BY _ck, {src}._bronze_id DESC"
    )

    # Build final column list for silver (including metadata)
    silver_cols = [
        spec.ck_field, *spec.columns,
        "_bronze_id", "_load_batch_id", "_dq_status",
    ]

    # For sales, add original snapshot columns
    is_sales = spec.name == "sales"
    if is_sales:
        silver_cols.extend(["_orig_qty_shipped", "_orig_qty_ordered", "_orig_qty"])

    insert_sql = (
        f"INSERT INTO {qident(silver_table)} ("
        + ", ".join(qident(c) for c in silver_cols)
        + ") SELECT "
        + ", ".join([
            f"r.{qident(spec.ck_field)}",
            *[f"r.{qident(c)}" for c in spec.columns],
            "r._bronze_id",
            f"{batch_id}",
            "'pending'",
        ])
    )
    if is_sales:
        insert_sql += ", r.qty_shipped, r.qty_ordered, r.qty"

    insert_sql += (
        f" FROM ({dedup_inner}) sub "
        f"JOIN LATERAL (SELECT "
        + ", ".join(typed_cols)
        + f", sub._bronze_id FROM (VALUES(1)) v) r ON TRUE"
    )

    # Simplified approach: use a subquery with typed expressions directly
    typed_col_list = [
        f"{ck_expr} AS {qident(spec.ck_field)}",
        *[f"{typed_expr(c, spec, src)} AS {qident(c)}" for c in spec.columns],
    ]

    # Use DISTINCT ON for dedup, with type casting inline
    final_sql = (
        f"INSERT INTO {qident(silver_table)} ("
        + ", ".join(qident(c) for c in silver_cols)
        + f") SELECT "
        + ", ".join([
            f"sub.{qident(spec.ck_field)}",
            *[f"sub.{qident(c)}" for c in spec.columns],
            "sub._bronze_id",
            f"{batch_id}",
            "'pending'",
        ])
    )
    if is_sales:
        final_sql += ", sub.qty_shipped, sub.qty_ordered, sub.qty"

    final_sql += (
        f" FROM ("
        f"SELECT DISTINCT ON (_ck) "
        + ", ".join(typed_col_list)
        + f", {src}._bronze_id, {ck_expr} AS _ck"
        + f" FROM {qident(bronze_table)} {src}"
        + f" WHERE {src}._load_batch_id = %s"
        + f" ORDER BY _ck, {src}._bronze_id DESC"
        + f") sub"
    )

    cur.execute(final_sql, [batch_id])
    inserted = cur.rowcount

    return inserted, 0  # quarantine count populated by DQ gate


# ---------------------------------------------------------------------------
# Phase 3: Silver DQ Gate
# ---------------------------------------------------------------------------

def run_silver_dq_gate(cur, spec: DomainSpec, batch_id: int) -> dict:
    """Run blocking DQ checks on silver rows. Returns gate result dict.

    Quarantines rows that fail blocking checks.
    """
    silver_table = f"silver_{spec.name}"
    cfg = _config()
    gates = cfg.get("promotion_gates", {})
    blocking = set(gates.get("blocking_checks", []))
    min_pass = float(gates.get("min_pass_rate", 95.0))

    total_quarantined = 0

    # Count total rows for this batch
    cur.execute(
        f"SELECT count(*) FROM {qident(silver_table)} WHERE _load_batch_id = %s",
        [batch_id],
    )
    total_rows = cur.fetchone()[0]
    if total_rows == 0:
        return {"passed": True, "total": 0, "quarantined": 0, "pass_rate": 100.0}

    # --- Completeness check: quarantine rows with NULL in PK columns ---
    if "completeness" in blocking:
        pk_cols = list(spec.key_fields)
        for col in pk_cols:
            where = f"{qident(col)} IS NULL AND _load_batch_id = %s AND _dq_status = 'pending'"
            # Get quarantine candidates
            cur.execute(
                f"SELECT _silver_id, _bronze_id, row_to_json(t.*) "
                f"FROM {qident(silver_table)} t "
                f"WHERE {where}",
                [batch_id],
            )
            bad_rows = cur.fetchall()
            for silver_id, bronze_id, raw_json in bad_rows:
                cur.execute(
                    """INSERT INTO silver_quarantine
                       (domain, _bronze_id, _load_batch_id, rejection_reason,
                        rejection_details, raw_row)
                       VALUES (%s, %s, %s, 'null_pk', %s, %s)""",
                    [spec.name, bronze_id, batch_id,
                     json.dumps({"column": col}),
                     json.dumps(raw_json) if isinstance(raw_json, dict) else raw_json],
                )
            if bad_rows:
                cur.execute(
                    f"UPDATE {qident(silver_table)} SET _dq_status = 'quarantined' "
                    f"WHERE {where}",
                    [batch_id],
                )
                total_quarantined += len(bad_rows)

    # --- Range check: quarantine rows with out-of-range numeric values ---
    if "range" in blocking:
        range_cfg = load_config("data_quality").get("checks", {}).get("range", {})
        domain_ranges = range_cfg.get(spec.name, {})
        if isinstance(domain_ranges, dict) and "columns" in domain_ranges:
            for col_def in domain_ranges["columns"]:
                col = col_def["column"]
                col_min = col_def.get("min")
                col_max = col_def.get("max")
                conditions = []
                if col_min is not None:
                    conditions.append(f"{qident(col)} < {col_min}")
                if col_max is not None:
                    conditions.append(f"{qident(col)} > {col_max}")
                if conditions:
                    where = (
                        f"({' OR '.join(conditions)}) "
                        f"AND _load_batch_id = %s AND _dq_status = 'pending'"
                    )
                    cur.execute(
                        f"SELECT _silver_id, _bronze_id, row_to_json(t.*) "
                        f"FROM {qident(silver_table)} t WHERE {where}",
                        [batch_id],
                    )
                    bad_rows = cur.fetchall()
                    for silver_id, bronze_id, raw_json in bad_rows:
                        cur.execute(
                            """INSERT INTO silver_quarantine
                               (domain, _bronze_id, _load_batch_id, rejection_reason,
                                rejection_details, raw_row)
                               VALUES (%s, %s, %s, 'range_violation', %s, %s)""",
                            [spec.name, bronze_id, batch_id,
                             json.dumps({"column": col, "min": col_min, "max": col_max}),
                             json.dumps(raw_json) if isinstance(raw_json, dict) else raw_json],
                        )
                    if bad_rows:
                        cur.execute(
                            f"UPDATE {qident(silver_table)} SET _dq_status = 'quarantined' "
                            f"WHERE {where}",
                            [batch_id],
                        )
                        total_quarantined += len(bad_rows)

    # --- Mark remaining pending rows as passed ---
    cur.execute(
        f"UPDATE {qident(silver_table)} SET _dq_status = 'passed' "
        f"WHERE _load_batch_id = %s AND _dq_status = 'pending'",
        [batch_id],
    )
    passed_count = cur.rowcount

    pass_rate = (passed_count / total_rows * 100) if total_rows > 0 else 100.0
    gate_passed = pass_rate >= min_pass

    return {
        "passed": gate_passed,
        "total": total_rows,
        "quarantined": total_quarantined,
        "passed_count": passed_count,
        "pass_rate": round(pass_rate, 2),
        "min_pass_rate": min_pass,
    }


# ---------------------------------------------------------------------------
# Phase 4: Silver DQ Fixes with Audit
# ---------------------------------------------------------------------------

def apply_silver_fixes(cur, spec: DomainSpec, batch_id: int) -> dict:
    """Apply auto-fix strategies to silver layer with full audit trail.

    For sales: snapshots original values before any corrections.
    Returns summary dict.
    """
    cfg = _config()
    fix_cfg = cfg.get("auto_fix", {}).get(spec.name, {})
    if not fix_cfg.get("enabled", False):
        return {"fixes_applied": 0, "domain": spec.name}

    silver_table = f"silver_{spec.name}"
    strategies = fix_cfg.get("strategies", [])
    preserve_original = fix_cfg.get("preserve_original", False)
    total_fixes = 0

    # For sales: snapshot originals before fixing
    if preserve_original and spec.name == "sales":
        cur.execute(
            f"UPDATE {qident(silver_table)} "
            f"SET _orig_qty_shipped = qty_shipped, "
            f"    _orig_qty_ordered = qty_ordered, "
            f"    _orig_qty = qty "
            f"WHERE _load_batch_id = %s AND _dq_status = 'passed' AND NOT _is_corrected",
            [batch_id],
        )

    dq_cfg = load_config("data_quality")

    for strategy in strategies:
        if strategy == "range":
            total_fixes += _fix_range_with_audit(
                cur, spec, silver_table, batch_id, dq_cfg
            )
        elif strategy == "completeness":
            total_fixes += _fix_completeness_with_audit(
                cur, spec, silver_table, batch_id, dq_cfg
            )
        elif strategy == "outliers":
            total_fixes += _fix_outliers_with_audit(
                cur, spec, silver_table, batch_id
            )
        elif strategy == "lead_time":
            total_fixes += _fix_lead_time_with_audit(
                cur, spec, silver_table, batch_id
            )

    # Mark corrected rows
    if total_fixes > 0 and preserve_original:
        cur.execute(
            f"UPDATE {qident(silver_table)} SET _is_corrected = TRUE "
            f"WHERE _load_batch_id = %s AND _dq_status = 'passed' "
            f"AND ({qident(spec.ck_field)}) IN ("
            f"  SELECT DISTINCT row_key FROM audit_dq_corrections "
            f"  WHERE load_batch_id = %s AND domain = %s"
            f")",
            [batch_id, batch_id, spec.name],
        )

    return {"fixes_applied": total_fixes, "domain": spec.name, "strategies": strategies}


def _record_correction(cur, domain: str, table: str, row_key: str,
                       column: str, old_val: Any, new_val: Any,
                       fix_type: str, fix_strategy: str,
                       batch_id: int, metadata: dict | None = None) -> None:
    """Write a single correction record to the audit table."""
    cur.execute(
        """INSERT INTO audit_dq_corrections
           (domain, table_name, row_key, column_name, old_value, new_value,
            fix_type, fix_strategy, applied_by, load_batch_id, metadata)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'system', %s, %s)""",
        [domain, table, row_key, column,
         str(old_val) if old_val is not None else None,
         str(new_val) if new_val is not None else None,
         fix_type, fix_strategy, batch_id,
         json.dumps(metadata) if metadata else None],
    )


def _fix_range_with_audit(cur, spec: DomainSpec, silver_table: str,
                          batch_id: int, dq_cfg: dict) -> int:
    """Clamp out-of-range values and record corrections."""
    range_checks = dq_cfg.get("checks", {}).get("range", {})
    domain_cfg = range_checks.get(spec.name, {})
    if not isinstance(domain_cfg, dict) or "columns" not in domain_cfg:
        return 0

    total = 0
    for col_def in domain_cfg["columns"]:
        col = col_def["column"]
        col_min = col_def.get("min")
        col_max = col_def.get("max")

        # Find rows needing clamping
        conditions = []
        if col_min is not None:
            conditions.append(f"{qident(col)} < {col_min}")
        if col_max is not None:
            conditions.append(f"{qident(col)} > {col_max}")
        if not conditions:
            continue

        cur.execute(
            f"SELECT {qident(spec.ck_field)}, {qident(col)} "
            f"FROM {qident(silver_table)} "
            f"WHERE ({' OR '.join(conditions)}) "
            f"AND _load_batch_id = %s AND _dq_status = 'passed'",
            [batch_id],
        )
        for row_key, old_val in cur.fetchall():
            if old_val is not None:
                new_val = old_val
                if col_min is not None and old_val < col_min:
                    new_val = col_min
                if col_max is not None and old_val > col_max:
                    new_val = col_max
                _record_correction(
                    cur, spec.name, silver_table, str(row_key), col,
                    old_val, new_val, "clamp", "range", batch_id,
                    {"min": col_min, "max": col_max},
                )
                total += 1

        # Apply the actual clamp UPDATE
        if col_min is not None:
            cur.execute(
                f"UPDATE {qident(silver_table)} SET {qident(col)} = %s "
                f"WHERE {qident(col)} < %s AND _load_batch_id = %s AND _dq_status = 'passed'",
                [col_min, col_min, batch_id],
            )
        if col_max is not None:
            cur.execute(
                f"UPDATE {qident(silver_table)} SET {qident(col)} = %s "
                f"WHERE {qident(col)} > %s AND _load_batch_id = %s AND _dq_status = 'passed'",
                [col_max, col_max, batch_id],
            )

    return total


def _fix_completeness_with_audit(cur, spec: DomainSpec, silver_table: str,
                                 batch_id: int, dq_cfg: dict) -> int:
    """Impute NULLs (numeric→median, categorical→mode) with audit."""
    comp_checks = dq_cfg.get("checks", {}).get("completeness", {})
    domain_cfg = comp_checks.get(spec.name, {})
    if not isinstance(domain_cfg, dict) or "columns" not in domain_cfg:
        return 0

    total = 0
    for col_def in domain_cfg["columns"]:
        col = col_def["column"]
        threshold = col_def.get("null_pct_threshold", 0)
        if threshold == 0:
            continue  # Skip PK columns (threshold=0 means no nulls allowed)

        is_numeric = col in spec.float_fields or col in spec.int_fields

        if is_numeric:
            # Compute median for imputation
            cur.execute(
                f"SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY {qident(col)}) "
                f"FROM {qident(silver_table)} "
                f"WHERE {qident(col)} IS NOT NULL "
                f"AND _load_batch_id = %s AND _dq_status = 'passed'",
                [batch_id],
            )
            median_val = cur.fetchone()[0]
            if median_val is None:
                continue

            # Record corrections
            cur.execute(
                f"SELECT {qident(spec.ck_field)} "
                f"FROM {qident(silver_table)} "
                f"WHERE {qident(col)} IS NULL "
                f"AND _load_batch_id = %s AND _dq_status = 'passed'",
                [batch_id],
            )
            for (row_key,) in cur.fetchall():
                _record_correction(
                    cur, spec.name, silver_table, str(row_key), col,
                    None, median_val, "median_impute", "completeness", batch_id,
                )
                total += 1

            # Apply
            cur.execute(
                f"UPDATE {qident(silver_table)} SET {qident(col)} = %s "
                f"WHERE {qident(col)} IS NULL "
                f"AND _load_batch_id = %s AND _dq_status = 'passed'",
                [median_val, batch_id],
            )
        else:
            # Mode imputation for categoricals
            cur.execute(
                f"SELECT {qident(col)}, count(*) "
                f"FROM {qident(silver_table)} "
                f"WHERE {qident(col)} IS NOT NULL "
                f"AND _load_batch_id = %s AND _dq_status = 'passed' "
                f"GROUP BY 1 ORDER BY 2 DESC LIMIT 1",
                [batch_id],
            )
            row = cur.fetchone()
            if not row:
                continue
            mode_val = row[0]

            cur.execute(
                f"SELECT {qident(spec.ck_field)} "
                f"FROM {qident(silver_table)} "
                f"WHERE {qident(col)} IS NULL "
                f"AND _load_batch_id = %s AND _dq_status = 'passed'",
                [batch_id],
            )
            for (row_key,) in cur.fetchall():
                _record_correction(
                    cur, spec.name, silver_table, str(row_key), col,
                    None, mode_val, "mode_impute", "completeness", batch_id,
                )
                total += 1

            cur.execute(
                f"UPDATE {qident(silver_table)} SET {qident(col)} = %s "
                f"WHERE {qident(col)} IS NULL "
                f"AND _load_batch_id = %s AND _dq_status = 'passed'",
                [mode_val, batch_id],
            )

    return total


def _fix_outliers_with_audit(cur, spec: DomainSpec, silver_table: str,
                             batch_id: int) -> int:
    """Winsorise statistical outliers (IQR method) with audit."""
    total = 0
    numeric_cols = list(spec.float_fields | spec.int_fields)
    # Only fix columns that are actual measure columns (not IDs)
    skip = set(spec.key_fields) | {"type", "lag", "execution_lag"}
    target_cols = [c for c in numeric_cols if c in spec.columns and c not in skip]

    for col in target_cols:
        cur.execute(
            f"SELECT percentile_cont(0.25) WITHIN GROUP (ORDER BY {qident(col)}), "
            f"       percentile_cont(0.75) WITHIN GROUP (ORDER BY {qident(col)}) "
            f"FROM {qident(silver_table)} "
            f"WHERE {qident(col)} IS NOT NULL "
            f"AND _load_batch_id = %s AND _dq_status = 'passed'",
            [batch_id],
        )
        row = cur.fetchone()
        if not row or row[0] is None or row[1] is None:
            continue
        q1, q3 = float(row[0]), float(row[1])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Record corrections for rows outside bounds
        cur.execute(
            f"SELECT {qident(spec.ck_field)}, {qident(col)} "
            f"FROM {qident(silver_table)} "
            f"WHERE ({qident(col)} < %s OR {qident(col)} > %s) "
            f"AND {qident(col)} IS NOT NULL "
            f"AND _load_batch_id = %s AND _dq_status = 'passed'",
            [lower, upper, batch_id],
        )
        for row_key, old_val in cur.fetchall():
            new_val = max(lower, min(upper, float(old_val)))
            _record_correction(
                cur, spec.name, silver_table, str(row_key), col,
                old_val, new_val, "winsorise", "outliers", batch_id,
                {"q1": q1, "q3": q3, "iqr": iqr, "lower": lower, "upper": upper},
            )
            total += 1

        # Apply clamp
        cur.execute(
            f"UPDATE {qident(silver_table)} SET {qident(col)} = %s "
            f"WHERE {qident(col)} < %s "
            f"AND _load_batch_id = %s AND _dq_status = 'passed'",
            [lower, lower, batch_id],
        )
        cur.execute(
            f"UPDATE {qident(silver_table)} SET {qident(col)} = %s "
            f"WHERE {qident(col)} > %s "
            f"AND _load_batch_id = %s AND _dq_status = 'passed'",
            [upper, upper, batch_id],
        )

    return total


def _fix_lead_time_with_audit(cur, spec: DomainSpec, silver_table: str,
                              batch_id: int) -> int:
    """Replace extreme lead_time_days with per-item median."""
    if "lead_time_days" not in spec.columns:
        return 0

    total = 0
    # Global median fallback
    cur.execute(
        f"SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY lead_time_days) "
        f"FROM {qident(silver_table)} "
        f"WHERE lead_time_days > 0 AND lead_time_days <= 730 "
        f"AND _load_batch_id = %s AND _dq_status = 'passed'",
        [batch_id],
    )
    global_median = cur.fetchone()[0] or 7

    # Find extreme values
    cur.execute(
        f"SELECT {qident(spec.ck_field)}, lead_time_days "
        f"FROM {qident(silver_table)} "
        f"WHERE (lead_time_days < 0 OR lead_time_days > 730) "
        f"AND _load_batch_id = %s AND _dq_status = 'passed'",
        [batch_id],
    )
    for row_key, old_val in cur.fetchall():
        _record_correction(
            cur, spec.name, silver_table, str(row_key), "lead_time_days",
            old_val, global_median, "median_replace", "lead_time", batch_id,
        )
        total += 1

    # Apply
    cur.execute(
        f"UPDATE {qident(silver_table)} SET lead_time_days = %s "
        f"WHERE (lead_time_days < 0 OR lead_time_days > 730) "
        f"AND _load_batch_id = %s AND _dq_status = 'passed'",
        [global_median, batch_id],
    )

    return total


# ---------------------------------------------------------------------------
# Phase 5: Gold Promotion
# ---------------------------------------------------------------------------

def promote_to_gold(cur, spec: DomainSpec, batch_id: int,
                    replace_mode: bool = False) -> dict:
    """Promote passed silver rows to gold (production) tables.

    For sales: writes both corrected → fact_sales_monthly and
    original → fact_sales_monthly_original.

    Returns summary dict.
    """
    silver_table = f"silver_{spec.name}"
    gold_table = spec.table
    is_sales = spec.name == "sales"
    cfg = _config()
    preserve = cfg.get("auto_fix", {}).get(spec.name, {}).get("preserve_original", False)

    # Build column list (gold columns = ck + spec.columns + load_ts + modified_ts)
    gold_cols = [spec.ck_field, *spec.columns]

    # Clear gold table
    if replace_mode and spec.name == "forecast":
        cur.execute(f"DELETE FROM {qident(gold_table)} WHERE model_id = 'external'")
    else:
        cur.execute(f"TRUNCATE TABLE {qident(gold_table)}")

    # INSERT corrected values from silver
    col_list = ", ".join(qident(c) for c in gold_cols)
    cur.execute(
        f"INSERT INTO {qident(gold_table)} ({col_list}) "
        f"SELECT {col_list} FROM {qident(silver_table)} "
        f"WHERE _load_batch_id = %s AND _dq_status = 'passed'",
        [batch_id],
    )
    gold_count = cur.rowcount

    # For sales with preserve_original: also write to fact_sales_monthly_original
    original_count = 0
    if is_sales and preserve:
        # Check if original table exists
        cur.execute("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'fact_sales_monthly_original'
            )
        """)
        if cur.fetchone()[0]:
            cur.execute(f"TRUNCATE TABLE fact_sales_monthly_original")

            # Use original snapshot values where available, else use current values
            cur.execute(
                f"INSERT INTO fact_sales_monthly_original "
                f"({col_list}) "
                f"SELECT {qident(spec.ck_field)}, dmdunit, dmdgroup, loc, startdate, type, "
                f"COALESCE(_orig_qty_shipped, qty_shipped), "
                f"COALESCE(_orig_qty_ordered, qty_ordered), "
                f"COALESCE(_orig_qty, qty), "
                f"file_dt "
                f"FROM {qident(silver_table)} "
                f"WHERE _load_batch_id = %s AND _dq_status = 'passed'",
                [batch_id],
            )
            original_count = cur.rowcount

    # Mark silver rows as promoted
    cur.execute(
        f"UPDATE {qident(silver_table)} SET _promoted_at = now() "
        f"WHERE _load_batch_id = %s AND _dq_status = 'passed'",
        [batch_id],
    )

    return {
        "gold_table": gold_table,
        "gold_count": gold_count,
        "original_count": original_count,
    }


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------

def write_lineage(cur, spec: DomainSpec, batch_id: int) -> int:
    """Write row lineage records for the current batch. Returns count."""
    silver_table = f"silver_{spec.name}"
    cur.execute(
        f"INSERT INTO audit_row_lineage "
        f"(domain, load_batch_id, bronze_id, silver_id, business_key, layer_reached) "
        f"SELECT %s, %s, _bronze_id, _silver_id, {qident(spec.ck_field)}, "
        f"  CASE WHEN _promoted_at IS NOT NULL THEN 'gold' "
        f"       WHEN _dq_status = 'quarantined' THEN 'quarantined' "
        f"       ELSE 'silver' END "
        f"FROM {qident(silver_table)} WHERE _load_batch_id = %s",
        [spec.name, batch_id, batch_id],
    )
    return cur.rowcount


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def prune_old_batches(cur, domain: str | None = None) -> dict:
    """Delete bronze/silver data older than retention config."""
    cfg = _config()
    bronze_days = cfg.get("layers", {}).get("bronze", {}).get("retention_days", 90)
    silver_days = cfg.get("layers", {}).get("silver", {}).get("retention_days", 30)

    from common.domain_specs import DOMAIN_SPECS

    domains = [domain] if domain else list(DOMAIN_SPECS.keys())
    total_bronze = 0
    total_silver = 0

    for d in domains:
        # Bronze pruning
        cur.execute(
            f"DELETE FROM bronze_{d} WHERE _load_batch_id IN ("
            f"  SELECT batch_id FROM audit_load_batch "
            f"  WHERE domain = %s AND started_at < now() - interval '%s days'"
            f")",
            [d, bronze_days],
        )
        total_bronze += cur.rowcount

        # Silver pruning
        cur.execute(
            f"DELETE FROM silver_{d} WHERE _load_batch_id IN ("
            f"  SELECT batch_id FROM audit_load_batch "
            f"  WHERE domain = %s AND started_at < now() - interval '%s days'"
            f")",
            [d, silver_days],
        )
        total_silver += cur.rowcount

    return {"bronze_deleted": total_bronze, "silver_deleted": total_silver}
