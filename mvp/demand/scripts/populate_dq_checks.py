"""Populate dim_dq_check_catalog from data_quality_config.yaml.

Reads all check definitions (freshness, completeness, uniqueness, range,
volume_delta, referential_integrity) from config/data_quality_config.yaml,
generates a unique check_name per check, builds a SQL template string, and
UPSERTs into dim_dq_check_catalog (ON CONFLICT on check_name DO UPDATE).

CLI:
    uv run python scripts/populate_dq_checks.py
    uv run python scripts/populate_dq_checks.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import get_db_params
from common.utils import _ts, load_config

CONFIG_NAME = "data_quality_config.yaml"


# ---------------------------------------------------------------------------
# SQL template builders — one per check type
# ---------------------------------------------------------------------------

def _freshness_template(table: str) -> str:
    """SQL that returns hours since the latest load_ts in a table."""
    return (
        f"SELECT EXTRACT(EPOCH FROM (NOW() - MAX(load_ts))) / 3600.0 "
        f"AS hours_since_load FROM {table}"
    )


def _completeness_template(table: str, column: str) -> str:
    """SQL that returns the null percentage for a column."""
    return (
        f"SELECT 100.0 * COUNT(*) FILTER (WHERE {column} IS NULL) "
        f"/ NULLIF(COUNT(*), 0) AS null_pct FROM {table}"
    )


def _uniqueness_template(table: str, key_columns: list[str]) -> str:
    """SQL that returns the count of duplicate key combinations."""
    cols = ", ".join(key_columns)
    return (
        f"SELECT COUNT(*) AS duplicate_count FROM ("
        f"SELECT {cols} FROM {table} "
        f"GROUP BY {cols} HAVING COUNT(*) > 1"
        f") sub"
    )


def _range_template(table: str, column: str, min_val: float, max_val: float) -> str:
    """SQL that returns the count of rows outside [min, max]."""
    return (
        f"SELECT COUNT(*) AS out_of_range FROM {table} "
        f"WHERE {column} < {min_val} OR {column} > {max_val}"
    )


def _volume_delta_template(table: str) -> str:
    """SQL that returns current row count for volume comparison."""
    return f"SELECT COUNT(*) AS row_count FROM {table}"


def _referential_integrity_template(
    source_table: str,
    source_columns: list[str],
    target_table: str,
    target_columns: list[str],
) -> str:
    """SQL that returns orphan count (source rows not in target)."""
    src_cols = ", ".join(f"s.{c}" for c in source_columns)
    join_cond = " AND ".join(
        f"s.{sc} = t.{tc}" for sc, tc in zip(source_columns, target_columns)
    )
    tgt_null = f"t.{target_columns[0]}"
    return (
        f"SELECT COUNT(*) AS orphan_count FROM {source_table} s "
        f"LEFT JOIN {target_table} t ON {join_cond} "
        f"WHERE {tgt_null} IS NULL"
    )


# ---------------------------------------------------------------------------
# Config parser — yields flat list of check dicts ready for upsert
# ---------------------------------------------------------------------------

def parse_checks(config: dict) -> list[dict]:
    """Parse all check definitions from the config into a flat list.

    Each item is a dict with keys: check_name, check_type, domain,
    sql_template, threshold, severity, enabled.
    """
    checks_cfg = config.get("checks", {})
    defaults = config.get("global_defaults", {})
    default_severity = defaults.get("severity", "warning")
    default_enabled = defaults.get("enabled", True)

    rows: list[dict] = []

    # ── Freshness ──────────────────────────────────────────────
    for domain, spec in checks_cfg.get("freshness", {}).items():
        table = spec["table"]
        rows.append({
            "check_name": f"freshness_{domain}",
            "check_type": "freshness",
            "domain": domain,
            "table_name": table,
            "sql_template": _freshness_template(table),
            "threshold": spec["max_hours_since_load"],
            "severity": spec.get("severity", default_severity),
            "enabled": spec.get("enabled", default_enabled),
        })

    # ── Completeness ───────────────────────────────────────────
    for domain, spec in checks_cfg.get("completeness", {}).items():
        table = spec["table"]
        for col_spec in spec.get("columns", []):
            column = col_spec["column"]
            rows.append({
                "check_name": f"completeness_{domain}_{column}",
                "check_type": "completeness",
                "domain": domain,
                "table_name": table,
                "sql_template": _completeness_template(table, column),
                "threshold": col_spec["null_pct_threshold"],
                "severity": col_spec.get("severity", default_severity),
                "enabled": col_spec.get("enabled", default_enabled),
            })

    # ── Uniqueness ─────────────────────────────────────────────
    for domain, spec in checks_cfg.get("uniqueness", {}).items():
        table = spec["table"]
        key_columns = spec["key_columns"]
        rows.append({
            "check_name": f"uniqueness_{domain}",
            "check_type": "uniqueness",
            "domain": domain,
            "table_name": table,
            "sql_template": _uniqueness_template(table, key_columns),
            "threshold": 0,  # any duplicates = fail
            "severity": spec.get("severity", default_severity),
            "enabled": spec.get("enabled", default_enabled),
        })

    # ── Range ──────────────────────────────────────────────────
    for domain, spec in checks_cfg.get("range", {}).items():
        table = spec["table"]
        for col_spec in spec.get("columns", []):
            column = col_spec["column"]
            min_val = col_spec["min"]
            max_val = col_spec["max"]
            rows.append({
                "check_name": f"range_{domain}_{column}",
                "check_type": "range",
                "domain": domain,
                "table_name": table,
                "sql_template": _range_template(table, column, min_val, max_val),
                "threshold": 0,  # any out-of-range = fail
                "severity": col_spec.get("severity", default_severity),
                "enabled": col_spec.get("enabled", default_enabled),
            })

    # ── Volume delta ───────────────────────────────────────────
    for domain, spec in checks_cfg.get("volume_delta", {}).items():
        table = spec["table"]
        rows.append({
            "check_name": f"volume_delta_{domain}",
            "check_type": "volume_delta",
            "domain": domain,
            "table_name": table,
            "sql_template": _volume_delta_template(table),
            "threshold": spec["max_pct_change"],
            "severity": spec.get("severity", default_severity),
            "enabled": spec.get("enabled", default_enabled),
        })

    # ── Referential integrity ──────────────────────────────────
    for check_key, spec in checks_cfg.get("referential_integrity", {}).items():
        # Domain inferred from the source table (fact_sales → sales, dim_dfu → dfu)
        source_table = spec["source_table"]
        domain = _domain_from_table(source_table)
        rows.append({
            "check_name": f"ri_{check_key}",
            "check_type": "referential_integrity",
            "domain": domain,
            "table_name": source_table,
            "sql_template": _referential_integrity_template(
                spec["source_table"],
                spec["source_columns"],
                spec["target_table"],
                spec["target_columns"],
            ),
            "threshold": 0,  # any orphans = fail
            "severity": spec.get("severity", default_severity),
            "enabled": spec.get("enabled", default_enabled),
        })

    return rows


def _domain_from_table(table_name: str) -> str:
    """Derive a domain label from a table name.

    Examples:
        fact_sales_monthly -> sales
        dim_dfu -> dfu
        fact_inventory_snapshot -> inventory
        fact_external_forecast_monthly -> forecast
    """
    name = table_name.replace("fact_", "").replace("dim_", "")
    # Map known suffixed names to short domain names
    mapping = {
        "sales_monthly": "sales",
        "external_forecast_monthly": "forecast",
        "inventory_snapshot": "inventory",
    }
    return mapping.get(name, name)


# ---------------------------------------------------------------------------
# Upsert logic
# ---------------------------------------------------------------------------

UPSERT_SQL = """
    INSERT INTO dim_dq_check_catalog
        (check_name, check_type, domain, table_name, sql_template, threshold, severity, enabled, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
    ON CONFLICT (check_name) DO UPDATE SET
        check_type   = EXCLUDED.check_type,
        domain       = EXCLUDED.domain,
        table_name   = EXCLUDED.table_name,
        sql_template = EXCLUDED.sql_template,
        threshold    = EXCLUDED.threshold,
        severity     = EXCLUDED.severity,
        enabled      = EXCLUDED.enabled,
        updated_at   = NOW()
"""


def upsert_checks(conn, checks: list[dict], dry_run: bool = False) -> int:
    """Upsert parsed checks into dim_dq_check_catalog. Returns count."""
    count = 0
    with conn.cursor() as cur:
        for chk in checks:
            if not dry_run:
                cur.execute(UPSERT_SQL, (
                    chk["check_name"],
                    chk["check_type"],
                    chk["domain"],
                    chk.get("table_name"),
                    chk["sql_template"],
                    chk["threshold"],
                    chk["severity"],
                    chk["enabled"],
                ))
            count += 1
    if not dry_run:
        conn.commit()
    return count


# ---------------------------------------------------------------------------
# Public entry point (callable from API endpoint)
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> dict:
    """Parse config and upsert all DQ checks. Returns summary dict.

    Returns
    -------
    dict
        {"total": int, "by_type": dict[str, int], "dry_run": bool}
    """
    import psycopg
    from dotenv import load_dotenv
    load_dotenv()

    config = load_config(CONFIG_NAME)
    checks = parse_checks(config)

    # Summary by type
    by_type: dict[str, int] = {}
    for chk in checks:
        by_type[chk["check_type"]] = by_type.get(chk["check_type"], 0) + 1

    mode = "(dry-run)" if dry_run else ""
    print(f"[{_ts()}] Parsed {len(checks)} checks from {CONFIG_NAME} {mode}")
    for ctype, cnt in sorted(by_type.items()):
        print(f"  {ctype}: {cnt}")

    with psycopg.connect(**get_db_params()) as conn:
        upserted = upsert_checks(conn, checks, dry_run=dry_run)

    verb = "Would upsert" if dry_run else "Upserted"
    print(f"[{_ts()}] {verb} {upserted} checks into dim_dq_check_catalog")

    return {"total": upserted, "by_type": by_type, "dry_run": dry_run}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate dim_dq_check_catalog from data_quality_config.yaml"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and print checks without writing to DB",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
