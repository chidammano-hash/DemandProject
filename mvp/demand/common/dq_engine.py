"""Data Quality Engine for Demand Studio (Spec 08-01).

Runs configurable SQL-based data quality checks and produces domain health scores.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from common.db import get_db_params
from common.utils import load_config, reset_config

_CONFIG_NAME = "data_quality_config.yaml"


# ---------------------------------------------------------------------------
# Config (thread-safe via common.utils.load_config)
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    return load_config(_CONFIG_NAME)


def _reset_config():
    reset_config(_CONFIG_NAME)


# ---------------------------------------------------------------------------
# Check types
# ---------------------------------------------------------------------------
def _check_freshness(conn, table_name: str, max_hours: int) -> dict:
    """Check if a table has been loaded within max_hours."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT max(load_ts) FROM {table_name} WHERE load_ts IS NOT NULL"
        )
        row = cur.fetchone()
    last_load = row[0] if row and row[0] else None
    if last_load is None:
        return {"status": "fail", "metric_value": None, "details": {"message": "No load_ts found"}}

    hours_ago = (datetime.now(timezone.utc) - last_load.replace(tzinfo=timezone.utc)).total_seconds() / 3600
    status = "pass" if hours_ago <= max_hours else "fail"
    return {"status": status, "metric_value": round(hours_ago, 2), "details": {"hours_since_load": round(hours_ago, 2)}}


def _check_completeness(conn, table_name: str, column: str, max_null_pct: float) -> dict:
    """Check null percentage of a column."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT count(*) AS total, "
            f"count(*) FILTER (WHERE {column} IS NULL) AS nulls "
            f"FROM {table_name}"
        )
        row = cur.fetchone()
    total, nulls = row[0], row[1]
    if total == 0:
        return {"status": "warn", "metric_value": 0, "details": {"message": "Empty table"}}
    null_pct = round(100.0 * nulls / total, 4)
    status = "pass" if null_pct <= max_null_pct else "fail"
    return {"status": status, "metric_value": null_pct, "details": {"total": total, "nulls": nulls, "null_pct": null_pct}}


def _check_row_count(conn, table_name: str, min_rows: int = 1) -> dict:
    """Check table has minimum expected row count."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table_name}")
        count = cur.fetchone()[0]
    status = "pass" if count >= min_rows else "fail"
    return {"status": status, "metric_value": count, "details": {"row_count": count, "min_expected": min_rows}}


def _check_uniqueness(conn, table_name: str, key_columns: list[str]) -> dict:
    """Check for duplicate keys."""
    cols = ", ".join(key_columns)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT count(*) FROM ("
            f"  SELECT {cols} FROM {table_name} GROUP BY {cols} HAVING count(*) > 1"
            f") dupes"
        )
        dup_count = cur.fetchone()[0]
    status = "pass" if dup_count == 0 else "fail"
    return {"status": status, "metric_value": dup_count, "details": {"duplicate_groups": dup_count}}


# ---------------------------------------------------------------------------
# DQEngine class
# ---------------------------------------------------------------------------
CHECK_FUNCTIONS = {
    "freshness": _check_freshness,
    "completeness": _check_completeness,
    "row_count": _check_row_count,
    "uniqueness": _check_uniqueness,
}


class DQEngine:
    """Data Quality check runner."""

    def __init__(self):
        self.config = _load_config()

    def run_all_checks(self, domain: str | None = None) -> list[dict]:
        """Run all configured checks, optionally filtered by domain."""
        import psycopg
        results = []
        checks = self.config.get("checks", [])

        with psycopg.connect(**get_db_params()) as conn:
            for check in checks:
                if domain and check.get("domain") != domain:
                    continue
                if not check.get("enabled", True):
                    continue
                result = self._run_single(conn, check)
                results.append(result)
                self._record_result(conn, result)
            conn.commit()

        return results

    def _run_single(self, conn, check: dict) -> dict:
        """Run a single check definition."""
        check_type = check.get("check_type", "row_count")
        check_name = check.get("check_name", "unknown")
        table_name = check.get("table_name", "")
        severity = check.get("severity", "warning")

        try:
            if check_type == "freshness":
                result = _check_freshness(conn, table_name, check.get("max_hours", 48))
            elif check_type == "completeness":
                result = _check_completeness(
                    conn, table_name, check.get("column", ""), check.get("max_null_pct", 5.0)
                )
            elif check_type == "row_count":
                result = _check_row_count(conn, table_name, check.get("min_rows", 1))
            elif check_type == "uniqueness":
                result = _check_uniqueness(conn, table_name, check.get("key_columns", []))
            else:
                result = {"status": "skip", "metric_value": None, "details": {"message": f"Unknown check type: {check_type}"}}
        except Exception as e:
            result = {"status": "error", "metric_value": None, "details": {"error": str(e)}}

        return {
            "check_name": check_name,
            "check_type": check_type,
            "domain": check.get("domain", ""),
            "table_name": table_name,
            "severity": severity,
            **result,
        }

    def _record_result(self, conn, result: dict) -> None:
        """Write check result to fact_dq_check_results."""
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO fact_dq_check_results
                       (check_name, domain, table_name, severity, status, metric_value, threshold, details)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        result["check_name"],
                        result.get("domain", ""),
                        result.get("table_name", ""),
                        result.get("severity", "warning"),
                        result["status"],
                        result.get("metric_value"),
                        None,
                        json.dumps(result.get("details")),
                    ),
                )
        except Exception:
            pass  # Best-effort recording

    def get_domain_score(self, domain: str) -> dict:
        """Get health score for a domain (weighted pass rate)."""
        import psycopg
        with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT status, count(*)
                   FROM fact_dq_check_results
                   WHERE domain = %s
                     AND run_ts >= now() - interval '24 hours'
                   GROUP BY status""",
                (domain,),
            )
            rows = cur.fetchall()

        counts = {r[0]: r[1] for r in rows}
        total = sum(counts.values())
        passed = counts.get("pass", 0)
        score = round(100.0 * passed / total, 1) if total > 0 else 100.0

        return {"domain": domain, "score": score, "total_checks": total, "passed": passed, "failed": counts.get("fail", 0), "warnings": counts.get("warn", 0)}

    def get_pipeline_health(self) -> dict:
        """Get freshness status across all key tables."""
        import psycopg
        tables = ["dim_item", "dim_location", "dim_dfu", "fact_sales_monthly", "fact_external_forecast_monthly"]
        results = []
        with psycopg.connect(**get_db_params()) as conn:
            for table in tables:
                try:
                    r = _check_freshness(conn, table, 48)
                    results.append({"table": table, **r})
                except Exception:
                    results.append({"table": table, "status": "error", "metric_value": None})
        return {"tables": results}
