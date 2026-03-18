"""Data Quality Engine for Demand Studio (Spec 08-01).

Runs configurable SQL-based data quality checks and produces domain health scores.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from common.db import get_db_params
from common.planning_date import get_planning_date
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
    """Check if a table has been loaded within max_hours of the planning date."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT max(load_ts) FROM {table_name} WHERE load_ts IS NOT NULL"
        )
        row = cur.fetchone()
    last_load = row[0] if row and row[0] else None
    if last_load is None:
        return {"status": "fail", "metric_value": None, "details": {"message": "No load_ts found"}}

    # Compare against planning date (frozen dev date), not wall-clock time
    planning_dt = datetime.combine(get_planning_date(), datetime.min.time(), tzinfo=timezone.utc)
    hours_ago = (planning_dt - last_load.replace(tzinfo=timezone.utc)).total_seconds() / 3600
    hours_ago = max(hours_ago, 0)  # Don't report negative if load_ts is after planning date
    status = "pass" if hours_ago <= max_hours else "fail"
    return {"status": status, "metric_value": round(hours_ago, 2), "details": {"hours_since_load": round(hours_ago, 2), "planning_date": str(get_planning_date())}}


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


def _check_range(conn, table_name: str, column: str, min_val: float | None, max_val: float | None) -> dict:
    """Check that numeric column values are within min/max bounds."""
    clauses = []
    if min_val is not None:
        clauses.append(f"{column} < {min_val}")
    if max_val is not None:
        clauses.append(f"{column} > {max_val}")
    if not clauses:
        return {"status": "skip", "metric_value": None, "details": {"message": "No min or max specified"}}

    where = " OR ".join(clauses)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT count(*) AS total, "
            f"count(*) FILTER (WHERE {where}) AS outliers "
            f"FROM {table_name} WHERE {column} IS NOT NULL"
        )
        row = cur.fetchone()
    total, outliers = row[0], row[1]
    if total == 0:
        return {"status": "warn", "metric_value": 0, "details": {"message": "No non-null rows"}}
    outlier_pct = round(100.0 * outliers / total, 4)
    status = "pass" if outliers == 0 else "fail"
    return {
        "status": status,
        "metric_value": outliers,
        "details": {"total": total, "outliers": outliers, "outlier_pct": outlier_pct, "min": min_val, "max": max_val},
    }


def _check_volume_delta(conn, table_name: str, max_pct_change: float) -> dict:
    """Check max % change in row count between the latest two loads."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT load_ts::date AS load_date, count(*) AS cnt "
            f"FROM {table_name} WHERE load_ts IS NOT NULL "
            f"GROUP BY load_ts::date ORDER BY load_ts::date DESC LIMIT 2"
        )
        rows = cur.fetchall()
    if len(rows) < 2:
        return {"status": "skip", "metric_value": None, "details": {"message": "Fewer than 2 loads found"}}

    latest_count, prev_count = rows[0][1], rows[1][1]
    if prev_count == 0:
        return {"status": "warn", "metric_value": None, "details": {"message": "Previous load had 0 rows"}}
    pct_change = round(100.0 * abs(latest_count - prev_count) / prev_count, 2)
    status = "pass" if pct_change <= max_pct_change else "fail"
    return {
        "status": status,
        "metric_value": pct_change,
        "details": {"latest_count": latest_count, "prev_count": prev_count, "pct_change": pct_change, "max_pct_change": max_pct_change},
    }


def _check_referential_integrity(
    conn, source_table: str, source_columns: list[str],
    target_table: str, target_columns: list[str],
) -> dict:
    """Check that all FK values in source exist in target."""
    src_cols = ", ".join(f"s.{c}" for c in source_columns)
    tgt_cols = ", ".join(f"t.{c}" for c in target_columns)
    join_cond = " AND ".join(f"s.{sc} = t.{tc}" for sc, tc in zip(source_columns, target_columns))
    null_filter = " AND ".join(f"s.{c} IS NOT NULL" for c in source_columns)

    with conn.cursor() as cur:
        cur.execute(
            f"SELECT count(DISTINCT ({src_cols})) "
            f"FROM {source_table} s "
            f"LEFT JOIN {target_table} t ON {join_cond} "
            f"WHERE {null_filter} AND t.{target_columns[0]} IS NULL"
            if len(target_columns) == 1 else
            f"SELECT count(*) FROM ("
            f"  SELECT DISTINCT {src_cols} FROM {source_table} s "
            f"  WHERE {null_filter} "
            f"  EXCEPT "
            f"  SELECT DISTINCT {tgt_cols} FROM {target_table} t"
            f") orphans"
        )
        orphan_count = cur.fetchone()[0]
    status = "pass" if orphan_count == 0 else "fail"
    return {
        "status": status,
        "metric_value": orphan_count,
        "details": {
            "orphan_keys": orphan_count,
            "source_table": source_table,
            "target_table": target_table,
        },
    }


# ---------------------------------------------------------------------------
# Statistical DQ checks (Expert checks)
# ---------------------------------------------------------------------------

def _check_statistical_outlier(
    conn, table_name: str, column: str, method: str = "iqr", threshold: float = 1.5,
) -> dict:
    """Detect statistical outliers using IQR or Z-score method.

    IQR method: outliers outside [Q1 - threshold*IQR, Q3 + threshold*IQR].
    Z-score method: outliers beyond ±threshold standard deviations.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT count(*) AS total, "
            f"avg({column}) AS mean, "
            f"stddev_pop({column}) AS stddev, "
            f"percentile_cont(0.25) WITHIN GROUP (ORDER BY {column}) AS q1, "
            f"percentile_cont(0.50) WITHIN GROUP (ORDER BY {column}) AS median, "
            f"percentile_cont(0.75) WITHIN GROUP (ORDER BY {column}) AS q3, "
            f"min({column}) AS col_min, "
            f"max({column}) AS col_max "
            f"FROM {table_name} WHERE {column} IS NOT NULL"
        )
        row = cur.fetchone()

    total = row[0]
    if total == 0:
        return {"status": "warn", "metric_value": 0, "details": {"message": "No non-null rows"}}

    mean, stddev = float(row[1] or 0), float(row[2] or 0)
    q1, median, q3 = float(row[3] or 0), float(row[4] or 0), float(row[5] or 0)
    col_min, col_max = float(row[6] or 0), float(row[7] or 0)
    iqr = q3 - q1

    if method == "zscore":
        lower = mean - threshold * stddev if stddev > 0 else col_min
        upper = mean + threshold * stddev if stddev > 0 else col_max
    else:  # iqr
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr

    # Count outliers
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT count(*) FROM {table_name} "
            f"WHERE {column} IS NOT NULL AND ({column} < %s OR {column} > %s)",
            (lower, upper),
        )
        outliers = cur.fetchone()[0]

    outlier_pct = round(100.0 * outliers / total, 4) if total > 0 else 0
    status = "pass" if outlier_pct < 1.0 else ("warn" if outlier_pct < 5.0 else "fail")
    return {
        "status": status,
        "metric_value": outliers,
        "details": {
            "method": method,
            "threshold": threshold,
            "total": total,
            "outliers": outliers,
            "outlier_pct": outlier_pct,
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "mean": round(mean, 4),
            "median": round(median, 4),
            "stddev": round(stddev, 4),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "col_min": round(col_min, 4),
            "col_max": round(col_max, 4),
        },
    }


def _check_distribution_drift(
    conn, table_name: str, column: str, max_drift: float = 0.1,
) -> dict:
    """Detect distribution drift between two most recent load batches.

    Uses a simplified two-sample comparison: compares mean, stddev, and
    quantiles between latest and previous load batches. Drift score is
    the max normalised shift across {mean, stddev, median}.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT DISTINCT load_ts::date FROM {table_name} "
            f"WHERE load_ts IS NOT NULL ORDER BY 1 DESC LIMIT 2"
        )
        dates = cur.fetchall()

    if len(dates) < 2:
        return {"status": "skip", "metric_value": None, "details": {"message": "Fewer than 2 load batches"}}

    stats = {}
    for label, d in [("latest", dates[0][0]), ("previous", dates[1][0])]:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT count(*), avg({column}), stddev_pop({column}), "
                f"percentile_cont(0.5) WITHIN GROUP (ORDER BY {column}) "
                f"FROM {table_name} WHERE {column} IS NOT NULL AND load_ts::date = %s",
                (d,),
            )
            row = cur.fetchone()
        stats[label] = {
            "count": row[0],
            "mean": float(row[1] or 0),
            "stddev": float(row[2] or 0),
            "median": float(row[3] or 0),
        }

    prev = stats["previous"]
    curr = stats["latest"]

    # Normalised shift: |curr - prev| / max(|prev|, 1) for each stat
    def _norm_shift(c: float, p: float) -> float:
        denom = max(abs(p), 1.0)
        return abs(c - p) / denom

    mean_shift = _norm_shift(curr["mean"], prev["mean"])
    stddev_shift = _norm_shift(curr["stddev"], prev["stddev"])
    median_shift = _norm_shift(curr["median"], prev["median"])
    drift_score = round(max(mean_shift, stddev_shift, median_shift), 4)

    status = "pass" if drift_score <= max_drift else ("warn" if drift_score <= max_drift * 2 else "fail")
    return {
        "status": status,
        "metric_value": drift_score,
        "details": {
            "drift_score": drift_score,
            "max_drift": max_drift,
            "mean_shift": round(mean_shift, 4),
            "stddev_shift": round(stddev_shift, 4),
            "median_shift": round(median_shift, 4),
            "latest": curr,
            "previous": prev,
            "latest_date": str(dates[0][0]),
            "previous_date": str(dates[1][0]),
        },
    }


def _check_temporal_gaps(
    conn, table_name: str, date_column: str, grain: str = "month",
) -> dict:
    """Detect gaps (missing periods) in a time-series table.

    Grain can be 'month' or 'day'.  Returns the number of missing periods.
    """
    trunc = "month" if grain == "month" else "day"
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT date_trunc('{trunc}', {date_column})::date AS period "
            f"FROM {table_name} WHERE {date_column} IS NOT NULL "
            f"GROUP BY 1 ORDER BY 1"
        )
        periods = [r[0] for r in cur.fetchall()]

    if len(periods) < 2:
        return {"status": "skip", "metric_value": None, "details": {"message": "Fewer than 2 periods"}}

    from dateutil.relativedelta import relativedelta
    delta = relativedelta(months=1) if grain == "month" else relativedelta(days=1)

    gaps = []
    for i in range(1, len(periods)):
        expected = periods[i - 1] + delta
        # Walk forward to find the gap
        while expected < periods[i]:
            gaps.append(str(expected))
            expected += delta
            if len(gaps) > 100:
                break
        if len(gaps) > 100:
            break

    gap_count = len(gaps)
    status = "pass" if gap_count == 0 else ("warn" if gap_count <= 3 else "fail")
    return {
        "status": status,
        "metric_value": gap_count,
        "details": {
            "gap_count": gap_count,
            "grain": grain,
            "date_column": date_column,
            "first_period": str(periods[0]),
            "last_period": str(periods[-1]),
            "total_periods": len(periods),
            "missing_periods": gaps[:20],  # First 20 gaps
        },
    }


def _check_cross_column(
    conn, table_name: str, rule: str, description: str = "",
) -> dict:
    """Validate a logical cross-column relationship.

    ``rule`` is a SQL boolean expression evaluated per row (e.g., ``qty_shipped <= qty``).
    Returns the count of rows violating the rule.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT count(*) AS total, "
            f"count(*) FILTER (WHERE NOT ({rule})) AS violations "
            f"FROM {table_name}"
        )
        row = cur.fetchone()

    total, violations = row[0], row[1]
    if total == 0:
        return {"status": "warn", "metric_value": 0, "details": {"message": "Empty table"}}
    violation_pct = round(100.0 * violations / total, 4)
    status = "pass" if violations == 0 else ("warn" if violation_pct < 1.0 else "fail")
    return {
        "status": status,
        "metric_value": violations,
        "details": {
            "total": total,
            "violations": violations,
            "violation_pct": violation_pct,
            "rule": rule,
            "description": description,
        },
    }


def _check_cardinality_anomaly(
    conn, table_name: str, column: str, max_change_pct: float = 10.0,
) -> dict:
    """Detect anomalous changes in distinct value counts between load batches.

    Flags when new values appear or old values disappear beyond the threshold.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT DISTINCT load_ts::date FROM {table_name} "
            f"WHERE load_ts IS NOT NULL ORDER BY 1 DESC LIMIT 2"
        )
        dates = cur.fetchall()

    if len(dates) < 2:
        return {"status": "skip", "metric_value": None, "details": {"message": "Fewer than 2 load batches"}}

    latest_date, prev_date = dates[0][0], dates[1][0]

    with conn.cursor() as cur:
        # Distinct values in latest batch
        cur.execute(
            f"SELECT count(DISTINCT {column}) FROM {table_name} "
            f"WHERE load_ts::date = %s AND {column} IS NOT NULL",
            (latest_date,),
        )
        latest_count = cur.fetchone()[0]

        # Distinct values in previous batch
        cur.execute(
            f"SELECT count(DISTINCT {column}) FROM {table_name} "
            f"WHERE load_ts::date = %s AND {column} IS NOT NULL",
            (prev_date,),
        )
        prev_count = cur.fetchone()[0]

        # New values (in latest but not in previous)
        cur.execute(
            f"SELECT count(*) FROM ("
            f"  SELECT DISTINCT {column} FROM {table_name} WHERE load_ts::date = %s AND {column} IS NOT NULL "
            f"  EXCEPT "
            f"  SELECT DISTINCT {column} FROM {table_name} WHERE load_ts::date = %s AND {column} IS NOT NULL"
            f") n",
            (latest_date, prev_date),
        )
        new_values = cur.fetchone()[0]

        # Dropped values (in previous but not in latest)
        cur.execute(
            f"SELECT count(*) FROM ("
            f"  SELECT DISTINCT {column} FROM {table_name} WHERE load_ts::date = %s AND {column} IS NOT NULL "
            f"  EXCEPT "
            f"  SELECT DISTINCT {column} FROM {table_name} WHERE load_ts::date = %s AND {column} IS NOT NULL"
            f") d",
            (prev_date, latest_date),
        )
        dropped_values = cur.fetchone()[0]

    change_pct = round(100.0 * (new_values + dropped_values) / max(prev_count, 1), 2)
    status = "pass" if change_pct <= max_change_pct else ("warn" if change_pct <= max_change_pct * 2 else "fail")
    return {
        "status": status,
        "metric_value": change_pct,
        "details": {
            "latest_distinct": latest_count,
            "previous_distinct": prev_count,
            "new_values": new_values,
            "dropped_values": dropped_values,
            "change_pct": change_pct,
            "max_change_pct": max_change_pct,
            "latest_date": str(latest_date),
            "previous_date": str(prev_date),
        },
    }


# ---------------------------------------------------------------------------
# DQEngine class
# ---------------------------------------------------------------------------
CHECK_FUNCTIONS = {
    "freshness": _check_freshness,
    "completeness": _check_completeness,
    "row_count": _check_row_count,
    "uniqueness": _check_uniqueness,
    "range": _check_range,
    "volume_delta": _check_volume_delta,
    "referential_integrity": _check_referential_integrity,
    "statistical_outlier": _check_statistical_outlier,
    "distribution_drift": _check_distribution_drift,
    "temporal_gaps": _check_temporal_gaps,
    "cross_column": _check_cross_column,
    "cardinality_anomaly": _check_cardinality_anomaly,
}


class DQEngine:
    """Data Quality check runner."""

    def __init__(self):
        self.config = _load_config()

    def run_all_checks(self, domain: str | None = None) -> list[dict]:
        """Run all configured checks, optionally filtered by domain."""
        import psycopg
        results = []
        checks = self._flatten_checks()

        with psycopg.connect(**get_db_params(), autocommit=True) as conn:
            for check in checks:
                if domain and check.get("domain") != domain:
                    continue
                if not check.get("enabled", True):
                    continue
                result = self._run_single(conn, check)
                results.append(result)
                self._record_result(conn, result)

        return results

    def _flatten_checks(self) -> list[dict]:
        """Flatten nested config structure into a list of individual check dicts.

        Config layout:
            checks:
              <check_type>:        # freshness, completeness, uniqueness, range, ...
                <domain_key>:      # item, location, sales, ...
                  table: ...
                  ...

        Completeness and range checks have a nested ``columns`` list — each
        column entry becomes its own check dict.
        """
        global_defaults = self.config.get("global_defaults", {})
        raw = self.config.get("checks", {})
        if isinstance(raw, list):
            return raw  # already flat (backward compat)

        flat: list[dict] = []
        for check_type, domains_map in raw.items():
            if not isinstance(domains_map, dict):
                continue
            for domain_key, spec in domains_map.items():
                if not isinstance(spec, dict):
                    continue
                table_name = spec.get("table", spec.get("source_table", ""))
                severity = spec.get("severity", global_defaults.get("severity", "warning"))
                enabled = spec.get("enabled", global_defaults.get("enabled", True))

                # --- Completeness & Range have per-column sub-checks ----------
                if check_type in ("completeness", "range") and "columns" in spec:
                    for col_spec in spec["columns"]:
                        col_name = col_spec.get("column", "")
                        if check_type == "completeness":
                            flat.append({
                                "check_type": "completeness",
                                "check_name": f"completeness_{domain_key}_{col_name}",
                                "domain": domain_key,
                                "table_name": table_name,
                                "column": col_name,
                                "max_null_pct": col_spec.get("null_pct_threshold", 5.0),
                                "severity": col_spec.get("severity", severity),
                                "enabled": enabled,
                            })
                        elif check_type == "range":
                            flat.append({
                                "check_type": "range",
                                "check_name": f"range_{domain_key}_{col_name}",
                                "domain": domain_key,
                                "table_name": table_name,
                                "column": col_name,
                                "min": col_spec.get("min"),
                                "max": col_spec.get("max"),
                                "severity": col_spec.get("severity", severity),
                                "enabled": enabled,
                            })
                    continue

                # --- Freshness ------------------------------------------------
                if check_type == "freshness":
                    flat.append({
                        "check_type": "freshness",
                        "check_name": f"freshness_{domain_key}",
                        "domain": domain_key,
                        "table_name": table_name,
                        "max_hours": spec.get("max_hours_since_load", 48),
                        "severity": severity,
                        "enabled": enabled,
                    })
                    continue

                # --- Uniqueness -----------------------------------------------
                if check_type == "uniqueness":
                    flat.append({
                        "check_type": "uniqueness",
                        "check_name": f"uniqueness_{domain_key}",
                        "domain": domain_key,
                        "table_name": table_name,
                        "key_columns": spec.get("key_columns", []),
                        "severity": severity,
                        "enabled": enabled,
                    })
                    continue

                # --- Volume delta ---------------------------------------------
                if check_type == "volume_delta":
                    flat.append({
                        "check_type": "volume_delta",
                        "check_name": f"volume_delta_{domain_key}",
                        "domain": domain_key,
                        "table_name": table_name,
                        "max_pct_change": spec.get("max_pct_change", 50.0),
                        "severity": severity,
                        "enabled": enabled,
                    })
                    continue

                # --- Referential integrity ------------------------------------
                if check_type == "referential_integrity":
                    flat.append({
                        "check_type": "referential_integrity",
                        "check_name": f"referential_integrity_{domain_key}",
                        "domain": domain_key,
                        "table_name": spec.get("source_table", ""),
                        "source_table": spec.get("source_table", ""),
                        "source_columns": spec.get("source_columns", []),
                        "target_table": spec.get("target_table", ""),
                        "target_columns": spec.get("target_columns", []),
                        "severity": severity,
                        "enabled": enabled,
                    })
                    continue

                # --- Statistical outlier (per-column) -------------------------
                if check_type == "statistical_outlier" and "columns" in spec:
                    for col_spec in spec["columns"]:
                        col_name = col_spec.get("column", "")
                        flat.append({
                            "check_type": "statistical_outlier",
                            "check_name": f"statistical_outlier_{domain_key}_{col_name}",
                            "domain": domain_key,
                            "table_name": table_name,
                            "column": col_name,
                            "method": col_spec.get("method", "iqr"),
                            "threshold": col_spec.get("threshold", 1.5),
                            "severity": col_spec.get("severity", severity),
                            "enabled": enabled,
                        })
                    continue

                # --- Distribution drift (per-column) --------------------------
                if check_type == "distribution_drift" and "columns" in spec:
                    for col_spec in spec["columns"]:
                        col_name = col_spec.get("column", "")
                        flat.append({
                            "check_type": "distribution_drift",
                            "check_name": f"distribution_drift_{domain_key}_{col_name}",
                            "domain": domain_key,
                            "table_name": table_name,
                            "column": col_name,
                            "max_drift": col_spec.get("max_drift", 0.1),
                            "severity": col_spec.get("severity", severity),
                            "enabled": enabled,
                        })
                    continue

                # --- Temporal gaps --------------------------------------------
                if check_type == "temporal_gaps":
                    flat.append({
                        "check_type": "temporal_gaps",
                        "check_name": f"temporal_gaps_{domain_key}",
                        "domain": domain_key,
                        "table_name": table_name,
                        "date_column": spec.get("date_column", "startdate"),
                        "grain": spec.get("grain", "month"),
                        "severity": severity,
                        "enabled": enabled,
                    })
                    continue

                # --- Cross-column consistency (per-rule) ----------------------
                if check_type == "cross_column" and "rules" in spec:
                    for ri, rule_spec in enumerate(spec["rules"]):
                        rule_name = rule_spec.get("name", f"rule_{ri}")
                        flat.append({
                            "check_type": "cross_column",
                            "check_name": f"cross_column_{domain_key}_{rule_name}",
                            "domain": domain_key,
                            "table_name": table_name,
                            "rule": rule_spec.get("expression", "TRUE"),
                            "description": rule_spec.get("description", ""),
                            "severity": rule_spec.get("severity", severity),
                            "enabled": enabled,
                        })
                    continue

                # --- Cardinality anomaly (per-column) -------------------------
                if check_type == "cardinality_anomaly" and "columns" in spec:
                    for col_spec in spec["columns"]:
                        col_name = col_spec.get("column", "")
                        flat.append({
                            "check_type": "cardinality_anomaly",
                            "check_name": f"cardinality_anomaly_{domain_key}_{col_name}",
                            "domain": domain_key,
                            "table_name": table_name,
                            "column": col_name,
                            "max_change_pct": col_spec.get("max_change_pct", 10.0),
                            "severity": col_spec.get("severity", severity),
                            "enabled": enabled,
                        })
                    continue

                # --- Row count (generic fallback) -----------------------------
                flat.append({
                    "check_type": check_type,
                    "check_name": f"{check_type}_{domain_key}",
                    "domain": domain_key,
                    "table_name": table_name,
                    "severity": severity,
                    "enabled": enabled,
                    **{k: v for k, v in spec.items() if k not in ("table", "severity", "enabled")},
                })

        return flat

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
            elif check_type == "range":
                result = _check_range(
                    conn, table_name, check.get("column", ""),
                    check.get("min"), check.get("max"),
                )
            elif check_type == "volume_delta":
                result = _check_volume_delta(
                    conn, table_name, check.get("max_pct_change", 50.0),
                )
            elif check_type == "referential_integrity":
                result = _check_referential_integrity(
                    conn,
                    check.get("source_table", table_name),
                    check.get("source_columns", []),
                    check.get("target_table", ""),
                    check.get("target_columns", []),
                )
            elif check_type == "statistical_outlier":
                result = _check_statistical_outlier(
                    conn, table_name, check.get("column", ""),
                    check.get("method", "iqr"), check.get("threshold", 1.5),
                )
            elif check_type == "distribution_drift":
                result = _check_distribution_drift(
                    conn, table_name, check.get("column", ""),
                    check.get("max_drift", 0.1),
                )
            elif check_type == "temporal_gaps":
                result = _check_temporal_gaps(
                    conn, table_name, check.get("date_column", "startdate"),
                    check.get("grain", "month"),
                )
            elif check_type == "cross_column":
                result = _check_cross_column(
                    conn, table_name, check.get("rule", "TRUE"),
                    check.get("description", ""),
                )
            elif check_type == "cardinality_anomaly":
                result = _check_cardinality_anomaly(
                    conn, table_name, check.get("column", ""),
                    check.get("max_change_pct", 10.0),
                )
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
