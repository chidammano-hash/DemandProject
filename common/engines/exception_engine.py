"""Exception detection and scoring engine — Feature 40: Planner Storyboard.

Pure functions for detecting each exception type, scoring severity, and generating
human-readable headlines. The `run_exception_detection` function orchestrates the
full pipeline against a live DB connection.

Detection functions follow a consistent contract:
  - Inputs: item_id, loc, signal data, config dict
  - Output: dict with keys {exception_type, item_id, loc, severity, financial_impact,
            headline, supporting_data, month_start} or None if no exception found.

Usage (from scripts):
    from common.exception_engine import run_exception_detection
    result = run_exception_detection(conn, config, month_start=date(2026, 3, 1))
"""
from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure detection functions
# ---------------------------------------------------------------------------

def detect_forecast_bias(
    item_id: str,
    loc: str,
    bias_history: list[dict],
    config: dict,
) -> dict | None:
    """Detect sustained forecast bias over trailing N months.

    bias_history: list of dicts with keys {month, forecast_sum, actual_sum}.
    Returns exception dict or None.
    """
    cfg = config.get("thresholds", {}).get("forecast_bias", {})
    bias_pct_threshold = float(cfg.get("bias_pct_threshold", 20.0))
    critical_pct = float(cfg.get("critical_pct_threshold", 40.0))
    min_months = int(cfg.get("min_months", 3))
    min_actual_units = float(cfg.get("min_actual_units", 100))

    if len(bias_history) < min_months:
        return None

    # Check recent N months for consistent bias
    recent = bias_history[-min_months:]
    total_actual = sum(r["actual_sum"] for r in recent)
    total_forecast = sum(r["forecast_sum"] for r in recent)

    if total_actual < min_actual_units:
        return None

    if total_actual == 0:
        return None

    bias_pct = ((total_forecast / total_actual) - 1.0) * 100.0
    abs_bias = abs(bias_pct)

    if abs_bias < bias_pct_threshold:
        return None

    # Check that bias is consistent in direction (not oscillating)
    consistent = all(
        ((r["forecast_sum"] / max(r["actual_sum"], 1) - 1.0) * 100.0) * bias_pct >= 0
        for r in recent
        if r["actual_sum"] > 0
    )
    if not consistent:
        return None

    # Compute severity score (0.0–1.0)
    if abs_bias >= critical_pct:
        rule_score = min(1.0, abs_bias / (critical_pct * 2))
    else:
        rule_score = min(0.74, abs_bias / (bias_pct_threshold * 2 + 10))

    direction = "over-forecast" if bias_pct > 0 else "under-forecast"
    avg_monthly = total_actual / min_months

    headline = (
        f"Forecast Bias: Item {item_id} @ {loc} is "
        f"{abs_bias:.0f}% {direction} over {min_months} months"
    )

    severity_score = score_exception(
        {"rule_score": rule_score, "urgency": min(1.0, abs_bias / 100)},
        financial_impact=None,
        config=config,
    )

    return {
        "exception_type": "forecast_bias",
        "item_id": item_id,
        "loc": loc,
        "severity": round(severity_score, 4),
        "financial_impact": None,
        "headline": headline,
        "supporting_data": {
            "bias_pct": round(bias_pct, 2),
            "direction": direction,
            "months_evaluated": min_months,
            "total_actual": round(total_actual, 2),
            "total_forecast": round(total_forecast, 2),
            "avg_monthly_actual": round(avg_monthly, 2),
        },
        "month_start": recent[-1].get("month"),
    }


def detect_stockout_risk(
    item_id: str,
    loc: str,
    dos: float,
    is_below_ss: bool,
    config: dict,
) -> dict | None:
    """Detect stockout risk from days-of-supply and safety stock position.

    dos: current days of supply.
    is_below_ss: whether inventory is already below safety stock.
    Returns exception dict or None.
    """
    cfg = config.get("thresholds", {}).get("stockout_risk", {})
    dos_threshold = float(cfg.get("dos_threshold", 14))
    critical_dos = float(cfg.get("critical_dos_threshold", 7))

    if dos >= dos_threshold and not is_below_ss:
        return None

    # rule_score: 1.0 when DOS=0, 0.0 when DOS=dos_threshold
    if dos_threshold > 0:
        rule_score = max(0.0, min(1.0, (dos_threshold - dos) / dos_threshold))
    else:
        rule_score = 1.0

    if dos <= critical_dos:
        urgency = 1.0
    elif dos <= dos_threshold:
        urgency = max(0.0, (dos_threshold - dos) / dos_threshold)
    else:
        urgency = 0.3  # below_ss only

    severity_score = score_exception(
        {"rule_score": rule_score, "urgency": urgency},
        financial_impact=None,
        config=config,
    )

    headline = (
        f"Stockout Risk: Item {item_id} @ {loc} has "
        f"{dos:.1f} days of supply"
        + (" (below safety stock)" if is_below_ss else "")
    )

    return {
        "exception_type": "stockout_risk",
        "item_id": item_id,
        "loc": loc,
        "severity": round(severity_score, 4),
        "financial_impact": None,
        "headline": headline,
        "supporting_data": {
            "dos": round(dos, 2),
            "dos_threshold": dos_threshold,
            "is_below_safety_stock": is_below_ss,
        },
        "month_start": None,
    }


def detect_accuracy_drop(
    item_id: str,
    loc: str,
    recent_wape: float,
    baseline_wape: float,
    config: dict,
) -> dict | None:
    """Detect significant forecast accuracy degradation vs baseline.

    recent_wape: WAPE over the most recent period (e.g. last month).
    baseline_wape: WAPE over the baseline period (e.g. prior 3 months avg).
    Returns exception dict or None.
    """
    cfg = config.get("thresholds", {}).get("accuracy_drop", {})
    drop_pct = float(cfg.get("accuracy_drop_pct", 15.0))
    critical_drop = float(cfg.get("critical_drop_pct", 25.0))
    min_recent_wape = float(cfg.get("min_recent_wape", 40.0))

    wape_delta = recent_wape - baseline_wape

    flagged = (wape_delta >= drop_pct) or (recent_wape >= min_recent_wape)
    if not flagged:
        return None

    if wape_delta >= critical_drop:
        rule_score = min(1.0, wape_delta / (critical_drop * 2))
        urgency = 1.0
    elif wape_delta >= drop_pct:
        rule_score = min(0.74, wape_delta / (drop_pct * 2))
        urgency = 0.6
    else:
        # Only flagged because recent_wape is high
        rule_score = min(0.5, recent_wape / 100)
        urgency = 0.4

    severity_score = score_exception(
        {"rule_score": rule_score, "urgency": urgency},
        financial_impact=None,
        config=config,
    )

    headline = (
        f"Accuracy Drop: Item {item_id} @ {loc} WAPE rose "
        f"{wape_delta:.1f}pp (from {baseline_wape:.1f}% to {recent_wape:.1f}%)"
    )

    return {
        "exception_type": "accuracy_drop",
        "item_id": item_id,
        "loc": loc,
        "severity": round(severity_score, 4),
        "financial_impact": None,
        "headline": headline,
        "supporting_data": {
            "recent_wape": round(recent_wape, 2),
            "baseline_wape": round(baseline_wape, 2),
            "wape_delta_pp": round(wape_delta, 2),
        },
        "month_start": None,
    }


def detect_excess_risk(
    item_id: str,
    loc: str,
    dos: float,
    config: dict,
) -> dict | None:
    """Detect excess inventory risk from high days-of-supply.

    dos: current days of supply.
    Returns exception dict or None.
    """
    cfg = config.get("thresholds", {}).get("excess_risk", {})
    excess_dos = float(cfg.get("excess_dos_threshold", 90))
    critical_dos = float(cfg.get("critical_dos_threshold", 180))

    if dos < excess_dos:
        return None

    if dos >= critical_dos:
        rule_score = min(1.0, dos / (critical_dos * 2))
        urgency = 1.0
    else:
        rule_score = min(0.74, (dos - excess_dos) / (excess_dos * 2))
        urgency = 0.5

    severity_score = score_exception(
        {"rule_score": rule_score, "urgency": urgency},
        financial_impact=None,
        config=config,
    )

    headline = (
        f"Excess Risk: Item {item_id} @ {loc} has "
        f"{dos:.0f} days of supply (threshold: {excess_dos:.0f}d)"
    )

    return {
        "exception_type": "excess_risk",
        "item_id": item_id,
        "loc": loc,
        "severity": round(severity_score, 4),
        "financial_impact": None,
        "headline": headline,
        "supporting_data": {
            "dos": round(dos, 2),
            "excess_dos_threshold": excess_dos,
            "critical_dos_threshold": critical_dos,
        },
        "month_start": None,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_exception(
    exception_data: dict,
    financial_impact: float | None,
    config: dict,
) -> float:
    """Compute normalized severity score (0.0–1.0) from rule + financial signals.

    exception_data must contain:
      - rule_score (float 0-1): how strongly the rule fired
      - urgency (float 0-1): time-based urgency
    financial_impact: dollar impact (or None if unknown)
    config: exception_engine config dict
    """
    weights = config.get("severity_weights", {})
    w_financial = float(weights.get("financial_impact", 0.4))
    w_rule = float(weights.get("rule_score", 0.4))
    w_urgency = float(weights.get("urgency", 0.2))

    rule_score = float(exception_data.get("rule_score", 0.5))
    urgency = float(exception_data.get("urgency", 0.5))

    # Normalize financial impact to 0-1 using log10 scale
    if financial_impact and financial_impact > 0:
        # log10(1)=0, log10(1M)=6 → normalize to [0,1] over [0, 1M]
        financial_score = min(1.0, math.log10(max(financial_impact, 1)) / 6.0)
    else:
        # No financial data — use rule + urgency only (rebalance weights)
        total = w_rule + w_urgency
        if total > 0:
            return min(1.0, (rule_score * w_rule + urgency * w_urgency) / total)
        return 0.5

    return min(
        1.0,
        rule_score * w_rule + financial_score * w_financial + urgency * w_urgency,
    )


# ---------------------------------------------------------------------------
# Headline generation (rule-based, no LLM)
# ---------------------------------------------------------------------------

def generate_headline(exception_type: str, data: dict) -> str:
    """Generate a concise human-readable headline for an exception.

    Uses rule-based templates — no LLM required.
    """
    item_id = data.get("item_id", "?")
    loc = data.get("loc", "?")
    sd = data.get("supporting_data", {})

    if exception_type == "forecast_bias":
        bias = sd.get("bias_pct", 0)
        direction = "over-forecast" if bias > 0 else "under-forecast"
        months = sd.get("months_evaluated", 3)
        return (
            f"Forecast Bias: Item {item_id} @ {loc} is "
            f"{abs(bias):.0f}% {direction} over {months} months"
        )
    if exception_type == "stockout_risk":
        dos = sd.get("dos", 0)
        return f"Stockout Risk: Item {item_id} @ {loc} has {dos:.1f} days of supply"
    if exception_type == "accuracy_drop":
        delta = sd.get("wape_delta_pp", 0)
        recent = sd.get("recent_wape", 0)
        return (
            f"Accuracy Drop: Item {item_id} @ {loc} WAPE rose "
            f"{delta:.1f}pp to {recent:.1f}%"
        )
    if exception_type == "excess_risk":
        dos = sd.get("dos", 0)
        return f"Excess Risk: Item {item_id} @ {loc} has {dos:.0f} days of supply"
    if exception_type == "model_drift":
        return f"Model Drift: Champion model unstable for Item {item_id} @ {loc}"
    if exception_type == "new_item":
        months = sd.get("history_months", "< 3")
        return f"New Item: Item {item_id} @ {loc} has only {months} months of history"
    return f"{exception_type}: Item {item_id} @ {loc}"


# ---------------------------------------------------------------------------
# DB orchestration
# ---------------------------------------------------------------------------

def run_exception_detection(
    conn: Any,
    config: dict,
    month_start: date,
    dry_run: bool = False,
) -> dict:
    """Full exception detection pipeline.

    Queries DB for signals, applies detection rules, inserts into exception_queue.

    Steps:
      1. Query fact_external_forecast_monthly for sustained forecast bias per item-loc
      2. Query mv_inventory_health_score for stockout risk (low DOS)
      3. Query aggregated accuracy for accuracy drop
      4. Query agg_inventory_monthly for excess risk (high DOS)
      5. Insert detected exceptions into exception_queue with dedupe check

    Returns {"detected": N, "inserted": N, "skipped_dedupe": N, "dry_run": bool}
    """
    engine_cfg = config.get("exception_engine", config)
    thresholds = engine_cfg.get("thresholds", {})
    dedupe_days = int(engine_cfg.get("dedupe_window_days", 7))
    max_exceptions = int(engine_cfg.get("max_exceptions_per_run", 500))
    ttl_days = int(engine_cfg.get("exception_ttl_days", 30))
    exception_types = engine_cfg.get("exception_types", [
        "forecast_bias", "stockout_risk", "accuracy_drop", "excess_risk",
        "model_drift", "new_item",
    ])

    expires_at = datetime.now(timezone.utc) + timedelta(days=ttl_days)
    dedupe_cutoff = datetime.now(timezone.utc) - timedelta(days=dedupe_days)

    exceptions_to_insert: list[dict] = []

    with conn.cursor() as cur:
        # ------------------------------------------------------------------
        # 1. Forecast bias detection
        # ------------------------------------------------------------------
        if "forecast_bias" in exception_types:
            bias_cfg = thresholds.get("forecast_bias", {})
            bias_threshold = float(bias_cfg.get("bias_pct_threshold", 20.0))
            min_months = int(bias_cfg.get("min_months", 3))
            min_actual = float(bias_cfg.get("min_actual_units", 100))

            cur.execute("""
                SELECT
                    item_id AS item_id,
                    loc,
                    COUNT(DISTINCT startdate)                    AS month_count,
                    SUM(tothist_dmd)                                     AS total_actual,
                    SUM(basefcst_pref)                            AS total_forecast,
                    (SUM(basefcst_pref) / NULLIF(SUM(tothist_dmd), 0) - 1) * 100 AS bias_pct
                FROM fact_external_forecast_monthly
                WHERE
                    startdate >= %s - INTERVAL '3 months'
                    AND startdate < %s
                    AND lag = 0
                    AND tothist_dmd > 0
                    AND model_id = 'external'
                GROUP BY item_id, loc
                HAVING
                    COUNT(DISTINCT startdate) >= %s
                    AND SUM(tothist_dmd) >= %s
                    AND ABS((SUM(basefcst_pref) / NULLIF(SUM(tothist_dmd), 0) - 1) * 100) >= %s
            """, (month_start, month_start, min_months, min_actual, bias_threshold))

            for row in cur.fetchall():
                item_id, loc, month_count, total_actual, total_forecast, bias_pct = row
                if total_actual is None or total_actual == 0:
                    continue

                bias_pct_val = float(bias_pct or 0)
                abs_bias = abs(bias_pct_val)
                critical_pct = float(bias_cfg.get("critical_pct_threshold", 40.0))

                if abs_bias >= critical_pct:
                    rule_score = min(1.0, abs_bias / (critical_pct * 2))
                else:
                    rule_score = min(0.74, abs_bias / (bias_threshold * 2 + 10))

                urgency = min(1.0, abs_bias / 100)
                severity_score = score_exception(
                    {"rule_score": rule_score, "urgency": urgency},
                    financial_impact=None,
                    config=engine_cfg,
                )

                direction = "over-forecast" if bias_pct_val > 0 else "under-forecast"
                sd = {
                    "bias_pct": round(bias_pct_val, 2),
                    "direction": direction,
                    "months_evaluated": int(month_count or min_months),
                    "total_actual": float(total_actual or 0),
                    "total_forecast": float(total_forecast or 0),
                }
                headline = generate_headline("forecast_bias", {
                    "item_id": item_id, "loc": loc, "supporting_data": sd,
                })

                exceptions_to_insert.append({
                    "exception_id": str(uuid.uuid4()),
                    "exception_type": "forecast_bias",
                    "item_id": str(item_id),
                    "loc": str(loc),
                    "severity": round(severity_score, 4),
                    "financial_impact": None,
                    "headline": headline,
                    "supporting_data": sd,
                    "month_start": month_start,
                    "expires_at": expires_at,
                })

        # ------------------------------------------------------------------
        # 2. Stockout risk detection (from mv_inventory_health_score)
        # ------------------------------------------------------------------
        if "stockout_risk" in exception_types:
            sr_cfg = thresholds.get("stockout_risk", {})
            dos_threshold = float(sr_cfg.get("dos_threshold", 14))
            critical_dos = float(sr_cfg.get("critical_dos_threshold", 7))

            cur.execute("""
                SELECT
                    item_id,
                    loc,
                    current_dos,
                    is_below_ss,
                    ss_coverage
                FROM mv_inventory_health_score
                WHERE
                    (current_dos < %s OR is_below_ss = TRUE)
                    AND current_dos IS NOT NULL
                LIMIT %s
            """, (dos_threshold, max_exceptions))

            for row in cur.fetchall():
                item_id, loc, dos, is_below_ss, ss_coverage_score = row
                dos_val = float(dos or 0)
                below_ss = bool(is_below_ss)

                if dos_val >= dos_threshold and not below_ss:
                    continue

                if dos_val <= critical_dos:
                    rule_score = min(1.0, (dos_threshold - dos_val) / dos_threshold)
                    urgency = 1.0
                elif dos_val < dos_threshold:
                    rule_score = min(0.74, (dos_threshold - dos_val) / dos_threshold)
                    urgency = 0.6
                else:
                    rule_score = 0.4
                    urgency = 0.3

                severity_score = score_exception(
                    {"rule_score": rule_score, "urgency": urgency},
                    financial_impact=None,
                    config=engine_cfg,
                )

                sd = {
                    "dos": round(dos_val, 2),
                    "dos_threshold": dos_threshold,
                    "is_below_safety_stock": below_ss,
                }
                headline = generate_headline("stockout_risk", {
                    "item_id": item_id, "loc": loc, "supporting_data": sd,
                })

                exceptions_to_insert.append({
                    "exception_id": str(uuid.uuid4()),
                    "exception_type": "stockout_risk",
                    "item_id": str(item_id),
                    "loc": str(loc),
                    "severity": round(severity_score, 4),
                    "financial_impact": None,
                    "headline": headline,
                    "supporting_data": sd,
                    "month_start": month_start,
                    "expires_at": expires_at,
                })

        # ------------------------------------------------------------------
        # 3. Accuracy drop detection
        # ------------------------------------------------------------------
        if "accuracy_drop" in exception_types:
            ad_cfg = thresholds.get("accuracy_drop", {})
            drop_pct = float(ad_cfg.get("accuracy_drop_pct", 15.0))
            min_recent_wape = float(ad_cfg.get("min_recent_wape", 40.0))

            cur.execute("""
                WITH recent AS (
                    SELECT
                        item_id AS item_id,
                        loc,
                        (1 - (SUM(ABS(basefcst_pref - tothist_dmd)) / NULLIF(ABS(SUM(tothist_dmd)), 0))) * 100 AS recent_accuracy
                    FROM fact_external_forecast_monthly
                    WHERE
                        startdate >= %s - INTERVAL '1 month'
                        AND startdate < %s
                        AND lag = 0
                        AND model_id = 'external'
                    GROUP BY item_id, loc
                ),
                baseline AS (
                    SELECT
                        item_id AS item_id,
                        loc,
                        (1 - (SUM(ABS(basefcst_pref - tothist_dmd)) / NULLIF(ABS(SUM(tothist_dmd)), 0))) * 100 AS baseline_accuracy
                    FROM fact_external_forecast_monthly
                    WHERE
                        startdate >= %s - INTERVAL '4 months'
                        AND startdate < %s - INTERVAL '1 month'
                        AND lag = 0
                        AND model_id = 'external'
                    GROUP BY item_id, loc
                )
                SELECT
                    r.item_id,
                    r.loc,
                    r.recent_accuracy,
                    b.baseline_accuracy,
                    b.baseline_accuracy - r.recent_accuracy AS accuracy_drop_pp
                FROM recent r
                JOIN baseline b ON r.item_id = b.item_id AND r.loc = b.loc
                WHERE
                    b.baseline_accuracy - r.recent_accuracy >= %s
                    OR (100 - r.recent_accuracy) >= %s
                LIMIT %s
            """, (
                month_start, month_start,
                month_start, month_start,
                drop_pct, min_recent_wape, max_exceptions,
            ))

            for row in cur.fetchall():
                item_id, loc, recent_acc, baseline_acc, acc_drop = row
                recent_wape = float(100 - (recent_acc or 0))
                baseline_wape = float(100 - (baseline_acc or 0))
                wape_delta = float(acc_drop or 0)

                critical_drop = float(ad_cfg.get("critical_drop_pct", 25.0))
                if wape_delta >= critical_drop:
                    rule_score = min(1.0, wape_delta / (critical_drop * 2))
                    urgency = 1.0
                elif wape_delta >= drop_pct:
                    rule_score = min(0.74, wape_delta / (drop_pct * 2))
                    urgency = 0.6
                else:
                    rule_score = min(0.5, recent_wape / 100)
                    urgency = 0.4

                severity_score = score_exception(
                    {"rule_score": rule_score, "urgency": urgency},
                    financial_impact=None,
                    config=engine_cfg,
                )

                sd = {
                    "recent_wape": round(recent_wape, 2),
                    "baseline_wape": round(baseline_wape, 2),
                    "wape_delta_pp": round(wape_delta, 2),
                }
                headline = generate_headline("accuracy_drop", {
                    "item_id": item_id, "loc": loc, "supporting_data": sd,
                })

                exceptions_to_insert.append({
                    "exception_id": str(uuid.uuid4()),
                    "exception_type": "accuracy_drop",
                    "item_id": str(item_id),
                    "loc": str(loc),
                    "severity": round(severity_score, 4),
                    "financial_impact": None,
                    "headline": headline,
                    "supporting_data": sd,
                    "month_start": month_start,
                    "expires_at": expires_at,
                })

        # ------------------------------------------------------------------
        # 4. Excess risk detection (from agg_inventory_monthly)
        # ------------------------------------------------------------------
        if "excess_risk" in exception_types:
            er_cfg = thresholds.get("excess_risk", {})
            excess_dos = float(er_cfg.get("excess_dos_threshold", 90))
            critical_dos_er = float(er_cfg.get("critical_dos_threshold", 180))

            cur.execute("""
                SELECT
                    item_id,
                    loc,
                    eom_qty_on_hand / NULLIF(avg_daily_sls, 0) AS dos
                FROM agg_inventory_monthly
                WHERE
                    month_start = %s
                    AND eom_qty_on_hand / NULLIF(avg_daily_sls, 0) > %s
                LIMIT %s
            """, (month_start, excess_dos, max_exceptions))

            for row in cur.fetchall():
                item_id, loc, dos = row
                dos_val = float(dos or 0)

                if dos_val < excess_dos:
                    continue

                if dos_val >= critical_dos_er:
                    rule_score = min(1.0, dos_val / (critical_dos_er * 2))
                    urgency = 1.0
                else:
                    rule_score = min(0.74, (dos_val - excess_dos) / (excess_dos * 2))
                    urgency = 0.5

                severity_score = score_exception(
                    {"rule_score": rule_score, "urgency": urgency},
                    financial_impact=None,
                    config=engine_cfg,
                )

                sd = {
                    "dos": round(dos_val, 2),
                    "excess_dos_threshold": excess_dos,
                    "critical_dos_threshold": critical_dos_er,
                }
                headline = generate_headline("excess_risk", {
                    "item_id": item_id, "loc": loc, "supporting_data": sd,
                })

                exceptions_to_insert.append({
                    "exception_id": str(uuid.uuid4()),
                    "exception_type": "excess_risk",
                    "item_id": str(item_id),
                    "loc": str(loc),
                    "severity": round(severity_score, 4),
                    "financial_impact": None,
                    "headline": headline,
                    "supporting_data": sd,
                    "month_start": month_start,
                    "expires_at": expires_at,
                })

    # ------------------------------------------------------------------
    # Deduplicate + insert
    # ------------------------------------------------------------------
    detected = len(exceptions_to_insert)

    # Apply per-run cap, ranked by severity desc
    exceptions_to_insert.sort(key=lambda e: e["severity"], reverse=True)
    exceptions_to_insert = exceptions_to_insert[:max_exceptions]

    if dry_run:
        return {
            "detected": detected,
            "inserted": 0,
            "skipped_dedupe": 0,
            "dry_run": True,
            "sample": [
                {"exception_type": e["exception_type"], "item_id": e["item_id"],
                 "loc": e["loc"], "severity": e["severity"], "headline": e["headline"]}
                for e in exceptions_to_insert[:5]
            ],
        }

    with conn.cursor() as cur:
        # Batch dedupe: find all (item_id, loc, exception_type) combos that
        # already have an open/investigating exception within the dedupe window.
        # Use a temp table to avoid building huge IN-lists.
        cur.execute("""
            CREATE TEMP TABLE _exc_candidates (
                idx int, item_id text, loc text, exception_type text
            ) ON COMMIT DROP
        """)
        candidate_rows = [
            (i, exc["item_id"], exc["loc"], exc["exception_type"])
            for i, exc in enumerate(exceptions_to_insert)
        ]
        if candidate_rows:
            cur.executemany(
                "INSERT INTO _exc_candidates (idx, item_id, loc, exception_type) "
                "VALUES (%s, %s, %s, %s)",
                candidate_rows,
            )

        # Find indices that are duplicates of existing open exceptions
        cur.execute("""
            SELECT DISTINCT c.idx
            FROM _exc_candidates c
            JOIN exception_queue eq
              ON eq.item_id = c.item_id
             AND eq.loc = c.loc
             AND eq.exception_type = c.exception_type
             AND eq.status IN ('open', 'investigating')
             AND eq.generated_at > %s
        """, (dedupe_cutoff,))
        dupe_indices = {row[0] for row in cur.fetchall()}

        # Build rows to insert, skipping duplicates
        insert_rows = []
        for i, exc in enumerate(exceptions_to_insert):
            if i in dupe_indices:
                continue
            insert_rows.append((
                exc["exception_id"],
                exc["exception_type"],
                exc["item_id"],
                exc["loc"],
                exc["severity"],
                exc.get("financial_impact"),
                exc["headline"],
                json.dumps(exc.get("supporting_data") or {}),
                exc.get("expires_at"),
                exc.get("month_start"),
            ))

        if insert_rows:
            cur.executemany("""
                INSERT INTO exception_queue
                    (exception_id, exception_type, item_id, loc,
                     severity, financial_impact, headline, supporting_data,
                     status, generated_at, expires_at, month_start)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, 'open', NOW(), %s, %s)
                ON CONFLICT (exception_id) DO NOTHING
            """, insert_rows)

        inserted = len(insert_rows)
        skipped = len(dupe_indices)

    conn.commit()

    return {
        "detected": detected,
        "inserted": inserted,
        "skipped_dedupe": skipped,
        "dry_run": False,
    }
