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

import hashlib
import json
import logging
import math
import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gen-4 SC-8: Root-cause grouping + SLA helpers
# ---------------------------------------------------------------------------

_SEVERITY_BANDS_DEFAULT: dict[str, float] = {
    "critical": 0.75,
    "high":     0.50,
    "medium":   0.25,
}
_RESPONSE_HOURS_DEFAULT: dict[str, int] = {
    "critical": 4, "high": 24, "medium": 72, "low": 168,
}


def derive_severity_band(
    severity: float,
    band_thresholds: dict[str, float] | None = None,
) -> str:
    """Map a severity score (0-1) to a categorical band.

    Returns one of: critical, high, medium, low.
    """
    bands = band_thresholds or _SEVERITY_BANDS_DEFAULT
    if severity >= float(bands.get("critical", 0.75)):
        return "critical"
    if severity >= float(bands.get("high", 0.50)):
        return "high"
    if severity >= float(bands.get("medium", 0.25)):
        return "medium"
    return "low"


def compute_sla_due_at(
    generated_at: datetime,
    severity_band: str,
    response_hours: dict[str, int] | None = None,
) -> datetime:
    """Compute SLA response deadline from generation time + band-specific hours.

    Args:
        generated_at:    When the exception was generated (TZ-aware recommended).
        severity_band:   critical | high | medium | low.
        response_hours:  Optional override; defaults to Gen-4 SC-8 values.
    """
    hours = response_hours or _RESPONSE_HOURS_DEFAULT
    h = int(hours.get(severity_band, hours.get("low", 168)))
    return generated_at + timedelta(hours=h)


def compute_root_cause_key(
    exception_type: str,
    supporting_data: dict[str, Any] | None,
    key_fields: list[str] | None = None,
) -> str:
    """Deterministic hash grouping exceptions with the same underlying cause.

    Default fields include `exception_type` and a stable bucket label derived
    from the primary metric (e.g. bias direction, DOS band). Callers can supply
    the exact fields via config/exception_sla.yaml.

    Returns a 16-char hex digest.
    """
    sd = supporting_data or {}
    parts: list[str] = [exception_type]

    # Common derived buckets — cheap + deterministic.
    if "bias_pct" in sd:
        direction = "over" if float(sd.get("bias_pct", 0) or 0) > 0 else "under"
        parts.append(f"bias:{direction}")
    if "dos" in sd:
        dos_val = float(sd.get("dos") or 0)
        # coarse DOS bucket — stable enough to group repeat occurrences
        if dos_val < 7:
            parts.append("dos:lt7")
        elif dos_val < 14:
            parts.append("dos:lt14")
        elif dos_val < 30:
            parts.append("dos:lt30")
        elif dos_val < 90:
            parts.append("dos:lt90")
        else:
            parts.append("dos:ge90")

    # Honour caller-supplied key fields when provided.
    for f in key_fields or []:
        v = sd.get(f)
        if v is not None:
            parts.append(f"{f}:{v}")

    digest = hashlib.sha1("|".join(parts).encode("utf-8"), usedforsecurity=False)
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Financial impact estimators (Gen-4 Roadmap 1.8)
# ---------------------------------------------------------------------------
#
# Every detector must pass a financial_impact to score_exception(). The prior
# behavior passed None everywhere, leaving the 0.4 weight on financial_impact
# dead. Detectors now estimate a dollar impact from the signal + unit economics
# the caller provides.
#
# All estimators take optional unit_cost / unit_margin (USD) and return a
# non-negative float or None if enough economics isn't known. Callers use the
# result directly as the `financial_impact` argument to score_exception.


def estimate_financial_impact_bias(
    abs_bias_pct: float,
    total_actual_units: float,
    unit_cost: float | None,
    unit_margin: float | None,
) -> float | None:
    """Forecast-bias financial impact.

    Treats bias as mis-allocated plan. Dollar impact = |bias%| * actual_units *
    (unit_margin if we have it else unit_cost) — margin is preferred because
    over-forecast ties up cash (margin * excess) and under-forecast misses a
    sale (margin * miss), both land on the margin scale.
    """
    if unit_cost is None and unit_margin is None:
        return None
    if total_actual_units <= 0:
        return None
    dollars_per_unit = unit_margin if unit_margin is not None else unit_cost
    if dollars_per_unit is None or dollars_per_unit <= 0:
        return None
    return max(0.0, (abs_bias_pct / 100.0) * total_actual_units * dollars_per_unit)


def estimate_financial_impact_stockout(
    dos: float,
    daily_demand_rate: float | None,
    unit_margin: float | None,
    unit_cost: float | None,
    horizon_days: int = 7,
) -> float | None:
    """Stockout-risk financial impact = lost-sales margin over horizon.

    Lost-sales margin = max(0, horizon - dos) * daily_demand * unit_margin.
    Falls back to unit_cost if margin missing (lower bound).
    """
    if daily_demand_rate is None or daily_demand_rate <= 0:
        return None
    dollars_per_unit = unit_margin if unit_margin is not None else unit_cost
    if dollars_per_unit is None or dollars_per_unit <= 0:
        return None
    exposure_days = max(0.0, float(horizon_days) - max(0.0, float(dos)))
    return float(exposure_days * daily_demand_rate * dollars_per_unit)


def estimate_financial_impact_excess(
    dos: float,
    excess_dos_threshold: float,
    daily_demand_rate: float | None,
    unit_cost: float | None,
    carrying_cost_rate: float,
) -> float | None:
    """Excess-risk financial impact = carrying cost on excess inventory (annualized).

    Excess units = max(0, dos - threshold) * daily_demand.
    Annual carrying cost = excess_units * unit_cost * carrying_cost_rate.
    Monthly carrying cost = annual / 12 — returned as the exception impact.
    """
    if (daily_demand_rate is None or daily_demand_rate <= 0
            or unit_cost is None or unit_cost <= 0):
        return None
    excess_days = max(0.0, float(dos) - float(excess_dos_threshold))
    excess_units = excess_days * float(daily_demand_rate)
    annual_cost = excess_units * unit_cost * float(carrying_cost_rate)
    return float(annual_cost / 12.0)


def estimate_financial_impact_accuracy(
    wape_delta_pp: float,
    recent_actual_units: float,
    unit_margin: float | None,
    unit_cost: float | None,
) -> float | None:
    """Accuracy-drop financial impact = delta_wape * recent_actual * dollars_per_unit.

    Interprets WAPE delta as extra planning error volume that will land in
    either stockout lost margin or excess carry.
    """
    if recent_actual_units <= 0:
        return None
    dollars_per_unit = unit_margin if unit_margin is not None else unit_cost
    if dollars_per_unit is None or dollars_per_unit <= 0:
        return None
    return max(0.0, (wape_delta_pp / 100.0) * recent_actual_units * dollars_per_unit)


# ---------------------------------------------------------------------------
# Pure detection functions
# ---------------------------------------------------------------------------

def detect_forecast_bias(
    item_id: str,
    loc: str,
    bias_history: list[dict],
    config: dict,
    unit_cost: float | None = None,
    unit_margin: float | None = None,
) -> dict | None:
    """Detect sustained forecast bias over trailing N months.

    bias_history: list of dicts with keys {month, forecast_sum, actual_sum}.
    unit_cost/unit_margin: optional unit economics for dollar impact (Gen-4 1.8).
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

    financial_impact = estimate_financial_impact_bias(
        abs_bias, total_actual, unit_cost, unit_margin
    )

    severity_score = score_exception(
        {"rule_score": rule_score, "urgency": min(1.0, abs_bias / 100)},
        financial_impact=financial_impact,
        config=config,
    )

    return {
        "exception_type": "forecast_bias",
        "item_id": item_id,
        "loc": loc,
        "severity": round(severity_score, 4),
        "financial_impact": financial_impact,
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
    daily_demand_rate: float | None = None,
    unit_cost: float | None = None,
    unit_margin: float | None = None,
) -> dict | None:
    """Detect stockout risk from days-of-supply and safety stock position.

    dos: current days of supply.
    is_below_ss: whether inventory is already below safety stock.
    daily_demand_rate/unit_cost/unit_margin: optional unit economics for the
        dollar-impact score (Gen-4 1.8).
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

    financial_impact = estimate_financial_impact_stockout(
        dos, daily_demand_rate, unit_margin, unit_cost,
        horizon_days=int(cfg.get("financial_horizon_days", 7)),
    )

    severity_score = score_exception(
        {"rule_score": rule_score, "urgency": urgency},
        financial_impact=financial_impact,
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
        "financial_impact": financial_impact,
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
    recent_actual_units: float = 0.0,
    unit_cost: float | None = None,
    unit_margin: float | None = None,
) -> dict | None:
    """Detect significant forecast accuracy degradation vs baseline.

    recent_wape: WAPE over the most recent period (e.g. last month).
    baseline_wape: WAPE over the baseline period (e.g. prior 3 months avg).
    recent_actual_units/unit_cost/unit_margin: optional unit economics for
        the dollar-impact score (Gen-4 1.8).
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

    financial_impact = estimate_financial_impact_accuracy(
        wape_delta, recent_actual_units, unit_margin, unit_cost
    )

    severity_score = score_exception(
        {"rule_score": rule_score, "urgency": urgency},
        financial_impact=financial_impact,
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
        "financial_impact": financial_impact,
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
    daily_demand_rate: float | None = None,
    unit_cost: float | None = None,
) -> dict | None:
    """Detect excess inventory risk from high days-of-supply.

    dos: current days of supply.
    daily_demand_rate/unit_cost: optional unit economics for dollar-impact
        (carrying cost; Gen-4 1.8).
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

    financial_impact = estimate_financial_impact_excess(
        dos, excess_dos, daily_demand_rate, unit_cost,
        carrying_cost_rate=float(cfg.get("carrying_cost_rate", 0.25)),
    )

    severity_score = score_exception(
        {"rule_score": rule_score, "urgency": urgency},
        financial_impact=financial_impact,
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
        "financial_impact": financial_impact,
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

    expires_at = datetime.now(UTC) + timedelta(days=ttl_days)
    dedupe_cutoff = datetime.now(UTC) - timedelta(days=dedupe_days)

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

            margin_assumption = float(engine_cfg.get("unit_margin_assumption", 0.30))
            cur.execute("""
                SELECT
                    f.item_id                                     AS item_id,
                    f.loc,
                    COUNT(DISTINCT f.startdate)                    AS month_count,
                    SUM(f.tothist_dmd)                             AS total_actual,
                    SUM(f.basefcst_pref)                           AS total_forecast,
                    (SUM(f.basefcst_pref) / NULLIF(SUM(f.tothist_dmd), 0) - 1) * 100 AS bias_pct,
                    MAX(eoq.unit_cost)                             AS unit_cost
                FROM fact_external_forecast_monthly f
                LEFT JOIN fact_eoq_targets eoq
                  ON eoq.item_id = f.item_id AND eoq.loc = f.loc
                WHERE
                    f.startdate >= %s - INTERVAL '3 months'
                    AND f.startdate < %s
                    AND f.lag = 0
                    AND f.tothist_dmd > 0
                    AND f.model_id = 'external'
                GROUP BY f.item_id, f.loc
                HAVING
                    COUNT(DISTINCT f.startdate) >= %s
                    AND SUM(f.tothist_dmd) >= %s
                    AND ABS((SUM(f.basefcst_pref) / NULLIF(SUM(f.tothist_dmd), 0) - 1) * 100) >= %s
            """, (month_start, month_start, min_months, min_actual, bias_threshold))

            for row in cur.fetchall():
                item_id, loc, month_count, total_actual, total_forecast, bias_pct, unit_cost = row
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
                unit_cost_f = float(unit_cost) if unit_cost is not None else None
                unit_margin_f = (
                    unit_cost_f * margin_assumption if unit_cost_f is not None else None
                )
                financial_impact = estimate_financial_impact_bias(
                    abs_bias, float(total_actual), unit_cost_f, unit_margin_f
                )
                severity_score = score_exception(
                    {"rule_score": rule_score, "urgency": urgency},
                    financial_impact=financial_impact,
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
                    "financial_impact": financial_impact,
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

            # Join unit economics (fact_eoq_targets) so every stockout exception
            # gets a dollar impact (Gen-4 1.8). unit_margin is derived as
            # unit_cost * margin_assumption when an explicit margin isn't stored.
            margin_assumption = float(engine_cfg.get("unit_margin_assumption", 0.30))
            horizon_days = int(sr_cfg.get("financial_horizon_days", 7))
            cur.execute("""
                SELECT
                    h.item_id,
                    h.loc,
                    h.current_dos,
                    h.is_below_ss,
                    h.ss_coverage,
                    eoq.unit_cost,
                    h.avg_daily_sls
                FROM mv_inventory_health_score h
                LEFT JOIN fact_eoq_targets eoq
                  ON eoq.item_id = h.item_id AND eoq.loc = h.loc
                WHERE
                    (h.current_dos < %s OR h.is_below_ss = TRUE)
                    AND h.current_dos IS NOT NULL
                LIMIT %s
            """, (dos_threshold, max_exceptions))

            for row in cur.fetchall():
                item_id, loc, dos, is_below_ss, ss_coverage_score, unit_cost, daily_demand = row
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

                unit_cost_f = float(unit_cost) if unit_cost is not None else None
                unit_margin_f = (
                    unit_cost_f * margin_assumption if unit_cost_f is not None else None
                )
                daily_demand_f = float(daily_demand) if daily_demand is not None else None
                financial_impact = estimate_financial_impact_stockout(
                    dos_val, daily_demand_f, unit_margin_f, unit_cost_f, horizon_days
                )

                severity_score = score_exception(
                    {"rule_score": rule_score, "urgency": urgency},
                    financial_impact=financial_impact,
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
                    "financial_impact": financial_impact,
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

            margin_assumption = float(engine_cfg.get("unit_margin_assumption", 0.30))
            cur.execute("""
                WITH recent AS (
                    SELECT
                        item_id AS item_id,
                        loc,
                        (1 - (SUM(ABS(basefcst_pref - tothist_dmd)) / NULLIF(ABS(SUM(tothist_dmd)), 0))) * 100 AS recent_accuracy,
                        SUM(tothist_dmd) AS recent_actual
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
                    b.baseline_accuracy - r.recent_accuracy AS accuracy_drop_pp,
                    r.recent_actual,
                    eoq.unit_cost
                FROM recent r
                JOIN baseline b ON r.item_id = b.item_id AND r.loc = b.loc
                LEFT JOIN fact_eoq_targets eoq
                  ON eoq.item_id = r.item_id AND eoq.loc = r.loc
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
                item_id, loc, recent_acc, baseline_acc, acc_drop, recent_actual, unit_cost = row
                recent_wape = float(100 - (recent_acc or 0))
                baseline_wape = float(100 - (baseline_acc or 0))
                wape_delta = float(acc_drop or 0)
                recent_actual_f = float(recent_actual or 0)
                unit_cost_f = float(unit_cost) if unit_cost is not None else None
                unit_margin_f = (
                    unit_cost_f * margin_assumption if unit_cost_f is not None else None
                )

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

                financial_impact = estimate_financial_impact_accuracy(
                    wape_delta, recent_actual_f, unit_margin_f, unit_cost_f
                )
                severity_score = score_exception(
                    {"rule_score": rule_score, "urgency": urgency},
                    financial_impact=financial_impact,
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
                    "financial_impact": financial_impact,
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

            carrying_cost_rate = float(er_cfg.get("carrying_cost_rate", 0.25))
            cur.execute("""
                SELECT
                    agg.item_id,
                    agg.loc,
                    agg.eom_qty_on_hand / NULLIF(agg.avg_daily_sls, 0) AS dos,
                    agg.avg_daily_sls,
                    eoq.unit_cost
                FROM agg_inventory_monthly agg
                LEFT JOIN fact_eoq_targets eoq
                  ON eoq.item_id = agg.item_id AND eoq.loc = agg.loc
                WHERE
                    agg.month_start = %s
                    AND agg.eom_qty_on_hand / NULLIF(agg.avg_daily_sls, 0) > %s
                LIMIT %s
            """, (month_start, excess_dos, max_exceptions))

            for row in cur.fetchall():
                item_id, loc, dos, daily_demand, unit_cost = row
                dos_val = float(dos or 0)

                if dos_val < excess_dos:
                    continue

                if dos_val >= critical_dos_er:
                    rule_score = min(1.0, dos_val / (critical_dos_er * 2))
                    urgency = 1.0
                else:
                    rule_score = min(0.74, (dos_val - excess_dos) / (excess_dos * 2))
                    urgency = 0.5

                unit_cost_f = float(unit_cost) if unit_cost is not None else None
                daily_demand_f = float(daily_demand) if daily_demand is not None else None
                financial_impact = estimate_financial_impact_excess(
                    dos_val, excess_dos, daily_demand_f, unit_cost_f, carrying_cost_rate
                )

                severity_score = score_exception(
                    {"rule_score": rule_score, "urgency": urgency},
                    financial_impact=financial_impact,
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
                    "financial_impact": financial_impact,
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

        # Gen-4 SC-8: load SLA config (graceful fallback to defaults)
        sla_cfg: dict[str, Any] = {}
        try:
            from common.utils import load_config as _lc
            sla_cfg = (_lc("exception_sla") or {}).get("exception_sla", {})
        except Exception:
            sla_cfg = {}
        band_thresholds = sla_cfg.get("severity_bands") or _SEVERITY_BANDS_DEFAULT
        response_hours = sla_cfg.get("response_hours") or _RESPONSE_HOURS_DEFAULT
        key_fields = sla_cfg.get("root_cause_key_fields") or []

        # Build rows to insert, skipping duplicates
        insert_rows = []
        now_utc = datetime.now(UTC)
        for i, exc in enumerate(exceptions_to_insert):
            if i in dupe_indices:
                continue
            # Gen-4 SC-8: enrich with root_cause_key / severity_band / sla_due_at
            band = derive_severity_band(float(exc["severity"]), band_thresholds)
            sla_due = compute_sla_due_at(now_utc, band, response_hours)
            rc_key = compute_root_cause_key(
                exc["exception_type"], exc.get("supporting_data"), key_fields
            )
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
                rc_key,
                band,
                sla_due,
            ))

        if insert_rows:
            # Gen-4 SC-8: extended insert with root_cause_key / severity_band / sla_due_at.
            # Columns added by sql/149; fall back to legacy insert if the migration
            # hasn't been applied yet.
            try:
                cur.execute("SAVEPOINT sc8_insert")
                cur.executemany("""
                    INSERT INTO exception_queue
                        (exception_id, exception_type, item_id, loc,
                         severity, financial_impact, headline, supporting_data,
                         status, generated_at, expires_at, month_start,
                         root_cause_key, severity_band, sla_due_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, 'open', NOW(), %s, %s,
                            %s, %s, %s)
                    ON CONFLICT (exception_id) DO NOTHING
                """, insert_rows)
            except Exception:
                cur.execute("ROLLBACK TO SAVEPOINT sc8_insert")
                legacy_rows = [r[:10] for r in insert_rows]
                cur.executemany("""
                    INSERT INTO exception_queue
                        (exception_id, exception_type, item_id, loc,
                         severity, financial_impact, headline, supporting_data,
                         status, generated_at, expires_at, month_start)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, 'open', NOW(), %s, %s)
                    ON CONFLICT (exception_id) DO NOTHING
                """, legacy_rows)

        inserted = len(insert_rows)
        skipped = len(dupe_indices)

    conn.commit()

    return {
        "detected": detected,
        "inserted": inserted,
        "skipped_dedupe": skipped,
        "dry_run": False,
    }
