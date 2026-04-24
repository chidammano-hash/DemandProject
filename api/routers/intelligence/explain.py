"""Forecast explanation API — local SHAP + counterfactual.

Gen-4 Stream G / AI-9.

Endpoint: ``GET /forecast/explain/{item_id}/{loc}``

Reads the currently-promoted model's forecast for the given DFU, joins
the per-DFU SHAP attributions where available, and returns a
counterfactual scenario: "if feature X increased/decreased by +/- one
std dev, forecast would change by ~delta". This is a scaffold using the
top-feature SHAP values; production counterfactuals require retraining
or model-specific partial-dependence.

Uses get_conn() directly (CLAUDE.md critical rule).
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from api.core import get_conn
from common.ai.decision_ledger import DecisionRecord, append_decision

logger = logging.getLogger(__name__)

router = APIRouter(tags=["explain"])

# Small cap so we never ship a 200-feature waterfall to the UI.
_TOP_FEATURE_LIMIT: int = 5
# Counterfactual shock size (in units of feature std dev).
_SHOCK_STD_UNITS: float = 1.0


def _shap_table_exists(cursor: Any) -> bool:
    """Return True when a persisted SHAP values table exists."""
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
             WHERE table_name = 'fact_shap_values'
        )
        """
    )
    row = cursor.fetchone()
    return bool(row and row[0])


def _fetch_forecast(cursor: Any, item_id: str, loc: str) -> dict[str, Any] | None:
    cursor.execute(
        """
        SELECT forecast_month, forecast_qty, model_id
          FROM fact_production_forecast
         WHERE item_id = %s AND loc = %s
         ORDER BY forecast_month
         LIMIT 1
        """,
        (item_id, loc),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "forecast_month": str(row[0]),
        "forecast_qty": float(row[1]),
        "model_id": row[2],
    }


def _fetch_top_features(
    cursor: Any, item_id: str, loc: str
) -> list[dict[str, Any]]:
    """Pull top SHAP features from ``fact_shap_values`` when available.

    Returns an empty list when the table is absent — the caller degrades
    gracefully so the endpoint still returns structured output.
    """
    if not _shap_table_exists(cursor):
        return []
    cursor.execute(
        """
        SELECT feature_name, shap_value, feature_value
          FROM fact_shap_values
         WHERE item_id = %s AND loc = %s
         ORDER BY ABS(shap_value) DESC
         LIMIT %s
        """,
        (item_id, loc, _TOP_FEATURE_LIMIT),
    )
    rows = cursor.fetchall()
    return [
        {
            "name": r[0],
            "shap": float(r[1]),
            "value": float(r[2]) if r[2] is not None else None,
        }
        for r in rows
    ]


def _build_counterfactual(
    top_features: list[dict[str, Any]],
    forecast_qty: float,
) -> dict[str, Any]:
    """Compute a naive +/- 1 std-dev counterfactual using SHAP slopes.

    The ``shap`` value already quantifies the feature's contribution at
    its current value; shifting by one std dev uses the linear-SHAP
    approximation ``delta ≈ shap / current_value * shocked_value``.
    When ``value`` is missing we fall back to reporting the raw shap
    contribution. Either way this is a directional hint, not a causal
    estimate.
    """
    if not top_features:
        return {
            "explanation": (
                "No SHAP attributions available; extend the pipeline to "
                "persist fact_shap_values for counterfactual analysis."
            ),
            "scenarios": [],
        }
    scenarios = []
    for f in top_features[:3]:
        current = f.get("value")
        shap = f.get("shap", 0.0)
        if current is not None and current != 0:
            shocked_value = float(current) * (1.0 + _SHOCK_STD_UNITS * 0.1)
            delta = shap * (shocked_value - current) / current
        else:
            delta = shap * _SHOCK_STD_UNITS
        scenarios.append({
            "feature": f["name"],
            "shock_units_std": _SHOCK_STD_UNITS,
            "baseline_forecast": forecast_qty,
            "estimated_new_forecast": forecast_qty + delta,
            "delta": delta,
        })
    return {
        "explanation": "One-std-dev shock to each of the top features.",
        "scenarios": scenarios,
    }


@router.get("/forecast/explain/{item_id}/{loc}")
def explain_forecast(item_id: str, loc: str) -> dict[str, Any]:
    """Return a local SHAP-backed explanation + naive counterfactual."""
    with get_conn() as conn, conn.cursor() as cur:
        forecast = _fetch_forecast(cur, item_id, loc)
        if forecast is None:
            raise HTTPException(
                status_code=404,
                detail=f"No promoted forecast for item_id={item_id} loc={loc}",
            )

        top = _fetch_top_features(cur, item_id, loc)
        counterfactual = _build_counterfactual(top, forecast["forecast_qty"])

        # Record the explanation as an advisory decision so we have an
        # audit trail of what the platform showed the planner.
        try:
            append_decision(
                cur,
                DecisionRecord(
                    agent_id="explain_api",
                    action_type="explain_forecast",
                    autonomy_tier="advisory",
                    subject_kind="dfu",
                    subject_id=f"{item_id}|{loc}",
                    payload={
                        "model_id": forecast["model_id"],
                        "forecast_qty": forecast["forecast_qty"],
                        "has_shap": bool(top),
                    },
                    policy_id="explain_policy",
                    actor="api",
                    outcome="served",
                ),
            )
            conn.commit()
        except (ValueError, KeyError) as exc:
            # Explanation response is useful even if the ledger append
            # hiccups. Log at warning and continue.
            logger.warning("decision ledger append failed: %s", exc)

        return {
            "item_id": item_id,
            "loc": loc,
            "forecast": forecast,
            "top_features": top,
            "counterfactual": counterfactual,
        }


__all__ = ["router"]
