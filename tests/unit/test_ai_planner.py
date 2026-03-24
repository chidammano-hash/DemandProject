"""Unit tests for common/ai_planner.py — IPAIfeature1.

Tests tool functions and agent dispatch with mocked DB pool.
No real DB or LLM API calls are made.
"""
from __future__ import annotations

import json
import sys
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pool(fetchall_return=None, fetchone_return=None, description=None):
    """Return (pool, conn, cursor) triple with configurable mock returns."""
    cursor = MagicMock()
    cursor.fetchall.return_value = fetchall_return or []
    cursor.fetchone.return_value = fetchone_return
    cursor.description = description or []
    cursor.rowcount = 1

    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = conn
    return pool, conn, cursor


# ---------------------------------------------------------------------------
# Tool: get_dfu_full_context
# ---------------------------------------------------------------------------

def test_get_dfu_full_context_found():
    from common.ai_planner import get_dfu_full_context

    pool, conn, cursor = _make_pool(
        fetchone_return=("100320", "1401-BULK", "A", "L", "high_volume_steady", None, False,
                         42.0, 14, 1000.0, 33.0, 250.0, 3000.0, 500.0, 3.0, 8000.0,
                         "continuous_rop", "rop", 7, 0.95, True, 28.5),
        description=[
            ("item_id",), ("loc",), ("abc_vol",), ("variability_class",), ("cluster_assignment",),
            ("seasonality_profile",), ("is_yearly_seasonal",),
            ("current_dos",), ("total_lt_days",), ("avg_on_hand",), ("avg_daily_sales",),
            ("eoq_effective",), ("annual_demand",), ("total_annual_cost",),
            ("eoq_months_supply",), ("cycle_stock_value",),
            ("current_policy_id",), ("policy_type",), ("review_period_days",),
            ("service_level",), ("use_safety_stock",), ("champion_wape",),
        ],
    )

    result = get_dfu_full_context(pool, "100320", "1401-BULK")
    assert result["item_id"] == "100320"
    assert result["loc"] == "1401-BULK"
    assert result["abc_vol"] == "A"
    assert result["current_dos"] == 42.0


def test_get_dfu_full_context_not_found():
    from common.ai_planner import get_dfu_full_context

    pool, conn, cursor = _make_pool(fetchone_return=None)
    result = get_dfu_full_context(pool, "MISSING", "LOC")
    assert "error" in result


# ---------------------------------------------------------------------------
# Tool: get_forecast_performance
# ---------------------------------------------------------------------------

def test_get_forecast_performance_returns_list():
    from common.ai_planner import get_forecast_performance

    pool, conn, cursor = _make_pool(
        fetchall_return=[
            ("2026-01-01", 100.0, 90.0, 11.11, 11.11),
            ("2026-02-01", 120.0, 130.0, -7.69, 7.69),
        ],
        description=[
            ("startdate",), ("forecast_qty",), ("actual_qty",),
            ("bias_pct",), ("abs_err_pct",),
        ],
    )

    result = get_forecast_performance(pool, "100320", "1401-BULK", months=6)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["startdate"] == "2026-01-01"
    assert result[0]["bias_pct"] == 11.11


# ---------------------------------------------------------------------------
# Tool: get_portfolio_exceptions
# ---------------------------------------------------------------------------

def test_get_portfolio_exceptions_returns_list():
    from common.ai_planner import get_portfolio_exceptions

    pool, conn, cursor = _make_pool(
        fetchall_return=[
            ("100320", "1401-BULK", "A", "L", "high_volume_steady", 10.0, 14, 52.5, True, False, True),
        ],
        description=[
            ("item_id",), ("loc",), ("abc_vol",), ("variability_class",),
            ("cluster_assignment",), ("avg_dos",), ("total_lt_days",),
            ("champion_wape",), ("stockout_risk",), ("excess_flag",), ("high_wape_flag",),
        ],
    )

    result = get_portfolio_exceptions(pool, limit=10)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["stockout_risk"] is True


# ---------------------------------------------------------------------------
# Tool: compute_bias_trend
# ---------------------------------------------------------------------------

def test_compute_bias_trend_returns_dict():
    from common.ai_planner import compute_bias_trend

    pool, conn, cursor = _make_pool(
        fetchone_return=(25.0, 28.0, 4, 0),
        description=[
            ("bias_6m_avg",), ("bias_3m_avg",),
            ("over_forecast_months",), ("under_forecast_months",),
        ],
    )

    result = compute_bias_trend(pool, "100320", "1401-BULK")
    assert result["bias_6m_avg"] == 25.0
    assert result["over_forecast_months"] == 4


def test_compute_bias_trend_no_data():
    from common.ai_planner import compute_bias_trend

    pool, conn, cursor = _make_pool(fetchone_return=None)
    result = compute_bias_trend(pool, "100320", "1401-BULK")
    assert result == {}


# ---------------------------------------------------------------------------
# Tool: get_inventory_trend
# ---------------------------------------------------------------------------

def test_get_inventory_trend_returns_list():
    from common.ai_planner import get_inventory_trend

    pool, conn, cursor = _make_pool(
        fetchall_return=[
            ("2026-02-01", 42.0, 1000.0, 33.0, 14),
            ("2026-01-01", 38.0, 900.0, 33.0, 14),
        ],
        description=[
            ("month_start",), ("avg_dos",), ("avg_on_hand",),
            ("avg_daily_sales",), ("total_lt_days",),
        ],
    )

    result = get_inventory_trend(pool, "100320", "1401-BULK", months=6)
    assert len(result) == 2
    assert result[0]["avg_dos"] == 42.0


# ---------------------------------------------------------------------------
# Tool: get_eoq_context
# ---------------------------------------------------------------------------

def test_get_eoq_context_returns_dict():
    from common.ai_planner import get_eoq_context

    pool, conn, cursor = _make_pool(
        fetchone_return=(250, 250, 3000.0, 0.17, 125.0, 500.0, 3.0, 8000.0, "2026-03-01"),
        description=[
            ("eoq_qty",), ("eoq_effective",), ("annual_demand",),
            ("ordering_cost_per_unit",), ("holding_cost_annual",),
            ("total_annual_cost",), ("eoq_months_supply",), ("cycle_stock_value",),
            ("computed_at",),
        ],
    )

    result = get_eoq_context(pool, "100320", "1401-BULK")
    assert result["eoq_effective"] == 250
    assert result["eoq_months_supply"] == 3.0


def test_get_eoq_context_no_data():
    from common.ai_planner import get_eoq_context

    pool, conn, cursor = _make_pool(fetchone_return=None)
    result = get_eoq_context(pool, "100320", "1401-BULK")
    assert "error" in result


# ---------------------------------------------------------------------------
# Tool: check_stockout_history
# ---------------------------------------------------------------------------

def test_check_stockout_history_returns_dict():
    from common.ai_planner import check_stockout_history

    pool, conn, cursor = _make_pool(
        fetchone_return=(2, 0, 18.5, 6),
        description=[
            ("stockout_months",), ("excess_months",), ("avg_dos",), ("total_months",),
        ],
    )

    result = check_stockout_history(pool, "100320", "1401-BULK")
    assert result["stockout_months"] == 2
    assert result["total_months"] == 6


# ---------------------------------------------------------------------------
# Tool: get_portfolio_health_summary
# ---------------------------------------------------------------------------

def test_get_portfolio_health_summary_returns_dict():
    from common.ai_planner import get_portfolio_health_summary

    pool, conn, cursor = _make_pool(
        fetchone_return=(500, 42.0, 25, 10, 28.5, 7),
        description=[
            ("total_dfus",), ("avg_dos",), ("stockout_risk_count",),
            ("excess_count",), ("portfolio_champion_wape",), ("open_insights",),
        ],
    )

    result = get_portfolio_health_summary(pool)
    assert result["total_dfus"] == 500
    assert result["open_insights"] == 7


# ---------------------------------------------------------------------------
# Tool: create_insight
# ---------------------------------------------------------------------------

def test_create_insight_returns_id():
    from common.ai_planner import create_insight

    pool, conn, cursor = _make_pool(fetchone_return=(42,))

    result = create_insight(
        pool,
        insight_type="stockout_risk",
        severity="high",
        item_id="100320",
        loc="1401-BULK",
        summary="DOS 18d below lead time 21d — stockout risk within 3 days.",
        recommendation="Trigger emergency reorder of 250 units immediately and review policy.",
        reasoning="DOS 18 < LT 21",
        financial_impact_estimate=5000.0,
    )
    assert result == 42
    conn.commit.assert_called()


def test_create_insight_no_row_returns_minus1():
    from common.ai_planner import create_insight

    pool, conn, cursor = _make_pool(fetchone_return=None)

    result = create_insight(
        pool,
        insight_type="excess_inventory",
        severity="medium",
        item_id="999",
        loc="LOC",
        summary="DOS 247d is 6.5× peer average — excess stock accumulating.",
        recommendation="Suspend planned orders until DOS falls below 120d and review policy.",
    )
    assert result == -1


def test_create_insight_validation_rejects_bad_summary():
    """Summary without any digit fails Pydantic validation; returns -1."""
    from common.ai_planner import create_insight

    pool, conn, cursor = _make_pool(fetchone_return=(99,))

    result = create_insight(
        pool,
        insight_type="forecast_bias",
        severity="medium",
        item_id="100320",
        loc="LOC",
        summary="Persistent over-forecast detected this month",  # no digit → invalid
        recommendation="Apply a multiplier adjustment and review the champion model.",
    )
    assert result == -1


def test_create_insight_validation_rejects_bad_type():
    """Invalid insight_type fails Pydantic validation; returns -1."""
    from common.ai_planner import create_insight

    pool, conn, cursor = _make_pool(fetchone_return=(99,))

    result = create_insight(
        pool,
        insight_type="inventory_risk",  # not a valid Literal
        severity="high",
        item_id="100320",
        loc="LOC",
        summary="DOS 12d below LT 14d — stockout risk.",
        recommendation="Reorder 200 units immediately and switch policy.",
    )
    assert result == -1


# ---------------------------------------------------------------------------
# Helpers: configs and agent constructors
# ---------------------------------------------------------------------------

_DEFAULT_CFG = {"provider": "openai", "model": "gpt-4o", "max_tokens": 4096, "temperature": 0.2}
_ANTHROPIC_CFG = {"provider": "anthropic", "model": "claude-opus-4-6", "max_tokens": 4096, "temperature": 0.2}


def _make_openai_agent(pool):
    """Return an AIPlannerAgent using a mocked OpenAI client."""
    mock_oai = MagicMock()
    with patch.dict("sys.modules", {"openai": mock_oai}):
        from common.ai_planner import AIPlannerAgent
        agent = AIPlannerAgent(pool, _DEFAULT_CFG)
    return agent


def _make_anthropic_agent(pool):
    """Return an AIPlannerAgent using a mocked Anthropic client."""
    mock_ant = MagicMock()
    with patch.dict("sys.modules", {"anthropic": mock_ant}):
        from common.ai_planner import AIPlannerAgent
        agent = AIPlannerAgent(pool, _ANTHROPIC_CFG)
    return agent


# ---------------------------------------------------------------------------
# AIPlannerAgent: _dispatch_tool
# ---------------------------------------------------------------------------

def test_dispatch_tool_unknown():
    pool, _, _ = _make_pool()
    agent = _make_openai_agent(pool)
    result = agent._dispatch_tool("nonexistent_tool", {})
    assert "error" in result


def test_dispatch_tool_get_portfolio_health_summary():
    pool, conn, cursor = _make_pool(
        fetchone_return=(100, 35.0, 5, 2, 30.0, 3),
        description=[
            ("total_dfus",), ("avg_dos",), ("stockout_risk_count",),
            ("excess_count",), ("portfolio_champion_wape",), ("open_insights",),
        ],
    )
    agent = _make_openai_agent(pool)
    result = agent._dispatch_tool("get_portfolio_health_summary", {})
    assert isinstance(result, dict)
    assert "total_dfus" in result


def test_dispatch_tool_create_insight_injects_scan_run_id():
    pool, conn, cursor = _make_pool(fetchone_return=(99,))
    agent = _make_openai_agent(pool)

    captured = {}

    def fake_create_insight(p, **kwargs):
        captured.update(kwargs)
        return 99

    with patch("common.ai_planner.create_insight", side_effect=fake_create_insight):
        agent._dispatch_tool("create_insight", {
            "insight_type": "stockout_risk",
            "severity": "high",
            "item_id": "100320",
            "loc": "1401-BULK",
            "summary": "test",
            "recommendation": "test rec",
            "scan_run_id": "my-scan-id",
        })

    assert captured["insight_type"] == "stockout_risk"


# ---------------------------------------------------------------------------
# AIPlannerAgent: _run_openai_loop (mocked OpenAI)
# ---------------------------------------------------------------------------

def _make_oai_response(finish_reason, content=None, tool_calls=None):
    """Build a mock openai ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []

    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message = msg

    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.total_tokens = 150

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _make_oai_tool_call(name, arguments_dict, call_id="call_001"):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments_dict)
    return tc


def test_openai_loop_stop():
    """Loop returns text immediately on finish_reason=stop."""
    pool, _, _ = _make_pool()
    agent = _make_openai_agent(pool)

    mock_resp = _make_oai_response("stop", content="All good.")
    with patch.object(agent.client.chat.completions, "create", return_value=mock_resp):
        text, ids = agent._run_agentic_loop("Analyse DFU X", "scan-001")

    assert text == "All good."
    assert ids == []


def test_openai_loop_tool_call_then_stop():
    """Loop dispatches tool call, appends result, then finishes."""
    pool, conn, cursor = _make_pool(
        fetchone_return=("100320", "1401-BULK", "A", "L", "c", None, False,
                         42.0, 14, 1000.0, 33.0, 250.0, 3000.0, 500.0, 3.0,
                         8000.0, "rop", "rop", 7, 0.95, True, 28.5),
        description=[
            ("item_id",), ("loc",), ("abc_vol",), ("variability_class",),
            ("cluster_assignment",), ("seasonality_profile",), ("is_yearly_seasonal",),
            ("current_dos",), ("total_lt_days",), ("avg_on_hand",), ("avg_daily_sales",),
            ("eoq_effective",), ("annual_demand",), ("total_annual_cost",),
            ("eoq_months_supply",), ("cycle_stock_value",),
            ("current_policy_id",), ("policy_type",), ("review_period_days",),
            ("service_level",), ("use_safety_stock",), ("champion_wape",),
        ],
    )
    agent = _make_openai_agent(pool)

    tool_resp = _make_oai_response(
        "tool_calls",
        tool_calls=[_make_oai_tool_call("get_dfu_full_context", {"item_id": "100320", "loc": "1401-BULK"})],
    )
    final_resp = _make_oai_response("stop", content="Analysis done.")

    with patch.object(agent.client.chat.completions, "create", side_effect=[tool_resp, final_resp]):
        text, ids = agent._run_agentic_loop("Analyse DFU 100320@1401-BULK", "scan-001")

    assert text == "Analysis done."
    assert ids == []


# ---------------------------------------------------------------------------
# AIPlannerAgent: _run_anthropic_loop (mocked Anthropic)
# ---------------------------------------------------------------------------

def _make_anthropic_text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_anthropic_response(stop_reason, content):
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content
    return resp


def test_anthropic_loop_end_turn():
    """Anthropic loop returns final text on end_turn."""
    pool, _, _ = _make_pool()
    agent = _make_anthropic_agent(pool)

    mock_resp = _make_anthropic_response("end_turn", [_make_anthropic_text_block("Done.")])
    with patch.object(agent.client.messages, "create", return_value=mock_resp):
        text, ids = agent._run_agentic_loop("Analyse DFU X", "scan-001")

    assert text == "Done."
    assert ids == []
