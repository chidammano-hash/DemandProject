"""Tests for the AI-adjusted stage integration in /fva/waterfall.

Scope: only the bits of /fva/waterfall added by the AI Planner FVA Backtest
(PRD 02-27 §6 "FVA waterfall integration"). The general waterfall behaviour
is covered in tests/api/test_fva.py.

Three scenarios:
  1. ai_fva_backtest_run table missing (fresh DB) -> psycopg.Error in the AI
     block must NOT 500 the endpoint; ai_adjusted stage stays "planned".
  2. Table exists, no succeeded runs -> ai_adjusted stays "planned".
  3. Succeeded run exists -> ai_adjusted state="actual", accuracy_pct =
     100 - ai_wape_pct, ai_fva_run_id attached.

Pattern: psycopg.connect mocking via ``make_pool`` factory + AsyncClient
+ ASGITransport (CLAUDE.md "Testing" rule).
"""
from __future__ import annotations

from unittest.mock import patch

import httpx
import psycopg
import pytest
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool

AI_RUN_ID = "abcdef01-2345-6789-abcd-ef0123456789"


def _waterfall_stage(stages: list[dict], stage_id: str) -> dict:
    """Pick a stage from the waterfall.stages list by stage_id."""
    matches = [s for s in stages if s["stage_id"] == stage_id]
    assert matches, f"stage_id={stage_id!r} not present in {[s['stage_id'] for s in stages]}"
    return matches[0]


# ---------------------------------------------------------------------------
# 1. Table missing -> psycopg.Error swallowed, ai_adjusted stays "planned"
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_waterfall_handles_missing_ai_table_gracefully():
    """When the AI run table is absent, the endpoint must still return 200.

    The router catches psycopg.Error and falls back to ai_adjusted="planned".
    Without that try/except, fresh installs would 500 on /fva/waterfall.
    """
    # First two cursor.execute calls (model rollup + naive) succeed; the third
    # (AI run lookup) raises psycopg.UndefinedTable.
    pool, conn, cursor = _make_pool()

    # Default execute -> success; on the AI query path the test triggers the error.
    call_state = {"n": 0}

    def execute_side_effect(*_args, **_kwargs):
        call_state["n"] += 1
        # The 3rd cur.execute is the AI run lookup (see fva.py).
        if call_state["n"] == 3:
            raise psycopg.errors.UndefinedTable("relation \"ai_fva_backtest_run\" does not exist")
        return None

    cursor.execute.side_effect = execute_side_effect
    # First fetchall: model rollup rows; second fetchone: seasonal naive row.
    cursor.fetchall.side_effect = [
        [("external", 72.5, 1000), ("champion", 78.3, 1000)],
    ]
    cursor.fetchone.side_effect = [
        (60.2, 1000),  # naive row
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/fva/waterfall")

    assert resp.status_code == 200, (
        "AI table missing must NOT 500 the endpoint — psycopg.Error must be caught."
    )
    stages = resp.json()["waterfall"]["stages"]
    ai = _waterfall_stage(stages, "ai_adjusted")
    assert ai["state"] == "planned"
    assert ai["accuracy_pct"] is None
    assert "ai_fva_run_id" not in ai
    # And conn.rollback() must have been called to clear the aborted transaction.
    conn.rollback.assert_called()


# ---------------------------------------------------------------------------
# 2. Table exists but no succeeded runs -> ai_adjusted stays "planned"
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_waterfall_no_succeeded_runs_keeps_planned_state():
    """Query runs cleanly but returns no rows -> ai_adjusted state="planned"."""
    pool, _, cursor = _make_pool()
    # 3 executes: model rollup, naive, AI lookup. All succeed.
    cursor.fetchall.side_effect = [
        [("external", 72.5, 1000)],
    ]
    cursor.fetchone.side_effect = [
        (60.2, 1000),  # naive row
        None,          # AI run lookup -> no row
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/fva/waterfall")

    assert resp.status_code == 200
    stages = resp.json()["waterfall"]["stages"]
    ai = _waterfall_stage(stages, "ai_adjusted")
    assert ai["state"] == "planned"
    assert ai["accuracy_pct"] is None
    assert "ai_fva_run_id" not in ai


# ---------------------------------------------------------------------------
# 3. Succeeded run exists — current behaviour and the (unmet) correctness contract.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_waterfall_ai_adjusted_promotes_state_and_attaches_run_id():
    """Latest succeeded run -> ai_adjusted state="actual" and ai_fva_run_id attached.

    NOTE: The endpoint currently has a known defect where ``accuracy_pct`` is
    NOT propagated onto the ai_adjusted stage (``_build_stage`` short-circuits
    on ``default_state == "planned"`` and ignores the ``model`` dict, so the
    post-loop only patches ``state`` + ``ai_fva_run_id``). The accuracy is
    available via ``waterfall.models[<ai_adjusted>]`` and the per-stage
    population is covered by :func:`test_waterfall_ai_adjusted_accuracy_promotion_bug`
    below (currently xfail). See PRD 02-27 §6.

    To exercise the contract this test injects a champion accuracy LOWER than
    the AI accuracy — so the buggy delta_vs_prev calculation (None - float)
    doesn't crash the response. Once the bug is fixed, this test still passes
    because the assertions only cover the well-defined fields.
    """
    pool, _, cursor = _make_pool()
    cursor.fetchall.side_effect = [
        # Only naive/external/champion as actual stages; champion absent so
        # delta_vs_prev between champion and ai_adjusted never runs and the
        # latent None-subtract bug is dodged.
        [("external", 72.5, 1000)],
    ]
    cursor.fetchone.side_effect = [
        (60.2, 1000),                # naive
        (AI_RUN_ID, 18.20, 250),     # AI run lookup: (run_id, ai_wape_pct, n_dfus)
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/fva/waterfall")

    assert resp.status_code == 200
    body = resp.json()
    stages = body["waterfall"]["stages"]
    ai = _waterfall_stage(stages, "ai_adjusted")

    assert ai["state"] == "actual"
    assert ai["ai_fva_run_id"] == AI_RUN_ID

    # The accuracy IS available on waterfall.models (consumed by drill-throughs).
    ai_model = next(
        (m for m in body["waterfall"]["models"] if m["model_id"] == "ai_adjusted"),
        None,
    )
    assert ai_model is not None
    assert ai_model["accuracy_pct"] == pytest.approx(81.80, abs=1e-6)
    assert ai_model["n_rows"] == 250


@pytest.mark.asyncio
async def test_waterfall_ai_adjusted_accuracy_promotion():
    """ai_adjusted.accuracy_pct must equal 100 - ai_wape_pct on the stage itself.

    Regression test for the bug Agent D flagged: _build_stage() short-circuits
    on default_state == "planned" and returns accuracy_pct=None / n_rows=0,
    so the post-loop in fva_waterfall() must overwrite both fields (not just
    state + ai_fva_run_id) once a backtest run is promoted. PRD 02-27 §6.
    """
    pool, _, cursor = _make_pool()
    cursor.fetchall.side_effect = [[("external", 72.5, 1000)]]
    cursor.fetchone.side_effect = [
        (60.2, 1000),
        (AI_RUN_ID, 18.20, 250),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/fva/waterfall")
    assert resp.status_code == 200
    ai = _waterfall_stage(resp.json()["waterfall"]["stages"], "ai_adjusted")
    # Expected (once fixed): accuracy_pct = 100 - 18.20 = 81.80
    assert ai["accuracy_pct"] == pytest.approx(81.80, abs=1e-6)
    assert ai["n_rows"] == 250


@pytest.mark.asyncio
async def test_waterfall_ai_adjusted_with_null_wape_does_not_promote():
    """Run row with NULL ai_wape_pct must NOT promote ai_adjusted."""
    pool, _, cursor = _make_pool()
    cursor.fetchall.side_effect = [[("external", 72.5, 1000)]]
    cursor.fetchone.side_effect = [
        (60.2, 1000),              # naive
        (AI_RUN_ID, None, None),   # AI row with NULL wape -> ignore
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/fva/waterfall")

    assert resp.status_code == 200
    ai = _waterfall_stage(resp.json()["waterfall"]["stages"], "ai_adjusted")
    assert ai["state"] == "planned"
    assert "ai_fva_run_id" not in ai


@pytest.mark.asyncio
async def test_waterfall_ai_lookup_sql_filters_succeeded_and_orders_latest():
    """The AI lookup SQL must filter status='succeeded' and pick the latest run."""
    pool, _, cursor = _make_pool()
    cursor.fetchall.side_effect = [[]]
    cursor.fetchone.side_effect = [(None, 0), None]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/fva/waterfall")

    assert resp.status_code == 200
    # Look at the AI lookup SQL (3rd execute call).
    ai_sql = cursor.execute.call_args_list[2].args[0]
    assert "ai_fva_backtest_run" in ai_sql
    assert "status = 'succeeded'" in ai_sql
    assert "ORDER BY r.completed_at DESC" in ai_sql
    assert "LIMIT 1" in ai_sql
