"""API tests for the AI FVA backtest router.

Scope (per agent contract):
  - Math passthrough only — assert the endpoint preserves the SQL-computed
    floats / Nones exactly, without truncating, int-casting, or treating 0.0
    as "missing".
  - Boundary cases on the /summary endpoint (None lift, exact-zero lift,
    sign-convention pinning).
  - /by-month returns BOTH baseline_wape_pct AND ai_wape_pct — the frontend
    computes lift on the fly, so the backend must surface both columns.

NOT covered here:
  - Walk-forward sampling / cost estimation (other agents)
  - LLM client / recommender (tests/unit/test_fva_recommender.py)
  - Frontend rendering / E2E (other agents)
"""
from __future__ import annotations

from collections import namedtuple
from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool

# psycopg's cur.description rows expose ``.name`` — the FVA router reads
# ``c.name for c in cur.description``. Tuples like ``("col",)`` don't have a
# ``.name``, so we use a tiny namedtuple instead.
_Col = namedtuple("_Col", ["name"])


def _cols(*names: str) -> list[_Col]:
    return [_Col(n) for n in names]


# ---------------------------------------------------------------------------
# /runs/{run_id}/summary — math passthrough
# ---------------------------------------------------------------------------

class TestSummaryEndpointMathPassthrough:
    """The endpoint MUST preserve numeric exactness from the MV.

    Specifically:
      - floats stay floats (no int-cast)
      - Nones stay Nones (no coercion to 0)
      - exact 0.0 stays 0.0 (NOT confused with "no data")
      - negative lift_pct flows through unchanged (SQL allows negatives)
    """

    @pytest.mark.asyncio
    async def test_floats_preserved_not_int_cast(self):
        pool, _, cursor = _make_pool()
        cursor.fetchone.return_value = (
            72.5,    # baseline_wape_pct
            70.25,   # ai_wape_pct  (lower accuracy than baseline)
            -2.25,   # lift_pct  (SQL: ai - baseline; negative => AI WORSE here)
            100,     # n_dfus
            60,      # n_winners
            30,      # n_losers
            10,      # n_ties
            60.0,    # win_rate_pct
        )

        with patch("api.core._get_pool", return_value=pool):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/ai-planner/fva-backtest/runs/abc-123/summary")

        assert resp.status_code == 200
        body = resp.json()
        assert body["baseline_wape_pct"] == 72.5
        assert body["ai_wape_pct"] == 70.25
        # Negative lift preserved exactly — not flipped to absolute value or 0.
        assert body["lift_pct"] == -2.25
        assert body["n_dfus"] == 100
        assert body["n_winners"] == 60
        assert body["n_losers"] == 30
        assert body["n_ties"] == 10
        assert body["win_rate_pct"] == 60.0
        # Types — make sure floats stayed floats, ints stayed ints.
        assert isinstance(body["baseline_wape_pct"], float)
        assert isinstance(body["lift_pct"], float)
        assert isinstance(body["n_dfus"], int)

    @pytest.mark.asyncio
    async def test_nones_preserved_not_coerced_to_zero(self):
        # MV returns NULL when |sum(actual)| = 0 (NULLIF). Endpoint must not
        # coerce that to 0.0 or "no summary" — the row exists, the metric is
        # legitimately undefined.
        pool, _, cursor = _make_pool()
        cursor.fetchone.return_value = (None, None, None, 50, 0, 0, 50, None)

        with patch("api.core._get_pool", return_value=pool):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/ai-planner/fva-backtest/runs/abc-123/summary")

        assert resp.status_code == 200
        body = resp.json()
        assert body["baseline_wape_pct"] is None
        assert body["ai_wape_pct"] is None
        assert body["lift_pct"] is None
        assert body["win_rate_pct"] is None
        # Counts still present.
        assert body["n_dfus"] == 50
        assert body["n_ties"] == 50

    @pytest.mark.asyncio
    async def test_exact_zero_lift_returned_not_treated_as_missing(self):
        # 0.0 is a legitimate value (perfect tie, or matching WAPEs). It must
        # NOT be returned as None.
        pool, _, cursor = _make_pool()
        cursor.fetchone.return_value = (80.0, 80.0, 0.0, 10, 0, 0, 10, 0.0)

        with patch("api.core._get_pool", return_value=pool):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/ai-planner/fva-backtest/runs/abc-123/summary")

        assert resp.status_code == 200
        body = resp.json()
        # Critical: 0.0 must NOT be falsy-collapsed to None.
        assert body["lift_pct"] == 0.0
        assert body["lift_pct"] is not None
        assert body["win_rate_pct"] == 0.0
        assert body["win_rate_pct"] is not None

    @pytest.mark.asyncio
    async def test_missing_run_returns_friendly_payload_not_404(self):
        # The summary endpoint chooses to return ``summary: None`` (with a
        # helpful message) rather than 404 when the MV doesn't have the row
        # yet (e.g. backfill still in progress). Pin that contract.
        pool, _, cursor = _make_pool()
        cursor.fetchone.return_value = None

        with patch("api.core._get_pool", return_value=pool):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/ai-planner/fva-backtest/runs/abc-123/summary")

        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == "abc-123"
        assert body["summary"] is None
        assert "message" in body

    @pytest.mark.asyncio
    async def test_positive_lift_pinned_to_sql_convention(self):
        """Sign-convention pin: in mv_ai_fva_overall, lift_pct > 0 means AI BETTER.

        WAPE in this codebase is *accuracy* (higher = better), so the SQL's
        ``lift = ai_wape - baseline_wape`` is positive when AI wins. This
        matches spec §14 *intent* but not the spec's literal formula text
        ("Lift = Baseline WAPE - AI WAPE") — the spec text has a sign bug.
        The report.html footer at api/routers/forecasting/ai_fva_backtest.py:412
        carries the same incorrect formula text. Endpoint passes the value
        through as-is; consumers must trust the SQL convention.
        """
        pool, _, cursor = _make_pool()
        # Hand-computed: baseline=80, ai=85 -> SQL lift = ai - baseline = +5
        cursor.fetchone.return_value = (80.0, 85.0, 5.0, 10, 7, 2, 1, 70.0)

        with patch("api.core._get_pool", return_value=pool):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/ai-planner/fva-backtest/runs/abc-123/summary")

        body = resp.json()
        # SQL: positive lift_pct => AI WAPE higher => AI is BETTER (accuracy-style WAPE).
        assert body["lift_pct"] == 5.0
        assert body["ai_wape_pct"] > body["baseline_wape_pct"]


# ---------------------------------------------------------------------------
# /runs/{run_id}/by-month — backend returns both WAPEs (FE computes lift)
# ---------------------------------------------------------------------------

class TestByMonthEndpoint:
    @pytest.mark.asyncio
    async def test_returns_both_baseline_and_ai_wape_per_month(self):
        # Frontend computes lift = baseline - ai (spec convention) on the fly,
        # so the backend MUST return both columns plus n_dfus per month.
        pool, _, cursor = _make_pool()
        cursor.fetchall.return_value = [
            ("2026-01-01", 70.0, 72.0, 500),
            ("2026-02-01", 65.0, 60.0, 510),
            ("2026-03-01", None, None, 0),  # zero-actuals month
        ]
        cursor.description = _cols(
            "forecast_run_month", "baseline_wape_pct", "ai_wape_pct", "n_dfus"
        )

        with patch("api.core._get_pool", return_value=pool):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/ai-planner/fva-backtest/runs/abc/by-month")

        assert resp.status_code == 200
        rows = resp.json()["rows"]
        assert len(rows) == 3
        # Every row must carry both metric columns so the FE can compute lift.
        for r in rows:
            assert "baseline_wape_pct" in r
            assert "ai_wape_pct" in r
            assert "n_dfus" in r
            assert "forecast_run_month" in r
        # Both values preserved exactly.
        assert rows[0]["baseline_wape_pct"] == 70.0
        assert rows[0]["ai_wape_pct"] == 72.0
        assert rows[1]["baseline_wape_pct"] == 65.0
        assert rows[1]["ai_wape_pct"] == 60.0
        # None pair preserved on the zero-actuals month.
        assert rows[2]["baseline_wape_pct"] is None
        assert rows[2]["ai_wape_pct"] is None

    @pytest.mark.asyncio
    async def test_empty_result_returns_empty_rows(self):
        pool, _, cursor = _make_pool()
        cursor.fetchall.return_value = []
        cursor.description = _cols(
            "forecast_run_month", "baseline_wape_pct", "ai_wape_pct", "n_dfus"
        )

        with patch("api.core._get_pool", return_value=pool):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/ai-planner/fva-backtest/runs/abc/by-month")

        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == "abc"
        assert body["rows"] == []


# ---------------------------------------------------------------------------
# /runs/{run_id}/by-recommendation — math passthrough on per-rec rollup
# ---------------------------------------------------------------------------

class TestByRecommendationEndpoint:
    @pytest.mark.asyncio
    async def test_preserves_floats_and_nulls_across_rec_codes(self):
        pool, _, cursor = _make_pool()
        cursor.fetchall.return_value = [
            # rec_code, baseline_wape, ai_wape, lift_pct, n_obs, avg_conf, avg_pct
            # SCALE_UP: AI WORSE (lower accuracy WAPE) -> negative lift in SQL.
            ("SCALE_UP", 70.0, 65.0, -5.0, 120, 0.82, 12.5),
            ("KEEP", 75.0, 75.0, 0.0, 200, 0.91, None),
            ("REPLACE", None, None, None, 0, None, None),
        ]
        cursor.description = _cols(
            "recommendation_code",
            "baseline_wape_pct",
            "ai_wape_pct",
            "lift_pct",
            "n_obs",
            "avg_confidence",
            "avg_pct_change",
        )

        with patch("api.core._get_pool", return_value=pool):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/ai-planner/fva-backtest/runs/abc/by-recommendation")

        assert resp.status_code == 200
        rows = resp.json()["rows"]
        assert len(rows) == 3
        scale_up = next(r for r in rows if r["recommendation_code"] == "SCALE_UP")
        keep = next(r for r in rows if r["recommendation_code"] == "KEEP")
        replace = next(r for r in rows if r["recommendation_code"] == "REPLACE")

        # SCALE_UP: AI worse on this code -> negative lift in SQL convention.
        assert scale_up["lift_pct"] == -5.0
        assert scale_up["avg_pct_change"] == 12.5
        # KEEP: exact zero lift NOT coerced to None.
        assert keep["lift_pct"] == 0.0
        assert keep["lift_pct"] is not None
        assert keep["avg_pct_change"] is None  # legitimately None for KEEP
        # REPLACE: all None metrics still surfaced.
        assert replace["baseline_wape_pct"] is None
        assert replace["ai_wape_pct"] is None
        assert replace["lift_pct"] is None
        assert replace["n_obs"] == 0


# ---------------------------------------------------------------------------
# /runs (list) — null cost / null sample counts preserved
# ---------------------------------------------------------------------------

class TestListRunsEndpoint:
    @pytest.mark.asyncio
    async def test_running_status_with_null_metrics(self):
        # A still-running backtest has null completed_at, null n_dfus_sampled,
        # null actual_cost_usd. The endpoint must surface those as null, not
        # default-fill them.
        pool, _, cursor = _make_pool()
        cursor.fetchall.return_value = [
            (
                "run-uuid-1", "running",
                "2026-05-18T10:00:00+0000", None,
                10, "2026-04-01", 3,
                "ollama", "qwen2.5:32b",
                None, None, None, None, None,
            ),
        ]
        cursor.description = _cols(
            "run_id", "status", "started_at", "completed_at",
            "window_months", "as_of_date", "horizon_months",
            "provider", "ai_model",
            "n_dfus_sampled", "n_recommendations",
            "estimated_cost_usd", "actual_cost_usd", "error_message",
        )

        with patch("api.core._get_pool", return_value=pool):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/ai-planner/fva-backtest/runs")

        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        run = body["runs"][0]
        assert run["status"] == "running"
        assert run["completed_at"] is None
        assert run["n_dfus_sampled"] is None
        assert run["actual_cost_usd"] is None


# ---------------------------------------------------------------------------
# POST /runs — 202 + background dispatch (no DB I/O, no LLM call)
# ---------------------------------------------------------------------------

class TestStartRunEndpoint:
    @pytest.mark.asyncio
    async def test_returns_202_and_launches_background_thread(self):
        # Math is irrelevant here — this test just pins that the endpoint
        # returns 202 + doesn't synchronously block on a long-running runner.
        pool, _, _ = _make_pool()

        with (
            patch("api.core._get_pool", return_value=pool),
            patch(
                "api.routers.forecasting.ai_fva_backtest._launch_backtest_thread"
            ) as mock_launch,
        ):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/ai-planner/fva-backtest/runs",
                    json={"window_months": 10, "limit_dfus": 100},
                    headers={"X-API-Key": "test-key"},
                )

        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "accepted"
        # Background launcher was invoked exactly once with the parsed request.
        assert mock_launch.call_count == 1
        called_req = mock_launch.call_args[0][0]
        assert called_req.window_months == 10
        assert called_req.limit_dfus == 100
