"""Contract tests for /ai-planner/fva-backtest/* router.

Scope (this file only): API contract, auth, request/response shape, error
paths. Walk-forward math, LLM client behaviour, and MV correctness are
covered by other test modules.

Spec: docs/specs/02-forecasting/27-ai-fva-backtest.md §6 (API Surface).
Follows tests/api/conftest.py `make_pool` + httpx.AsyncClient + ASGITransport
pattern (CLAUDE.md "Testing" rule).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RUN_ID = "11111111-2222-3333-4444-555555555555"
RUN_ID_2 = "22222222-3333-4444-5555-666666666666"
RUN_ID_3 = "33333333-4444-5555-6666-777777777777"


def _run_row(
    run_id: str = RUN_ID,
    status: str = "succeeded",
    started_at: str | None = "2026-05-01T10:00:00+00",
    completed_at: str | None = "2026-05-01T10:30:00+00",
    window_months: int = 12,
    as_of_date: str = "2026-05-01",
    horizon_months: int = 3,
    provider: str = "ollama",
    ai_model: str = "qwen2.5:7b",
    n_dfus_sampled: int | None = 200,
    n_recommendations: int | None = 180,
    estimated_cost_usd: float | None = 0.0,
    actual_cost_usd: float | None = 0.0,
    error_message: str | None = None,
) -> tuple:
    """Mirror the 14-column SELECT in list_runs / get_run."""
    return (
        run_id, status, started_at, completed_at,
        window_months, as_of_date, horizon_months,
        provider, ai_model,
        n_dfus_sampled, n_recommendations,
        estimated_cost_usd, actual_cost_usd, error_message,
    )


# ============================================================================
# POST /ai-planner/fva-backtest/runs  (write endpoint)
# ============================================================================

@pytest.mark.asyncio
async def test_post_runs_accepted_with_valid_body():
    """Valid body -> 202 accepted; background thread is launched."""
    pool, _, _ = _make_pool()
    fake_thread = MagicMock()
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.forecasting.ai_fva_backtest.threading.Thread",
               return_value=fake_thread) as thread_cls:
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/ai-planner/fva-backtest/runs",
                json={"window_months": 12, "horizon_months": 3, "provider": "ollama"},
            )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "accepted"
    assert "run_id" in body["message"] or "Poll" in body["message"]
    # Verify background launch was triggered exactly once.
    thread_cls.assert_called_once()
    fake_thread.start.assert_called_once()


@pytest.mark.asyncio
async def test_post_runs_accepts_empty_body():
    """All fields are optional — empty body should still be accepted (202)."""
    pool, _, _ = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.forecasting.ai_fva_backtest.threading.Thread",
               return_value=MagicMock()):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/ai-planner/fva-backtest/runs", json={})
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_post_runs_requires_api_key_when_configured(monkeypatch):
    """When API_KEY env is set, omitting X-API-Key returns 401/403."""
    monkeypatch.setenv("API_KEY", "secret-key")
    pool, _, _ = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.forecasting.ai_fva_backtest.threading.Thread",
               return_value=MagicMock()):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/ai-planner/fva-backtest/runs", json={})
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_post_runs_accepts_with_valid_api_key(monkeypatch):
    """When API_KEY is set and header matches, request is accepted."""
    monkeypatch.setenv("API_KEY", "secret-key")
    pool, _, _ = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.forecasting.ai_fva_backtest.threading.Thread",
               return_value=MagicMock()):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/ai-planner/fva-backtest/runs",
                json={},
                headers={"X-API-Key": "secret-key"},
            )
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_post_runs_extra_fields_rejected():
    """`extra='forbid'` on StartRunRequest -> unknown fields yield 422."""
    pool, _, _ = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.forecasting.ai_fva_backtest.threading.Thread",
               return_value=MagicMock()):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/ai-planner/fva-backtest/runs",
                json={"window_months": 12, "definitely_not_a_field": "x"},
            )
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("window_months", 0),     # below ge=1
        ("window_months", 37),    # above le=36
        ("horizon_months", 0),    # below ge=1
        ("horizon_months", 13),   # above le=12
        ("limit_dfus", 0),        # below ge=1
        ("limit_dfus", 50_001),   # above le=50_000
        ("notes", "x" * 1001),    # > max_length=1000
        ("provider", "totally-not-a-provider"),  # not in pattern
    ],
)
@pytest.mark.asyncio
async def test_post_runs_validation_rejects_out_of_range(field, value):
    pool, _, _ = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.forecasting.ai_fva_backtest.threading.Thread",
               return_value=MagicMock()):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/ai-planner/fva-backtest/runs",
                json={field: value},
            )
    assert resp.status_code == 422, f"{field}={value!r} should be 422 but got {resp.status_code}"


@pytest.mark.parametrize("provider", ["ollama", "anthropic", "openai", "openai_compat"])
@pytest.mark.asyncio
async def test_post_runs_all_supported_providers_accepted(provider):
    """The pattern enumerates exactly four providers — all four must pass."""
    pool, _, _ = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.forecasting.ai_fva_backtest.threading.Thread",
               return_value=MagicMock()):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/ai-planner/fva-backtest/runs",
                json={"provider": provider},
            )
    assert resp.status_code == 202


# ============================================================================
# GET /ai-planner/fva-backtest/runs  (list)
# ============================================================================

def _named_cols(*names: str) -> list:
    """Build mock cursor.description entries that expose ``.name`` (psycopg-style)."""
    out = []
    for n in names:
        m = MagicMock()
        m.name = n
        out.append(m)
    return out


@pytest.mark.asyncio
async def test_get_runs_empty_db():
    """No rows -> 200 with empty list and count=0."""
    pool, _, cursor = _make_pool(fetchall_return=[])
    cursor.description = _named_cols(
        "run_id", "status", "started_at", "completed_at",
        "window_months", "as_of_date", "horizon_months",
        "provider", "ai_model",
        "n_dfus_sampled", "n_recommendations",
        "estimated_cost_usd", "actual_cost_usd", "error_message",
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/ai-planner/fva-backtest/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"runs": [], "count": 0}


@pytest.mark.asyncio
async def test_get_runs_passes_through_rows_in_order():
    """Rows returned by the DB should appear unmodified in the response.

    The endpoint relies on SQL `ORDER BY started_at DESC`; the test asserts
    the response preserves whatever order the DB returns (no extra Python
    sort).
    """
    rows = [
        _run_row(run_id=RUN_ID,   started_at="2026-05-03T10:00:00+00"),
        _run_row(run_id=RUN_ID_2, started_at="2026-05-02T10:00:00+00"),
        _run_row(run_id=RUN_ID_3, started_at="2026-05-01T10:00:00+00"),
    ]
    pool, _, cursor = _make_pool(fetchall_return=rows)
    cursor.description = _named_cols(
        "run_id", "status", "started_at", "completed_at",
        "window_months", "as_of_date", "horizon_months",
        "provider", "ai_model",
        "n_dfus_sampled", "n_recommendations",
        "estimated_cost_usd", "actual_cost_usd", "error_message",
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/ai-planner/fva-backtest/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert [r["run_id"] for r in body["runs"]] == [RUN_ID, RUN_ID_2, RUN_ID_3]
    # Confirm SQL has ORDER BY started_at DESC.
    sql = cursor.execute.call_args.args[0]
    assert "ORDER BY started_at DESC" in sql


@pytest.mark.asyncio
async def test_get_runs_status_filter_passed_to_sql():
    """`?status=succeeded` -> SQL adds a WHERE clause and binds (status, limit).

    The list_runs endpoint builds the query conditionally to avoid the
    psycopg3 IndeterminateDatatype error that `%s IS NULL` triggers.
    """
    pool, _, cursor = _make_pool(fetchall_return=[])
    cursor.description = _named_cols("run_id")
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(
                "/ai-planner/fva-backtest/runs",
                params={"status": "succeeded", "limit": 10},
            )
    assert resp.status_code == 200
    sql, params = cursor.execute.call_args.args
    assert params == ("succeeded", 10)
    assert "WHERE status = %s" in sql


@pytest.mark.asyncio
async def test_get_runs_no_status_filter_omits_where_clause():
    """No `?status=` -> SQL omits the WHERE clause and binds (limit,) only."""
    pool, _, cursor = _make_pool(fetchall_return=[])
    cursor.description = _named_cols("run_id")
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/ai-planner/fva-backtest/runs", params={"limit": 5})
    assert resp.status_code == 200
    sql, params = cursor.execute.call_args.args
    assert params == (5,)
    assert "WHERE" not in sql


@pytest.mark.asyncio
async def test_get_runs_limit_max_500():
    """`limit=501` -> 422 (le=500 on Query)."""
    pool, _, _ = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/ai-planner/fva-backtest/runs", params={"limit": 501})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_get_runs_limit_at_boundary_accepted():
    """`limit=500` exactly is the max — should be accepted."""
    pool, _, cursor = _make_pool(fetchall_return=[])
    cursor.description = _named_cols("run_id")
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/ai-planner/fva-backtest/runs", params={"limit": 500})
    assert resp.status_code == 200


# ============================================================================
# GET /ai-planner/fva-backtest/runs/{run_id}
# ============================================================================

@pytest.mark.asyncio
async def test_get_run_invalid_uuid_returns_400():
    """Non-UUID path param -> 400 with explicit detail."""
    pool, _, _ = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/ai-planner/fva-backtest/runs/not-a-uuid")
    assert resp.status_code == 400
    assert "Invalid run_id" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_get_run_unknown_returns_404():
    """Valid UUID but no row -> 404."""
    pool, _, cursor = _make_pool()
    # make_pool's default ``fetchone_return=None`` is interpreted as
    # "use default" -> (0,). Force a real None explicitly.
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(f"/ai-planner/fva-backtest/runs/{RUN_ID}")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()
    # Cursor was hit — SQL parameter was the UUID string.
    assert cursor.execute.call_args.args[1] == (RUN_ID,)


@pytest.mark.asyncio
async def test_get_run_returns_run_summary_with_float_costs():
    """Valid run -> RunSummary; Decimal costs cast to float."""
    import decimal
    row = _run_row(
        estimated_cost_usd=decimal.Decimal("1.2345"),
        actual_cost_usd=decimal.Decimal("0.9876"),
    )
    pool, _, _ = _make_pool(fetchone_return=row)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(f"/ai-planner/fva-backtest/runs/{RUN_ID}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == RUN_ID
    assert body["status"] == "succeeded"
    assert body["window_months"] == 12
    assert body["horizon_months"] == 3
    # Floats, not Decimals — JSON serialisable.
    assert isinstance(body["estimated_cost_usd"], float)
    assert isinstance(body["actual_cost_usd"], float)
    assert body["estimated_cost_usd"] == pytest.approx(1.2345)
    assert body["actual_cost_usd"] == pytest.approx(0.9876)


@pytest.mark.asyncio
async def test_get_run_handles_null_costs():
    """NULL costs in DB -> None in response (no crash)."""
    row = _run_row(estimated_cost_usd=None, actual_cost_usd=None)
    pool, _, _ = _make_pool(fetchone_return=row)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(f"/ai-planner/fva-backtest/runs/{RUN_ID}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["estimated_cost_usd"] is None
    assert body["actual_cost_usd"] is None


# ============================================================================
# GET /ai-planner/fva-backtest/runs/{run_id}/summary
# ============================================================================

@pytest.mark.asyncio
async def test_summary_missing_mv_returns_graceful_payload():
    """When mv_ai_fva_overall has no row, return 200 with summary=None.

    Spec'd: no 404 — the run may exist but the MV may not be refreshed yet.
    """
    pool, _, cursor = _make_pool()
    cursor.fetchone.return_value = None  # explicit None, not the conftest default (0,)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(f"/ai-planner/fva-backtest/runs/{RUN_ID}/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == RUN_ID
    assert body["summary"] is None
    assert "message" in body and "FVA summary" in body["message"]


@pytest.mark.asyncio
async def test_summary_with_data_returns_floats():
    """Numeric columns are returned as floats (not Decimal)."""
    import decimal
    row = (
        decimal.Decimal("25.5"),  # baseline_wape_pct
        decimal.Decimal("18.2"),  # ai_wape_pct
        decimal.Decimal("7.3"),   # lift_pct
        200,                       # n_dfus
        120,                       # n_winners
        50,                        # n_losers
        30,                        # n_ties
        decimal.Decimal("60.0"),  # win_rate_pct
    )
    pool, _, _ = _make_pool(fetchone_return=row)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(f"/ai-planner/fva-backtest/runs/{RUN_ID}/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == RUN_ID
    assert body["baseline_wape_pct"] == pytest.approx(25.5)
    assert body["ai_wape_pct"] == pytest.approx(18.2)
    assert body["lift_pct"] == pytest.approx(7.3)
    assert body["n_dfus"] == 200
    assert body["win_rate_pct"] == pytest.approx(60.0)


# ============================================================================
# GET /by-recommendation, /by-month, /dfus
# ============================================================================

@pytest.mark.asyncio
async def test_by_recommendation_empty_returns_200_not_404():
    pool, _, cursor = _make_pool(fetchall_return=[])
    cursor.description = _named_cols(
        "recommendation_code", "baseline_wape_pct", "ai_wape_pct", "lift_pct",
        "n_obs", "avg_confidence", "avg_pct_change",
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(f"/ai-planner/fva-backtest/runs/{RUN_ID}/by-recommendation")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"run_id": RUN_ID, "rows": []}


@pytest.mark.asyncio
async def test_by_month_empty_returns_200_not_404():
    pool, _, cursor = _make_pool(fetchall_return=[])
    cursor.description = _named_cols(
        "forecast_run_month", "baseline_wape_pct", "ai_wape_pct", "n_dfus",
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(f"/ai-planner/fva-backtest/runs/{RUN_ID}/by-month")
    assert resp.status_code == 200
    assert resp.json() == {"run_id": RUN_ID, "rows": []}


@pytest.mark.asyncio
async def test_dfus_empty_returns_200_with_zero_count():
    pool, _, cursor = _make_pool(fetchall_return=[])
    cursor.description = _named_cols(
        "item_id", "loc", "sae_baseline", "sae_ai", "abs_error_reduction", "n_obs",
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(f"/ai-planner/fva-backtest/runs/{RUN_ID}/dfus")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"run_id": RUN_ID, "rows": [], "count": 0}


@pytest.mark.asyncio
async def test_dfus_sort_param_rejects_invalid_values():
    """`?sort=foo` -> 422 (pattern on Query)."""
    pool, _, _ = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(
                f"/ai-planner/fva-backtest/runs/{RUN_ID}/dfus",
                params={"sort": "alphabetical"},
            )
    assert resp.status_code == 422


@pytest.mark.parametrize("sort,expected_order_by", [
    ("error_reduction", "abs_error_reduction DESC"),
    ("item_id",        "item_id ASC, loc ASC"),
])
@pytest.mark.asyncio
async def test_dfus_sort_param_drives_order_by(sort, expected_order_by):
    """Verify the sort param toggles the ORDER BY (and only between two safe options)."""
    pool, _, cursor = _make_pool(fetchall_return=[])
    cursor.description = _named_cols(
        "item_id", "loc", "sae_baseline", "sae_ai", "abs_error_reduction", "n_obs",
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(
                f"/ai-planner/fva-backtest/runs/{RUN_ID}/dfus",
                params={"sort": sort},
            )
    assert resp.status_code == 200
    sql = cursor.execute.call_args.args[0]
    assert expected_order_by in sql


@pytest.mark.asyncio
async def test_dfus_limit_validation():
    """limit must be 1..2000."""
    pool, _, _ = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(
                f"/ai-planner/fva-backtest/runs/{RUN_ID}/dfus",
                params={"limit": 2001},
            )
    assert resp.status_code == 422


# ============================================================================
# GET /ai-planner/fva-backtest/runs/{run_id}/report.html
# ============================================================================

def _report_meta_row(run_id: str = RUN_ID) -> tuple:
    """13-column SELECT used in run_report_html (meta)."""
    return (
        run_id, "succeeded", "2026-05-01 10:00:00", "2026-05-01 10:30:00",
        12, "2026-05-01", 3,
        "ollama", "qwen2.5:7b",
        200, 180,
        0.0,
        "test notes",
    )


@pytest.mark.asyncio
async def test_report_html_invalid_uuid_returns_400():
    pool, _, _ = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/ai-planner/fva-backtest/runs/not-a-uuid/report.html")
    assert resp.status_code == 400
    assert "Invalid run_id" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_report_html_unknown_run_returns_404():
    """Meta fetchone returns None -> 404 before any subsequent fetch."""
    pool, _, cursor = _make_pool()
    # First fetchone (meta) is None -> 404 raised, no other queries run.
    cursor.fetchone.side_effect = [None]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(f"/ai-planner/fva-backtest/runs/{RUN_ID}/report.html")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_report_html_valid_run_returns_html_with_placeholders_when_mv_empty():
    """Valid run + no MV data -> still 200, text/html, placeholders rendered."""
    pool, _, cursor = _make_pool()
    # Sequence of cursor calls in run_report_html:
    #   fetchone -> meta row    (1)
    #   fetchone -> overall row (2) -> None (no MV data)
    #   fetchall -> by_rec      (1)
    #   fetchall -> by_month    (2)
    cursor.fetchone.side_effect = [_report_meta_row(), None]
    cursor.fetchall.side_effect = [[], []]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(f"/ai-planner/fva-backtest/runs/{RUN_ID}/report.html")
    assert resp.status_code == 200
    ct = resp.headers["content-type"]
    assert ct.startswith("text/html")
    html = resp.text
    # Run metadata is rendered.
    assert RUN_ID in html
    assert "ollama" in html
    assert "qwen2.5:7b" in html
    assert "2026-05-01 10:00:00" in html
    # The print button appears.
    assert "Print / Save as PDF" in html
    # Placeholder messages when MVs are empty.
    assert "No FVA summary yet" in html
    assert "No data yet." in html


@pytest.mark.asyncio
async def test_report_html_renders_overall_metrics_when_present():
    """Verify the headline result section contains computed values."""
    pool, _, cursor = _make_pool()
    overall_row = (25.5, 18.2, 7.3, 200, 120, 50, 30, 60.0)
    cursor.fetchone.side_effect = [_report_meta_row(), overall_row]
    cursor.fetchall.side_effect = [
        [("INCREASE_5_10", 30.0, 18.0, 12.0, 50, 0.85, 7.5)],
        [("2026-04", 22.0, 16.0, 100)],
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(f"/ai-planner/fva-backtest/runs/{RUN_ID}/report.html")
    assert resp.status_code == 200
    html = resp.text
    assert "Baseline WAPE" in html
    assert "AI WAPE" in html
    assert "INCREASE_5_10" in html  # recommendation code rendered
    assert "2026-04" in html         # month rendered
    # Positive lift class is rendered.
    assert "lift positive" in html


# ============================================================================
# GET /ai-planner/fva-backtest/runs/{run_id}/dfu-detail
# ============================================================================

@pytest.mark.asyncio
async def test_dfu_detail_happy_path_computes_wape_and_lift():
    """3 lag rows with actuals + 1 recommendation -> 200, correct WAPE math."""
    lag_rows = [
        # (forecast_run_month, target_month, lag, baseline_qty, ai_qty, actual_qty)
        ("2025-11-01", "2025-11-01", 1, 100.0,  80.0, 90.0),   # base err 10, ai err 10
        ("2025-11-01", "2025-12-01", 2, 100.0,  80.0, 70.0),   # base err 30, ai err 10
        ("2025-11-01", "2026-01-01", 3, 100.0,  80.0, 60.0),   # base err 40, ai err 20
    ]
    rec_rows = [
        ("2025-11-01", "SCALE_DOWN", -20.0, 0.85, "downward trend", ["recent_drop"]),
    ]
    # actuals sum = 220, sae_base = 80, sae_ai = 40
    # base_wape = 100 - 100*80/220  = 63.636...
    # ai_wape   = 100 - 100*40/220  = 81.818...
    # lift      = ai - base         = 18.18 pp
    summary_row = (3, 220.0, 80.0, 40.0)

    pool, _, _ = _make_pool(
        fetchall_returns=[lag_rows, rec_rows],
        fetchone_return=summary_row,
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(
                f"/ai-planner/fva-backtest/runs/{RUN_ID}/dfu-detail",
                params={"item_id": "916045", "loc": "1401-BULK"},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == RUN_ID
    assert body["item_id"] == "916045"
    assert body["loc"] == "1401-BULK"
    assert body["summary"]["n_obs"] == 3
    assert body["summary"]["baseline_wape_pct"] == pytest.approx(63.6363636, rel=1e-4)
    assert body["summary"]["ai_wape_pct"]       == pytest.approx(81.8181818, rel=1e-4)
    assert body["summary"]["lift_pp"]           == pytest.approx(18.1818182, rel=1e-4)
    assert len(body["lags"]) == 3
    assert body["lags"][0]["target_month"] == "2025-11-01"
    assert body["lags"][0]["lag"] == 1
    assert len(body["recommendations"]) == 1
    assert body["recommendations"][0]["recommendation_code"] == "SCALE_DOWN"
    assert body["recommendations"][0]["pct_change"] == -20.0
    assert body["recommendations"][0]["evidence_keys"] == ["recent_drop"]


@pytest.mark.asyncio
async def test_dfu_detail_zero_actuals_yields_none_wape_not_crash():
    """All actual_qty NULL -> abs_actual=0 -> WAPE None (avoid div-by-zero)."""
    lag_rows = [("2025-11-01", "2025-11-01", 1, 100.0, 80.0, None)]
    rec_rows = []
    summary_row = (0, 0.0, 0.0, 0.0)
    pool, _, _ = _make_pool(
        fetchall_returns=[lag_rows, rec_rows],
        fetchone_return=summary_row,
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(
                f"/ai-planner/fva-backtest/runs/{RUN_ID}/dfu-detail",
                params={"item_id": "X", "loc": "L1"},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["baseline_wape_pct"] is None
    assert body["summary"]["ai_wape_pct"] is None
    assert body["summary"]["lift_pp"] is None
    assert body["summary"]["n_obs"] == 0


@pytest.mark.asyncio
async def test_dfu_detail_unknown_dfu_returns_404():
    """No lag rows AND no rec rows -> 404."""
    pool, _, _ = _make_pool(
        fetchall_returns=[[], []],
        fetchone_return=(0, 0.0, 0.0, 0.0),
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(
                f"/ai-planner/fva-backtest/runs/{RUN_ID}/dfu-detail",
                params={"item_id": "nope", "loc": "nope"},
            )
    assert resp.status_code == 404
    assert "DFU not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_dfu_detail_invalid_run_id_returns_400():
    """Non-UUID run_id -> 400 from the same guard as get_run/report.html."""
    pool, _, _ = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(
                "/ai-planner/fva-backtest/runs/not-a-uuid/dfu-detail",
                params={"item_id": "X", "loc": "L1"},
            )
    assert resp.status_code == 400
    assert "Invalid run_id" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_dfu_detail_missing_query_params_return_422():
    """item_id and loc are required (min_length=1)."""
    pool, _, _ = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(f"/ai-planner/fva-backtest/runs/{RUN_ID}/dfu-detail")
    assert resp.status_code == 422
