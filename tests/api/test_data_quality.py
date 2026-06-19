"""API tests for data quality endpoints (Spec 08-01).

Tests all 5 data quality REST endpoints using httpx AsyncClient with
ASGITransport -- no running server needed.
"""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import patch, MagicMock

import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


_NOW = datetime.datetime(2026, 3, 1, 12, 0, 0)


# ---------------------------------------------------------------------------
# GET /data-quality/dashboard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dq_dashboard_200():
    # Row shape: (domain, passed, failed, warnings, skipped, info_fails,
    #             warning_fails, total). The single fail here is critical
    #             severity (info_fails=0, warning_fails=0).
    pool, conn, cursor = _make_pool(fetchall_return=[
        ("sales", 8, 1, 1, 0, 0, 0, 10),
        ("forecast", 5, 3, 2, 0, 0, 0, 10),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/dashboard")
    assert resp.status_code == 200
    data = resp.json()
    assert "domains" in data
    assert len(data["domains"]) == 2
    assert data["domains"][0]["domain"] == "sales"
    # No skips -> score is passed / (passed+failed+warnings) = 8/10 = 80.
    assert data["domains"][0]["score"] == 80.0
    assert data["domains"][0]["passed"] == 8
    assert data["domains"][0]["failed"] == 1
    assert data["domains"][0]["warnings"] == 1
    assert data["domains"][0]["skipped"] == 0


@pytest.mark.asyncio
async def test_dq_dashboard_excludes_skipped_from_score_denominator():
    """F7.1: a domain with all-passing scored checks reads 100% even when some
    checks were skipped. Skipped checks are surfaced explicitly and dropped from
    the score denominator so "10 pass / 0 fail / 0 warn" no longer reads 62.5%.
    The breakdown must reconcile: passed + failed + warnings + skipped == total."""
    # item: 10 pass / 0 fail / 0 warn / 6 skip / 0 info_fail / 0 warn_fail / 16 total.
    pool, conn, cursor = _make_pool(fetchall_return=[
        ("item", 10, 0, 0, 6, 0, 0, 16),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/dashboard")
    assert resp.status_code == 200
    d = resp.json()["domains"][0]
    assert d["domain"] == "item"
    assert d["score"] == 100.0  # 10 / (10+0+0), skip excluded from denominator
    assert d["passed"] == 10
    assert d["failed"] == 0
    assert d["warnings"] == 0
    assert d["skipped"] == 6
    assert d["total"] == 16
    # Reconciles end to end.
    assert d["passed"] + d["failed"] + d["warnings"] + d["skipped"] == d["total"]


@pytest.mark.asyncio
async def test_dq_dashboard_info_fails_excluded_from_score():
    """U8.3: an info-severity fail must not crater a domain to 0% alarm-red.

    `sku_to_item` has its only fail at INFO severity. It must score >= a domain
    whose only fail is critical/warning severity. Info fails are excluded from
    the score denominator (treated like skips) and surfaced as `info_fails`,
    while the raw `failed` count stays visible.

    Row shape: (domain, passed, failed, warnings, skipped, info_fails,
    warning_fails, total).
    """
    # Row shape: (domain, passed, failed, warnings, skipped, info_fails,
    #             warning_fails, total).
    pool, conn, cursor = _make_pool(fetchall_return=[
        # Only fail is info-severity: 0 passed, 2 failed, both info.
        ("sku_to_item", 0, 2, 0, 0, 2, 0, 2),
        # Only fail is a genuine critical-severity fail (not info, not warning).
        ("inventory", 10, 2, 0, 0, 0, 0, 12),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/dashboard")
    assert resp.status_code == 200
    domains = {d["domain"]: d for d in resp.json()["domains"]}

    info_domain = domains["sku_to_item"]
    real_fail_domain = domains["inventory"]

    # Info-only domain: 2 fails but all info -> excluded from the denominator so
    # it is NOT cratered to 0% alarm-red (U8.3). With 0 pass and nothing
    # scoreable it now reads a NEUTRAL null score (not a misleading green 100%,
    # U4.2) — the original intent (never red) is preserved.
    assert info_domain["score"] is None
    assert info_domain["info_fails"] == 2
    assert info_domain["failed"] == 2  # raw count still visible

    # A domain with a genuine (non-info) fail scores below 100.
    assert real_fail_domain["score"] < 100.0
    assert real_fail_domain["info_fails"] == 0

    # The info-only domain is neutral (null) rather than red — it never reads a
    # lower/redder score than the genuinely-failing domain.
    assert info_domain["score"] is None
    assert real_fail_domain["score"] is not None


@pytest.mark.asyncio
async def test_dq_dashboard_warning_fails_excluded_from_score():
    """F3.1: a domain whose only failures are WARNING severity must not crater to
    a red 0% badge that contradicts the Check Catalog's WARNING label for the same
    checks. Warning-severity fails are excluded from the score denominator (like
    info fails / skips), surfaced as `warning_fails`, and rolled into the amber
    warn bucket — never counted as a hard red critical fail.

    Row shape: (domain, passed, failed, warnings, skipped, info_fails,
    warning_fails, total).
    """
    pool, conn, cursor = _make_pool(fetchall_return=[
        # forecast_to_sku: only fail is warning-severity (0 pass, 2 warn-fail).
        ("forecast_to_sku", 0, 2, 0, 0, 0, 2, 2),
        # inventory: 8 warning-fails + 2 actual warns, plus 20 pass.
        ("inventory", 20, 8, 2, 0, 0, 8, 30),
        # a domain with a genuine CRITICAL fail (not warning, not info).
        ("hardfail", 1, 1, 0, 0, 0, 0, 2),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/dashboard")
    assert resp.status_code == 200
    domains = {d["domain"]: d for d in resp.json()["domains"]}

    warn_only = domains["forecast_to_sku"]
    # All failures are warnings -> excluded from the score denominator so the
    # domain is NOT cratered to a red 0% (F3.1). But with NOTHING scoreable and
    # 0 pass, it must NOT read a green 100% either (U4.2) — that would hide a
    # real warning-only integrity gap behind a "perfect" badge. Score is null
    # so the card renders a neutral "warn-only" state.
    assert warn_only["score"] is None
    assert warn_only["warning_fails"] == 2
    assert warn_only["failed"] == 2  # raw count still visible

    # inventory: warning-fails are excluded from the denominator like info/skip,
    # so they no longer crater the score; only the 2 genuine WARN-status rows
    # weight it. score = passed / (passed + critical_fails + warnings)
    # = 20 / (20 + 0 + 2) = 90.9 (was 66.7 when warning-fails were hard fails).
    inv = domains["inventory"]
    assert inv["score"] == 90.9
    assert inv["warning_fails"] == 8

    # A genuine critical fail still scores below 100 and counts as a hard fail.
    hard = domains["hardfail"]
    assert hard["score"] == 50.0  # 1 / (1 + 1)
    assert hard["warning_fails"] == 0


@pytest.mark.asyncio
async def test_dq_dashboard_warn_only_domain_is_neutral_not_green_100():
    """U4.2: a domain whose ONLY checks are failing warning/info checks (0 pass,
    nothing scoreable) must NOT show a green 100% — that would read identically
    to a domain where everything passed, hiding a real integrity gap. It returns
    a null score (neutral "warn-only") instead. A domain with genuine passes and
    no scored fails still earns a true green 100%.

    Row shape: (domain, passed, failed, warnings, skipped, info_fails,
    warning_fails, total).
    """
    pool, conn, cursor = _make_pool(fetchall_return=[
        # warn-only: 0 pass, both fails are warning-severity -> nothing scoreable.
        ("forecast_to_sku", 0, 2, 0, 0, 0, 2, 2),
        # info-only: 0 pass, both fails are info-severity -> nothing scoreable.
        ("sku_to_item", 0, 2, 0, 0, 2, 0, 2),
        # genuine perfect: 10 pass, 0 scored fails -> real green 100%.
        ("forecast", 10, 0, 0, 4, 0, 0, 14),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/dashboard")
    assert resp.status_code == 200
    domains = {d["domain"]: d for d in resp.json()["domains"]}

    # Warn-only and info-only domains: neutral (null), NOT green 100%.
    assert domains["forecast_to_sku"]["score"] is None
    assert domains["sku_to_item"]["score"] is None
    # A domain with real passes and no scored fails keeps its true 100%.
    assert domains["forecast"]["score"] == 100.0


@pytest.mark.asyncio
async def test_dq_dashboard_empty():
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/dashboard")
    assert resp.status_code == 200
    assert resp.json()["domains"] == []


# ---------------------------------------------------------------------------
# GET /data-quality/checks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dq_checks_200():
    pool, conn, cursor = _make_pool(fetchall_return=[
        (1, "freshness_sales", "freshness", "sales", "fact_sales_monthly",
         "critical", True, "pass", 1.0, _NOW),
        (2, "completeness_forecast", "completeness", "forecast",
         "fact_external_forecast_monthly", "warning", True, "warn", 0.95, _NOW),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/checks")
    assert resp.status_code == 200
    checks = resp.json()["checks"]
    assert len(checks) == 2
    assert checks[0]["check_name"] == "freshness_sales"
    assert checks[0]["check_type"] == "freshness"
    assert checks[0]["severity"] == "critical"
    assert checks[0]["enabled"] is True
    assert checks[0]["last_status"] == "pass"
    assert checks[0]["last_value"] == 1.0
    assert checks[0]["last_run"] is not None


@pytest.mark.asyncio
async def test_dq_checks_derives_from_results_when_catalog_empty():
    """F4.2/U4.1: when dim_dq_check_catalog is empty but fact_dq_check_results
    has rows, /checks must still return the distinct checks derived from the
    results table (not an empty list). The catalog dimension being unpopulated
    must NOT make the Check Catalog blank.

    The handler now runs a single results-derived query; the mock returns the
    same 10-column row shape the endpoint serializes.
    """
    pool, conn, cursor = _make_pool(fetchall_return=[
        (None, "freshness_sales", "freshness", "sales", "fact_sales_monthly",
         "critical", True, "pass", 1.0, _NOW),
        (None, "completeness_forecast", "completeness", "forecast",
         "fact_external_forecast_monthly", "warning", True, "warn", 0.95, _NOW),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/checks")
    assert resp.status_code == 200
    checks = resp.json()["checks"]
    # Must NOT be empty: the catalog dimension is empty but results exist.
    assert len(checks) == 2
    names = {c["check_name"] for c in checks}
    assert names == {"freshness_sales", "completeness_forecast"}
    # SQL existence must be driven by fact_dq_check_results, NOT by the
    # (empty) dim_dq_check_catalog. The catalog may only appear behind a LEFT
    # JOIN (enrichment), never as the FROM driver.
    executed_sql = " ".join(
        str(c.args[0]) for c in cursor.execute.call_args_list
    ).lower()
    assert "fact_dq_check_results" in executed_sql
    catalog_idx = executed_sql.find("dim_dq_check_catalog")
    if catalog_idx != -1:
        preceding = executed_sql[:catalog_idx]
        assert preceding.rstrip().endswith("left join"), (
            "catalog must be a LEFT JOIN enrichment, not the FROM driver"
        )


@pytest.mark.asyncio
async def test_dq_checks_null_metric():
    """Null metric_value and run_ts should be returned as None."""
    pool, conn, cursor = _make_pool(fetchall_return=[
        (1, "orphan_check", "referential", "sales", "fact_sales_monthly",
         "warning", False, None, None, None),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/checks")
    assert resp.status_code == 200
    check = resp.json()["checks"][0]
    assert check["last_value"] is None
    assert check["last_run"] is None


# ---------------------------------------------------------------------------
# GET /data-quality/history
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dq_history_200():
    pool, conn, cursor = _make_pool(fetchall_return=[
        (1, "freshness_sales", "sales", "fact_sales_monthly", "critical",
         "pass", 1.0, None, _NOW),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/history?days=7")
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    assert len(entries) == 1
    assert entries[0]["check_name"] == "freshness_sales"
    assert entries[0]["status"] == "pass"


@pytest.mark.asyncio
async def test_dq_history_with_domain_filter():
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/history?domain=sales&days=3&limit=10")
    assert resp.status_code == 200
    assert resp.json()["entries"] == []
    # Verify domain filter was included in the SQL params
    cursor.execute.assert_called_once()
    call_args = cursor.execute.call_args
    assert "sales" in call_args[0][1]  # domain param in the params list


@pytest.mark.asyncio
async def test_dq_history_invalid_days():
    """days < 1 should fail validation."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/history?days=0")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /data-quality/run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dq_run_200():
    """POST /data-quality/run triggers ad-hoc check run (manager+ role)."""
    pool, conn, cursor = _make_pool()
    mock_engine = MagicMock()
    mock_engine.run_all_checks.return_value = [
        {"check_name": "freshness_sales", "status": "pass"}
    ]
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.platform.data_quality.DQEngine", return_value=mock_engine, create=True), \
         patch.dict("sys.modules", {"common.engines.dq_engine": MagicMock(DQEngine=lambda: mock_engine)}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/data-quality/run")
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert data["total"] == 1
    assert data["triggered"] == 1
    assert data["message"] == "ok"


@pytest.mark.asyncio
async def test_dq_run_with_domain():
    """POST /data-quality/run?domain=sales passes domain to engine."""
    pool, conn, cursor = _make_pool()
    mock_engine = MagicMock()
    mock_engine.run_all_checks.return_value = []
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("sys.modules", {"common.engines.dq_engine": MagicMock(DQEngine=lambda: mock_engine)}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/data-quality/run?domain=sales")
    assert resp.status_code == 200
    mock_engine.run_all_checks.assert_called_once_with(domain="sales")


# ---------------------------------------------------------------------------
# GET /data-quality/fix/preview
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dq_fix_preview_200():
    """Preview returns indexed fix items."""
    pool, conn, cursor = _make_pool()
    mock_items = [
        {"id": 0, "fix_type": "range", "description": "Clamp t.col to [0, 100]",
         "affected_rows": 500, "recommendation": None, "status": "pending"},
        {"id": 1, "fix_type": "completeness", "description": "Impute t.col NULLs",
         "affected_rows": 100, "recommendation": None, "status": "pending"},
    ]
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("sys.modules", {"scripts.ops.fix_dq_issues": MagicMock(
             preview_all_fixes=MagicMock(return_value=mock_items),
             FIX_REGISTRY={"range": None, "completeness": None, "orphans": None},
         )}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/fix/preview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2
    assert data["items"][0]["id"] == 0
    assert data["items"][0]["fix_type"] == "range"
    assert data["items"][0]["affected_rows"] == 500


@pytest.mark.asyncio
async def test_dq_fix_preview_invalid_type():
    """Preview with unknown fix_type returns error."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("sys.modules", {"scripts.ops.fix_dq_issues": MagicMock(
             FIX_REGISTRY={"range": None, "completeness": None, "orphans": None},
         )}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/fix/preview?fix_type=bogus")
    assert resp.status_code == 200
    assert "error" in resp.json()


@pytest.mark.asyncio
async def test_dq_fix_preview_empty():
    """Preview with no fixable issues returns empty list."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("sys.modules", {"scripts.ops.fix_dq_issues": MagicMock(
             preview_all_fixes=MagicMock(return_value=[]),
             FIX_REGISTRY={"range": None},
         )}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/fix/preview")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0
    assert resp.json()["items"] == []


# ---------------------------------------------------------------------------
# POST /data-quality/fix/apply
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dq_fix_apply_200():
    """Apply selected fixes returns applied results."""
    pool, conn, cursor = _make_pool()
    mock_result = {
        "applied": [
            {"id": 0, "fix_type": "range", "description": "Clamp t.col",
             "affected_rows": 500, "recommendation": None, "status": "applied",
             "rows_fixed": 500},
        ],
        "skipped": [
            {"id": 1, "fix_type": "completeness", "description": "Impute",
             "affected_rows": 100, "recommendation": None, "status": "skipped"},
        ],
        "total_applied": 1,
        "total_skipped": 1,
        "total_rows_fixed": 500,
    }
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("sys.modules", {"scripts.ops.fix_dq_issues": MagicMock(
             apply_selected_fixes=MagicMock(return_value=mock_result),
         )}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/data-quality/fix/apply",
                                     json={"fix_ids": [0]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_applied"] == 1
    assert data["total_rows_fixed"] == 500
    assert len(data["applied"]) == 1
    assert data["applied"][0]["status"] == "applied"


@pytest.mark.asyncio
async def test_dq_fix_apply_empty_ids():
    """Apply with empty fix_ids returns error."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/data-quality/fix/apply",
                                     json={"fix_ids": []})
    assert resp.status_code == 200
    assert resp.json()["total_applied"] == 0
    assert "error" in resp.json()


@pytest.mark.asyncio
async def test_dq_fix_apply_missing_body():
    """Apply without body returns 422."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/data-quality/fix/apply")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Auth: write endpoints require X-API-Key when API_KEY is configured
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", [
    ("post", "/data-quality/run"),
    ("post", "/data-quality/fix/apply"),
    ("post", "/data-quality/fix"),
])
async def test_dq_write_endpoints_require_api_key(method, path, monkeypatch):
    """With API_KEY set, the mutating endpoints reject a request with no key."""
    monkeypatch.setenv("API_KEY", "secret-key")
    pool, _, _ = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await getattr(client, method)(path)
    assert resp.status_code in (401, 403)
