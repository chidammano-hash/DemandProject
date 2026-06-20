"""Tests for SQL Runner API — /sql-runner endpoints."""

import pytest
from unittest.mock import patch
import httpx
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


# ===========================================================================
# POST /sql-runner/execute — valid SELECT
# ===========================================================================

@pytest.mark.asyncio
async def test_execute_valid_select():
    """Valid SELECT returns columns and rows."""
    pool, conn, cursor = _make_pool()
    cursor.description = [("id",), ("name",)]
    cursor.fetchmany.return_value = [(1, "alpha"), (2, "beta")]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sql-runner/execute",
                json={"sql": "SELECT id, name FROM dim_item LIMIT 2"},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["columns"] == ["id", "name"]
    assert len(data["rows"]) == 2
    assert data["row_count"] == 2
    assert "elapsed_ms" in data


@pytest.mark.asyncio
async def test_execute_returns_truncated_flag():
    """When rows exceed max, truncated is True."""
    pool, conn, cursor = _make_pool()
    cursor.description = [("n",)]
    # Return max_rows + 1 to trigger truncation (default max is 1000)
    cursor.fetchmany.return_value = [(i,) for i in range(1002)]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sql-runner/execute",
                json={"sql": "SELECT generate_series(1,2000)", "max_rows": 1000},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["truncated"] is True
    assert data["row_count"] <= 1000


# ===========================================================================
# POST /sql-runner/execute — blocked write statements
# ===========================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO dim_item (sk) VALUES (1)",
        "UPDATE dim_item SET sk = 2",
        "DELETE FROM dim_item WHERE sk = 1",
        "DROP TABLE dim_item",
        "ALTER TABLE dim_item ADD COLUMN x INT",
        "CREATE TABLE evil (id INT)",
        "TRUNCATE dim_item",
        "GRANT ALL ON dim_item TO public",
        "REVOKE ALL ON dim_item FROM public",
    ],
)
async def test_execute_blocks_write_statements(sql):
    """DML/DDL statements are rejected with 400."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/sql-runner/execute", json={"sql": sql})
    assert resp.status_code == 400
    assert "blocked" in resp.json()["detail"].lower() or "Write" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_execute_empty_sql_rejected():
    """Empty SQL is rejected with 400/422."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/sql-runner/execute", json={"sql": ""})
    assert resp.status_code in (400, 422)


@pytest.mark.asyncio
async def test_execute_allows_select_with_keywords_in_strings():
    """SELECT with keywords inside string literals should not be blocked."""
    pool, conn, cursor = _make_pool()
    cursor.description = [("msg",)]
    cursor.fetchmany.return_value = [("DELETE is not allowed",)]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sql-runner/execute",
                json={"sql": "SELECT 'DELETE is not allowed' AS msg"},
            )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_execute_no_result_description():
    """Query returning no description (non-SELECT) raises 400."""
    pool, conn, cursor = _make_pool()
    cursor.description = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sql-runner/execute",
                json={"sql": "SELECT 1"},
            )
    assert resp.status_code == 400
    assert "did not return results" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_execute_db_error_returns_400():
    """Database errors surface as 400 with message."""
    pool, conn, cursor = _make_pool()
    # First two executes are SET statements, third is the actual query
    cursor.execute.side_effect = [None, None, Exception("relation does not exist")]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sql-runner/execute",
                json={"sql": "SELECT * FROM nonexistent_table"},
            )
    assert resp.status_code == 400
    assert "does not exist" in resp.json()["detail"]


# ===========================================================================
# GET /sql-runner/schema
# ===========================================================================

@pytest.mark.asyncio
async def test_schema_returns_tables():
    """GET /sql-runner/schema returns table list with columns."""
    pool, conn, cursor = _make_pool(fetchall_return=[
        ("public", "dim_item", "BASE TABLE", "sk", "integer", "NO"),
        ("public", "dim_item", "BASE TABLE", "ck", "text", "YES"),
        ("public", "dim_location", "BASE TABLE", "sk", "integer", "NO"),
    ])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sql-runner/schema")
    assert resp.status_code == 200
    data = resp.json()
    assert "tables" in data
    assert len(data["tables"]) == 2
    item_table = next(t for t in data["tables"] if t["table_name"] == "dim_item")
    assert len(item_table["columns"]) == 2
    assert item_table["columns"][0]["name"] == "sk"


# ===========================================================================
# Hard cap clamp — max_rows cannot exceed HARD_CAP
# ===========================================================================

@pytest.mark.asyncio
async def test_execute_hard_cap_rejects_oversized_request():
    """Requesting more rows than the hard cap (5000) is rejected by Pydantic validation."""
    pool, conn, cursor = _make_pool()
    cursor.description = [("n",)]
    cursor.fetchmany.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sql-runner/execute",
                json={"sql": "SELECT 1", "max_rows": 99_999},
            )
    # Pydantic le=5000 rejects values above the hard cap with 422
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_execute_hard_cap_clamps_default_via_config():
    """When max_rows is omitted the config default (1000) is clamped to at most HARD_CAP."""
    pool, conn, cursor = _make_pool()
    cursor.description = [("n",)]
    # Return more rows than the config default to verify clamp works at fetchmany level
    cursor.fetchmany.return_value = [(i,) for i in range(1001)]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sql-runner/execute",
                json={"sql": "SELECT generate_series(1, 2000)"},
            )
    assert resp.status_code == 200
    data = resp.json()
    # Should be capped at config default (1000), never exceed HARD_CAP
    assert data["row_count"] <= 5000
    assert data["truncated"] is True


@pytest.mark.asyncio
async def test_execute_hard_cap_allows_at_boundary():
    """Requesting exactly HARD_CAP rows is accepted."""
    pool, conn, cursor = _make_pool()
    cursor.description = [("n",)]
    cursor.fetchmany.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sql-runner/execute",
                json={"sql": "SELECT 1", "max_rows": 5000},
            )
    assert resp.status_code == 200


# ===========================================================================
# GET /sql-runner/history
# ===========================================================================

@pytest.mark.asyncio
async def test_history_returns_list():
    """GET /sql-runner/history returns history array."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sql-runner/history")
    assert resp.status_code == 200
    data = resp.json()
    assert "history" in data
    assert isinstance(data["history"], list)
