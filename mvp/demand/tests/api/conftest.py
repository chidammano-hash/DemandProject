"""Shared fixtures for API tests."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from contextlib import contextmanager

import httpx
from httpx import ASGITransport


def make_pool(fetchall_return=None, fetchone_return=None):
    """Shared factory for mock DB pool used across API tests.

    Args:
        fetchall_return: return value for cursor.fetchall() calls. Defaults to [].
        fetchone_return: return value for cursor.fetchone() calls. Defaults to (0,).
    """
    from unittest.mock import MagicMock
    cursor = MagicMock()
    cursor.fetchall.return_value = fetchall_return if fetchall_return is not None else []
    cursor.fetchone.return_value = fetchone_return if fetchone_return is not None else (0,)
    cursor.description = [("col",)]
    cursor.rowcount = 1

    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = conn
    return pool, conn, cursor


@pytest.fixture
def mock_pool():
    """Create a mock connection pool that returns mock cursors."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    mock_cursor.description = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = mock_conn

    return pool, mock_conn, mock_cursor


@pytest.fixture
async def client(mock_pool):
    """Async test client with mocked DB pool."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
