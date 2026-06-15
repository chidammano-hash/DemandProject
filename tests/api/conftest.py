"""Shared fixtures for API tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the global write rate-limiter between tests.

    The rate_limit_middleware (api/main.py) counts POST/PUT/DELETE per client_ip
    in a process-global singleton. Without this, cumulative write requests across
    the session eventually trip the 300/min limit and unrelated tests get 429s.
    """
    from common.services.rate_limiter import get_rate_limiter
    get_rate_limiter().reset()
    yield
    get_rate_limiter().reset()


def _make_async_cm(value):
    """Wrap a value in an awaitable async-context-manager that yields it.

    psycopg-pool's ``AsyncConnectionPool.connection()`` returns an async
    context manager — same shape for both pool.connection() and
    conn.cursor(). MagicMock can't fake that natively, so we wire up a tiny
    helper that exposes ``__aenter__``/``__aexit__`` returning ``value``.
    """
    cm = MagicMock()

    async def _aenter(_self):
        return value

    async def _aexit(_self, exc_type, exc, tb):
        return False

    cm.__aenter__ = _aenter
    cm.__aexit__ = _aexit
    return cm


def make_pool(
    fetchall_return=None,
    fetchone_return=None,
    *,
    fetchall_returns=None,
    fetchone_returns=None,
):
    """Shared factory for mock DB pool used across API tests.

    Args:
        fetchall_return: single return value for cursor.fetchall() calls. Defaults to [].
        fetchone_return: single return value for cursor.fetchone() calls. Defaults to (0,).
        fetchall_returns: optional list of per-call fetchall return values
            (wires up ``cursor.fetchall.side_effect``). Takes precedence over
            ``fetchall_return`` when set.
        fetchone_returns: optional list of per-call fetchone return values
            (wires up ``cursor.fetchone.side_effect``). Takes precedence over
            ``fetchone_return`` when set.

    Backwards compatible: callers passing only scalar ``fetchall_return`` /
    ``fetchone_return`` get identical behaviour to before the multi-call
    extension.
    """
    from unittest.mock import MagicMock
    cursor = MagicMock()

    if fetchall_returns is not None:
        # Sequential per-call fetchall; copy so callers can reuse the list.
        cursor.fetchall.side_effect = list(fetchall_returns)
    else:
        cursor.fetchall.return_value = fetchall_return if fetchall_return is not None else []

    if fetchone_returns is not None:
        cursor.fetchone.side_effect = list(fetchone_returns)
    else:
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


def make_async_pool(
    fetchall_return=None,
    fetchone_return=None,
    *,
    fetchall_returns=None,
    fetchone_returns=None,
):
    """Async sibling of :func:`make_pool`.

    Builds a mock pool whose ``connection()`` returns an async context
    manager yielding a connection whose ``cursor()`` is also an async
    context manager. ``cur.execute`` / ``cur.fetchone`` / ``cur.fetchall``
    are :class:`AsyncMock` so awaiting them works as expected.

    The (cursor, conn, pool) triple is returned in the same order as
    :func:`make_pool` so test code reads the same way.

    Behaviour:
        * Default ``fetchall`` -> ``[]``; default ``fetchone`` -> ``(0,)``
        * ``fetchall_returns`` / ``fetchone_returns`` set ``side_effect`` so
          per-call values are returned (mirrors the sync helper).
    """
    cursor = MagicMock()
    cursor.execute = AsyncMock(return_value=None)

    if fetchall_returns is not None:
        cursor.fetchall = AsyncMock(side_effect=list(fetchall_returns))
    else:
        cursor.fetchall = AsyncMock(
            return_value=fetchall_return if fetchall_return is not None else []
        )

    if fetchone_returns is not None:
        cursor.fetchone = AsyncMock(side_effect=list(fetchone_returns))
    else:
        cursor.fetchone = AsyncMock(
            return_value=fetchone_return if fetchone_return is not None else (0,)
        )

    cursor.description = [("col",)]
    cursor.rowcount = 1

    conn = MagicMock()
    conn.cursor = MagicMock(return_value=_make_async_cm(cursor))
    conn.commit = AsyncMock(return_value=None)
    # ``async with conn.transaction():`` opens a SAVEPOINT in real psycopg3.
    # Routers use this to isolate independent queries (e.g. the action-feed
    # multi-source feed) so one failing statement doesn't abort the others.
    # Mock it as a no-op async context manager so those code paths run.
    conn.transaction = MagicMock(return_value=_make_async_cm(None))

    pool = MagicMock()
    pool.connection = MagicMock(return_value=_make_async_cm(conn))
    return pool, conn, cursor


@pytest.fixture
def mock_pool():
    """Create a mock connection pool that returns mock cursors.

    Thin wrapper over :func:`make_pool` that keeps this fixture's historical
    defaults (``fetchone`` -> ``None`` and an empty ``description``), which a
    few of its 22 consumers rely on. The pool/conn/cursor plumbing is shared
    with ``make_pool`` so there is a single definition of it.
    """
    pool, conn, cursor = make_pool()
    # make_pool coerces a None fetchone default to (0,); this fixture's
    # contract is an explicit None with an empty column description.
    cursor.fetchone.return_value = None
    cursor.description = []
    return pool, conn, cursor


@pytest.fixture
async def client(mock_pool):
    """Async test client with mocked DB pool."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest_asyncio.fixture
async def async_client(mock_pool):
    """Shared async test client with mocked DB pool.

    Inject alongside ``mock_pool`` to configure cursor return values before
    making requests::

        async def test_something(mock_pool, async_client):
            _, _, cursor = mock_pool
            cursor.fetchall.return_value = [(...)]
            resp = await async_client.get("/some/endpoint")
    """
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
