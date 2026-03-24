"""Tests for tuning chat API endpoints."""

import json
import pytest
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport


@pytest.fixture
def mock_pool():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    mock_cursor.description = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.commit = MagicMock()
    mock_conn.close = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    pool = MagicMock()
    pool.connection.return_value = mock_conn
    return pool, mock_conn, mock_cursor


# ---------------------------------------------------------------------------
# POST /lgbm-tuning/chat/sessions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_session_returns_201(mock_pool):
    """POST /lgbm-tuning/chat/sessions creates a session."""
    pool, conn, cursor = mock_pool
    import uuid
    session_uuid = str(uuid.uuid4())
    # First fetchall for recent runs, then fetchone for INSERT RETURNING
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = (
        session_uuid, "Test Session", "active", "2026-03-23T10:00:00+00:00", "2026-03-23T10:00:00+00:00",
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/lgbm-tuning/chat/sessions",
                json={"title": "Test Session"},
            )
    assert resp.status_code == 201
    data = resp.json()
    assert "session" in data
    assert data["session"]["session_id"] == session_uuid
    assert data["session"]["title"] == "Test Session"
    assert data["session"]["status"] == "active"


# ---------------------------------------------------------------------------
# GET /lgbm-tuning/chat/sessions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_sessions_returns_200(mock_pool):
    """GET /lgbm-tuning/chat/sessions returns sessions list."""
    pool, _, cursor = mock_pool
    import uuid
    session_uuid = str(uuid.uuid4())
    cursor.fetchall.return_value = [
        (session_uuid, "My Session", "active", "2026-03-23T10:00:00+00:00", "2026-03-23T10:00:00+00:00", 5),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/lgbm-tuning/chat/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert "sessions" in data
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["title"] == "My Session"
    assert data["sessions"][0]["message_count"] == 5


# ---------------------------------------------------------------------------
# GET /lgbm-tuning/chat/sessions/{id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_session_returns_messages(mock_pool):
    """GET /lgbm-tuning/chat/sessions/{id} returns session and messages."""
    pool, _, cursor = mock_pool
    import uuid
    session_uuid = str(uuid.uuid4())

    # fetchone for session, fetchall for messages
    cursor.fetchone.return_value = (
        session_uuid, "Test Session", "active", None, "2026-03-23T10:00:00+00:00", "2026-03-23T10:00:00+00:00",
    )
    cursor.fetchall.return_value = [
        (1, session_uuid, "user", "Hello", "text", None, "2026-03-23T10:00:00+00:00"),
        (2, session_uuid, "assistant", "Hi there", "text", None, "2026-03-23T10:00:01+00:00"),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/lgbm-tuning/chat/sessions/{session_uuid}")
    assert resp.status_code == 200
    data = resp.json()
    assert "session" in data
    assert "messages" in data
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_get_session_404_missing(mock_pool):
    """GET /lgbm-tuning/chat/sessions/{id} returns 404 for missing session."""
    pool, _, cursor = mock_pool
    import uuid
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/lgbm-tuning/chat/sessions/{uuid.uuid4()}")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /lgbm-tuning/chat/sessions/{id}/messages
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_message_returns_ai_response(mock_pool):
    """POST .../messages returns user + AI response messages."""
    pool, conn, cursor = mock_pool
    import uuid
    session_uuid = str(uuid.uuid4())

    # Sequence: session check, user insert, messages fetch, AI messages insert, update timestamp
    call_count = [0]
    orig_fetchone = cursor.fetchone

    def fetchone_side_effect():
        call_count[0] += 1
        if call_count[0] == 1:
            # Session status check
            return ("active",)
        if call_count[0] == 2:
            # User message INSERT RETURNING
            return (
                1, session_uuid, "user", "What should I try?",
                "text", None, "2026-03-23T10:00:00+00:00",
            )
        if call_count[0] == 3:
            # AI message INSERT RETURNING
            return (
                2, session_uuid, "assistant", "Based on your runs...",
                "text", None, "2026-03-23T10:00:01+00:00",
            )
        return None

    cursor.fetchone.side_effect = fetchone_side_effect

    # Messages fetch (for AI context)
    cursor.fetchall.return_value = [
        (1, session_uuid, "user", "What should I try?", "text", None, "2026-03-23T10:00:00+00:00"),
    ]

    mock_advisor = MagicMock()
    mock_advisor.run_turn.return_value = ("Based on your runs...", [])

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "common.ai.tuning_advisor.TuningAdvisorAgent",
            return_value=mock_advisor,
        ),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/lgbm-tuning/chat/sessions/{session_uuid}/messages",
                json={"content": "What should I try?"},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert "messages" in data
    assert len(data["messages"]) >= 1


@pytest.mark.asyncio
async def test_send_message_empty_content_422(mock_pool):
    """POST .../messages with empty content returns 422."""
    pool, _, _ = mock_pool
    import uuid
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/lgbm-tuning/chat/sessions/{uuid.uuid4()}/messages",
                json={"content": ""},
            )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /lgbm-tuning/chat/sessions/{id}/confirm-run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_confirm_run_returns_run_id(mock_pool):
    """POST .../confirm-run triggers a backtest and returns run_id."""
    pool, conn, cursor = mock_pool
    import uuid
    session_uuid = str(uuid.uuid4())

    call_count = [0]

    def fetchone_side_effect():
        call_count[0] += 1
        if call_count[0] == 1:
            # Session check
            return ("active",)
        if call_count[0] == 2:
            # Recommendation message metadata
            rec = {"strategy_label": "test_exp", "overrides": {"learning_rate": 0.03}, "description": "Test"}
            return (json.dumps(rec),)
        if call_count[0] == 3:
            # Active run count
            return (0,)
        if call_count[0] == 4:
            # Message INSERT RETURNING
            return (
                10, session_uuid, "system", "Run #7 started",
                "run_started", None, "2026-03-23T10:00:00+00:00",
            )
        return None

    cursor.fetchone.side_effect = fetchone_side_effect

    mock_mgr = MagicMock()

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.ml.tuning_tracker.register_run", return_value=7),
        patch("common.job_registry.JobManager", return_value=mock_mgr),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/lgbm-tuning/chat/sessions/{session_uuid}/confirm-run",
                json={"recommendation_message_id": 5},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == 7
    assert data["status"] == "started"
    mock_mgr.submit_job.assert_called_once()
    call_kwargs = mock_mgr.submit_job.call_args
    assert call_kwargs[1]["job_type"] == "tuning_backtest" or call_kwargs[0][0] == "tuning_backtest"


@pytest.mark.asyncio
async def test_confirm_run_concurrent_409(mock_pool):
    """POST .../confirm-run returns 409 when a run is already active."""
    pool, conn, cursor = mock_pool
    import uuid
    session_uuid = str(uuid.uuid4())

    call_count = [0]

    def fetchone_side_effect():
        call_count[0] += 1
        if call_count[0] == 1:
            return ("active",)
        if call_count[0] == 2:
            rec = {"strategy_label": "test", "overrides": {"learning_rate": 0.03}, "description": "Test"}
            return (json.dumps(rec),)
        if call_count[0] == 3:
            return (1,)  # active run count = 1
        return None

    cursor.fetchone.side_effect = fetchone_side_effect

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/lgbm-tuning/chat/sessions/{session_uuid}/confirm-run",
                json={"recommendation_message_id": 5},
            )
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# GET /lgbm-tuning/chat/sessions/{id}/run-status/{run_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_status_running(mock_pool):
    """GET .../run-status/7 returns running status."""
    pool, _, cursor = mock_pool
    import uuid
    from datetime import datetime, timezone
    session_uuid = str(uuid.uuid4())

    started = datetime(2026, 3, 23, 10, 0, 0, tzinfo=timezone.utc)
    cursor.fetchone.return_value = (
        "running", started, None, None, None, None, None, None,
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/lgbm-tuning/chat/sessions/{session_uuid}/run-status/7",
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == 7
    assert data["status"] == "running"


@pytest.mark.asyncio
async def test_run_status_completed(mock_pool):
    """GET .../run-status/7 returns completed status with results."""
    pool, _, cursor = mock_pool
    import uuid
    from datetime import datetime, timezone
    session_uuid = str(uuid.uuid4())

    started = datetime(2026, 3, 23, 10, 0, 0, tzinfo=timezone.utc)
    completed = datetime(2026, 3, 23, 10, 30, 0, tzinfo=timezone.utc)
    cursor.fetchone.return_value = (
        "completed", started, completed, 71.79, 28.21, -0.012, 2725140, 50602,
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/lgbm-tuning/chat/sessions/{session_uuid}/run-status/7",
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["results"]["accuracy_pct"] == 71.79
    assert data["results"]["wape"] == 28.21
    assert data["elapsed_seconds"] == 1800


@pytest.mark.asyncio
async def test_run_status_404(mock_pool):
    """GET .../run-status/999 returns 404 for missing run."""
    pool, _, cursor = mock_pool
    import uuid
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/lgbm-tuning/chat/sessions/{uuid.uuid4()}/run-status/999",
            )
    assert resp.status_code == 404
