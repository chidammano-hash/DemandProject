"""Tests for chat endpoint and helper functions."""

import pytest
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport



class TestIsSafeSql:
    """Test the SQL safety checker that blocks non-SELECT statements."""

    def test_allows_select(self):
        from api.routers.chat import _is_safe_sql
        assert _is_safe_sql("SELECT * FROM dim_item LIMIT 10") is True

    def test_allows_select_with_subquery(self):
        from api.routers.chat import _is_safe_sql
        assert _is_safe_sql("SELECT * FROM (SELECT item_no FROM dim_item) t LIMIT 10") is True

    def test_blocks_insert(self):
        from api.routers.chat import _is_safe_sql
        assert _is_safe_sql("INSERT INTO dim_item VALUES ('foo')") is False

    def test_blocks_update(self):
        from api.routers.chat import _is_safe_sql
        assert _is_safe_sql("UPDATE dim_item SET item_desc = 'x'") is False

    def test_blocks_delete(self):
        from api.routers.chat import _is_safe_sql
        assert _is_safe_sql("DELETE FROM dim_item") is False

    def test_blocks_drop(self):
        from api.routers.chat import _is_safe_sql
        assert _is_safe_sql("DROP TABLE dim_item") is False

    def test_blocks_alter(self):
        from api.routers.chat import _is_safe_sql
        assert _is_safe_sql("ALTER TABLE dim_item ADD COLUMN x TEXT") is False

    def test_blocks_truncate(self):
        from api.routers.chat import _is_safe_sql
        assert _is_safe_sql("TRUNCATE dim_item") is False

    def test_blocks_select_then_drop(self):
        from api.routers.chat import _is_safe_sql
        assert _is_safe_sql("SELECT 1; DROP TABLE dim_item") is False

    def test_strips_comments(self):
        from api.routers.chat import _is_safe_sql
        assert _is_safe_sql("-- delete comment\nSELECT 1") is True

    def test_strips_block_comments(self):
        from api.routers.chat import _is_safe_sql
        assert _is_safe_sql("/* DROP TABLE */ SELECT 1") is True


class TestBuildSchemaSummary:
    """Test the schema summary builder."""

    def test_returns_string(self):
        from api.routers.chat import _build_schema_summary
        result = _build_schema_summary()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_tables(self):
        from api.routers.chat import _build_schema_summary
        result = _build_schema_summary()
        assert "dim_item" in result
        assert "fact_sales_monthly" in result


@pytest.mark.asyncio
async def test_chat_empty_question(mock_pool):
    """POST /chat with empty question should return 422."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/chat", json={"question": ""})
            assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_success(mock_pool):
    """POST /chat with mocked OpenAI should return 200 with answer."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("context chunk 1",)]
    cursor.description = [("source_text",)]

    mock_openai = MagicMock()
    # Mock embedding
    mock_embed_resp = MagicMock()
    mock_embed_resp.data = [MagicMock(embedding=[0.1] * 1536)]
    mock_openai.embeddings.create.return_value = mock_embed_resp
    # Mock chat completion
    mock_chat_resp = MagicMock()
    mock_chat_resp.choices = [MagicMock(message=MagicMock(content='{"answer": "Test answer", "sql": null}'))]
    mock_openai.chat.completions.create.return_value = mock_chat_resp

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("api.routers.chat.get_openai", return_value=mock_openai),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/chat", json={"question": "How many items?"})
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert data["answer"] == "Test answer"
