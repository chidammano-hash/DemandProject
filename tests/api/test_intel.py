"""Tests for market intelligence endpoint."""

import pytest
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport



@pytest.mark.asyncio
async def test_intel_missing_item(mock_pool):
    """POST /market-intelligence with empty item should return 422."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/market-intelligence",
                json={"item_id": "", "location_id": "LOC1"},
            )
            assert response.status_code == 422


@pytest.mark.asyncio
async def test_intel_missing_location(mock_pool):
    """POST /market-intelligence with empty location should return 422."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/market-intelligence",
                json={"item_id": "100320", "location_id": ""},
            )
            assert response.status_code == 422


@pytest.mark.asyncio
async def test_intel_success(mock_pool):
    """POST /market-intelligence with mocked OpenAI should return 200."""
    pool, _, cursor = mock_pool
    # Mock item lookup
    cursor.fetchone.side_effect = [
        ("Widget A", "BrandX", "Hardware"),  # dim_item
        ("CA", "West DC"),                   # dim_location
    ]
    cursor.fetchall.return_value = []  # No sales context

    mock_openai = MagicMock()
    mock_chat_resp = MagicMock()
    mock_chat_resp.choices = [MagicMock(message=MagicMock(content="Market briefing text"))]
    mock_openai.chat.completions.create.return_value = mock_chat_resp

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("api.routers.intel.get_openai", return_value=mock_openai),
        patch.dict("os.environ", {"GOOGLE_API_KEY": "", "GOOGLE_CX": ""}, clear=False),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/market-intelligence",
                json={"item_id": "100320", "location_id": "1401-BULK"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "narrative" in data
            assert "item_id" in data
            assert "location_id" in data
            assert data["item_id"] == "100320"
