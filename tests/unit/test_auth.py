"""Unit tests for common/auth.py (Spec 08-02)."""

import pytest
from fastapi import HTTPException

from api.auth import require_api_key
from common.auth import (
    CurrentUser,
    _extract_bearer,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    def test_hash_and_verify(self):
        pw = "securepassword123"
        hashed = hash_password(pw)
        assert hashed != pw
        assert verify_password(pw, hashed) is True

    def test_wrong_password(self):
        hashed = hash_password("correct")
        assert verify_password("wrong", hashed) is False

    def test_invalid_hash(self):
        assert verify_password("test", "not-a-valid-hash") is False


class TestJWTTokens:
    def test_access_token_roundtrip(self):
        token = create_access_token("user-123", "test@test.com", "planner")
        payload = decode_token(token)
        assert payload["sub"] == "user-123"
        assert payload["email"] == "test@test.com"
        assert payload["role"] == "planner"
        assert payload["type"] == "access"

    def test_refresh_token_roundtrip(self):
        token = create_refresh_token("user-456")
        payload = decode_token(token)
        assert payload["sub"] == "user-456"
        assert payload["type"] == "refresh"

    def test_invalid_token_raises(self):
        with pytest.raises(HTTPException) as exc_info:
            decode_token("invalid.token.here")
        assert exc_info.value.status_code == 401


class TestExtractBearer:
    def test_valid_bearer(self):
        assert _extract_bearer("Bearer abc123") == "abc123"

    def test_no_header(self):
        assert _extract_bearer(None) is None

    def test_wrong_scheme(self):
        assert _extract_bearer("Basic abc123") is None

    def test_malformed(self):
        assert _extract_bearer("Bearer") is None


class TestCurrentUser:
    def test_model(self):
        user = CurrentUser(user_id="u1", email="a@b.com", role="admin")
        assert user.user_id == "u1"
        assert user.email == "a@b.com"
        assert user.role == "admin"


class TestWriteAuthorization:
    @pytest.mark.asyncio
    async def test_planner_jwt_authorizes_write(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "service-secret")
        token = create_access_token("planner-1", "planner@example.com", "planner")

        await require_api_key(x_api_key=None, authorization=f"Bearer {token}")

    @pytest.mark.asyncio
    async def test_viewer_jwt_cannot_write(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "service-secret")
        token = create_access_token("viewer-1", "viewer@example.com", "viewer")

        with pytest.raises(HTTPException) as exc_info:
            await require_api_key(x_api_key=None, authorization=f"Bearer {token}")

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_service_api_key_still_authorizes_automation(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "service-secret")

        await require_api_key(x_api_key="service-secret", authorization=None)
