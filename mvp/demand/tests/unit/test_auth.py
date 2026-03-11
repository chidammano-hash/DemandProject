"""Unit tests for common/auth.py (Spec 08-02)."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException

from common.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    CurrentUser,
    _extract_bearer,
    _reset_config_cache,
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
