"""Unit tests for common/webhook_dispatcher.py (Spec 08-10)."""
from unittest.mock import patch, MagicMock

from common.webhook_dispatcher import _sign_payload, dispatch_webhook


class TestSignPayload:
    def test_produces_hex_digest(self):
        sig = _sign_payload('{"event":"test"}', "secret123")
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA-256 hex length

    def test_deterministic(self):
        s1 = _sign_payload("payload", "key")
        s2 = _sign_payload("payload", "key")
        assert s1 == s2

    def test_different_secret(self):
        s1 = _sign_payload("payload", "key1")
        s2 = _sign_payload("payload", "key2")
        assert s1 != s2


class TestDispatchWebhook:
    @patch("common.webhook_dispatcher.urllib.request.urlopen")
    def test_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = lambda s, *a: None
        mock_urlopen.return_value = mock_resp

        result = dispatch_webhook(
            url="http://example.com/webhook",
            secret="test-secret",
            event_type="test",
            payload={"msg": "hello"},
        )
        assert result["status"] == "delivered"
        assert result["status_code"] == 200
        assert result["attempt"] == 1

    @patch("common.webhook_dispatcher.urllib.request.urlopen")
    def test_failure_after_retries(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")

        result = dispatch_webhook(
            url="http://example.com/webhook",
            secret="test-secret",
            event_type="test",
            payload={"msg": "hello"},
            max_retries=1,
            backoff_base=0.01,
        )
        assert result["status"] == "failed"
        assert "error" in result
