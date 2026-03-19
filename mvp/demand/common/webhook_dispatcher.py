"""Webhook dispatcher for Supply Chain Command Center (Spec 08-10).

Dispatches signed webhook payloads to registered consumers with retry logic.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.request
from typing import Any


def _sign_payload(payload: str, secret: str) -> str:
    """Generate HMAC-SHA256 signature for webhook payload."""
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def dispatch_webhook(
    url: str,
    secret: str,
    event_type: str,
    payload: dict,
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> dict:
    """Send a signed webhook to a registered URL with retry logic."""
    body = json.dumps({"event_type": event_type, "data": payload, "timestamp": time.time()}, default=str)
    signature = _sign_payload(body, secret)
    headers = {
        "Content-Type": "application/json",
        "X-DS-Signature": f"sha256={signature}",
        "X-DS-Event": event_type,
    }

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, data=body.encode(), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return {
                    "status": "delivered",
                    "status_code": resp.status,
                    "attempt": attempt,
                }
        except Exception as e:
            if attempt < max_retries:
                time.sleep(backoff_base ** attempt)
            else:
                return {
                    "status": "failed",
                    "status_code": None,
                    "attempt": attempt,
                    "error": str(e),
                }

    return {"status": "failed", "attempt": max_retries}


class WebhookEngine:
    """Manages webhook registrations and dispatches events."""

    def dispatch_event(self, event_type: str, payload: dict) -> list[dict]:
        """Dispatch an event to all registered webhooks matching the event type."""
        results = []
        try:
            from api.core import get_conn
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """SELECT webhook_id, url, secret, event_types
                       FROM dim_webhook_registration
                       WHERE is_active = TRUE""",
                )
                registrations = cur.fetchall()
        except Exception:
            return [{"status": "error", "message": "Failed to load registrations"}]

        for reg in registrations:
            wh_id, url, secret, event_types = reg
            # Check if this webhook is subscribed to this event type
            subscribed = event_types or []
            if isinstance(subscribed, str):
                try:
                    subscribed = json.loads(subscribed)
                except Exception:
                    subscribed = []
            if subscribed and event_type not in subscribed:
                continue

            result = dispatch_webhook(url, secret, event_type, payload)
            result["webhook_id"] = wh_id
            results.append(result)

            # Log delivery
            self._log_delivery(wh_id, event_type, payload, result)

        return results

    def _log_delivery(self, webhook_id: int, event_type: str, payload: dict, result: dict):
        try:
            from api.core import get_conn
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO fact_webhook_delivery
                       (webhook_id, event_type, payload, status_code, attempt, status)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (
                        webhook_id,
                        event_type,
                        json.dumps(payload, default=str),
                        result.get("status_code"),
                        result.get("attempt", 1),
                        result["status"],
                    ),
                )
                conn.commit()
        except Exception:
            pass
