"""Notification engine for Supply Chain Command Center (Spec 08-04).

Multi-channel notification dispatch: Slack, Teams, Email, PagerDuty.
"""
from __future__ import annotations

import json
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from common.utils import load_config, reset_config

_CONFIG_NAME = "notification_config.yaml"


# ---------------------------------------------------------------------------
# Config (thread-safe via common.utils.load_config)
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    return load_config(_CONFIG_NAME)


def _reset_config():
    reset_config(_CONFIG_NAME)


# ---------------------------------------------------------------------------
# Rate limiter (in-process dedup)
# ---------------------------------------------------------------------------
_recent_events: dict[str, float] = {}


def _should_rate_limit(event_key: str, cooldown: int = 300) -> bool:
    now = time.time()
    last = _recent_events.get(event_key)
    if last and (now - last) < cooldown:
        return True
    _recent_events[event_key] = now
    return False


# ---------------------------------------------------------------------------
# Channel adapters
# ---------------------------------------------------------------------------
def _send_slack(webhook_url: str, subject: str, body: str) -> tuple[bool, str]:
    """Send a Slack notification via incoming webhook."""
    try:
        import urllib.request
        payload = json.dumps({"text": f"*{subject}*\n{body}"}).encode()
        req = urllib.request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200, ""
    except Exception as e:
        return False, str(e)


def _send_teams(webhook_url: str, subject: str, body: str) -> tuple[bool, str]:
    """Send a Teams notification via incoming webhook."""
    try:
        import urllib.request
        payload = json.dumps({
            "@type": "MessageCard",
            "summary": subject,
            "sections": [{"activityTitle": subject, "text": body}],
        }).encode()
        req = urllib.request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200, ""
    except Exception as e:
        return False, str(e)


def _send_email(smtp_config: dict, recipient: str, subject: str, body: str) -> tuple[bool, str]:
    """Send an email notification via SMTP."""
    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_config.get("from_address", "noreply@demandstudio.local")
        msg["To"] = recipient
        msg["Subject"] = f"[Supply Chain Command Center] {subject}"
        msg.attach(MIMEText(body, "plain"))

        host = smtp_config.get("host", "localhost")
        port = smtp_config.get("port", 587)
        with smtplib.SMTP(host, port, timeout=10) as server:
            user = smtp_config.get("user")
            password = smtp_config.get("password")
            if user and password:
                server.starttls()
                server.login(user, password)
            server.send_message(msg)
        return True, ""
    except Exception as e:
        return False, str(e)


def _send_pagerduty(routing_key: str, subject: str, body: str, severity: str = "warning") -> tuple[bool, str]:
    """Send a PagerDuty alert via Events API v2."""
    try:
        import urllib.request
        pd_severity = {"critical": "critical", "high": "error", "medium": "warning"}.get(severity, "info")
        payload = json.dumps({
            "routing_key": routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": subject,
                "severity": pd_severity,
                "source": "demand-studio",
                "custom_details": {"body": body},
            },
        }).encode()
        req = urllib.request.Request(
            "https://events.pagerduty.com/v2/enqueue",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 202), ""
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# NotificationEngine
# ---------------------------------------------------------------------------
CHANNEL_SENDERS = {
    "slack": lambda cfg, r, s, b, sev: _send_slack(cfg.get("webhook_url", ""), s, b),
    "teams": lambda cfg, r, s, b, sev: _send_teams(cfg.get("webhook_url", ""), s, b),
    "email": lambda cfg, r, s, b, sev: _send_email(cfg, r, s, b),
    "pagerduty": lambda cfg, r, s, b, sev: _send_pagerduty(cfg.get("routing_key", ""), s, b, sev),
}


class NotificationEngine:
    """Dispatches notifications to configured channels."""

    def __init__(self):
        self.config = _load_config()

    def send(
        self,
        event_type: str,
        severity: str,
        subject: str,
        body: str,
        recipient: str = "",
    ) -> list[dict]:
        """Send notification to all channels matching event_type + severity rules."""
        cfg = self.config
        rate_limits = cfg.get("rate_limits", {})
        cooldown = rate_limits.get("cooldown_seconds", 300)

        event_key = f"{event_type}:{severity}:{subject[:50]}"
        if _should_rate_limit(event_key, cooldown):
            return [{"channel": "all", "status": "rate_limited"}]

        routing = cfg.get("routing_rules", {})
        channels_cfg = cfg.get("channels", {})

        # Find matching channels
        rule = routing.get(event_type, routing.get("default", {}))
        target_channels = rule.get("channels", []) if isinstance(rule, dict) else []

        # Filter by severity
        min_severity_map = {"info": 0, "warning": 1, "high": 2, "critical": 3}
        event_sev = min_severity_map.get(severity, 0)
        min_sev = min_severity_map.get(rule.get("min_severity", "info") if isinstance(rule, dict) else "info", 0)
        if event_sev < min_sev:
            return [{"channel": "all", "status": "below_severity_threshold"}]

        results = []
        for ch_name in target_channels:
            ch_cfg = channels_cfg.get(ch_name, {})
            if not ch_cfg.get("enabled", True):
                continue

            sender = CHANNEL_SENDERS.get(ch_cfg.get("type", ch_name))
            if not sender:
                results.append({"channel": ch_name, "status": "unsupported"})
                continue

            success, error = sender(ch_cfg, recipient, subject, body, severity)
            results.append({
                "channel": ch_name,
                "status": "delivered" if success else "failed",
                "error": error or None,
            })

            # Log to DB (best-effort)
            self._log_delivery(ch_name, event_type, severity, recipient, subject, body, success, error)

        return results or [{"channel": "none", "status": "no_matching_channels"}]

    def _log_delivery(self, channel: str, event_type: str, severity: str,
                      recipient: str, subject: str, body: str, success: bool, error: str):
        """Record notification delivery attempt."""
        try:
            from api.core import get_conn
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO fact_notification_log
                       (event_type, severity, recipient, subject, body, status, error)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (event_type, severity, recipient, subject, body,
                     "delivered" if success else "failed", error or None),
                )
                conn.commit()
        except Exception:
            pass
