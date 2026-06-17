"""Unit tests for common.engines.override_ledger."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from common.engines.override_ledger import record_override_approval


@patch("common.engines.override_ledger.append_decision")
def test_record_override_approval_calls_ledger(mock_append: MagicMock) -> None:
    cursor = MagicMock()
    record_override_approval(
        cursor,
        override_id=42,
        item_id="100",
        loc="L1",
        override_month=date(2026, 5, 1),
        override_type="PROMO",
        actor="planner1",
        source="auto_approve",
    )
    mock_append.assert_called_once()
    record = mock_append.call_args[0][1]
    assert record.action_type == "forecast_override_approved"
    assert record.subject_id == "100-L1"
    assert record.payload["override_id"] == 42
    assert record.payload["source"] == "auto_approve"
