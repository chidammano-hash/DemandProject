"""Unit tests for agentic champion adjustment staging/approval (champion_adjust.py)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from common.ai.sku_chat import champion_adjust

_PREVIEW = {
    "item_id": "100320",
    "loc": "DC1",
    "recommendation_code": "SCALE_DOWN",
    "rec_pct_change": -15.0,
    "proposed_qty": None,
    "apply_horizon_months": 3,
    "confidence": 0.72,
    "rationale": "Q3 actuals are running ~15% below the champion forecast.",
    "evidence_keys": ["trend_break"],
    "months": [{"forecast_month": "2026-07-01", "champion_qty": 100, "ai_qty": 85}],
}


def _mock_pool(*, fetchone=None, description=None):
    cursor = MagicMock()
    cursor.description = description if description is not None else [("col",)]
    cursor.fetchone.return_value = fetchone
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    pool = MagicMock()
    pool.connection.return_value = conn
    return pool, cursor


def test_stage_adjustment_builds_preview_and_stages():
    pool, cur = _mock_pool()
    fake_preview = MagicMock()
    fake_preview.to_dict.return_value = _PREVIEW
    with patch.object(champion_adjust.svc, "adjust_dfu", return_value=fake_preview) as adj:
        out = champion_adjust.stage_adjustment(
            pool, session_id="s1", item_id="100320", customer_group="RETAIL",
            loc="DC1", rationale="reduce Q3",
        )
    adj.assert_called_once()
    assert out["approval_id"]
    assert out["preview"]["recommendation_code"] == "SCALE_DOWN"
    cur.execute.assert_called_once()  # INSERT into pending table


def test_stage_adjustment_no_champion_raises_adjustment_error():
    pool, _ = _mock_pool()
    # NoChampionForecast is a ValueError subclass — caught and re-raised.
    with patch.object(champion_adjust.svc, "adjust_dfu", side_effect=ValueError("no champ")):
        with pytest.raises(champion_adjust.AdjustmentError):
            champion_adjust.stage_adjustment(
                pool, session_id="s1", item_id="x", customer_group="", loc="DC1", rationale="",
            )


_PENDING_DESC = [
    ("approval_id",), ("session_id",), ("item_id",), ("customer_group",),
    ("loc",), ("preview",), ("status",),
]


def test_apply_adjustment_calls_save_and_marks_approved():
    pool, _ = _mock_pool(
        fetchone=("appr-1", "s1", "100320", "RETAIL", "DC1", _PREVIEW, "pending"),
        description=_PENDING_DESC,
    )
    with patch.object(champion_adjust.svc, "save_adjustment", return_value={"saved": True}) as save:
        out = champion_adjust.apply_adjustment(pool, "appr-1")

    # save_adjustment got the recommendation reconstructed from the staged preview
    _, kwargs = save.call_args
    rec = kwargs["recommendation"]
    assert rec["recommendation_code"] == "SCALE_DOWN"
    assert rec["pct_change"] == -15.0          # rec_pct_change -> pct_change
    assert out["status"] == "approved"
    assert out["result"] == {"saved": True}


def test_apply_adjustment_unknown_id_raises():
    pool, _ = _mock_pool(fetchone=None, description=_PENDING_DESC)
    with pytest.raises(champion_adjust.AdjustmentError):
        champion_adjust.apply_adjustment(pool, "missing")


def test_apply_adjustment_save_failure_raises():
    pool, _ = _mock_pool(
        fetchone=("appr-1", "s1", "100320", "RETAIL", "DC1", _PREVIEW, "pending"),
        description=_PENDING_DESC,
    )
    with patch.object(champion_adjust.svc, "save_adjustment", side_effect=ValueError("boom")):
        with pytest.raises(champion_adjust.AdjustmentError):
            champion_adjust.apply_adjustment(pool, "appr-1")


def test_reject_adjustment_marks_rejected():
    pool, cur = _mock_pool()
    out = champion_adjust.reject_adjustment(pool, "appr-1")
    assert out == {"approval_id": "appr-1", "status": "rejected"}
    cur.execute.assert_called_once()  # UPDATE ... status='rejected'
