"""Unit tests for the bounded live-forecast snapshot archive."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from common.services.forecast_snapshot import (
    cleanup_reconciliation_issues,
    missing_required_lags,
    select_top_contenders,
)


def test_select_top_contenders_uses_wape_then_deterministic_tiebreakers():
    rows = [
        {
            "model_id": "zeta",
            "backtest_run_id": 9,
            "wape": 12.0,
            "accuracy_pct": 88.0,
            "completed_at": datetime(2026, 6, 1, tzinfo=UTC),
        },
        {
            "model_id": "alpha",
            "backtest_run_id": 8,
            "wape": 12.0,
            "accuracy_pct": 88.0,
            "completed_at": datetime(2026, 6, 1, tzinfo=UTC),
        },
        {
            "model_id": "beta",
            "backtest_run_id": 7,
            "wape": 12.0,
            "accuracy_pct": 87.0,
            "completed_at": datetime(2026, 6, 2, tzinfo=UTC),
        },
        {
            "model_id": "winner",
            "backtest_run_id": 6,
            "wape": 10.0,
            "accuracy_pct": 80.0,
            "completed_at": datetime(2026, 5, 1, tzinfo=UTC),
        },
    ]

    selected = select_top_contenders(rows)

    assert [row["model_id"] for row in selected] == ["winner", "alpha", "zeta"]
    assert [row["contender_rank"] for row in selected] == [1, 2, 3]


def test_select_top_contenders_requires_three_eligible_models():
    rows = [
        {
            "model_id": "only_one",
            "backtest_run_id": 1,
            "wape": 10.0,
            "accuracy_pct": 90.0,
            "completed_at": datetime(2026, 6, 1, tzinfo=UTC),
        },
        {
            "model_id": "missing_wape",
            "backtest_run_id": 2,
            "wape": None,
            "accuracy_pct": 91.0,
            "completed_at": datetime(2026, 6, 1, tzinfo=UTC),
        },
    ]

    with pytest.raises(ValueError, match="three eligible"):
        select_top_contenders(rows)


def test_missing_required_lags_reports_only_incomplete_roster_models():
    counts = {
        "model_a": {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2},
        "model_b": {0: 2, 1: 2, 2: 2, 4: 2, 5: 2},
    }

    assert missing_required_lags(counts) == {"model_b": [3]}


def test_cleanup_reconciliation_ignores_unselected_models_but_requires_champion():
    expected = {("model_a", "run-a"): 12, ("model_b", "run-b"): 10}
    archived = {("model_a", "run-a"): 12, ("model_b", "run-b"): 10}

    assert cleanup_reconciliation_issues(expected, archived, champion_archive_count=4) == []
    assert cleanup_reconciliation_issues(expected, archived, champion_archive_count=0) == [
        "champion archive is missing"
    ]
