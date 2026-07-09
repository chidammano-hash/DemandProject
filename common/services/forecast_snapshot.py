"""Shared invariants for the bounded live-forecast snapshot archive."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

SNAPSHOT_LAGS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)


def select_top_contenders(
    runs: Iterable[Mapping[str, Any]],
    *,
    contender_count: int = 3,
) -> list[dict[str, Any]]:
    """Return the fixed top contenders, ranked by completed backtest WAPE.

    Callers supply one latest completed, database-loaded run per forecastable
    model. ``None`` WAPE runs are intentionally ineligible: ranking an unknown
    score would make the archive roster non-deterministic.
    """
    eligible = [dict(run) for run in runs if run.get("wape") is not None]
    eligible.sort(
        key=lambda run: (
            float(run["wape"]),
            -float(run.get("accuracy_pct") or float("-inf")),
            _descending_datetime_key(run.get("completed_at")),
            str(run["model_id"]),
        )
    )
    if len(eligible) < contender_count:
        raise ValueError(
            f"Expected three eligible contender models (configured count={contender_count}); "
            f"found {len(eligible)}"
        )

    selected = eligible[:contender_count]
    for rank, run in enumerate(selected, start=1):
        run["contender_rank"] = rank
    return selected


def _descending_datetime_key(value: Any) -> float:
    """Sort newer completed runs first without requiring a datetime subtype."""
    if value is None:
        return float("inf")
    timestamp = getattr(value, "timestamp", None)
    if callable(timestamp):
        return -float(timestamp())
    return float("inf")


def missing_required_lags(
    counts_by_model: Mapping[str, Mapping[int, int]],
) -> dict[str, list[int]]:
    """Return required archive lags that have no rows for each selected model."""
    missing: dict[str, list[int]] = {}
    for model_id, counts in counts_by_model.items():
        absent = [lag for lag in SNAPSHOT_LAGS if int(counts.get(lag, 0)) <= 0]
        if absent:
            missing[model_id] = absent
    return missing


def cleanup_reconciliation_issues(
    expected_contender_counts: Mapping[tuple[str, str], int],
    archived_contender_counts: Mapping[tuple[str, str], int],
    *,
    champion_archive_count: int,
) -> list[str]:
    """List selected-roster archive shortfalls that block staging cleanup."""
    issues = [
        (
            f"{model_id}/{run_id}: expected {expected}, "
            f"archived {int(archived_contender_counts.get((model_id, run_id), 0))}"
        )
        for (model_id, run_id), expected in expected_contender_counts.items()
        if int(archived_contender_counts.get((model_id, run_id), 0)) != int(expected)
    ]
    if champion_archive_count <= 0:
        issues.append("champion archive is missing")
    return issues
