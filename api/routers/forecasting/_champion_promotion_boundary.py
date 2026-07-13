"""Fail-closed API policy for retired manual champion promotion paths."""

from __future__ import annotations

from typing import Final, NoReturn

from fastapi import HTTPException

MANUAL_CHAMPION_PROMOTION_RETIRED_DETAIL: Final = {
    "code": "manual_champion_promotion_retired",
    "message": (
        "Manual champion promotion is retired. Run "
        "POST /jobs/pipelines/named/champion-refresh to create and atomically promote "
        "a governed champion."
    ),
}

RETIRED_PUBLIC_CHAMPION_JOB_TYPES: Final = frozenset({"champion_results_load"})


def raise_manual_champion_promotion_retired() -> NoReturn:
    """Reject production mutation through a legacy public API boundary."""
    raise HTTPException(
        status_code=410,
        detail=MANUAL_CHAMPION_PROMOTION_RETIRED_DETAIL,
    )


def reject_retired_champion_job_type(job_type: str) -> None:
    """Prevent generic job APIs from launching legacy champion mutation jobs."""
    if job_type in RETIRED_PUBLIC_CHAMPION_JOB_TYPES:
        raise_manual_champion_promotion_retired()
