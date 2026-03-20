"""APScheduler wrapper for the job engine (Feature 39).

Contains:
- APScheduler BackgroundScheduler factory (_make_scheduler)
- Cron/interval trigger creation utilities (make_trigger)

This module isolates all APScheduler-specific logic so that job_registry.py
remains focused on orchestration and the public JobManager API.
"""
from __future__ import annotations

import logging
from typing import Any

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


def make_scheduler() -> BackgroundScheduler:
    """Create and start a BackgroundScheduler with the standard project settings.

    Returns a *started* scheduler instance ready to accept jobs.
    """
    executors = {
        "default": ThreadPoolExecutor(max_workers=4),
    }
    job_defaults = {
        "coalesce": True,
        "max_instances": 1,
        "misfire_grace_time": 3600,
    }
    scheduler = BackgroundScheduler(
        executors=executors,
        job_defaults=job_defaults,
        timezone="UTC",
    )
    scheduler.start()
    logger.info("APScheduler BackgroundScheduler started (4 workers)")
    return scheduler


def make_trigger(cron: str | None, interval_minutes: int | None) -> Any:
    """Create an APScheduler trigger from a cron expression or interval.

    Exactly one of *cron* or *interval_minutes* must be provided.

    Returns a CronTrigger or IntervalTrigger instance.
    """
    if cron:
        return CronTrigger.from_crontab(cron)
    if interval_minutes is not None:
        return IntervalTrigger(minutes=interval_minutes)
    raise ValueError("Must specify either cron or interval_minutes")
