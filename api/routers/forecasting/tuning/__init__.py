"""Unified model-tuning router for LightGBM experiments.

The router is split into focused sub-modules; this package aggregates them
into the single ``router`` that ``api/main.py`` mounts at ``/model-tuning``.
"""
from __future__ import annotations

from fastapi import APIRouter

from . import (
    cancel_delete,
    cluster,
    compare,
    create,
    detail,
    lag,
    logs,
    month,
    promote,
    promote_results,
    promotions,
    templates,
)
from . import list as list_module

router = APIRouter()
router.include_router(list_module.router)
router.include_router(detail.router)
router.include_router(create.router)
router.include_router(compare.router)
router.include_router(cluster.router)
router.include_router(lag.router)
router.include_router(logs.router)
router.include_router(month.router)
router.include_router(promote.router)
router.include_router(promote_results.router)
router.include_router(cancel_delete.router)
router.include_router(templates.router)
router.include_router(promotions.router)

__all__ = ["router"]
