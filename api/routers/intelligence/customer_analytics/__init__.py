"""Customer Analytics endpoints — demand-aware geographic, segment, and channel analytics.

Joins fact_customer_demand_monthly with dim_customer / dim_item to provide
rich demand visualizations: map, treemap, heatmap, channel mix, segment trends,
ranking, and OOS impact.

This package combines the sub-routers (geo, segments, ranking, lifecycle,
kpis, recalculate) into one ``router`` so ``api.main`` can mount it as a
single unit.
"""
from __future__ import annotations

from fastapi import APIRouter

# Re-export internals that tests patch on this module path.
# (tests/api/test_customer_analytics.py patches
#  api.routers.intelligence.customer_analytics.{get_planning_date,_get_nomi})
from ._helpers import _get_nomi, get_planning_date  # noqa: F401
from . import assistant, geo, kpis, lifecycle, ranking, recalculate, segments

router = APIRouter()
router.include_router(assistant.router)
router.include_router(geo.router)
router.include_router(segments.router)
router.include_router(ranking.router)
router.include_router(lifecycle.router)
router.include_router(kpis.router)
router.include_router(recalculate.router)

__all__ = ["router"]
