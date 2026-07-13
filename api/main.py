"""Demand Unified MVP API — FastAPI application entry point.

All route handlers live in modular router files under api/routers/.
This file is responsible only for app creation, middleware, and router mounting.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, ORJSONResponse
from starlette.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware

from api.pool import (
    close_async_pool,
    close_async_read_pool,
    close_pool,
    close_read_pool,
    open_async_pool,
    open_async_read_pool,
    open_pool,
)
from common.services.rate_limiter import get_rate_limiter


logger = logging.getLogger("api.access")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler: open pool + start scheduler on startup, close on shutdown.

    Replaces the legacy ``@app.on_event("startup")`` / ``@app.on_event("shutdown")``
    decorators (deprecated in FastAPI 0.109+). The scheduler is started lazily
    via ``JobManager.start()`` — importing it here guarantees the APScheduler
    BackgroundScheduler is live before the first request.
    """
    # Anyio threadpool: FastAPI offloads sync `def` handlers (which is most of
    # this codebase) to this pool. Default is 40 tokens — bump to 100 so the
    # 13-endpoint Customer Analytics tab × multiple planners doesn't queue.
    # Total concurrency = N gunicorn workers × this number; keep it well under
    # the DB pool size × workers to avoid pool exhaustion.
    try:
        import os as _os
        from anyio import to_thread
        _tp_size = int(_os.getenv("ANYIO_THREADPOOL_SIZE", "100"))
        to_thread.current_default_thread_limiter().total_tokens = _tp_size
        logger.info("Anyio threadpool sized to %d tokens", _tp_size)
    except Exception as exc:  # noqa: BLE001 — best effort; default 40 is fine
        logger.warning("Could not resize anyio threadpool: %s", exc)

    # Pool: best-effort open — in offline/test modes where POSTGRES_PASSWORD is
    # unset we continue without a live pool so unit tests (which patch the pool)
    # can still import the app.
    try:
        open_pool()
        logger.info("DB connection pool opened on startup")
    except RuntimeError as exc:
        logger.warning("DB pool not opened on startup: %s", exc)

    # Async DB pool — used by routers converted to async def (Item 19 pilot).
    # Best-effort: failure to open the async pool should not abort startup
    # because sync handlers can still serve traffic.
    try:
        await open_async_pool()
        logger.info("Async DB connection pool opened on startup")
    except (RuntimeError, Exception) as exc:  # noqa: BLE001 — best effort
        logger.warning("Async DB pool not opened on startup: %s", exc)

    # Read-replica async pool (Item 24) — opened only when READ_REPLICA_URL is
    # set. open_async_read_pool() is a no-op (returns None) when the env var
    # is unset, so this block is free in single-node deployments. Failure to
    # open the replica pool is logged but never aborts startup; analytics
    # endpoints fall through to the primary async pool.
    try:
        replica = await open_async_read_pool()
        if replica is not None:
            logger.info("Async read-replica pool opened on startup")
    except Exception as exc:  # noqa: BLE001 — best effort
        logger.warning("Async read-replica pool not opened on startup: %s", exc)

    # Reap orphan integration jobs from any prior crashed/killed API process.
    # Safe to do unconditionally: this fresh process has no live workers, so any
    # row in 'queued'/'running' is provably abandoned.
    try:
        from common.services.integration_runner import IntegrationRunner
        from api.core import _get_pool
        IntegrationRunner(_get_pool()).reap_orphans()
    except Exception as exc:  # noqa: BLE001 — startup best-effort
        logger.warning("integration orphan reap skipped: %s", exc)

    # Scheduler: initialised lazily on demand. Pre-warm it so background jobs
    # start on boot rather than on the first /jobs request.
    job_manager = None
    try:
        from common.services.job_registry import JobManager
        job_manager = JobManager()
        job_manager.start()
        logger.info("APScheduler BackgroundScheduler started on startup")
    except Exception as exc:  # noqa: BLE001 — scheduler init is best-effort; app must still serve API if jobs backend is down
        logger.warning("Scheduler not started on startup: %s", exc)

    try:
        yield
    finally:
        if job_manager is not None:
            try:
                job_manager.shutdown()
            except Exception as exc:  # noqa: BLE001 — shutdown cleanup
                logger.warning("Scheduler shutdown raised: %s", exc)
        close_pool()
        # Read replica pools (Item 24) — both no-op if the corresponding
        # READ_REPLICA_URL pool was never opened.
        close_read_pool()
        try:
            await close_async_pool()
        except Exception as exc:  # noqa: BLE001 — shutdown cleanup
            logger.warning("Async DB pool shutdown raised: %s", exc)
        try:
            await close_async_read_pool()
        except Exception as exc:  # noqa: BLE001 — shutdown cleanup
            logger.warning("Async read-replica pool shutdown raised: %s", exc)
        logger.info("DB connection pool closed on shutdown")


app = FastAPI(
    title="Demand Unified MVP API",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)

# --- Middleware (outermost first) ---
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "Idempotency-Key"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limit write endpoints (POST/PUT/DELETE) to prevent abuse."""
    if request.method in ("POST", "PUT", "DELETE"):
        limiter = get_rate_limiter()
        client_ip = request.client.host if request.client else "unknown"
        limit = limiter.get_tier_limit("standard")
        allowed, remaining = limiter.check(client_ip, max_requests=limit, window_seconds=60)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": "60"},
            )
    return await call_next(request)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Stamp every request with a short request-id so slow-query logs and
    pool-exhaustion errors can be correlated end-to-end.

    Honours an inbound `X-Request-ID` header (e.g. from nginx / a load
    balancer) so the same id flows through the full stack.
    """
    from uuid import uuid4
    rid = request.headers.get("X-Request-ID") or uuid4().hex[:12]
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    rid = getattr(request.state, "request_id", "-")
    logger.info(
        "rid=%s %s %s %d %.1fms",
        rid, request.method, request.url.path, response.status_code, duration_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Router registration
#
# Order matters for path-parameter routes: routers with literal prefixes
# (e.g. /domains/forecast, /domains/dfu) are registered BEFORE routers
# with catch-all {domain} path params (domains.py) so that literal paths
# take precedence.
# ---------------------------------------------------------------------------
from api.routers.forecasting import accuracy   # noqa: E402
from api.routers.forecasting import analysis   # noqa: E402
from api.routers.forecasting import clusters   # noqa: E402
from api.routers.forecasting import competition  # noqa: E402
from api.routers.intelligence import intel      # noqa: E402
from api.routers.inventory import inv_backtest  # noqa: E402
from api.routers import inventory  # noqa: E402
from api.routers.forecasting import shap       # noqa: E402
from api.routers.inventory import inv_planning_variability    # noqa: E402
from api.routers.inventory import inv_planning_lead_time      # noqa: E402
from api.routers.inventory import inv_planning_eoq            # noqa: E402
from api.routers.inventory import inv_planning_policy         # noqa: E402
from api.routers.inventory import inv_planning_health         # noqa: E402
from api.routers.inventory import inv_planning_exceptions     # noqa: E402
from api.routers.inventory import inv_planning_abc_xyz        # noqa: E402
from api.routers.inventory import inv_planning_supplier       # noqa: E402
from api.routers.inventory import inv_planning_intramonth     # noqa: E402
from api.routers.inventory import inv_planning_demand_signals # noqa: E402
from api.routers.inventory import inv_planning_simulation     # noqa: E402
from api.routers.inventory import inv_planning_investment     # noqa: E402
from api.routers.inventory import inv_planning_safety_stock   # noqa: E402
from api.routers.inventory import inv_planning_replenishment  # noqa: E402
from api.routers.operations import fill_rate    # noqa: E402
from api.routers.operations import control_tower  # noqa: E402
from api.routers.intelligence import storyboard  # noqa: E402
from api.routers.forecasting import production_forecast  # noqa: E402
from api.routers.forecasting import consensus_plan      # noqa: E402
from api.routers.operations import supply      # noqa: E402
from api.routers.inventory import inv_planning_projection  # noqa: E402
from api.routers.forecasting import bias_corrections  # noqa: E402
from api.routers.operations import service_level     # noqa: E402
from api.routers.inventory import lead_time_learning  # noqa: E402
from api.routers.forecasting import blended_forecast  # noqa: E402
from api.routers.operations import echelon_planning  # noqa: E402
from api.routers.operations import financial_plan    # noqa: E402
from api.routers.operations import events            # noqa: E402
from api.routers.operations import supply_scenarios  # noqa: E402
from api.routers.inventory import inv_planning_rebalancing  # noqa: E402
from api.routers.inventory import inv_planning_insights     # noqa: E402
# --- Gen-4 subdirectory imports ---
from api.routers.operations import sop              # noqa: E402
from api.routers.intelligence import ai_planner      # noqa: E402
from api.routers.intelligence import sku_chat      # noqa: E402  # 06-07 SKU Chatbot (Claude Agent SDK)
from api.routers.intelligence import explain as explain_router  # noqa: E402  # Gen-4 G: forecast explain
from api.routers.intelligence import copilot as copilot_router  # noqa: E402  # Grounded planning Copilot
from api.routers.core import dashboard               # noqa: E402
from api.routers.core import pipeline_readiness      # noqa: E402
from api.routers.core import jobs                    # noqa: E402
from api.routers.platform import auth_router         # noqa: E402  # 08-02 RBAC
from api.routers.platform import users               # noqa: E402  # 08-02 User mgmt
from api.routers.platform import admin as admin_router  # noqa: E402  # Gen-4 J: LLM reset / tuning invalidation
# --- Spec 08-xx: Next-gen platform routers ---
from api.routers.platform import data_quality      # noqa: E402  # 08-01 DQ
from api.routers.platform import medallion         # noqa: E402  # Medallion lineage
from api.routers.platform import notifications     # noqa: E402  # 08-04 Alerts
from api.routers.platform import collaboration     # noqa: E402  # 08-05 Annotations
from api.routers.intelligence import external_signals  # noqa: E402  # 08-06 Demand signals
from api.routers.forecasting import fva              # noqa: E402  # 08-07 FVA tracking
from api.routers.platform import reports          # noqa: E402  # 08-08 Reporting
from api.routers.platform import webhooks         # noqa: E402  # 08-10 Webhooks
from api.routers.platform import config_manager   # noqa: E402  # Config management UI
from api.routers.platform import sql_runner       # noqa: E402  # SQL Runner
from api.routers.platform import integration      # noqa: E402  # Unified ETL integration jobs
from api.routers.platform import integration_chain  # noqa: E402  # Smart chain detection + sequential exec
from api.routers.inventory import sourcing as sourcing_router   # noqa: E402
from api.routers.inventory import purchase_orders as po_router  # noqa: E402
from api.routers.forecasting import accuracy_budget  # noqa: E402  # Accuracy budget
from api.routers.forecasting import lgbm_tuning  # noqa: E402  # LGBM tuning
from api.routers.forecasting import tuning_chat  # noqa: E402  # LGBM tuning chat
from api.routers.forecasting import cluster_eda  # noqa: E402  # Cluster EDA
from api.routers.forecasting import sampled_backtest  # noqa: E402  # Sampled backtest
from api.routers.forecasting import feature_lab  # noqa: E402  # Feature Lab
from api.routers.forecasting import tuning as unified_model_tuning  # noqa: E402  # Unified model tuning
from api.routers.forecasting import cluster_experiments  # noqa: E402  # Cluster experiments
from api.routers.forecasting import backtest_management  # noqa: E402  # Backtest management
from api.routers.forecasting import forecast_promotion  # noqa: E402  # Forecast release promotion
from api.routers.forecasting import forecast_release  # noqa: E402  # Planner release readiness
from api.routers.forecasting import champion_experiments  # noqa: E402  # Champion experiments
from api.routers.forecasting import champion_sweeps  # noqa: E402  # Champion strategy sweep (tournament)
from api.routers.forecasting import sku_features     # noqa: E402  # SKU feature explorer
from api.routers.forecasting import ai_champion      # noqa: E402  # AI Champion forward adjuster
from api.routers.forecasting import customer_forecast  # noqa: E402  # Customer-level forecasting
from api.routers.intelligence import customer_analytics  # noqa: E402  # Customer Analytics
from api.routers.inventory import demand_history  # noqa: E402  # Demand History Workbench
from api.routers.inventory import inv_planning_algorithm_comparison  # noqa: E402  # Algorithm Inventory Comparison
from api.routers.inventory import integrated_targets  # noqa: E402  # Integrated Planning Targets (SS+EOQ+ROP)
from api.routers.inventory import working_capital  # noqa: E402  # Gen-4 SC-10: DIO/DPO/DSO/C2C + rolling-13-week
from api.routers import domains    # noqa: E402

# Specific-path routers first
app.include_router(accuracy.router)
app.include_router(analysis.router)
app.include_router(clusters.router)
app.include_router(competition.router)
app.include_router(dashboard.router)
app.include_router(pipeline_readiness.router)
app.include_router(intel.router)
app.include_router(inv_backtest.router)
app.include_router(inventory.router)
app.include_router(inv_planning_variability.router)
app.include_router(inv_planning_lead_time.router)
app.include_router(inv_planning_eoq.router)
app.include_router(inv_planning_policy.router)
app.include_router(inv_planning_health.router)
app.include_router(inv_planning_exceptions.router)
app.include_router(inv_planning_abc_xyz.router)
app.include_router(inv_planning_supplier.router)
app.include_router(inv_planning_intramonth.router)
app.include_router(inv_planning_demand_signals.router)
app.include_router(inv_planning_simulation.router)
app.include_router(inv_planning_investment.router)
app.include_router(inv_planning_safety_stock.router)
app.include_router(inv_planning_replenishment.router)
app.include_router(fill_rate.router)
app.include_router(control_tower.router)
app.include_router(jobs.router)
app.include_router(ai_planner.router)
app.include_router(sku_chat.router)
app.include_router(storyboard.router)
app.include_router(production_forecast.router)
app.include_router(consensus_plan.router)
app.include_router(supply.router)
app.include_router(inv_planning_projection.router)
app.include_router(bias_corrections.router)
app.include_router(service_level.router)
app.include_router(lead_time_learning.router)
app.include_router(blended_forecast.router)
app.include_router(echelon_planning.router)
app.include_router(financial_plan.router)
app.include_router(sop.router)
app.include_router(events.router)
app.include_router(supply_scenarios.router)
app.include_router(inv_planning_rebalancing.router)
app.include_router(inv_planning_insights.router)
app.include_router(shap.router)

# --- Spec 08-xx: Next-gen platform routers ---
app.include_router(auth_router.router)
app.include_router(users.router)
app.include_router(admin_router.router)
app.include_router(data_quality.router)
app.include_router(medallion.router)
app.include_router(notifications.router)
app.include_router(collaboration.router)
app.include_router(external_signals.router)
app.include_router(fva.router)
app.include_router(reports.router)
app.include_router(webhooks.router)
app.include_router(config_manager.router)
app.include_router(sql_runner.router)
app.include_router(integration.router)
app.include_router(integration_chain.router)
app.include_router(sourcing_router.router)
app.include_router(po_router.router)

app.include_router(accuracy_budget.router)
app.include_router(lgbm_tuning.router)
app.include_router(tuning_chat.router)
app.include_router(cluster_eda.router)
app.include_router(sampled_backtest.router)
app.include_router(feature_lab.router)
app.include_router(unified_model_tuning.router, prefix="/model-tuning", tags=["model-tuning"])
app.include_router(cluster_experiments.router)
app.include_router(backtest_management.router)
app.include_router(forecast_promotion.router)
app.include_router(forecast_release.router)
app.include_router(champion_experiments.router)
app.include_router(champion_sweeps.router)
app.include_router(sku_features.router)
app.include_router(ai_champion.router)
app.include_router(customer_forecast.router)
app.include_router(customer_analytics.router)
app.include_router(demand_history.router)
app.include_router(inv_planning_algorithm_comparison.router)
app.include_router(integrated_targets.router)
app.include_router(working_capital.router)
app.include_router(explain_router.router)
app.include_router(copilot_router.router)

# domains.py has catch-all /domains/{domain}/* — mount last
app.include_router(domains.router)
