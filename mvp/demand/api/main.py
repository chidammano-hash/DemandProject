"""Demand Unified MVP API — FastAPI application entry point.

All route handlers live in modular router files under api/routers/.
This file is responsible only for app creation, middleware, and router mounting.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from starlette.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Demand Unified MVP API")

# --- Middleware (outermost first) ---
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Router registration
#
# Order matters for path-parameter routes: routers with literal prefixes
# (e.g. /domains/forecast, /domains/dfu) are registered BEFORE routers
# with catch-all {domain} path params (domains.py) so that literal paths
# take precedence.
# ---------------------------------------------------------------------------
from api.routers import accuracy   # noqa: E402
from api.routers import analysis   # noqa: E402
from api.routers import benchmark  # noqa: E402
from api.routers import chat       # noqa: E402
from api.routers import clusters   # noqa: E402
from api.routers import competition  # noqa: E402
from api.routers import dashboard  # noqa: E402
from api.routers import intel      # noqa: E402
from api.routers import inv_backtest  # noqa: E402
from api.routers import inventory  # noqa: E402
from api.routers import jobs       # noqa: E402
from api.routers import shap       # noqa: E402
from api.routers import inv_planning  # noqa: E402
from api.routers import fill_rate    # noqa: E402
from api.routers import control_tower  # noqa: E402
from api.routers import ai_planner  # noqa: E402
from api.routers import domains    # noqa: E402

# Specific-path routers first
app.include_router(accuracy.router)
app.include_router(analysis.router)
app.include_router(benchmark.router)
app.include_router(chat.router)
app.include_router(clusters.router)
app.include_router(competition.router)
app.include_router(dashboard.router)
app.include_router(intel.router)
app.include_router(inv_backtest.router)
app.include_router(inventory.router)
app.include_router(inv_planning.router)
app.include_router(fill_rate.router)
app.include_router(control_tower.router)
app.include_router(jobs.router)
app.include_router(ai_planner.router)
app.include_router(shap.router)

# domains.py has catch-all /domains/{domain}/* — mount last
app.include_router(domains.router)
