"""Compatibility shim: re-exports from split inv_planning_* routers.

This file exists so that test patches like
``patch("api.routers.inv_planning.open", ...)`` continue to resolve.
All actual route handlers live in the domain-specific router modules.
"""
from __future__ import annotations

# Re-export routers so any code that does
#   from api.routers import inv_planning; inv_planning.router
# still works (though main.py no longer mounts this router).
from api.routers.inv_planning_variability import router as _v_router  # noqa: F401
from api.routers.inv_planning_eoq import router as router  # noqa: F401

# Expose open so patch("api.routers.inv_planning.open", ...) resolves.
open = open  # noqa: A001, F841
