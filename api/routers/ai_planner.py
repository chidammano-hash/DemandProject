"""Backward-compat shim — ai_planner.py moved to api.routers.intelligence.ai_planner."""
from __future__ import annotations

import sys as _sys

from api.routers.intelligence import ai_planner as _moved

_sys.modules[__name__] = _moved
