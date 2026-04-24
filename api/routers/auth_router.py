"""Backward-compat shim — auth_router.py moved to api.routers.platform.auth_router."""
from __future__ import annotations

import sys as _sys

from api.routers.platform import auth_router as _moved

_sys.modules[__name__] = _moved
