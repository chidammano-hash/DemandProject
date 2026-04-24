"""Backward-compat shim — dashboard.py moved to api.routers.core.dashboard."""
from __future__ import annotations

import sys as _sys

from api.routers.core import dashboard as _moved

_sys.modules[__name__] = _moved
