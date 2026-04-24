"""Backward-compat shim — jobs.py moved to api.routers.core.jobs."""
from __future__ import annotations

import sys as _sys

from api.routers.core import jobs as _moved

_sys.modules[__name__] = _moved
