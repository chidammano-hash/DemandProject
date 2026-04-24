"""Backward-compat shim — users.py moved to api.routers.platform.users."""
from __future__ import annotations

import sys as _sys

from api.routers.platform import users as _moved

_sys.modules[__name__] = _moved
