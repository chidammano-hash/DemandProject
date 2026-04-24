"""Backward-compat shim — chat.py moved to api.routers.intelligence.chat."""
from __future__ import annotations

import sys as _sys

from api.routers.intelligence import chat as _moved

_sys.modules[__name__] = _moved
