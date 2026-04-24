"""Backward-compat shim — sop.py moved to api.routers.operations.sop.

Importing this module returns the same module object as
``api.routers.operations.sop`` so ``patch("api.routers.sop.xxx")`` and
``from api.routers.sop import yyy`` continue to work.
"""
from __future__ import annotations

import sys as _sys

from api.routers.operations import sop as _moved

_sys.modules[__name__] = _moved
