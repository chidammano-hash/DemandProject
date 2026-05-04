"""Backward-compat shim — medallion.py moved to api.routers.platform.medallion.

Importing this module returns the same module object as
``api.routers.platform.medallion`` so ``patch("api.routers.medallion.xxx")`` and
``from api.routers.medallion import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.platform import medallion as _moved

_sys.modules[__name__] = _moved
