"""Backward-compat shim — reports.py moved to api.routers.platform.reports.

Importing this module returns the same module object as
``api.routers.platform.reports`` so ``patch("api.routers.reports.xxx")`` and
``from api.routers.reports import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.platform import reports as _moved

_sys.modules[__name__] = _moved
