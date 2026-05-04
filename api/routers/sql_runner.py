"""Backward-compat shim — sql_runner.py moved to api.routers.platform.sql_runner.

Importing this module returns the same module object as
``api.routers.platform.sql_runner`` so ``patch("api.routers.sql_runner.xxx")`` and
``from api.routers.sql_runner import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.platform import sql_runner as _moved

_sys.modules[__name__] = _moved
