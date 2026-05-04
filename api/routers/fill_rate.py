"""Backward-compat shim — fill_rate.py moved to api.routers.operations.fill_rate.

Importing this module returns the same module object as
``api.routers.operations.fill_rate`` so ``patch("api.routers.fill_rate.xxx")`` and
``from api.routers.fill_rate import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.operations import fill_rate as _moved

_sys.modules[__name__] = _moved
