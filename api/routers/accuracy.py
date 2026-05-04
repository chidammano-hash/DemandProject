"""Backward-compat shim — accuracy.py moved to api.routers.forecasting.accuracy.

Importing this module returns the same module object as
``api.routers.forecasting.accuracy`` so ``patch("api.routers.accuracy.xxx")`` and
``from api.routers.accuracy import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.forecasting import accuracy as _moved

_sys.modules[__name__] = _moved
