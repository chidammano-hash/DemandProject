"""Backward-compat shim — blended_forecast.py moved to api.routers.forecasting.blended_forecast.

Importing this module returns the same module object as
``api.routers.forecasting.blended_forecast`` so ``patch("api.routers.blended_forecast.xxx")`` and
``from api.routers.blended_forecast import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.forecasting import blended_forecast as _moved

_sys.modules[__name__] = _moved
