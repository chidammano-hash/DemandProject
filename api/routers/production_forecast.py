"""Backward-compat shim — production_forecast.py moved to api.routers.forecasting.production_forecast.

Importing this module returns the same module object as
``api.routers.forecasting.production_forecast`` so ``patch("api.routers.production_forecast.xxx")`` and
``from api.routers.production_forecast import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.forecasting import production_forecast as _moved

_sys.modules[__name__] = _moved
