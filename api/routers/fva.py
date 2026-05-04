"""Backward-compat shim — fva.py moved to api.routers.forecasting.fva.

Importing this module returns the same module object as
``api.routers.forecasting.fva`` so ``patch("api.routers.fva.xxx")`` and
``from api.routers.fva import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.forecasting import fva as _moved

_sys.modules[__name__] = _moved
