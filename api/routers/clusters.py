"""Backward-compat shim — clusters.py moved to api.routers.forecasting.clusters.

Importing this module returns the same module object as
``api.routers.forecasting.clusters`` so ``patch("api.routers.clusters.xxx")`` and
``from api.routers.clusters import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.forecasting import clusters as _moved

_sys.modules[__name__] = _moved
