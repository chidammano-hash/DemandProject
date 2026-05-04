"""Backward-compat shim — analysis.py moved to api.routers.forecasting.analysis.

Importing this module returns the same module object as
``api.routers.forecasting.analysis`` so ``patch("api.routers.analysis.xxx")`` and
``from api.routers.analysis import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.forecasting import analysis as _moved

_sys.modules[__name__] = _moved
