"""Backward-compat shim — bias_corrections.py moved to api.routers.forecasting.bias_corrections.

Importing this module returns the same module object as
``api.routers.forecasting.bias_corrections`` so ``patch("api.routers.bias_corrections.xxx")`` and
``from api.routers.bias_corrections import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.forecasting import bias_corrections as _moved

_sys.modules[__name__] = _moved
