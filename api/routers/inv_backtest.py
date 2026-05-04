"""Backward-compat shim — inv_backtest.py moved to api.routers.inventory.inv_backtest.

Importing this module returns the same module object as
``api.routers.inventory.inv_backtest`` so ``patch("api.routers.inv_backtest.xxx")`` and
``from api.routers.inv_backtest import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.inventory import inv_backtest as _moved

_sys.modules[__name__] = _moved
