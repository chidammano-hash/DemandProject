"""Backward-compat shim — inv_planning_variability.py moved to api.routers.inventory.inv_planning_variability.

Importing this module returns the same module object as
``api.routers.inventory.inv_planning_variability`` so ``patch("api.routers.inv_planning_variability.xxx")`` and
``from api.routers.inv_planning_variability import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.inventory import inv_planning_variability as _moved

_sys.modules[__name__] = _moved
