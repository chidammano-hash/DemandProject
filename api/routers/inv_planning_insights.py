"""Backward-compat shim — inv_planning_insights.py moved to api.routers.inventory.inv_planning_insights.

Importing this module returns the same module object as
``api.routers.inventory.inv_planning_insights`` so ``patch("api.routers.inv_planning_insights.xxx")`` and
``from api.routers.inv_planning_insights import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.inventory import inv_planning_insights as _moved

_sys.modules[__name__] = _moved
