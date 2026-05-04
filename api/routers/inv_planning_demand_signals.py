"""Backward-compat shim — inv_planning_demand_signals.py moved to api.routers.inventory.inv_planning_demand_signals.

Importing this module returns the same module object as
``api.routers.inventory.inv_planning_demand_signals`` so ``patch("api.routers.inv_planning_demand_signals.xxx")`` and
``from api.routers.inv_planning_demand_signals import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.inventory import inv_planning_demand_signals as _moved

_sys.modules[__name__] = _moved
