"""Backward-compat shim — inv_planning_investment.py moved to api.routers.inventory.inv_planning_investment.

Importing this module returns the same module object as
``api.routers.inventory.inv_planning_investment`` so ``patch("api.routers.inv_planning_investment.xxx")`` and
``from api.routers.inv_planning_investment import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.inventory import inv_planning_investment as _moved

_sys.modules[__name__] = _moved
