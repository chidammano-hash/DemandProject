"""Backward-compat shim — inv_planning_eoq.py moved to api.routers.inventory.inv_planning_eoq.

Importing this module returns the same module object as
``api.routers.inventory.inv_planning_eoq`` so ``patch("api.routers.inv_planning_eoq.xxx")`` and
``from api.routers.inv_planning_eoq import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.inventory import inv_planning_eoq as _moved

_sys.modules[__name__] = _moved
