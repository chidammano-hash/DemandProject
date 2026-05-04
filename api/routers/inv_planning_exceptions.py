"""Backward-compat shim — inv_planning_exceptions.py moved to api.routers.inventory.inv_planning_exceptions.

Importing this module returns the same module object as
``api.routers.inventory.inv_planning_exceptions`` so ``patch("api.routers.inv_planning_exceptions.xxx")`` and
``from api.routers.inv_planning_exceptions import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.inventory import inv_planning_exceptions as _moved

_sys.modules[__name__] = _moved
