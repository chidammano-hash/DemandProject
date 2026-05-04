"""Backward-compat shim — supply_scenarios.py moved to api.routers.operations.supply_scenarios.

Importing this module returns the same module object as
``api.routers.operations.supply_scenarios`` so ``patch("api.routers.supply_scenarios.xxx")`` and
``from api.routers.supply_scenarios import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.operations import supply_scenarios as _moved

_sys.modules[__name__] = _moved
