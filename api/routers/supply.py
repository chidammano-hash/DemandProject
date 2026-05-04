"""Backward-compat shim — supply.py moved to api.routers.operations.supply.

Importing this module returns the same module object as
``api.routers.operations.supply`` so ``patch("api.routers.supply.xxx")`` and
``from api.routers.supply import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.operations import supply as _moved

_sys.modules[__name__] = _moved
