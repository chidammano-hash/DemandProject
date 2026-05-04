"""Backward-compat shim — control_tower.py moved to api.routers.operations.control_tower.

Importing this module returns the same module object as
``api.routers.operations.control_tower`` so ``patch("api.routers.control_tower.xxx")`` and
``from api.routers.control_tower import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.operations import control_tower as _moved

_sys.modules[__name__] = _moved
