"""Backward-compat shim — service_level.py moved to api.routers.operations.service_level.

Importing this module returns the same module object as
``api.routers.operations.service_level`` so ``patch("api.routers.service_level.xxx")`` and
``from api.routers.service_level import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.operations import service_level as _moved

_sys.modules[__name__] = _moved
