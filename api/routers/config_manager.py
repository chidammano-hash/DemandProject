"""Backward-compat shim — config_manager.py moved to api.routers.platform.config_manager.

Importing this module returns the same module object as
``api.routers.platform.config_manager`` so ``patch("api.routers.config_manager.xxx")`` and
``from api.routers.config_manager import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.platform import config_manager as _moved

_sys.modules[__name__] = _moved
