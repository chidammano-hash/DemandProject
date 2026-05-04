"""Backward-compat shim — notifications.py moved to api.routers.platform.notifications.

Importing this module returns the same module object as
``api.routers.platform.notifications`` so ``patch("api.routers.notifications.xxx")`` and
``from api.routers.notifications import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.platform import notifications as _moved

_sys.modules[__name__] = _moved
