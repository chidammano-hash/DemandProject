"""Backward-compat shim — events.py moved to api.routers.operations.events.

Importing this module returns the same module object as
``api.routers.operations.events`` so ``patch("api.routers.events.xxx")`` and
``from api.routers.events import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.operations import events as _moved

_sys.modules[__name__] = _moved
