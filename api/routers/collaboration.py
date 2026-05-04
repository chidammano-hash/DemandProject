"""Backward-compat shim — collaboration.py moved to api.routers.platform.collaboration.

Importing this module returns the same module object as
``api.routers.platform.collaboration`` so ``patch("api.routers.collaboration.xxx")`` and
``from api.routers.collaboration import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.platform import collaboration as _moved

_sys.modules[__name__] = _moved
