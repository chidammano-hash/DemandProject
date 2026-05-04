"""Backward-compat shim — data_quality.py moved to api.routers.platform.data_quality.

Importing this module returns the same module object as
``api.routers.platform.data_quality`` so ``patch("api.routers.data_quality.xxx")`` and
``from api.routers.data_quality import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.platform import data_quality as _moved

_sys.modules[__name__] = _moved
