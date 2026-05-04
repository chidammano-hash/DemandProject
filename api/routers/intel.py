"""Backward-compat shim — intel.py moved to api.routers.intelligence.intel.

Importing this module returns the same module object as
``api.routers.intelligence.intel`` so ``patch("api.routers.intel.xxx")`` and
``from api.routers.intel import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.intelligence import intel as _moved

_sys.modules[__name__] = _moved
