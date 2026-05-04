"""Backward-compat shim — storyboard.py moved to api.routers.intelligence.storyboard.

Importing this module returns the same module object as
``api.routers.intelligence.storyboard`` so ``patch("api.routers.storyboard.xxx")`` and
``from api.routers.storyboard import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.intelligence import storyboard as _moved

_sys.modules[__name__] = _moved
