"""Backward-compat shim — external_signals.py moved to api.routers.intelligence.external_signals.

Importing this module returns the same module object as
``api.routers.intelligence.external_signals`` so ``patch("api.routers.external_signals.xxx")`` and
``from api.routers.external_signals import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.intelligence import external_signals as _moved

_sys.modules[__name__] = _moved
