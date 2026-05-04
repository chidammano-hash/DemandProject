"""Backward-compat shim — lead_time_learning.py moved to api.routers.inventory.lead_time_learning.

Importing this module returns the same module object as
``api.routers.inventory.lead_time_learning`` so ``patch("api.routers.lead_time_learning.xxx")`` and
``from api.routers.lead_time_learning import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.inventory import lead_time_learning as _moved

_sys.modules[__name__] = _moved
