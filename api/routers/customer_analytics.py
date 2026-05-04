"""Backward-compat shim — customer_analytics.py moved to api.routers.intelligence.customer_analytics.

Importing this module returns the same module object as
``api.routers.intelligence.customer_analytics`` so ``patch("api.routers.customer_analytics.xxx")`` and
``from api.routers.customer_analytics import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.intelligence import customer_analytics as _moved

_sys.modules[__name__] = _moved
