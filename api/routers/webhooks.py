"""Backward-compat shim — webhooks.py moved to api.routers.platform.webhooks.

Importing this module returns the same module object as
``api.routers.platform.webhooks`` so ``patch("api.routers.webhooks.xxx")`` and
``from api.routers.webhooks import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.platform import webhooks as _moved

_sys.modules[__name__] = _moved
