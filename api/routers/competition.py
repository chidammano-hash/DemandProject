"""Backward-compat shim — competition.py moved to api.routers.forecasting.competition.

Importing this module returns the same module object as
``api.routers.forecasting.competition`` so ``patch("api.routers.competition.xxx")`` and
``from api.routers.competition import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.forecasting import competition as _moved

_sys.modules[__name__] = _moved
