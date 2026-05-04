"""Backward-compat shim — consensus_plan.py moved to api.routers.forecasting.consensus_plan.

Importing this module returns the same module object as
``api.routers.forecasting.consensus_plan`` so ``patch("api.routers.consensus_plan.xxx")`` and
``from api.routers.consensus_plan import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.forecasting import consensus_plan as _moved

_sys.modules[__name__] = _moved
