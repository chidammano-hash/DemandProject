"""Backward-compat shim — financial_plan.py moved to api.routers.operations.financial_plan.

Importing this module returns the same module object as
``api.routers.operations.financial_plan`` so ``patch("api.routers.financial_plan.xxx")`` and
``from api.routers.financial_plan import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.operations import financial_plan as _moved

_sys.modules[__name__] = _moved
