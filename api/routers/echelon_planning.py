"""Backward-compat shim — echelon_planning.py moved to api.routers.operations.echelon_planning.

Importing this module returns the same module object as
``api.routers.operations.echelon_planning`` so ``patch("api.routers.echelon_planning.xxx")`` and
``from api.routers.echelon_planning import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.operations import echelon_planning as _moved

_sys.modules[__name__] = _moved
