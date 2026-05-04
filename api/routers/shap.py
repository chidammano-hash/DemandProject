"""Backward-compat shim — shap.py moved to api.routers.forecasting.shap.

Importing this module returns the same module object as
``api.routers.forecasting.shap`` so ``patch("api.routers.shap.xxx")`` and
``from api.routers.shap import yyy`` continue to work.
"""

from __future__ import annotations

import sys as _sys

from api.routers.forecasting import shap as _moved

_sys.modules[__name__] = _moved
