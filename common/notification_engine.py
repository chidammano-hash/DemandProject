# common/notification_engine.py — backward-compatible shim
# Real implementation moved to common/services/notification_engine.py
import sys as _sys
import types as _types
import common.services.notification_engine as _real  # noqa: F401

_this = _sys.modules[__name__]

# Copy all non-dunder attributes from real module
for _attr in dir(_real):
    if not _attr.startswith("__"):
        setattr(_this, _attr, getattr(_real, _attr))


class _ShimModule(_types.ModuleType):
    """Module subclass that propagates attribute writes to the real module.

    This ensures unittest.mock.patch("common.notification_engine.X", ...) also
    sets X on common.services.notification_engine, where the actual functions
    read their globals from.
    """

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if not name.startswith("__"):
            setattr(_real, name, value)

    def __delattr__(self, name):
        super().__delattr__(name)
        if hasattr(_real, name):
            delattr(_real, name)


_shim = _ShimModule(__name__)
_shim.__dict__.update(_this.__dict__)
_shim.__file__ = __file__
_shim.__loader__ = getattr(_this, "__loader__", None)
_shim.__spec__ = getattr(_this, "__spec__", None)
_shim.__path__ = getattr(_this, "__path__", [])
_sys.modules[__name__] = _shim
