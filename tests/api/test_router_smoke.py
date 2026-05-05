"""Smoke tests for all routers — validates imports, route registration, and structural layout.

These tests catch structural regressions from the router restructure (flat root
to domain subdirectories). They run in <1s and do not require a database.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

# Domain subdirectories that should contain all router modules.
_DOMAIN_PACKAGES = (
    "api.routers.inventory",
    "api.routers.forecasting",
    "api.routers.operations",
    "api.routers.platform",
    "api.routers.intelligence",
    "api.routers.core",
)

# Modules legitimately allowed to live at the flat `api/routers/` root
# (not in a domain subdir). Add new entries here when introducing
# intentional flat-root routers (e.g. catch-all handlers).
_FLAT_ROOT_ALLOWLIST = frozenset(
    {
        "domains",  # catch-all generic domain endpoint, mounted last
    }
)


@pytest.mark.parametrize("pkg_name", _DOMAIN_PACKAGES)
def test_domain_package_modules_import_and_have_router(pkg_name: str):
    """Every module in each domain package imports cleanly and exposes ``router``."""
    pkg = importlib.import_module(pkg_name)
    failures: list[str] = []
    for _, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
        if ispkg or modname.startswith("_"):
            continue
        full = f"{pkg_name}.{modname}"
        mod = importlib.import_module(full)
        if not hasattr(mod, "router"):
            failures.append(f"{full}: missing 'router' attribute")
    assert not failures, "\n".join(failures)


def test_app_includes_routers_under_each_domain():
    """The main app registers a wide range of distinct route tags.

    Catches regressions where main.py drops imports during the restructure.
    """
    from api.main import app

    seen_tags: set[str] = set()
    for r in app.router.routes:
        tags = getattr(r, "tags", None) or []
        for t in tags:
            seen_tags.add(t.lower())

    assert len(seen_tags) >= 30, (
        f"only {len(seen_tags)} distinct tags found: {sorted(seen_tags)[:10]}..."
    )


def test_only_allowlisted_files_at_flat_root():
    """No router files should live at the flat api/routers/ root except allowlisted ones.

    All concrete routers must live in a domain subdir
    (inventory/, forecasting/, operations/, platform/, intelligence/, core/).
    """
    import api.routers as flat_pkg

    unexpected: list[str] = []
    for _, modname, ispkg in pkgutil.iter_modules(flat_pkg.__path__):
        if ispkg or modname in _FLAT_ROOT_ALLOWLIST:
            continue
        unexpected.append(f"api.routers.{modname}")
    assert not unexpected, (
        f"flat router(s) at api/routers/ root (move to a domain subdir, "
        f"or add to _FLAT_ROOT_ALLOWLIST if intentional): {unexpected}"
    )
