"""Smoke tests for all routers — validates imports, route registration, and shim integrity.

These tests catch structural regressions from the router restructure (flat root
to domain subdirectories with backward-compat shims). They run in <1s and
do not require a database.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

# Domain subdirectories that should contain non-shim router modules.
_DOMAIN_PACKAGES = (
    "api.routers.inventory",
    "api.routers.forecasting",
    "api.routers.operations",
    "api.routers.platform",
    "api.routers.intelligence",
    "api.routers.core",
)

# Modules legitimately allowed to live at the flat `api/routers/` root
# (not shims, not in a domain subdir). Add new entries here when introducing
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


def test_flat_router_shims_resolve_to_domain_modules():
    """Backward-compat shims at api.routers.<name> re-export the moved module."""
    # Sample a handful from each domain to confirm sys.modules redirection works.
    cases = [
        ("api.routers.inv_planning_eoq", "api.routers.inventory.inv_planning_eoq"),
        (
            "api.routers.inv_planning_safety_stock",
            "api.routers.inventory.inv_planning_safety_stock",
        ),
        ("api.routers.competition", "api.routers.forecasting.competition"),
        ("api.routers.shap", "api.routers.forecasting.shap"),
        ("api.routers.control_tower", "api.routers.operations.control_tower"),
        ("api.routers.supply", "api.routers.operations.supply"),
        ("api.routers.config_manager", "api.routers.platform.config_manager"),
        ("api.routers.notifications", "api.routers.platform.notifications"),
        ("api.routers.intel", "api.routers.intelligence.intel"),
        ("api.routers.storyboard", "api.routers.intelligence.storyboard"),
        ("api.routers.clusters", "api.routers.forecasting.clusters"),
        ("api.routers.financial_plan", "api.routers.operations.financial_plan"),
    ]
    for shim_path, expected_path in cases:
        mod = importlib.import_module(shim_path)
        assert mod.__name__ == expected_path, (
            f"shim {shim_path} should redirect to {expected_path}, got {mod.__name__}"
        )


def test_no_unmoved_router_at_flat_root():
    """All flat api/routers/*.py (except domains.py and __init__.py) should be shims."""
    import api.routers as flat_pkg

    non_shim: list[str] = []
    for _, modname, ispkg in pkgutil.iter_modules(flat_pkg.__path__):
        if ispkg or modname in _FLAT_ROOT_ALLOWLIST:
            continue
        full = f"api.routers.{modname}"
        mod = importlib.import_module(full)
        # A shim's __name__ is rewritten via sys.modules[__name__] = _moved,
        # so the imported module's __name__ won't match the requested path.
        if mod.__name__ == full:
            non_shim.append(full)
    assert not non_shim, (
        f"flat router(s) still at root (move to domain subdir + add shim, "
        f"or add to _FLAT_ROOT_ALLOWLIST if intentional): {non_shim}"
    )
