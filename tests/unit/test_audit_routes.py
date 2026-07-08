import importlib.util

from common.core.paths import PROJECT_ROOT


def _load_audit_routes_module():
    module_path = PROJECT_ROOT / "scripts" / "tools" / "audit_routes.py"
    spec = importlib.util.spec_from_file_location("audit_routes", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_package_subrouters_handles_multiline_relative_imports(tmp_path):
    init_file = tmp_path / "__init__.py"
    init_file.write_text(
        """
from fastapi import APIRouter

from . import (
    detail,
    list as list_module,
    promote_results,
)

router = APIRouter()
router.include_router(list_module.router)
router.include_router(detail.router)
router.include_router(promote_results.router)
""",
    )

    audit_routes = _load_audit_routes_module()

    assert audit_routes._collect_package_subrouters(init_file) == {
        str(tmp_path / "detail.py"),
        str(tmp_path / "list.py"),
        str(tmp_path / "promote_results.py"),
    }
