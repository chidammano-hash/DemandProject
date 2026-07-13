#!/usr/bin/env python3
"""Route registry audit — checks API router, Vite, and Nginx consistency.

Detects:
1. Router files not mounted in main.py
2. API prefixes missing from Vite or Nginx proxies
3. Nginx proxy matching bare collection paths as well as child paths
4. domains.py not mounted last
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "api" / "main.py").exists():
    raise RuntimeError("Run audit_routes.py from the DemandProject repository root")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _collect_package_subrouters(init_file: Path) -> set[str]:
    """Find sibling router modules aggregated by a package router."""
    text = init_file.read_text()
    imported_modules: dict[str, str] = {}

    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None:
            for alias in node.names:
                imported_modules[alias.asname or alias.name] = alias.name

    included_aliases = set(
        re.findall(r"router\.include_router\((\w+)\.router\)", text)
    )
    return {
        str(init_file.parent / f"{imported_modules[alias]}.py")
        for alias in included_aliases
        if alias in imported_modules
    }


def get_router_files():
    """Find all router .py files (excluding __init__.py).

    Files whose sole role is to be re-exported by a package ``__init__.py``
    (e.g. ``api/routers/inventory/inventory_main.py`` which is re-exported
    as ``api.routers.inventory.router``) are excluded so they don't count
    as unmounted.  We detect that case by scanning sibling ``__init__.py``
    files for a matching ``from .<stem> import router`` line.
    """
    router_dir = ROOT / "api" / "routers"
    reexported: set[str] = set()
    # Match either ``from .<stem> import router`` (relative) or
    # ``from api.routers.<pkg>.<stem> import router`` (absolute).
    rel_pat = re.compile(r"from\s+\.(\w+)\s+import\s+router")
    abs_pat = re.compile(r"from\s+api\.routers\.[\w.]+\.(\w+)\s+import\s+router")
    for init in router_dir.rglob("__init__.py"):
        text = init.read_text()
        reexported.update(_collect_package_subrouters(init))
        for m in rel_pat.finditer(text):
            reexported.add(str(init.parent / f"{m.group(1)}.py"))
        for m in abs_pat.finditer(text):
            reexported.add(str(init.parent / f"{m.group(1)}.py"))
    # Also handle the ``from api.routers.<pkg>.<stem> import router`` form
    # where the <pkg> is in a different directory than the __init__.  For
    # inventory_main this is used via ``from api.routers.inventory.inventory_main``.
    # Detect by scanning all __init__.py for any absolute import and mapping
    # to the last component of the module path.
    abs_pat_any = re.compile(r"from\s+api\.routers\.(\w+)\.(\w+)\s+import\s+router")
    for init in router_dir.rglob("__init__.py"):
        text = init.read_text()
        for m in abs_pat_any.finditer(text):
            pkg, stem = m.group(1), m.group(2)
            reexported.add(str(router_dir / pkg / f"{stem}.py"))

    return sorted(
        p for p in router_dir.rglob("*.py")
        if p.name != "__init__.py"
        and not p.name.startswith("_")
        and str(p) not in reexported
    )

def get_mounted_routers():
    """Parse app.include_router() calls from main.py."""
    main_py = ROOT / "api" / "main.py"
    text = main_py.read_text()
    pattern = re.compile(r'app\.include_router\(([^,)]+)')
    return pattern.findall(text)

def get_api_prefixes() -> list[str]:
    """Return concrete top-level prefixes from the assembled FastAPI app."""
    from api.main import app

    ignored = {"/docs", "/openapi.json", "/redoc", "/"}
    prefixes: set[str] = set()
    for route in app.routes:
        path = getattr(route, "path", "")
        if not path or path.rstrip("/") in {value.rstrip("/") for value in ignored}:
            continue
        first_segment = path.strip("/").split("/", 1)[0]
        if not first_segment or first_segment.startswith("{") or first_segment == "docs":
            continue
        prefixes.add(f"/{first_segment}")
    return sorted(prefixes)

def get_vite_proxies():
    """Extract proxy entries from vite.config.ts.

    Supports both the array-driven form (Gen-4 Coder-2 P0 refactor):
        const API_PATH_PREFIXES = ["/foo", "/bar", ...]
    and the legacy per-prefix block form:
        "/foo": { target: ..., changeOrigin: true }
    """
    vite_config = ROOT / "frontend" / "vite.config.ts"
    if not vite_config.exists():
        return []
    text = vite_config.read_text()

    array_match = re.search(r"API_PATH_PREFIXES[^=]*=\s*\[(.*?)\]", text, re.DOTALL)
    if array_match:
        prefixes = re.findall(r'"(/[^"]+)"', array_match.group(1))
        return sorted(set(prefixes))

    # Legacy fallback
    pattern = re.compile(r'["\'](/[^"\']+)["\']:\s*\{')
    return sorted(set(pattern.findall(text)))


def get_nginx_proxies() -> tuple[list[str], bool]:
    """Extract top-level API prefixes and boundary safety from Nginx config."""
    nginx_config = ROOT / "frontend" / "nginx.conf"
    if not nginx_config.exists():
        return [], False
    text = nginx_config.read_text()
    match = re.search(r"location\s+~\s+\^/\(([^)]+)\)([^\s{]*)", text)
    if not match:
        return [], False
    prefixes = sorted({f"/{value}" for value in match.group(1).split("|")})
    suffix = match.group(2)
    matches_bare_prefix = "$" in suffix
    return prefixes, matches_bare_prefix

def check_domains_last():
    """Verify domains.py is mounted last in main.py."""
    main_py = ROOT / "api" / "main.py"
    lines = main_py.read_text().splitlines()
    last_router_line = 0
    domains_line = 0
    for i, line in enumerate(lines, 1):
        if "include_router" in line:
            last_router_line = i
            if "domains" in line.lower():
                domains_line = i
    return domains_line == last_router_line

def main():
    issues = 0

    # 1. Check router file count vs mounted count (per-file verification).
    router_files = get_router_files()
    mounted_refs = get_mounted_routers()
    # Extract the leading identifier of each include_router() reference
    # (everything before the first '.'), which is the imported module alias.
    mounted_module_aliases = {ref.strip().split(".")[0] for ref in mounted_refs}

    # A router file is "mounted" if its stem appears in the mounted module
    # aliases OR if the alias matches any ``import ... as <alias>`` / direct
    # import found in main.py (to handle ``from ... import X as Y``).
    main_py_text = (ROOT / "api" / "main.py").read_text()
    import_alias_pat = re.compile(
        r"from\s+api\.routers(?:\.[\w.]+)?\s+import\s+(\w+)(?:\s+as\s+(\w+))?"
    )
    alias_to_stem: dict[str, str] = {}
    for m in import_alias_pat.finditer(main_py_text):
        imported, alias = m.group(1), m.group(2)
        alias_to_stem[alias or imported] = imported

    unmounted = []
    for rf in router_files:
        stem = rf.stem
        # Is any mounted alias's underlying import == this stem?
        mounted_via_alias = any(
            alias in mounted_module_aliases and alias_to_stem.get(alias) == stem
            for alias in alias_to_stem
        )
        if stem not in mounted_module_aliases and not mounted_via_alias:
            unmounted.append(rf)

    print(f"Router files: {len(router_files)}")
    print(f"Mounted routers: {len(mounted_refs)}")
    if unmounted:
        print(f"  WARNING: {len(unmounted)} router files appear UNMOUNTED:")
        for rf in unmounted:
            print(f"    - {rf.relative_to(ROOT)}")
        issues += len(unmounted)
    print()

    # 2. Check development and production proxy coverage.
    api_prefixes = get_api_prefixes()
    vite_proxies = get_vite_proxies()
    nginx_proxies, nginx_matches_bare = get_nginx_proxies()
    dynamic_prefixes = {"/domains"}
    missing_vite = [p for p in api_prefixes if p not in vite_proxies and p not in dynamic_prefixes]
    missing_nginx = [p for p in api_prefixes if p not in nginx_proxies and p not in dynamic_prefixes]
    if missing_vite:
        print(f"API prefixes MISSING from vite.config.ts proxy ({len(missing_vite)}):")
        for p in missing_vite:
            print(f"  - {p}")
        issues += len(missing_vite)
    else:
        print("All API prefixes have Vite proxy entries.")
    if missing_nginx:
        print(f"API prefixes MISSING from nginx.conf proxy ({len(missing_nginx)}):")
        for p in missing_nginx:
            print(f"  - {p}")
        issues += len(missing_nginx)
    else:
        print("All API prefixes have Nginx proxy entries.")
    if not nginx_matches_bare:
        print("WARNING: Nginx API matcher does not cover bare collection paths.")
        issues += 1
    print()

    # 3. Check domains.py mounted last
    if check_domains_last():
        print("domains.py is mounted last.")
    else:
        print("WARNING: domains.py is NOT mounted last! This will shadow other routes.")
        issues += 1

    print(f"\nTotal issues: {issues}")
    return 1 if issues > 0 else 0

if __name__ == "__main__":
    sys.exit(main())
