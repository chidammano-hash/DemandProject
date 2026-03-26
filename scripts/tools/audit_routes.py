#!/usr/bin/env python3
"""Route registry audit — checks consistency between API routers and Vite proxy.

Detects:
1. Router files not mounted in main.py
2. API prefixes missing from vite.config.ts proxy
3. domains.py not mounted last
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

def get_router_files():
    """Find all router .py files (excluding __init__.py)."""
    router_dir = ROOT / "api" / "routers"
    return sorted(
        p for p in router_dir.rglob("*.py")
        if p.name != "__init__.py"
    )

def get_mounted_routers():
    """Parse app.include_router() calls from main.py."""
    main_py = ROOT / "api" / "main.py"
    text = main_py.read_text()
    pattern = re.compile(r'app\.include_router\(([^,)]+)')
    return pattern.findall(text)

def get_api_prefixes():
    """Extract API path prefixes from main.py include_router calls."""
    main_py = ROOT / "api" / "main.py"
    text = main_py.read_text()
    pattern = re.compile(r'prefix\s*=\s*["\'](/[^"\']+)["\']')
    return sorted(set(pattern.findall(text)))

def get_vite_proxies():
    """Extract proxy entries from vite.config.ts."""
    vite_config = ROOT / "frontend" / "vite.config.ts"
    if not vite_config.exists():
        return []
    text = vite_config.read_text()
    pattern = re.compile(r'["\'](/[^"\']+)["\']:\s*\{')
    return sorted(set(pattern.findall(text)))

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

    # 1. Check router file count vs mounted count
    router_files = get_router_files()
    mounted = get_mounted_routers()
    print(f"Router files: {len(router_files)}")
    print(f"Mounted routers: {len(mounted)}")
    if len(router_files) > len(mounted) + 5:
        print(f"  WARNING: {len(router_files) - len(mounted)} router files may not be mounted")
        issues += 1
    print()

    # 2. Check Vite proxy coverage
    api_prefixes = get_api_prefixes()
    vite_proxies = get_vite_proxies()
    missing_proxies = [p for p in api_prefixes if p not in vite_proxies and not p.startswith("/domains")]
    if missing_proxies:
        print(f"API prefixes MISSING from vite.config.ts proxy ({len(missing_proxies)}):")
        for p in missing_proxies:
            print(f"  - {p}")
        issues += len(missing_proxies)
    else:
        print("All API prefixes have Vite proxy entries.")
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
