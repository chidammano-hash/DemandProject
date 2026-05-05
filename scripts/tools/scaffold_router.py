#!/usr/bin/env python3
"""Scaffold a new API router with all required wiring.

Usage:
    python scripts/tools/scaffold_router.py --domain forecasting --name consensus_variance
    python scripts/tools/scaffold_router.py --domain inventory --name rebalancing_v2 --prefix /inv-rebalance

Creates:
1. Router file in api/routers/{domain}/
2. Test file in tests/api/
3. Prints instructions for main.py and vite.config.ts wiring
"""
import argparse
import sys
from pathlib import Path

from common.core.paths import PROJECT_ROOT as ROOT
ROUTER_TEMPLATE = '''"""Router for {title}.

Part of the {domain} domain.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter

from api.core import get_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/{prefix}", tags=["{tag}"])


@router.get("/")
async def list_{name}(limit: int = 50, offset: int = 0):
    """List {title} records with pagination."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM {name}")
        total = cur.fetchone()[0]
        cur.execute(
            "SELECT * FROM {name} ORDER BY 1 LIMIT %s OFFSET %s",
            (limit, offset),
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return {{
        "total": total,
        "rows": [dict(zip(cols, r)) for r in rows],
    }}
'''

TEST_TEMPLATE = '''"""Tests for {name} router."""
import pytest
from unittest.mock import patch

import httpx
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


def _make_pool_helper(**kwargs):
    return _make_pool(**kwargs)


@pytest.mark.asyncio
async def test_list_{name}_empty():
    pool, conn, cursor = _make_pool_helper(
        fetchone_return=(0,),
        fetchall_return=[],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/{prefix}/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["rows"] == []
'''

def main():
    parser = argparse.ArgumentParser(description="Scaffold a new API router")
    parser.add_argument("--domain", required=True, choices=[
        "inventory", "forecasting", "operations", "platform", "intelligence", "core"
    ])
    parser.add_argument("--name", required=True, help="Router name (snake_case)")
    parser.add_argument("--prefix", help="API prefix (default: derived from name)")
    args = parser.parse_args()

    name = args.name
    domain = args.domain
    prefix = args.prefix or name.replace("_", "-")
    title = name.replace("_", " ").title()
    tag = f"{domain}/{name}"

    # Create router file
    router_dir = ROOT / "api" / "routers" / domain
    router_file = router_dir / f"{name}.py"
    if router_file.exists():
        print(f"ERROR: {router_file} already exists!")
        return 1
    router_file.write_text(ROUTER_TEMPLATE.format(
        title=title, domain=domain, prefix=prefix, tag=tag, name=name,
    ))
    print(f"Created: {router_file}")

    # Create test file
    test_file = ROOT / "tests" / "api" / f"test_{name}.py"
    if test_file.exists():
        print(f"WARNING: {test_file} already exists, skipping")
    else:
        test_file.write_text(TEST_TEMPLATE.format(
            name=name, prefix=prefix, title=title,
        ))
        print(f"Created: {test_file}")

    # Print wiring instructions
    print(f"\n--- Manual wiring steps ---")
    print(f"1. Add to api/main.py (BEFORE domains.py mount):")
    print(f'   from api.routers.{domain}.{name} import router as {name}_router')
    print(f'   app.include_router({name}_router)')
    print(f"")
    print(f"2. Add to frontend/vite.config.ts proxy:")
    print(f'   "/{prefix}": {{ target: "http://127.0.0.1:8000", changeOrigin: true }},')
    print(f"")
    print(f"3. Run: make audit-routers (to verify wiring)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
